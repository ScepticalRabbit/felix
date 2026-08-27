import subprocess
import sys
import shutil
import platform
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy

DIST_NAME = "felix"
PROJECT_ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------------------------
# Platform-specific utilities
# ------------------------------------------------------------------------------

def get_platform_info() -> dict[str, str]:
    """Return platform-specific shared-library extensions and RPATH tokens."""
    system = platform.system().lower()
    if system == "windows":
        return {
            "lib_ext": ".dll",
            "lib_prefix": "",
            "runtime_lib_dir": "",
        }
    elif system == "darwin":
        return {
            "lib_ext": ".dylib",
            "lib_prefix": "lib",
            "runtime_lib_dir": "@loader_path",
        }
    else:  # Linux and other Unix-like
        return {
            "lib_ext": ".so",
            "lib_prefix": "lib",
            "runtime_lib_dir": "$ORIGIN",
        }


PLATFORM_INFO = get_platform_info()


def lib_base_name(ext_full_name: str) -> str:
    return ext_full_name.rsplit(".", maxsplit=1)[-1]


def lib_link_name(ext_name: str) -> str:
    base = lib_base_name(ext_name)
    return (
        f"{PLATFORM_INFO['lib_prefix']}"
        f"{base}"
        f"{PLATFORM_INFO['lib_ext']}"
    )


def lib_link_aliases(
    ext_name: str,
    source_path: Path | None = None,
) -> list[str]:
    lib_names = {lib_link_name(ext_name)}
    if source_path is not None and source_path.suffix == ".zig":
        source_lib = (
            f"{PLATFORM_INFO['lib_prefix']}"
            f"{source_path.stem}"
            f"{PLATFORM_INFO['lib_ext']}"
        )
        lib_names.add(source_lib)
    return sorted(lib_names)


# ------------------------------------------------------------------------------
# Custom Multi-Build: builds both Zig and Cython extensions in one pass
# ------------------------------------------------------------------------------

class MultiBuildExt(build_ext):

    def run(self) -> None:
        print(80 * "=")
        print("MultiBuildExt: run pre-process")
        print(80 * "=")

        build_temp_path = Path(self.build_temp)
        build_temp_path.mkdir(exist_ok=True, parents=True)
        print(f"Temp build dir:\n    {build_temp_path}\n")

        build_lib_path = Path(self.build_lib)
        build_lib_path.mkdir(exist_ok=True, parents=True)
        print(f"Library build dir:\n    {build_lib_path}\n")

        is_windows = platform.system().lower() == "windows"
        if not is_windows:
            runtime_dir = PLATFORM_INFO["runtime_lib_dir"]
            if runtime_dir not in self.rpath:
                self.rpath.append(runtime_dir)

        # Collect all extension output directories
        ext_dirs = [
            str(
                Path(self.get_ext_fullpath(ee.name)).resolve().parent
            )
            for ee in self.extensions
        ]

        for dd in ext_dirs:
            if dd not in self.library_dirs:
                self.library_dirs.append(dd)
            if not is_windows and dd not in self.rpath:
                self.rpath.append(dd)
            for ee in self.extensions:
                if dd not in ee.library_dirs:
                    ee.library_dirs.append(dd)
                if not is_windows and dd not in ee.runtime_library_dirs:
                    ee.runtime_library_dirs.append(dd)

        for ee in self.extensions:
            print(80 * "-")
            print(f"Extension dirs pre-build: {ee.name}")
            print("  include_dirs:")
            for dd in ee.include_dirs:
                print(f"      {dd}")
            print("  library_dirs:")
            for dd in ee.library_dirs:
                print(f"      {dd}")
            print("  libraries:")
            for dd in ee.libraries:
                print(f"      {dd}")
            print()

        super().run()

        # When building inplace, copy zig lib aliases back into src/ so that
        # the cross-link step (and runtime loading) can find them.
        if self.inplace:
            for ee in self.extensions:
                if Path(ee.sources[0]).suffix != ".zig":
                    continue
                zig_src_path = (
                    Path(self.get_ext_fullpath(ee.name)).resolve()
                )
                zig_build_dir = (
                    Path(self.build_lib).resolve()
                    / self.get_ext_filename(ee.name)
                )
                zig_build_dir = zig_build_dir.parent
                for alias in lib_link_aliases(
                    ee.name,
                    Path(ee.sources[0]),
                ):
                    build_lib_path = zig_build_dir / alias
                    src_lib_path = zig_src_path.parent / alias
                    if build_lib_path.is_file():
                        shutil.copy2(build_lib_path, src_lib_path)
                        print(
                            f"Inplace zig lib copy:\n"
                            f"    {build_lib_path}\n"
                            f" -> {src_lib_path}"
                        )

        # Copy linked libraries next to the extensions that need them
        for ext_with_lib in self.extensions:
            if not ext_with_lib.libraries:
                continue
            for lib in ext_with_lib.libraries:
                for ext_link_lib in self.extensions:
                    if lib not in (
                        lib_base_name(ext_link_lib.name),
                        ext_link_lib.name,
                    ):
                        continue
                    print(
                        f"Cross-link: {ext_with_lib.name}"
                        f" -> {ext_link_lib.name}"
                    )
                    orig_dir = (
                        Path(
                            self.get_ext_fullpath(ext_link_lib.name)
                        ).resolve().parent
                    )
                    run_dir = (
                        Path(
                            self.get_ext_fullpath(ext_with_lib.name)
                        ).resolve().parent
                    )
                    for lib_name in lib_link_aliases(
                        ext_link_lib.name,
                        Path(ext_link_lib.sources[0]),
                    ):
                        orig_lib = orig_dir / lib_name
                        run_lib = run_dir / lib_name
                        print(
                            f"  Copying:\n"
                            f"    From: {orig_lib}\n"
                            f"    To  : {run_lib}\n"
                        )
                        shutil.copy2(orig_lib, run_lib)

    def build_extension(self, ext: Extension) -> None:
        print(80 * "=")
        print(f"MultiBuildExt: build_extension  [{ext.name}]")
        print(80 * "=")

        first_source = Path(ext.sources[0])

        # Re-sync all library_dirs (paths change between run/build when
        # --inplace is used)
        ext_dirs = [
            str(
                Path(self.get_ext_fullpath(ee.name)).resolve().parent
            )
            for ee in self.extensions
        ]
        for dd in ext_dirs:
            for ee in self.extensions:
                if dd not in ee.library_dirs:
                    ee.library_dirs.append(dd)

        output_ext_path = Path(self.get_ext_fullpath(ext.name))
        output_ext_dir = output_ext_path.parent
        output_ext_dir.mkdir(exist_ok=True, parents=True)

        if first_source.suffix == ".zig":
            assert len(ext.sources) == 1, (
                "Zig compiler expects a single root source file"
            )

            print(80 * "-")
            print(f"Zig build: {ext.name}")
            print(80 * "-")
            print(f"Root file: {first_source}")

            zig_python_output = self.get_ext_fullpath(ext.name)
            zig_lib_outputs = [
                output_ext_dir / name
                for name in lib_link_aliases(ext.name, first_source)
            ]

            is_windows = platform.system().lower() == "windows"
            zig_target_args: list[str] = []
            if is_windows:
                arch = platform.machine().lower()
                if arch in ("amd64", "x86_64"):
                    triple = "x86_64-windows-msvc"
                elif arch in ("arm64", "aarch64"):
                    triple = "aarch64-windows-msvc"
                else:
                    triple = "i386-windows-msvc"
                zig_target_args = ["-target", triple]

            zig_cmd = [
                "build-lib",
                "-dynamic",
                "-O", "ReleaseFast",
                "-lc",
                f"-femit-bin={zig_python_output}",
                *zig_target_args,
                *[f"-I{d}" for d in self.include_dirs],
                *ext.extra_compile_args,
                *ext.extra_link_args,
                str(first_source),
            ]

            print(f"zig {' '.join(zig_cmd)}\n")

            try:
                # Uses the ziglang PyPI package:
                # https://pypi.org/project/ziglang/
                subprocess.check_call(
                    [sys.executable, "-m", "ziglang"] + zig_cmd
                )
                print("Zig build successful\n")

                for zig_out in zig_lib_outputs:
                    shutil.copy2(zig_python_output, zig_out)
                    print(f"Copied to: {zig_out}")

                if is_windows:
                    # MSVC linker expects c_felix.lib
                    zig_lib_win = f"{first_source.stem}.lib"
                    src_lib_win = output_ext_dir / zig_lib_win
                    dst_lib_win = output_ext_dir / "c_felix.lib"
                    if src_lib_win.is_file():
                        shutil.copy2(src_lib_win, dst_lib_win)
                        print(f"Copied import lib: {dst_lib_win}")

            except subprocess.CalledProcessError as exc:
                print(f"Zig build failed: {exc}")
                raise

        elif first_source.suffix in (".c", ".pyx", ".py"):
            print(80 * "-")
            print(f"C/Cython build: {ext.name}")
            print(80 * "-")
            super().build_extension(ext)

        else:
            print(80 * "-")
            print(f"Default build: {ext.name}")
            print(80 * "-")
            super().build_extension(ext)

        print(f"\nbuild_extension complete: {ext.name}\n")


# ------------------------------------------------------------------------------
# Extensions
# ------------------------------------------------------------------------------

H_DIRS = [
    numpy.get_include(),
    str(PROJECT_ROOT / "src"),
    str(PROJECT_ROOT / "src" / "felix" / "cython"),
    str(PROJECT_ROOT / "src" / "felix" / "zig"),
]

is_windows = platform.system().lower() == "windows"

if is_windows:
    cython_compile_args = ["/fp:fast", "/O2"]
    cython_link_args = ["msvcrt.lib", "ucrt.lib", "vcruntime.lib"]
    zig_compile_args: list[str] = []
    runtime_lib_dirs: list[str] = []
else:
    cython_compile_args = ["-ffast-math", "-O3"]
    cython_link_args: list[str] = []
    zig_compile_args = []
    runtime_lib_dirs = [PLATFORM_INFO["runtime_lib_dir"]]

# Zig shared library (the Felix core)
ext_zig = Extension(
    name="felix.zig.c_felix",
    sources=["src/felix/zig/c-abi.zig"],
    extra_compile_args=zig_compile_args,
)

# Cython extension that wraps the Zig library
ext_cython = Extension(
    name="felix.cython.felix",
    sources=["src/felix/cython/felix.pyx"],
    include_dirs=H_DIRS,
    libraries=["c_felix"],
    library_dirs=[],            # populated dynamically by MultiBuildExt.run()
    runtime_library_dirs=runtime_lib_dirs,
    extra_compile_args=cython_compile_args,
    extra_link_args=cython_link_args,
)

ext_modules = [ext_zig] + cythonize(
    ext_cython,
    annotate=True,
    compiler_directives={"language_level": "3"},
)

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------

setup(
    name=DIST_NAME,
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": MultiBuildExt,
    },
    zip_safe=False,
    package_data={
        "felix": [f"*{PLATFORM_INFO['lib_ext']}"],
        "felix.cython": [f"*{PLATFORM_INFO['lib_ext']}"],
        "felix.zig": [f"*{PLATFORM_INFO['lib_ext']}"],
        "": [f"*{PLATFORM_INFO['lib_ext']}"],
    },
    include_package_data=True,
)
