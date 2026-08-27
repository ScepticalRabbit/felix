#!/usr/bin/env python
"""Run all Felix tests in isolated subprocesses to avoid VTK/PyVista crashes."""
import subprocess
import sys
from pathlib import Path

CORE_TESTS = [
    "src/felix/pytests/test_analytic_3d.py",
    "src/felix/pytests/test_derived_and_library.py",
    "src/felix/pytests/test_errgraph.py",
    "src/felix/pytests/test_field_transforms.py",
    "src/felix/pytests/test_kernels_and_quadrature.py",
    "src/felix/pytests/test_postprocessors.py",
    "src/felix/pytests/test_sensors_differential.py",
    "src/felix/pytests/test_sensors_spatial_verif.py",
    "src/felix/pytests/test_sensorsim.py",
    "src/felix/pytests/test_sensortools_and_errors.py",
    "src/felix/pytests/test_spatial_rotations.py",
    "src/felix/pytests/test_zig_facade_analytic.py",
]

PYVISTA_TESTS = [
    "src/felix/pytests/test_sensors_ray.py",
    "src/felix/pytests/test_sensors_spatial_ray.py",
    "src/felix/pytests/test_sensor_library.py",
    "src/felix/pytests/test_sensorsim_examples.py",
]

def run_tests(test_files, label):
    print(f"\n{'='*60}")
    print(f"Running {label} ({len(test_files)} files)")
    print(f"{'='*60}")
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest"] + test_files + ["-v"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    
    # Print last 50 lines of output
    output = result.stdout + "\n" + result.stderr
    for line in output.strip().split("\n")[-50:]:
        print(line)
    
    if result.returncode == 0:
        print(f"\n✓ {label} PASSED")
        return True
    else:
        print(f"\n✗ {label} FAILED (return code: {result.returncode})")
        return False


def run_tests_individual(test_files, label):
    print(f"\n{'='*60}")
    print(f"Running {label} (each test in isolated subprocess)")
    print(f"{'='*60}")
    
    all_passed = True
    for test_file in test_files:
        # First collect tests in this file
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "--collect-only", "-q"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        test_names = [line.strip() for line in result.stdout.strip().split("\n") 
                      if line.strip() and not line.startswith("=") and "::" in line]
        
        print(f"\n  {test_file}: {len(test_names)} tests")
        for test_name in test_names:
            print(f"    Running {test_name}...", end=" ", flush=True)
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_name, "-v"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                print("PASS")
            else:
                print(f"FAIL (code: {result.returncode})")
                all_passed = False
    
    if all_passed:
        print(f"\n✓ {label} ALL PASSED")
    else:
        print(f"\n✗ {label} SOME FAILED")
    return all_passed


def main():
    print("Felix Test Runner - Isolated Subprocess Mode")
    print("This avoids VTK/PyVista memory corruption by running each group in a separate process.")
    
    core_ok = run_tests(CORE_TESTS, "Core Tests")
    pyvista_ok = run_tests_individual(PYVISTA_TESTS, "PyVista/Ray Tests")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Core Tests:      {'PASS' if core_ok else 'FAIL'}")
    print(f"PyVista Tests:   {'PASS' if pyvista_ok else 'FAIL'}")
    
    if core_ok and pyvista_ok:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())