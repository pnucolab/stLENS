#!/usr/bin/env python3

import os
import subprocess
import sys
import platform
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

class CustomBuildPy(build_py):
    """Custom build_py command to compile C++ files before building the package"""
    
    def run(self):
        # Compile C++ files first
        self.compile_cpp_files()
        # Then run the normal build_py
        super().run()
    
    def compile_cpp_files(self):
        """Compile C++ files to shared libraries"""
        # Get the source directory
        src_dir = Path(__file__).parent / "src" / "stLENS"
        
        # Define the C++ files and their corresponding shared library names
        cpp_files = [
            ("random_matrix.cpp", "random_matrix.so"),
            ("perturb_omp.cpp", "perturb_omp.so")
        ]
        
        # Check if g++ is available
        try:
            subprocess.run(["g++", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: g++ compiler not found. C++ extensions will not be compiled.")
            print("Install a C++ compiler (g++) to enable optimized performance.")
            return
        
        for cpp_file, so_file in cpp_files:
            cpp_path = src_dir / cpp_file
            so_path = src_dir / so_file
            
            if not cpp_path.exists():
                print(f"Warning: {cpp_path} not found, skipping...")
                continue
                
            print(f"Compiling {cpp_file} -> {so_file}")
            
            # Base compilation command
            cmd = [
                "g++",
                "-shared",
                "-fPIC",
                "-O3",
                "-std=c++11",
                str(cpp_path),
                "-o", str(so_path)
            ]
            
            # Add OpenMP flag for perturb_omp.cpp
            if cpp_file == "perturb_omp.cpp":
                cmd.insert(-3, "-fopenmp")
                # On macOS, we might need to use libomp
                if platform.system() == "Darwin":
                    # Try to find OpenMP library path
                    try:
                        # Common paths where OpenMP might be installed via Homebrew
                        omp_paths = [
                            "/opt/homebrew/lib",
                            "/usr/local/lib",
                            "/opt/local/lib"
                        ]
                        for path in omp_paths:
                            if os.path.exists(f"{path}/libomp.dylib"):
                                cmd.extend(["-L" + path, "-lomp"])
                                break
                    except Exception:
                        pass
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"Successfully compiled {so_file}")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to compile {cpp_file}: {e}")
                print(f"Command: {' '.join(cmd)}")
                if e.stderr:
                    print(f"Error output: {e.stderr}")
                # Don't fail the installation, just warn
                continue

if __name__ == "__main__":
    setup(
        cmdclass={
            'build_py': CustomBuildPy,
        },
    )
