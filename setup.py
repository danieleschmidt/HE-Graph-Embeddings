"""Setup script for HE-Graph-Embeddings"""


from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys

class CMakeExtension(Extension):
    """CMakeExtension class."""
    def __init__(self, name, sourcedir=''):
        """  Init  ."""
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    """CMakeBuild class."""
    def run(self) -> None:
        """Run."""
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            logger.error(f"Error in operation: {e}")
            raise RuntimeError("CMake must be installed to build the extensions")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext) -> None:
        """Build Extension."""
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DBUILD_TESTS=OFF',
            '-DBUILD_BENCHMARKS=OFF',
            '-DBUILD_PYTHON_BINDINGS=ON'
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
        build_args += ['--', f'-j{os.cpu_count()}']

        env = os.environ.copy()
        env['CXXFLAGS'] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                            cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                            cwd=self.build_temp)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = [
    "torch>=2.0.0",
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "networkx>=2.6",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "matplotlib>=3.4.0",
    "tqdm>=4.62.0",
    "pyyaml>=5.4.0",
    "jsonschema>=3.2.0",
    "python-dotenv>=0.19.0",
]

dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
    "pytest-benchmark>=3.4.0",
    "pytest-asyncio>=0.18.0",
    "black>=22.0.0",
    "isort>=5.10.0",
    "ruff>=0.1.0",
    "mypy>=0.990",
    "pylint>=2.12.0",
    "sphinx>=4.3.0",
    "sphinx-rtd-theme>=1.0.0",
    "pre-commit>=2.16.0",
    "jupyterlab>=3.2.0",
    "wandb>=0.12.0",
    "tensorboard>=2.7.0",
]

setup(
    name="he-graph-embeddings",
    version="0.1.0",
    author="Daniel Schmidt",
    author_email="daniel@example.com",
    description="GPU-accelerated homomorphic encryption for graph neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/HE-Graph-Embeddings",
    project_urls={
        "Bug Tracker": "https://github.com/danieleschmidt/HE-Graph-Embeddings/issues",
        "Documentation": "https://he-graph-embeddings.readthedocs.io",
        "Source Code": "https://github.com/danieleschmidt/HE-Graph-Embeddings",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[CMakeExtension("he_graph_cpp")],
    cmdclass={"build_ext": CMakeBuild},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "all": requirements + dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "he-graph=he_graph.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)