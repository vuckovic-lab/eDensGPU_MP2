"""Package configuration; install with ``python -m pip install .``."""

from pathlib import Path

from setuptools import setup


setup(
    name="eDensGPU_MP2",
    version="1.2",
    description="MP2 energy density generator for GPU-based implementation",
    long_description=Path(__file__).with_name("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license_files=["LICENSE*"],
    python_requires=">=3.11,<3.13",
    package_dir={"eDensGPU_MP2": "."},
    packages=["eDensGPU_MP2"],
    py_modules=["eDens_mp2"],
    install_requires=[
        "psutil>=5.9", "pyscf==2.14.0", "numpy>=1.26.4,<2.3", "scipy==1.15.3",
        "h5py==3.14.0", "opt_einsum==3.4.0",
        "gpu4pyscf-cuda12x==1.8.1", "cupy-cuda12x==13.4.1", "cutensor-cu12==2.2.0",
    ],
    data_files=[("share/MP2_edens_GPU", ["example_colab.ipynb"])],
)
