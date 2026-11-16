# eDensGPU_MP2

Regularized MP2 correlation energy density generator using the GPU-based implementation of `eDensGPU`. Obtains correlation energies and correlation energy densities per particle for all spin channels of open- and closed-shell chemical systems. Requires a PySCF molecule and a converged SCF calculation.

Version: 1.3 

## Install

Clone the repository to a work environment that uses Python 3.11/3.12 and an NVIDIA GPU with a CUDA 12 compatible driver:

```bash
git clone https://github.com/vuckovic-lab/eDensGPU_MP2.git
```

Run the pip installation locally:

```bash
cd eDensGPU_MP2
python -m pip install .
```

The `example_colab.ipynb` example notebook automatically installs the module on a GPU-enabled runtime. Click [here](https://colab.research.google.com/github/vuckovic-lab/eDensGPU_MP2/blob/main/example_colab.ipynb) to open the notebook directly in Google Colab. 

## Example

### Molecule setup

Choose your molecular system and its parameters for the PySCF evaluation. 
Note that PySCF uses `spin=2S`. Set `scf_type` to `rhf` for a closed-shell system or `uhf` for an open-shell system. 

```python
import cupy as cp
from pyscf import gto
from gpu4pyscf import dft

###==== Define molecule and basis set ====###
struc = """
O   0.000000   0.000000   0.000000
H   0.000000  -0.757000   0.587000
H   0.000000   0.757000   0.587000
"""
charge = 0
spin = 0
basis = "def2qzvppd"
scf_type = "rhf"       # "rhf" or "uhf"
xc_fun = 'hf'


###==== Run SCF calculation ==###
mol = gto.M(atom=struc, basis=basis, charge=charge, spin=spin, cart=False)
if scf_type == "rhf":
    mf = dft.RKS(mol)
    mf.xc = xc_fun
elif scf_type == "uhf":
    mf = dft.UKS(mol)
    mf.xc = xc_fun
cp.cuda.runtime.deviceSynchronize()
mf.kernel()
cp.cuda.runtime.deviceSynchronize()
print("HF energy (a.u.):", repr(float(mf.e_tot)))
```

### Run `eDensGPU_MP2`
MP2 correlation energy density generation `ec_mp2_gen()` is evaluated on the created grid `coords` and integration weights `weights` from `make_grid()`, where the `grid_level` is a user-given choice. A correlation-fitting basis is based on the chosen PySCF basis, `basis+'ri'`. Regularization uses `kappa=2.0` as the default.

```python
import time
import cupy as cp
from eDens_mp2 import ec_mp2_gen, make_grid

###==== eDensGPU_MP2 parameters ====###
grid_level = 3
kappa = 2.0
auxbasis = basis + "ri"
coords, weights = make_grid(mol, level=grid_level)

###==== Run eDensGPU_MP2 calculation ====###
cp.cuda.runtime.deviceSynchronize()
mp2_started = time.perf_counter()
ec_out = ec_mp2_gen(mf, mol, coords, weights, kappa=kappa, auxbasis=auxbasis, verbose=True)
cp.cuda.runtime.deviceSynchronize()
mp2_seconds = time.perf_counter() - mp2_started
print(f"Generation elapsed: {mp2_seconds:.3f} s")

print("Correlation energies (a.u.):")
print("Total:", repr(ec_out.tot))
print("OS:   ", repr(ec_out.os))
```

The result class object contains (in a.u.):

| Attribute | Contents |
| --- | --- |
| `tot` and `os` | Total and opposite-spin (OS) correlation energies|
| `dens_tot` and `dens_os` |  Total and OS energy densities per particle |
| `rho`, `weights` and `coords` | Electron density, quadrature weights and coordinates |
| `kappa` and `auxbasis` | Regularization parameter and correlation-fitting auxiliary basis. |

For same-spin (SS) correlation energies and correlation energy densities per particle simply subtract the OS part from the total results:

```python
print(f"SS:    {ec_out.tot-ec_out.os}")
```

## Reference

The original MP2 correlation energy density generator is available on [Zenodo](https://doi.org/10.5281/zenodo.17390040). Please also cite the corresponding original publication:

Elias Polak, Heng Zhao and Stefan Vuckovic, **Real-space machine learning of
correlation density functionals**, *Nature Communications* **16**, 11306 (2025).
[Published article and DOI](https://doi.org/10.1038/s41467-025-66450-z).


## License

This repository includes the [GNU General Public License, version 3](LICENSE).
Consult that file for the redistribution terms and warranty disclaimer.
Separately installed dependencies retain their own licenses; this repository's
license does not replace them.

## Contact

Research group: [Vuckovic group](https://www.stefanvuckovic.com/).
