# edens_mp2_gpu

Regularized MP2 correlation energy density generator for GPU-based implementation. Obtains correlation energies and correlation energy densities per particle for all spin channels of open- and closed-shell chemical systems. Requires PySCF molecular object and SCF resolution. 

Version: 1.0 

## Install

Use Python 3.11/3.12 and an NVIDIA GPU with a CUDA 12 compatible driver:

```bash
python -m pip install .
```

See the `example_colab.ipynb` example notebook for installation on Google Colab within a runtime allocated GPU. 

## Run

Given a SCF converged `mf` and its PySCF molecule `mol`:

```python
from edens_mp2 import ec_mp2_gen, make_grid

coords, weights = make_grid(mol, level=3)
ec_out = ec_mp2_gen(mf, mol, coords, weights,
                     kappa=2.0, auxbasis=basis + 'ri')
print(f"Total: {ec_out.tot}")
print(f"OS:    {ec_out.os}")
```

The returned class-object contains:

| Attribute | Contents |
| --- | --- |
| `tot` and `os` | Total and opposite-spin (os) correlation energies in Hartree (a.u.). |
| `dens_tot` and `dens_os` |  Total and os energy densities per particle (float64 CuPy arrays). |
| `rho`, `weights` and `coords` | Electron density, quadrature weights and coordinates in a.u. (float64 CuPy arrays) |
| `kappa` and `auxbasis` | Regularization parameter and used correlation auxiliary basis. |

For same-spin (ss) correlation energies and correlation energy densities per particle simply subtract the os part from the total results:

```python
print(f"SS:    {ec_out.tot-ec_out.ss}")
```


## Files

- [example_colab.ipynb](example_colab.ipynb): pip installation, molecule/SCF setup and evaluation.
- `edens_mp2/__init__.py`: the result/generator class and `make_grid`.
- `edens_mp2/_core.py`: GPU kernels and memory planning.


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

- Author: Elias Polak — [Elias-Py09 on GitHub](https://github.com/Elias-Py09).

- Research group: [Vuckovic group](https://www.stefanvuckovic.com/).
