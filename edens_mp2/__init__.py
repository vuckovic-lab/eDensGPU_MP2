"""GPU local-slope κMP2: ``ec_out = ec_mp2_gen(mf, mol, coords, weights)``."""

from numbers import Real

import numpy as np
from pyscf import dft

from . import _core

__all__ = ["ec_mp2_gen", "make_grid"]


def make_grid(mol, level=3):
    """Return (coordinates in bohr, weights) using the parent's Becke recipe."""
    from pyscf.dft import radi

    grid = dft.gen_grid.Grids(mol)
    grid.level = int(level)
    grid.atomic_radii = radi.BRAGG_RADII
    grid.radii_adjust = radi.treutler_atomic_radii_adjust
    grid.radi_method = radi.treutler_ahlrichs
    grid.becke_scheme = dft.gen_grid.original_becke
    grid.prune = dft.gen_grid.nwchem_prune
    grid.alignment = 8
    grid.build(sort_grids=True)
    return np.asarray(grid.coords), np.asarray(grid.weights)


class ec_mp2_gen:
    """Run GPU strat1 κMP2 and retain total and opposite-spin results.

    ``tot`` and ``os`` are Python floats in Hartree.
    ``dens_tot`` and ``dens_os`` are float64 CuPy
    arrays in Hartree per particle, normalized by the total density.
    For example, ``tot = cupy.sum(rho * weights * dens_tot)``.
    ``rho``, ``weights`` and ``coords`` are also float64 CuPy arrays.

    Supply converged real RHF, UHF or CUHF orbitals; SCF is not rerun.
    ``grid`` accepts coordinates (N,3) with separate weights (N,), or a built
    PySCF Grids object. Memory allowances are decimal GB; None retains the
    parent's automatic planning. A CUDA GPU is required; no CPU fallback.
    """

    def __init__(
        self,
        mf,
        mol,
        grid,
        weights=None,
        *,
        kappa=2.0,
        auxbasis="def2tzvpri",
        gpu_mem_gb=None,
        host_mem_gb=None,
        verbose=False,
    ):
        if isinstance(kappa, (bool, np.bool_)) or not isinstance(kappa, Real):
            raise ValueError("kappa must be a positive finite number")
        if not np.isfinite(kappa) or kappa <= 0:
            raise ValueError("kappa must be a positive finite number")
        _core.validate_memory(gpu_mem_gb, host_mem_gb)
        _core.validate(mf)
        reference = mf.mol
        if mol is not reference and (
            mol.charge != reference.charge
            or mol.spin != reference.spin
            or mol.cart != reference.cart
            or any(
                not np.array_equal(getattr(mol, attr), getattr(reference, attr))
                for attr in ("_atm", "_bas", "_env", "_ecpbas")
            )
        ):
            raise ValueError("mol must describe the same molecule and basis as mf.mol")
        if hasattr(grid, "coords"):
            if weights is not None:
                raise ValueError("A PySCF grid already supplies its weights")
            grid, weights = grid.coords, grid.weights
        if weights is None:
            raise ValueError("Supply grid coordinates and weights, or a built PySCF grid")
        coords = np.asarray(_core.host(grid), dtype=np.float64)
        weights = np.asarray(_core.host(weights), dtype=np.float64)
        if (
            coords.ndim != 2
            or coords.shape[1] != 3
            or not len(coords)
            or weights.shape != (len(coords),)
        ):
            raise ValueError("Expected nonempty grid (N,3) and weights (N,)")
        if not (np.isfinite(coords).all() and np.isfinite(weights).all()):
            raise ValueError("Grid coordinates and weights must be finite")

        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("ec_mp2_gen requires an NVIDIA CUDA GPU")
        # Let GPU4PySCF initialize its allocator before entering a private pool.
        import gpu4pyscf  # noqa: F401

        pool = cp.cuda.MemoryPool()
        try:
            with cp.cuda.using_allocator(pool.malloc):
                fields, density = _evaluate(
                    mf,
                    coords,
                    float(kappa),
                    auxbasis,
                    gpu_mem_gb,
                    host_mem_gb,
                    verbose,
                )
        finally:
            cp.cuda.get_current_stream().synchronize()
            pool.free_all_blocks()

        # Returned device arrays use the restored allocator, so they outlive
        # the private calculation pool. Host staging preserves streamed MP2.
        self.kappa = float(kappa)
        self.auxbasis = auxbasis
        self.coords = cp.asarray(coords, dtype=cp.float64)
        self.weights = cp.asarray(weights, dtype=cp.float64)
        self.rho = cp.asarray(density, dtype=cp.float64)
        self.dens_os = cp.asarray(fields["ec_kmp2_os_array"], dtype=cp.float64)
        self.dens_tot = self.dens_os + cp.asarray(fields["ec_kmp2_ss_array"], dtype=cp.float64)
        self.tot = float(cp.dot(self.weights, self.rho * self.dens_tot).item())
        self.os = float(cp.dot(self.weights, self.rho * self.dens_os).item())
        if not (
            np.isfinite((self.tot, self.os)).all()
            and all(
                bool(cp.isfinite(value).all())
                for value in (
                    self.rho,
                    self.dens_tot,
                    self.dens_os,
                )
            )
        ):
            raise FloatingPointError("Nonfinite density or κMP2 result")


def _evaluate(mf, coords, kappa, auxbasis, gpu_mem_gb, host_mem_gb, verbose):
    """Device workspaces die here, before the caller releases its private pool."""
    engine = _core.Engine(mf, gpu_mem_gb=gpu_mem_gb)
    points = engine.xp.asarray(coords)
    # Preserve the parent's density contractions and native AO ordering.
    semilocal = engine.semilocal(points)
    density = _core.host(semilocal["rho"]).sum(0)
    if verbose:
        print(f"Evaluating κMP2 strat1 on {len(coords)} grid points ...", flush=True)
    fields = _core.df_fields(
        engine,
        points,
        engine.xp.asarray(density),
        auxbasis,
        kappa=kappa,
        gpu_mem_gb=gpu_mem_gb,
        host_mem_gb=host_mem_gb,
        progress=verbose,
    )
    engine.sync()
    return fields, density
