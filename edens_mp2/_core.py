"""Internal float64 strat1 kernels, consolidated from edens_all_gpu.

Sections follow the data flow: orbitals, storage, density/integrals, and MP2.
Contraction order, CUDA kernels, and GPU/RAM/disk planning are retained.
The host-output adapters derive from GPU4PySCF (Apache-2.0; see LICENSE_GPU4PYSCF).
Only the public generator in __init__.py is intended for normal use.
"""

import ctypes
import math
import shutil
import tempfile
from contextlib import AbstractContextManager
from math import ceil, isfinite, isqrt, prod
from pathlib import Path
from time import perf_counter

import numpy as np

# Validation of the supplied orbitals
# ----------------------------------------------------------------------------


def validate(mf):
    if not getattr(mf, "converged", False):
        raise ValueError("Supply a converged molecular mean-field calculation")
    if mf.mol.cart:
        raise NotImplementedError("This reference recipe requires spherical molecular bases")
    if (
        getattr(mf, "cell", None) is not None
        or getattr(mf, "kpts", None) is not None
        or hasattr(mf.mol, "lattice_vectors")
    ):
        raise NotImplementedError("Periodic calculations are not supported")
    for name in ("mo_coeff", "mo_occ", "mo_energy"):
        a = host(getattr(mf, name))
        if np.iscomplexobj(a) or not np.all(np.isfinite(a)):
            raise ValueError("Real, finite orbitals and energies are required")
    occ = host(mf.mo_occ)
    if occ.ndim not in (1, 2) or (occ.ndim == 2 and occ.shape[0] != 2):
        raise ValueError("Expected RHF or spin-resolved UHF/CUHF orbital layout")
    allowed = (0.0, 2.0) if occ.ndim == 1 else (0.0, 1.0)
    if not np.isin(occ, allowed).all():
        raise NotImplementedError("Local MP2 requires integer RHF/UHF/CUHF occupations")


# Memory allowances
# ----------------------------------------------------------------------------


def validate_gpu_mem_gb(value):
    if value is None:
        return
    if isinstance(value, (bool, str)):
        raise ValueError("gpu_mem_gb must be None or a positive finite number (decimal GB)")
    try:
        valid = math.isfinite(float(value)) and float(value) > 0
    except (ValueError, TypeError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("gpu_mem_gb must be None or a positive finite number (decimal GB)")


def host_available_bytes():
    """Respect Linux cgroups when present; None on unsupported systems."""
    try:
        info = dict(s.split(":", 1) for s in Path("/proc/meminfo").read_text().splitlines())
        available = int(info["MemAvailable"].split()[0]) * 1024
        limit = Path("/sys/fs/cgroup/memory.max")
        if limit.exists() and limit.read_text().strip() != "max":
            available = min(
                available,
                max(
                    0,
                    int(limit.read_text()) - int(Path("/sys/fs/cgroup/memory.current").read_text()),
                ),
            )
        return available
    except (OSError, KeyError, ValueError):
        return None


# Native integral index bounds
# ----------------------------------------------------------------------------


def native_point_limit(sorted_shells):
    """Largest legal fake/auxiliary int32 pair index (no accuracy cutoff)."""
    limit = np.iinfo(np.int32).max
    return (isqrt(int(sorted_shells) ** 2 + 4 * limit) - int(sorted_shells)) // 2


# Lossless factor storage and contraction planning
# ----------------------------------------------------------------------------


MIB = 1024**2


def largest_rows(total, per_row, available, fixed=0):
    n = min(int(total), (int(available) - int(fixed)) // max(1, int(per_row)))
    if total and n < 1:
        raise MemoryError("MP2 budget cannot fit the minimum tile plus fixed workspaces")
    return max(1, n)


def pair_plan(nj, na, nb, naux, available, cap=None, resident=False, host_available=None):
    """Maximal j block, preferring full virtual axes, within this live-array bound.

    Three partner-factor layouts and eight pair-sized arrays cover direct,
    exchange, denominator, regularizer, layout copies and weighted products.
    Resident factors/kernel buffers are deducted by the caller.
    """
    from math import ceil

    amax = min(na, int(cap)) if cap else na
    bmax = min(nb, int(cap)) if cap else nb
    fixed = 4096
    # Enumerate distinct ceil(na/a) intervals, retaining the largest a in each.
    # For any fixed j,a the largest feasible b cannot require more tiles.
    candidates = []
    avals = []
    a = 1
    while a <= amax:
        blocks = ceil(na / a)
        end = amax if blocks == 1 else min(amax, (na - 1) // (blocks - 1))
        avals.extend((a, end))
        a = end + 1
    jmax = (
        nj
        if resident or host_available is None
        else min(nj, max(0, (host_available - 4096) // max(1, nb * naux * 8)))
    )
    for j in range(1, jmax + 1):
        for a in sorted(set(avals)):
            # Full virtual slices are contiguous views; partial slices need
            # layout copies. A streamed partner factor is one further owner.
            layouts = (0 if resident else 1) + (0 if a == na and bmax == nb else 2)
            factor = layouts * nb * naux * 8
            b = min(bmax, max(0, ((int(available) - fixed) // j - factor) // (64 * a)))
            if b < 1:
                continue
            if b < nb and layouts < (2 if resident else 3):
                layouts = 2 if resident else 3
                factor = layouts * nb * naux * 8
                b = min(bmax, max(0, ((int(available) - fixed) // j - factor) // (64 * a)))
            if b < 1:
                continue
            cost = factor + a * b * 64
            count = ceil(nj / j) * ceil(na / a) * ceil(nb / b)
            candidates.append((count, ceil(na / a) * ceil(nb / b), -j, -a * b, j, a, b, cost))
    if not candidates:
        raise MemoryError("MP2 budgets cannot fit one partner occupied tile")
    count, _, _, _, j, a, b, cost = min(candidates)
    return {
        "j": j,
        "a": a,
        "b": b,
        "bytes_per_j": cost,
        "estimated_bytes": fixed + j * cost,
        "total_blocks_per_observation": count,
        "resident_factors": resident,
    }


class TrackedPool:
    def __init__(self, xp, budget):
        self.pool = xp.cuda.MemoryPool()
        self.pool.set_limit(size=int(budget))
        self.peak_live = 0
        self.peak_reserved = 0

    def malloc(self, n):
        ptr = self.pool.malloc(n)
        self.peak_live = max(self.peak_live, self.pool.used_bytes())
        self.peak_reserved = max(self.peak_reserved, self.pool.total_bytes())
        return ptr

    def used_bytes(self):
        return self.pool.used_bytes()

    def total_bytes(self):
        return self.pool.total_bytes()

    def free_all_blocks(self):
        self.pool.free_all_blocks()


class HostArena:
    """Explicit RAM allowance; uncompressed HDF5 spill, never lossy compression.

    Scratch files are deleted on success and failure. Disk reads are explicit
    bounded slices rather than mmap residency growing with the complete tensor.
    """

    def __init__(self, host_bytes=None, directory=None):

        available = host_available_bytes()
        if available is None:
            try:
                import psutil

                available = psutil.virtual_memory().available
            except ImportError:
                available = 2 * 1024**3
        if host_bytes is None:
            host_bytes = int(0.8 * available)
        if host_bytes <= 0 or host_bytes > available:
            raise MemoryError(
                "MP2 host scratch budget must be positive and fit currently available CPU RAM"
            )
        self.limit = int(host_bytes)
        self.live = 0
        self.peak = 0
        self.disk_live = 0
        self.disk_peak = 0
        self.directory = Path(tempfile.mkdtemp(prefix="edens-mp2-", dir=directory))
        self.stores = []

    @property
    def remaining(self):
        return max(0, self.limit - self.live)

    @property
    def disk_available(self):
        allocated = sum(
            s.path.stat().st_blocks * 512
            for s in self.stores
            if not s.closed and s.path is not None and s.path.exists()
        )
        pending = max(0, self.disk_live - allocated)
        return max(0, shutil.disk_usage(self.directory).free - 2 * 1024**3 - pending)

    def allocate(self, shape, name, force=None):
        store = ArrayStore(self, shape, name, force)
        self.stores.append(store)
        return store

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for s in self.stores:
            s.close()
        shutil.rmtree(self.directory, ignore_errors=True)


class ArrayStore:
    def __init__(self, arena, shape, name, force=None):
        self.arena = arena
        self.shape = tuple(shape)
        self.nbytes = prod(shape) * 8
        self.name = name
        self.closed = False
        self.path = None
        self.file = None
        self.kind = "ram" if self.nbytes <= arena.remaining and force != "disk" else "disk"
        if force == "ram" and self.kind != "ram":
            raise MemoryError("MP2 host budget cannot fit requested raw factor array")
        if self.kind == "ram":
            self.array = np.empty(shape, dtype=np.float64)
            arena.live += self.nbytes
            arena.peak = max(arena.peak, arena.live)
        else:
            import h5py

            if self.nbytes > arena.disk_available:
                raise MemoryError("MP2 temporary disk space cannot fit one storage group")
            self.path = arena.directory / (name + ".h5")
            self.file = h5py.File(self.path, "w", rdcc_nbytes=1024**2)
            self.array = self.file.create_dataset("data", shape=shape, dtype="f8")
            arena.disk_live += self.nbytes
            arena.disk_peak = max(arena.disk_peak, arena.disk_live)

    def read(self, index):
        return self.array[index]

    def write(self, index, value):
        self.array[index] = value

    def close(self):
        if self.closed:
            return
        if self.kind == "ram":
            self.arena.live -= self.nbytes
        else:
            self.file.close()
            self.path.unlink(missing_ok=True)
            self.arena.disk_live -= self.nbytes
        self.array = None
        self.closed = True


def grid_plan(
    no,
    points,
    kernel_per_i,
    gpu_per_r,
    host_per_r,
    gpu_available,
    host_available,
    fixed=0,
    max_points_per_block=None,
):
    """Minimize total grid passes, jointly choosing occupied and r tiles.

    Exact enumeration of the occupied choices under the stated live-array
    bounds. Ties prefer fewer occupied passes, then larger r. If the full
    workload fits both budgets, the answer is one occupied and one r block.
    """
    from math import ceil

    candidates = []
    for i in range(1, no + 1):
        r = min(
            points,
            (gpu_available - fixed - i * kernel_per_i) // max(1, gpu_per_r),
            (host_available - fixed) // max(1, host_per_r),
        )
        if max_points_per_block is not None:
            r = min(r, int(max_points_per_block))
        if r > 0:
            nk = ceil(no / i)
            nr = ceil(points / r)
            candidates.append((nk * nr, nk, -r, -i, i, r))
    if not candidates:
        raise MemoryError("MP2 budgets cannot fit one occupied kernel and one grid point")
    count, nk, _, _, i, r = min(candidates)
    # Having fixed the fewest passes, balance their sizes. A maximal first
    # block plus a tiny tail uses more memory without reducing block count.
    nr = ceil(points / r)
    i = ceil(no / nk)
    r = ceil(points / nr)
    return dict(
        occupied_block=i,
        grid_block=r,
        kernel_blocks=nk,
        total_grid_blocks=count,
        gpu_estimated_bytes=fixed + i * kernel_per_i + r * gpu_per_r,
        host_estimated_bytes=fixed + r * host_per_r,
        full_gpu_estimated_bytes=fixed + no * kernel_per_i + points * gpu_per_r,
        full_host_estimated_bytes=fixed + points * host_per_r,
    )


class DeviceStore:
    """A single GPU owner, never mirrored by a full host array."""

    def __init__(self, array, name):
        self.array = array
        self.shape = array.shape
        self.nbytes = array.nbytes
        self.name = name
        self.kind = "gpu"
        self.closed = False

    def read(self, index):
        return self.array[index]

    def write(self, index, value):
        self.array[index] = value

    def close(self):
        self.array = None
        self.closed = True


class AuxiliarySink:
    """Native integral output adapter with contiguous auxiliary-major disk I/O.

    Only one native output tile and its sorted host layout coexist. There is
    no full (occupied,virtual,auxiliary) host array or repeated integral pass.
    """

    def __init__(self, store, no, nv, qcart):
        self.store = store
        self.shape = (no, nv, qcart)

    def __setitem__(self, index, value):
        order = np.argsort(index[2])
        ids = np.asarray(index[2])[order]
        # Sorted fancy indexing is required by h5py; no value is discarded.
        self.store.write((ids, slice(None)), value.reshape(-1, len(ids)).T[order])


# Bounded grid workspaces
# ----------------------------------------------------------------------------


_GRID_POOLS = []


def available_host():
    value = host_available_bytes()
    if value is None:
        try:
            import psutil

            value = psutil.virtual_memory().available
        except ImportError:
            import os

            value = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return int(value)


def plan_grid(n, gpu_per_point, host_per_point, gpu_available, host_available):
    values = (n, gpu_per_point, host_per_point, gpu_available, host_available)
    if any(isinstance(v, bool) or int(v) != v or v < 0 for v in values):
        raise ValueError("Grid sizes and allowances must be nonnegative integer bytes")
    n, gp, hp, g, h = map(int, values)
    if not n:
        return dict(points=0, max_points=0, blocks=0, limiting_resource="empty")
    limits = {"whole grid": n, "GPU": n if not gp else g // gp, "host": n if not hp else h // hp}
    which = min(limits, key=limits.get)
    cap = limits[which]
    if cap < 1:
        raise MemoryError("Grid allowance cannot fit one point and its workspaces")
    blocks = math.ceil(n / cap)
    points = math.ceil(n / blocks)
    return dict(
        points=points,
        max_points=cap,
        blocks=blocks,
        limiting_resource=which,
        gpu_per_point=gp,
        host_per_point=hp,
    )


def execution_plan(n, gp, hp, g, h, l2_bytes):
    """Memory admission followed by a hardware-scaled allocation working set.

    A memory ceiling is not a performance target: allocating tens of GiB for
    one grid tile is slower than reusing a smaller tile on A100. The 64-cache
    working set was selected by paired sweeps, not by changing AO screening.
    Keep the memory-only minimum in metadata so extra execution tiles are
    visible. Devices not reporting an L2 size retain the memory-only plan.
    """
    memory = plan_grid(n, gp, hp, g, h)
    working = min(g, 64 * int(l2_bytes)) if l2_bytes else g
    # A performance preference must never reject a memory-feasible point.
    working = min(g, max(working, gp))
    execution = plan_grid(n, gp, hp, working, h)
    return memory, execution, working


def slices(coords, points):
    if not len(coords):
        return
    if points < 1:
        raise ValueError("Nonempty grid requires a positive planned tile")
    for start in range(0, len(coords), points):
        stop = min(start + points, len(coords))
        yield slice(start, stop), coords[start:stop]


class GridMemory(AbstractContextManager):
    """A bounded private CUDA pool, admitted host outputs, and OOM-safe tiles."""

    def __init__(self, engine, label, output_bytes):
        self.engine = engine
        self.cp = engine.xp
        self.label = label
        validate_gpu_mem_gb(engine.gpu_mem_gb)
        self.free = int(self.cp.cuda.runtime.memGetInfo()[0])
        self.budget = int(0.8 * self.free if engine.gpu_mem_gb is None else engine.gpu_mem_gb * 1e9)
        if self.budget > self.free:
            raise MemoryError(f"{label} GPU allowance exceeds currently free memory")
        self.host_budget = int(0.8 * available_host())
        self.outputs = int(output_bytes)
        if self.outputs > self.host_budget:
            raise MemoryError(f"{label} returned fields cannot fit available host RAM")
        self.reserve = 16 * MIB
        if self.budget <= self.reserve:
            raise MemoryError(f"{label} GPU allowance cannot fit native workspace reserve")
        self.stream_ordered = bool(self.cp.cuda.Device().attributes.get("MemoryPoolsSupported", 0))
        allocator = self.cp.cuda.malloc_async if self.stream_ordered else self.cp.cuda.malloc
        # Keep MemoryPool's HARD allocation limit. The stream-ordered base
        # allocator alone has only a cache threshold, not a working-byte cap.
        self.pool = self.cp.cuda.MemoryPool(allocator)
        self.pool.set_limit(size=self.budget - self.reserve)
        self.peak = 0
        self.reserved_peak = 0
        self.metadata = {
            "gpu_budget_bytes": self.budget,
            "gpu_free_at_entry": self.free,
            "native_reserve_bytes": self.reserve,
            "host_budget_bytes": self.host_budget,
            "host_output_bytes": self.outputs,
            "actual_batches": [],
            "retries": [],
            "allocator": "cuda_malloc_async" if self.stream_ordered else "cuda_malloc",
            "policy": "GPU/host admission; cache-scaled execution tiles within that admission",
        }

    def malloc(self, size):
        m = self.pool.malloc(size)
        self.peak = max(self.peak, self.pool.used_bytes())
        self.reserved_peak = max(self.reserved_peak, self.pool.total_bytes())
        return m

    def __enter__(self):
        _GRID_POOLS[:] = [p for p in _GRID_POOLS if p.used_bytes()]
        self.context = self.cp.cuda.using_allocator(self.malloc)
        self.context.__enter__()
        return self

    def __exit__(self, *args):
        try:
            self.cp.cuda.get_current_stream().synchronize()
        finally:
            self.context.__exit__(*args)
        self.pool.free_all_blocks()
        if self.stream_ordered:
            # Return unused driver-cache pages too; never leave a previous
            # family's large cache outside the next family's free-memory view.
            self.cp.cuda.get_current_stream().synchronize()
            handle = self.cp.cuda.runtime.deviceGetDefaultMemPool(self.cp.cuda.Device().id)
            self.cp.cuda.runtime.memPoolTrimTo(handle, 0)
        if self.pool.used_bytes():
            _GRID_POOLS.append(self.pool)
        self.metadata.update(
            peak_pool_bytes=self.peak,
            peak_reserved_pool_bytes=self.reserved_peak,
            actual_blocks=len(self.metadata["actual_batches"]),
        )
        self.engine.grid_memory = getattr(self.engine, "grid_memory", {})
        self.engine.grid_memory[self.label] = self.metadata

    def device_outputs_fit(self, n, per_point):
        """Keep complete outputs on-device only without shrinking a work tile.

        Otherwise stream rows to their admitted host destination. This avoids
        many small transfers when output residency is inexpensive, without
        making the full returned field a GPU memory requirement.
        """
        allowed = max(0, self.budget - self.reserve - self.pool.used_bytes())
        _, plan, _ = execution_plan(
            n, per_point, 0, allowed, 0, int(self.cp.cuda.Device().attributes.get("L2CacheSize", 0))
        )
        answer = self.outputs + plan["points"] * per_point <= allowed
        self.metadata["output_residency"] = "GPU" if answer else "host streamed"
        return answer

    def run(self, coords, gpu_per_point, host_per_point, evaluate):
        allowed = self.budget - self.reserve - self.pool.used_bytes()
        l2 = int(self.cp.cuda.Device().attributes.get("L2CacheSize", 0))
        minimum, p, working = execution_plan(
            len(coords),
            int(gpu_per_point),
            int(host_per_point),
            max(0, allowed),
            max(0, self.host_budget - self.outputs),
            l2,
        )
        self.metadata.update(
            memory_only_plan=minimum, l2_cache_bytes=l2, execution_workspace_bytes=working
        )
        self.metadata["plan"] = p
        size = p["points"]
        start = 0
        while start < len(coords):
            stop = min(start + size, len(coords))
            failed = False
            try:
                block = evaluate(coords[start:stop], slice(start, stop))
            except self.cp.cuda.memory.OutOfMemoryError as error:
                failed = True
                reason = str(error)
            if failed:
                import gc

                gc.collect()
                self.pool.free_all_blocks()
                self.metadata["retries"].append(
                    {"start": start, "points": stop - start, "error": reason}
                )
                if size == 1:
                    raise MemoryError(
                        f"{self.label} cannot allocate one grid point within the GPU allowance"
                    )
                size = max(1, size // 2)
                continue
            self.metadata["actual_batches"].append(stop - start)
            yield slice(start, stop), block
            del block
            start = stop


# AO evaluation and density preparation
# ----------------------------------------------------------------------------


def host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


class Engine:
    """Device orbitals and GPU AO/density operations; no CPU evaluator."""

    def __init__(self, mf, *, gpu_mem_gb=None):
        import cupy as cp
        from gpu4pyscf.dft.numint import eval_ao, _GDFTOpt

        validate(mf)
        self.mol = mf.mol
        if max(self.mol.bas_angular(i) for i in range(self.mol.nbas)) > 4:
            raise NotImplementedError("GPU orbital integrals support angular momentum through g")
        self.xp = cp
        self._eval_ao = eval_ao
        self._opt_factory = _GDFTOpt.from_mol
        validate_gpu_mem_gb(gpu_mem_gb)
        self.gpu_mem_gb = gpu_mem_gb
        self.grid_memory = {}
        self._ao_opts = {}
        self.host_occ = host(mf.mo_occ)
        self.c = cp.asarray(mf.mo_coeff, dtype=cp.float64)
        self.occ = cp.asarray(mf.mo_occ, dtype=cp.float64)
        self.energy = cp.asarray(mf.mo_energy, dtype=cp.float64)
        self.restricted = self.c.ndim == 2

    def sync(self):
        self.xp.cuda.get_current_stream().synchronize()

    def ao(self, mol, coords, deriv=0):
        key = id(mol)
        if key not in self._ao_opts:
            self._ao_opts[key] = self._opt_factory(mol)
        return self._eval_ao(mol, coords, deriv=deriv, gdftopt=self._ao_opts[key])

    def blocks(self, coords, *, planned_points):
        yield from slices(coords, planned_points)

    def semilocal(self, coords):
        return semilocal(self, coords)


# Parent density contractions in native GPU AO order
# ----------------------------------------------------------------------------


def _opt(engine):
    key = id(engine.mol)
    if key not in engine._ao_opts:
        engine._ao_opts[key] = engine._opt_factory(engine.mol)
    return engine._ao_opts[key]


def _sorted_coeff(engine, opt, coeff):
    """Apply GPU4PySCF 1.8.1's AO permutation without an identity GEMM.

    Its ``opt.coeff`` is ``eye(mol.nao)[opt._ao_idx]``. CuPy's gather
    (including the padded indices) therefore gives the identical transform.
    No coefficient is screened, rounded or otherwise changed.
    """
    cp = engine.xp
    return cp.ascontiguousarray(coeff[cp.asarray(opt._ao_idx)].T)


def _copy_rows(cp, source, destination, start, width, row_pitch):
    """Copy contiguous device rows directly into final strided host rows.

    Sizes and offsets are float64 elements. Synchronize the current stream
    before its source owners can be reused; no pageable host staging array.
    """
    if not source.size:
        return
    cp.cuda.runtime.memcpy2DAsync(
        destination.ctypes.data + start * 8,
        row_pitch * 8,
        source.data.ptr,
        width * 8,
        width * 8,
        source.size // width,
        cp.cuda.runtime.memcpyDeviceToHost,
        cp.cuda.get_current_stream().ptr,
    )
    cp.cuda.get_current_stream().synchronize()


_SEMILOCAL = r"""
extern "C" __global__ void semilocal_reduce(
 const double* m, const double* lap, int no, int nb,
 long long n, long long offset, double* rho, double* grad,
 double* tau, double* laprho) {
 int r=blockDim.x*blockIdx.x+threadIdx.x;
 if(r>=nb) return;
 double den=0., gx=0., gy=0., gz=0., kin=0., dl=0.;
 long long stride=(long long)no*nb;
 for(int i=0;i<no;++i) {
   long long j=(long long)i*nb+r;
   double v=m[j], x=m[stride+j], y=m[2*stride+j], z=m[3*stride+j];
   den+=v*v; gx+=v*x; gy+=v*y; gz+=v*z;
   kin+=x*x+y*y+z*z; dl+=v*lap[j];
 }
 long long k=offset+r;
 rho[k]=den; grad[3*k]=2*gx; grad[3*k+1]=2*gy; grad[3*k+2]=2*gz;
 tau[k]=0.5*kin; laprho[k]=2*(dl+kin);
}
"""
_DENSITY_REDUCE = None
_TRACE = None
_LAPLACIAN = r"""
extern "C" __global__ void derivative_trace(double* a, long long size) {
 long long k=(long long)blockDim.x*blockIdx.x+threadIdx.x;
 if(k<size) a[4*size+k]=(a[4*size+k]+a[7*size+k])+a[9*size+k];
}
"""


def semilocal(engine, coords):
    global _DENSITY_REDUCE, _TRACE

    cp = engine.xp
    n = len(coords)
    shapes = {"rho": (2, n), "grad_rho": (2, n, 3), "laplacian": (2, n), "tau": (2, n)}
    with GridMemory(engine, "semilocal", 12 * n * 8) as memory:
        out = {k: np.empty(shape, dtype=np.float64) for k, shape in shapes.items()}
        if not n:
            return out
        opt = _opt(engine)
        coeff = [engine.c] if engine.restricted else engine.c
        occ = [engine.host_occ] if engine.restricted else engine.host_occ
        # Allocate persistent native AO data before the large transient AO
        # owners, so it cannot strand slices of those owners in the pool.
        opt.envs_cache
        occupied = [
            _sorted_coeff(engine, opt, c[:, np.asarray(o) != 0]) for c, o in zip(coeff, occ)
        ]
        ns = len(opt._ao_idx)
        if _DENSITY_REDUCE is None:
            _DENSITY_REDUCE = cp.RawKernel(_SEMILOCAL, "semilocal_reduce")
        if _TRACE is None:
            _TRACE = cp.RawKernel(_LAPLACIAN, "derivative_trace")
        per_point = 8 * (10 * ns + 5 * max(len(c) for c in occupied) + 12 + 32)
        resident = memory.device_outputs_fit(n, per_point)
        if resident:
            gpu_out = {k: cp.empty(shape, dtype=cp.float64) for k, shape in shapes.items()}

        def evaluate(points, sl):
            nb = len(points)
            a = engine._eval_ao(opt._sorted_mol, points, deriv=2, gdftopt=opt, transpose=False)
            # The individual second derivatives are not used after the trace.
            # Preserve (xx + yy) + zz exactly, while reusing the xx buffer.
            # A direct kernel avoids CuPy's conservative overlap-copy for
            # distinct derivative views sharing the same base allocation.
            size = ns * nb
            _TRACE(((size + 255) // 256,), (256,), (a, np.int64(size)))
            lap = a[4]
            values = (
                gpu_out
                if resident
                else {
                    k: cp.empty(
                        (len(occupied), nb, 3) if k == "grad_rho" else (len(occupied), nb),
                        dtype=cp.float64,
                    )
                    for k in out
                }
            )
            for spin, c in enumerate(occupied):
                m = cp.matmul(c, a[:4])
                ml = c @ lap
                _DENSITY_REDUCE(
                    ((nb + 127) // 128,),
                    (128,),
                    (
                        m,
                        ml,
                        np.int32(len(c)),
                        np.int32(nb),
                        np.int64(nb),
                        np.int64(spin * n + sl.start if resident else spin * nb),
                        values["rho"],
                        values["grad_rho"],
                        values["tau"],
                        values["laplacian"],
                    ),
                )
                del m, ml
            if not resident:
                for k, v in values.items():
                    components = 3 if k == "grad_rho" else 1
                    _copy_rows(
                        cp, v, out[k], sl.start * components, nb * components, n * components
                    )

        # Ten AO derivatives, occupied values/derivatives,
        # outputs and native coordinate/workspace owners; all float64.
        for sl, _ in memory.run(coords, per_point, 0, evaluate):
            pass
        if resident:
            for key in out:
                if engine.restricted:
                    gpu_out[key][1] = gpu_out[key][0]
                gpu_out[key].get(out=out[key])
            del gpu_out
        elif engine.restricted:
            for value in out.values():
                value[1] = value[0]
        del occupied
    return out


# Finite-Gaussian Coulomb potentials
# ----------------------------------------------------------------------------


class ExchangeCoulombTiles:
    """Reusable rectangular integral workspace on the CUDA default stream.

    GPU4PySCF 1.8.1's native fill_int2c2e kernel launches on that stream. Callers
    must order input/output dependencies with it (see hartree_potential).
    """

    def __init__(self, aux, block_points):
        import cupy as cp
        from pyscf import gto
        from gpu4pyscf.gto.mole import SortedGTO, PBCIntEnvVars, PTR_BAS_COORD, _scale_sp_ctr_coeff
        from gpu4pyscf.pbc.df.int2c2e import int2c2e_scheme, libpbc

        self.cp = cp
        self.lib = libpbc
        self.aux = SortedGTO.from_mol(aux)
        self.ncart = int(self.aux.p_ao_loc[-1])
        lmax = int(self.aux.uniq_l_ctr[:, 0].max())
        if lmax > 6:
            raise NotImplementedError("Upstream two-centre CUDA kernel supports auxiliary l <= 6")
        # The caller has planned the COMPLETE live allocation set. Unlike
        # MRF's helper, do not introduce a second independent memory cap.
        self.block_points = int(block_points)
        if self.block_points < 1:
            raise ValueError("Exchange Coulomb tile must contain at least one point")
        if self.block_points * (self.block_points + self.aux.nbas) > np.iinfo(np.int32).max:
            raise MemoryError("Exchange native shell-pair indices exceed int32")
        fake = gto.fakemol_for_charges(np.zeros((self.block_points, 3)), expnt=1e16)
        combined = fake + self.aux
        combined._bas[:, PTR_BAS_COORD] = combined._atm[
            combined._bas[:, gto.ATOM_OF], gto.PTR_COORD
        ]
        # fill_int2c2e uses these values solely as output row/column addresses.
        # All tasks have fake-charge bra and auxiliary ket, so overlapping
        # row/column numbering is intentional. The sentinel is the row stride.
        loc = np.concatenate(
            (np.arange(self.block_points, dtype=np.int32), self.aux.p_ao_loc)
        ).astype(np.int32)
        self.env = PBCIntEnvVars.new(
            combined.natm,
            combined.nbas,
            1,
            1,
            combined._atm,
            combined._bas,
            _scale_sp_ctr_coeff(combined),
            loc,
            cp.zeros((1, 3)),
        )
        self.coord_indices = cp.asarray(
            combined._atm[: self.block_points, gto.PTR_COORD, None] + np.arange(3)
        )
        nsp, stride, shared = int2c2e_scheme(0, gout_width=60)
        self.stride = cp.asarray(stride, dtype=np.int32)
        self.shared = int(shared[0, : lmax + 1].max())
        offsets = np.append(0, np.cumsum(self.aux.l_ctr_counts))
        pairs = []
        batches = []
        count = 0
        for group, (l, nprim) in enumerate(self.aux.uniq_l_ctr):
            js = cp.arange(offsets[group], offsets[group + 1], dtype=np.int32) + self.block_points
            ids = (
                cp.arange(self.block_points, dtype=np.int32)[:, None] * combined.nbas + js
            ).ravel()
            pairs.append(ids)
            batches.append(cp.arange(count, count + len(ids), int(nsp[0, l]) * 4, dtype=np.int32))
            count += len(ids)
        self.pairs = cp.concatenate(pairs)
        self.batches = cp.concatenate(batches + [cp.asarray([count], dtype=np.int32)])
        self.values = cp.empty((self.block_points, self.ncart), dtype=cp.float64)
        self.memory = {
            "block_points": self.block_points,
            "integrals_mib": self.values.nbytes / 1024**2,
            "shell_pairs_mib": self.pairs.nbytes / 1024**2,
        }

    def evaluate(self, points):
        cp = self.cp
        n = len(points)
        if n == 0:
            return self.values[:0]
        if n > self.block_points:
            raise ValueError("Exchange tile exceeds its allocated workspace")
        env = self.env._env_ref_holder[2]
        env[self.coord_indices[:n]] = cp.asarray(points, dtype=cp.float64)
        # Padding reuses a real point; its rows are discarded, never sampled.
        if n < self.block_points:
            env[self.coord_indices[n:]] = cp.asarray(points[-1], dtype=cp.float64)
        ptr = lambda a: ctypes.c_void_p(a.data.ptr)
        err = self.lib.fill_int2c2e(
            ptr(self.values),
            ctypes.byref(self.env),
            ctypes.c_double(0.0),
            ctypes.c_double(1.0),
            ctypes.c_double(1.0),
            ctypes.c_int(self.shared),
            ctypes.c_int(len(self.batches) - 1),
            ptr(self.batches),
            ptr(self.pairs),
            ptr(self.stride),
        )
        if err:
            raise RuntimeError("CUDA exchange two-centre integral evaluation failed")
        return self.values[:n]


# GPU4PySCF host-output integral adapters
# ----------------------------------------------------------------------------


def host_cart_factors(
    intopt, occ_coeff_set, vir_coeff_set, j3c_ovl_cart_set, aux_batch_size, log=None
):
    """Get 3-center overlap integrals in Cartesian basis on GPU by batch of auxiliary basis.

    Parameters
    ----------
    intopt : gpu4pyscf.df.int3c2e_bdiv.Int3c2eOpt
    occ_coeff_set : list of cupy.ndarray | list of numpy.ndarray
    vir_coeff_set : list of cupy.ndarray | list of numpy.ndarray
    j3c_ovl_cart_set : list of cupy.ndarray | list of numpy.ndarray
    aux_batch_size : int | None
    log : gpupyscf.lib.logger.Logger, optional

    See also
    --------
    get_j3c_ovl_gpu_bdiv
    """
    from gpu4pyscf.mp.dfmp2_addons import cp, np, cupy, logger, ndarray, contract

    mol = intopt.mol.mol
    aux = intopt.auxmol.mol
    on_gpu = isinstance(j3c_ovl_cart_set[0], cp.ndarray)
    if log is None:
        log = logger.new_logger(mol, verbose=mol.verbose)
    t0 = log.init_timer()
    t1 = t0

    # determine the number of tasks (spins/properties)
    nset = len(j3c_ovl_cart_set)
    assert len(occ_coeff_set) == len(vir_coeff_set) == nset

    # check dimensionality of input arrays
    naux_cart = aux.nao_cart()
    aux_batch_size = min(aux_batch_size, naux_cart)

    for occ_coeff, vir_coeff, j3c_ovl_cart in zip(occ_coeff_set, vir_coeff_set, j3c_ovl_cart_set):
        nao = mol.nao
        nocc, nvir, _ = j3c_ovl_cart.shape
        assert occ_coeff.shape == (nao, nocc)
        assert vir_coeff.shape == (nao, nvir)
        assert j3c_ovl_cart.shape[2] == naux_cart

    # allocate temporary buffers
    nao_cart = mol.nao_cart()
    buffer_size = nao_cart * nao_cart * aux_batch_size
    cache1 = cp.empty(buffer_size, dtype=np.float64)
    cache2 = cp.empty(buffer_size, dtype=np.float64)
    cache_j3c = cache2 if nset == 1 else cp.empty(buffer_size, dtype=np.float64)

    # if all auxiliary basis can fit into memory, we should not use a finite value to split batch
    aux_batch_size_evaluator = aux_batch_size if aux_batch_size < aux.nao_cart() else None
    int3c2e_gen, aux_sorting, ao_pair_offsets, aux_offsets = intopt.int3c2e_evaluator(
        cart=True, reorder_aux=True, aux_batch_size=aux_batch_size_evaluator
    )
    assert len(ao_pair_offsets) == 2, "AO pair should not be sliced."

    rows, cols = divmod(
        intopt.pair_and_diag_indices(cart=True, original_ao_order=False)[0], nao_cart
    )
    aux_resorting = np.argsort(aux_sorting)
    if not on_gpu:
        aux_resorting = aux_resorting.get()

    # transform MO coefficients to cartesian basis (also apply AO sorting)
    occ_coeff_cart_set = []
    vir_coeff_cart_set = []
    for iset in range(nset):
        occ_coeff_cart_set.append(intopt.mol.C_dot_mat(occ_coeff_set[iset]))
        vir_coeff_cart_set.append(intopt.mol.C_dot_mat(vir_coeff_set[iset]))

    t1 = log.timer_debug1("prepare for j3c_ovl_cart_bdiv", *t1)

    nbatch_aux = len(aux_offsets) - 1
    for ibatch_aux, (p0, p1) in enumerate(zip(aux_offsets[:-1], aux_offsets[1:])):
        naux_batch = p1 - p0
        # step 1: evaluate compressed 3c-2e ERI
        j3c_raw = int3c2e_gen(aux_batch_id=ibatch_aux, out=cache1)
        cupy.cuda.stream.get_current_stream().synchronize()
        t1 = log.timer_debug1(f"compute int3c2e for aux batch {ibatch_aux}/{nbatch_aux}", *t1)
        # step 2: decompress to full 3c-2e ERI (3-dimension tensor)
        j3c_expand_pair = ndarray(
            [nao_cart, nao_cart, naux_batch], dtype=np.float64, buffer=cache_j3c
        )
        j3c_expand_pair[:] = 0.0
        j3c_expand_pair[rows, cols] = j3c_raw
        j3c_expand_pair[cols, rows] = j3c_raw
        j3c_raw = None
        cupy.cuda.stream.get_current_stream().synchronize()
        t1 = log.timer_debug1(f"decompress int3c2e for aux batch {ibatch_aux}/{nbatch_aux}", *t1)
        for iset in range(nset):
            occ_coeff_cart = occ_coeff_cart_set[iset]
            vir_coeff_cart = vir_coeff_cart_set[iset]
            nocc = occ_coeff_cart.shape[1]
            nvir = vir_coeff_cart.shape[1]
            # step 3: contract over first AO index (nao_cart -> nocc)
            j3c_obx_sorted = ndarray([nocc, nao_cart, naux_batch], dtype=np.float64, buffer=cache1)
            contract("uvP, ui -> ivP", j3c_expand_pair, occ_coeff_cart, out=j3c_obx_sorted)
            # step 4: contract over second AO index (nao_cart -> nvir)
            j3c_ovl_sorted = ndarray([nocc, nvir, naux_batch], dtype=np.float64, buffer=cache2)
            contract("ivP, va -> iaP", j3c_obx_sorted, vir_coeff_cart, out=j3c_ovl_sorted)
            j3c_obx_sorted = None
            # step 5: inplace store the result with aux resorting
            if on_gpu:
                j3c_ovl_cart_set[iset][:, :, aux_resorting[p0:p1]] = j3c_ovl_sorted
            else:
                j3c_ovl_cart_set[iset][:, :, aux_resorting[p0:p1]] = j3c_ovl_sorted.get(
                    blocking=True
                )
            j3c_ovl_sorted = None
            cupy.cuda.stream.get_current_stream().synchronize()
            t1 = log.timer_debug1(
                f"contract for aux batch {ibatch_aux}/{nbatch_aux}, set {iset}/{nset}", *t1
            )
    t0 = log.timer("get_j3c_ovl_cart_bdiv_gpu", *t0)
    return nbatch_aux  # Observability only: count actual shell-aligned batches.


def host_spherical_factors(intopt, j3c_ovl_cart_set, batch_ov_size, j3c_ovl_set=None, log=None):
    """Apply AO transformation (sorting, sph2cart) for j3c_ovl_cart_set in batch of occupied-virtual pairs.

    Parameters
    ----------
    intopt : gpu4pyscf.df.int3c2e_bdiv.Int3c2eOpt
    j3c_ovl_cart_set : list of cupy.ndarray | list of numpy.ndarray
    batch_ov_size : int
    j3c_ovl_set : list of cupy.ndarray | list of numpy.ndarray, optional
    log : gpu4pyscf.lib.logger.Logger, optional

    Returns
    -------
    list of cupy.ndarray | list of numpy.ndarray

    See also
    --------
    get_j3c_ovl_gpu_bdiv
    """
    from gpu4pyscf.mp.dfmp2_addons import cp, np, cupy, logger, ndarray, contract

    mol = intopt.mol.mol
    aux = intopt.auxmol.mol
    if log is None:
        log = logger.new_logger(mol, verbose=mol.verbose)
    t0 = log.init_timer()

    # get maximum nocc * nvir size
    nov = 0
    for j3c_ovl_cart in j3c_ovl_cart_set:
        nocc, nvir, _ = j3c_ovl_cart.shape
        nov = max(nov, nocc * nvir)
    batch_ov_size = min(batch_ov_size, nov)

    naux = aux.nao
    cache = cp.empty([batch_ov_size, naux], dtype=np.float64)
    on_gpu = (
        isinstance(j3c_ovl_set[0], cp.ndarray)
        if j3c_ovl_set is not None
        else isinstance(j3c_ovl_cart_set[0], cp.ndarray)
    )

    # create j3c_ovl_set if not given
    if j3c_ovl_set is not None:
        assert len(j3c_ovl_set) == len(j3c_ovl_cart_set)
    else:
        j3c_ovl_set = []
        for j3c_ovl_cart in j3c_ovl_cart_set:
            nocc, nvir, _ = j3c_ovl_cart.shape
            if on_gpu:
                j3c_ovl = ndarray([nocc, nvir, naux], dtype=j3c_ovl_cart.dtype, buffer=j3c_ovl_cart)
            else:
                j3c_ovl = np.ndarray(
                    shape=[nocc, nvir, naux], dtype=j3c_ovl_cart.dtype, buffer=j3c_ovl_cart
                )
            j3c_ovl_set.append(j3c_ovl)

    for j3c_ovl_cart, j3c_ovl in zip(j3c_ovl_cart_set, j3c_ovl_set):
        assert j3c_ovl_cart.ndim == 3
        nocc, nvir, naux_cart = j3c_ovl_cart.shape
        assert (
            naux_cart == aux.nao_cart()
        ), f"Auxiliary basis inconsistent: {naux_cart} != {aux.nao_cart()}."
        j3c_ovl_cart = j3c_ovl_cart.reshape(nocc * nvir, naux_cart)
        j3c_ovl = j3c_ovl.reshape(nocc * nvir, naux)
        for idx_ov in range(0, nocc * nvir, batch_ov_size):
            nbatch_ov = min(nocc * nvir - idx_ov, batch_ov_size)
            slc = range(idx_ov, idx_ov + nbatch_ov)
            cache[:] = 0.0
            cache_ovl = intopt.auxmol.mat_dot_C(j3c_ovl_cart[slc], out=cache)
            if on_gpu:
                j3c_ovl[slc] = cache_ovl
            else:
                j3c_ovl[slc] = cache_ovl.get(blocking=True)
            cache_ovl = None
    cupy.cuda.stream.get_current_stream().synchronize()
    t0 = log.timer("sph2cart_j3c_ovl_bdiv", *t0)
    return j3c_ovl_set


# Internal field names
# ----------------------------------------------------------------------------


def field_key(family, channel, kappa=None, multiple=False):
    base = family if not multiple else f"{family}_k{repr(float(kappa))}"
    return f'ec_{base}{"" if channel=="tot" else "_"+channel}_array'


# GPU DF amplitude evaluation
# ----------------------------------------------------------------------------


# GPU4PySCF caches device arrays (e.g. bdiv launch schemes) across calls.
# PooledMemory keeps only a weak reference to its pool. Keep the owning pool
# alive while any such allocation survives, even after freeing unused blocks.
_DF_POOLS = []


def validate_memory(gpu_mem_gb, host_mem_gb):
    """Validate optional scratch budgets in decimal GB."""
    for name, value in (("gpu_mem_gb", gpu_mem_gb), ("host_mem_gb", host_mem_gb)):
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a positive number or None")
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def _orbitals(engine):
    coeff = [engine.c] if engine.restricted else list(engine.c)
    occ = [host(engine.occ)] if engine.restricted else host(engine.occ)
    eps = [engine.energy] if engine.restricted else list(engine.energy)
    co, cv, eo, ev = [], [], [], []
    for c, n, e in zip(coeff, occ, eps):
        oi = np.flatnonzero(n > 0)
        vi = np.flatnonzero(n == 0)
        co.append(c[:, oi])
        cv.append(c[:, vi])
        eo.append(e[oi])
        ev.append(e[vi])
    return co, cv, eo, ev


def df_fields(
    engine,
    coords,
    density,
    aux_basis,
    *,
    kappa=2.0,
    gpu_mem_gb=None,
    host_mem_gb=None,
    progress=False,
):
    """Evaluate regularized OS/SS strat1 fields with a bounded CuPy pool."""
    validate_memory(gpu_mem_gb, host_mem_gb)
    ks = (float(kappa),)
    requests = [("kmp2", ks[0], ("os", "ss"))]
    cp = engine.xp
    original_ao_keys = set(engine._ao_opts)
    _DF_POOLS[:] = [p for p in _DF_POOLS if p.used_bytes()]
    free = cp.cuda.runtime.memGetInfo()[0]
    budget = int(0.8 * free) if gpu_mem_gb is None else int(float(gpu_mem_gb) * 1e9)
    if budget <= 0 or budget > free:
        raise MemoryError(f"strat1 requested {budget/1e9:.3f} GB; only {free/1e9:.3f} GB free")
    pool = TrackedPool(cp, budget)
    try:
        with cp.cuda.using_allocator(pool.malloc):
            out, meta = streamed_fields(
                engine,
                coords,
                density,
                aux_basis,
                requests,
                ks,
                budget,
                None,
                None,
                progress,
                pool,
                host_mem_gb,
            )
            engine.sync()
            meta.update(
                gpu_pool_reserved_at_exit_bytes=pool.total_bytes(),
                gpu_pool_live_at_exit_bytes=pool.used_bytes(),
                gpu_pool_peak_live_bytes=pool.peak_live,
                gpu_pool_peak_reserved_bytes=pool.peak_reserved,
            )
    finally:
        engine.sync()
        for key in set(engine._ao_opts) - original_ao_keys:
            engine._ao_opts.pop(key)
        pool.free_all_blocks()
        if pool.used_bytes():
            _DF_POOLS.append(pool)
    engine.mp2_details = meta
    return out


# GPU grid contractions
# ----------------------------------------------------------------------------


_FIELD_REDUCE = None

# CUDA scheduling crossover measured on the target A100; this is a tensor
# layout threshold, never a numerical screening or density threshold.
_PACKED_KERNEL_MAX_BYTES = 256 * 1024**2


def contract_field(potential, kernel, occupied, virtual, out, available, stats):
    """Larger FP64 GEMM plus a bounded fused reduction of the same orbital sum.

    Small kernels combine occupied rows using the largest legal scratch
    matrix. Large kernels retain the original occupied GEMMs, which measured
    faster on A100 while reusing the same resident data and grid tile. No full grid x occupied x virtual array survives a
    batch and no such array is transferred to the host.
    """
    import cupy as cp

    global _FIELD_REDUCE
    ni, nv, nq = kernel.shape
    nr = len(potential)
    batch = min(ni, max(1, (int(available) - 16 * 1024**2) // max(1, nr * nv * 8)))
    if ni == 1 or batch == 1 or kernel.nbytes > _PACKED_KERNEL_MAX_BYTES:
        stats["field_contraction_mode"] = "occupied GEMMs"
        for i in range(ni):
            out += occupied[:, i] * cp.sum(virtual * (potential @ kernel[i].T), axis=1)
        stats["field_gemm_blocks"] = stats.get("field_gemm_blocks", 0) + ni
        return
    stats["field_contraction_mode"] = "packed GEMM and FP64 reduction"
    if _FIELD_REDUCE is None:
        _FIELD_REDUCE = cp.RawKernel(
            r"""
        extern "C" __global__ void reduce_field(
            const double* work, const double* po, const double* pv,
            double* out, long long ni, long long nv,
            long long i0, long long count) {
            long long r=blockIdx.x, width=count*nv;
            double sum=0.;
            for(long long x=threadIdx.x;x<width;x+=blockDim.x) {
                long long i=x/nv,a=x%nv;
                sum+=work[r*width+x]*pv[r*nv+a]*po[r*ni+i0+i];
            }
            __shared__ double partial[256];
            partial[threadIdx.x]=sum; __syncthreads();
            for(int stride=128;stride;stride/=2) {
                if(threadIdx.x<stride)partial[threadIdx.x]+=partial[threadIdx.x+stride];
                __syncthreads();
            }
            if(threadIdx.x==0)out[r]+=partial[0];
        }
        """,
            "reduce_field",
            options=("--std=c++11",),
        )
    stats["field_gemm_max_occupied"] = max(stats.get("field_gemm_max_occupied", 0), batch)
    for start in range(0, ni, batch):
        count = min(batch, ni - start)
        work = potential @ kernel[start : start + count].reshape(count * nv, nq).T
        _FIELD_REDUCE((nr,), (256,), (work, occupied, virtual, out, ni, nv, start, count))
        stats["field_gemm_blocks"] = stats.get("field_gemm_blocks", 0) + 1
        del work


def potential_layout(aux):
    from gpu4pyscf.gto.mole import SortedGTO

    if max(aux.bas_angular(i) for i in range(aux.nbas)) > 6:
        raise NotImplementedError(
            "GPU auxiliary potentials support angular momentum through i (l <= 6)"
        )
    sorted_aux = SortedGTO.from_mol(aux)
    return dict(
        ncart=int(sorted_aux.p_ao_loc[-1]),
        nshells=int(sorted_aux.nbas),
        point_limit=native_point_limit(sorted_aux.nbas),
    )


def potential_blocks(engine, aux, coords, stats, layout, *, planned_points):
    import cupy as cp

    current, null = cp.cuda.get_current_stream(), cp.cuda.Stream.null
    if current.ptr != null.ptr:
        null.wait_event(current.record())
    tile_points = min(planned_points, len(coords), layout["point_limit"])
    with null:
        tiles = ExchangeCoulombTiles(aux, tile_points)
    try:
        for sl, points in engine.blocks(coords, planned_points=planned_points):
            if current.ptr != null.ptr:
                null.wait_event(current.record())
            with null:
                if len(points) <= tile_points:
                    value = tiles.aux.mat_dot_C(tiles.evaluate(points))
                    stats["cuda_potential_blocks"] = stats.get("cuda_potential_blocks", 0) + 1
                else:
                    value = cp.empty((len(points), aux.nao), dtype=cp.float64)
                    for begin in range(0, len(points), tile_points):
                        end = min(begin + tile_points, len(points))
                        piece = tiles.aux.mat_dot_C(tiles.evaluate(points[begin:end]))
                        value[begin:end] = piece
                        del piece
                        stats["cuda_potential_blocks"] = stats.get("cuda_potential_blocks", 0) + 1
                ready = null.record()
            current.wait_event(ready)
            yield sl, points, value
            del value
    finally:
        current.synchronize()
        null.synchronize()
        del tiles


# Disk-backed pair contractions
# ----------------------------------------------------------------------------


def disk_plan(
    no,
    nv,
    q,
    nkeys,
    partners,
    available,
    host_available,
    *,
    kernels_resident=False,
    start=0,
    cap=None,
):
    """Minimize observation batches times partner/virtual tiles jointly.

    For each distinct ceil(no / batch), its smallest sufficient batch leaves
    the most space for partner tiles. Testing those choices is equivalent to
    testing every observation-batch size under this live-array bound.
    """
    plane = nv * q * 8
    fixed = 16 * MIB + max(nkeys, 3) * plane
    extra_keys = 0 if kernels_resident else nkeys
    maximum = min(no, max(0, (host_available - 4096) // max(1, plane)))
    candidates = []
    previous = None
    for count in range(1, maximum + 1):
        groups = ceil(no / count)
        if groups == previous:
            continue
        previous = groups
        owned = count * (1 + extra_keys) * plane
        free = available - fixed - owned
        if free < 4096:
            continue
        plans = []
        try:
            for p in partners:
                resident = p["same"] and start == 0 and count == p["no"]
                plan = pair_plan(
                    p["no"],
                    nv,
                    p["nv"],
                    q,
                    free,
                    cap=cap,
                    resident=resident,
                    host_available=host_available,
                )
                plans.append(dict(spin=p["spin"], **plan))
        except MemoryError:
            continue
        pair_count = sum(p["total_blocks_per_observation"] for p in plans)
        score = (groups * pair_count, groups, -count)
        gpu = fixed + owned + max((p["estimated_bytes"] for p in plans), default=0)
        host = max(
            [count * plane + 4096]
            + [
                (0 if p["resident_factors"] else p["j"] * t["nv"] * q * 8 + 4096)
                for p, t in zip(plans, partners)
            ]
        )
        candidates.append(
            (
                score,
                dict(
                    occupied_block=count,
                    observation_batches=groups,
                    total_pair_batches=groups * pair_count,
                    pairs=plans,
                    gpu_estimated_bytes=gpu,
                    host_estimated_bytes=host,
                    gpu_available_bytes=available,
                    host_available_bytes=host_available,
                ),
            )
        )
    if not candidates:
        raise MemoryError("MP2 budgets cannot fit one observation and one partner tile")
    return min(candidates, key=lambda x: x[0])[1]


def accumulate(
    engine, factors, eo, ev, chol_t, keys, arena, pool, budget, pair_cap, progress, stats, kstores
):
    import cupy as cp
    import numpy as np
    from cupyx.scipy.linalg import solve_triangular

    q = chol_t.shape[0]
    analytic = {key: 0.0 for key in keys}
    stats.setdefault("observation_plans", [])
    stats.setdefault("disk_observation_batches", 0)
    stats.setdefault("factor_upload_blocks", 0)
    stats.setdefault("factor_upload_bytes", 0)
    stats.setdefault("reused_factor_tiles", 0)
    reported = set()

    def block(s, i0, count, plan, partners):
        no, nv, _ = factors[s].shape
        i1 = i0 + count
        kernels = {
            key: (
                kstores[s][key].array[i0:i1]
                if kstores[s][key].kind == "gpu"
                else cp.empty((count, nv, q), dtype=cp.float64)
            )
            for key in keys
        }
        for value in kernels.values():
            value.fill(0)
        with _FactorUploader(max(0, arena.remaining - 4096)) as staging:
            fi_batch = staging.upload(factors[s], slice(i0, i1))
            stats["factor_upload_blocks"] += 1
            stats["factor_upload_bytes"] += fi_batch.nbytes
            for partner, p in zip(partners, plan["pairs"]):
                t = partner["spin"]
                no2, nv2, _ = factors[t].shape
                allowos = partner["os"]
                allowss = partner["ss"]
                wants = partner["wants"]
                signature = (s, t, p["j"], p["a"], p["b"], p["resident_factors"])
                if signature not in reported:
                    reported.add(signature)
                    stats["pair_plans"].append(
                        dict(
                            spin=s,
                            partner=t,
                            occupied_observation=i0 + 1,
                            observation_count=count,
                            host_factor_tile_bytes=(
                                0 if p["resident_factors"] else p["j"] * nv2 * q * 8
                            ),
                            **{k: v for k, v in p.items() if k != "spin"},
                        )
                    )
                for j0 in range(0, no2, p["j"]):
                    j1 = min(no2, j0 + p["j"])
                    nj = j1 - j0
                    _clear(engine, pool)
                    if t == s and i0 <= j0 and j1 <= i1:
                        fj = fi_batch[j0 - i0 : j1 - i0]
                        stats["reused_factor_tiles"] += 1
                    else:
                        fj = staging.upload(factors[t], slice(j0, j1))
                        stats["factor_upload_blocks"] += 1
                        stats["factor_upload_bytes"] += fj.nbytes
                    if progress:
                        print(
                            f'Strategy 1 partner block {j0//p["j"]+1}/{ceil(no2/p["j"])}; occupied {i0+1}-{i1}',
                            flush=True,
                        )
                    for ii in range(count):
                        i = i0 + ii
                        fi = fi_batch[ii]
                        _clear(engine, pool)
                        for a0 in range(0, nv, p["a"]):
                            aa = slice(a0, min(nv, a0 + p["a"]))
                            na = aa.stop - aa.start
                            for b0 in range(0, nv2, p["b"]):
                                bb = slice(b0, min(nv2, b0 + p["b"]))
                                nb = bb.stop - bb.start
                                right = cp.ascontiguousarray(fj[:, bb]).reshape(nj * nb, q)
                                g = fi[aa] @ right.T
                                den = (
                                    ev[s][aa, None]
                                    + cp.broadcast_to(ev[t][bb], (nj, nb)).reshape(1, -1)
                                    - eo[s][i]
                                    - cp.repeat(eo[t][j0:j1], nb)[None, :]
                                )
                                ex = None
                                if allowss and any(c in ("ss", "tot") for _, c in wants):
                                    if na == nv and nb == nv2 and nv == nv2:
                                        ex = (
                                            g.reshape(nv, nj, nv)
                                            .transpose(2, 1, 0)
                                            .reshape(nv, nj * nv)
                                        )
                                    else:
                                        ex = (
                                            (
                                                fi[bb]
                                                @ cp.ascontiguousarray(fj[:, aa])
                                                .reshape(nj * na, q)
                                                .T
                                            )
                                            .reshape(nb, nj, na)
                                            .transpose(2, 1, 0)
                                            .reshape(na, nj * nb)
                                        )
                                increments = {}
                                for key in wants:
                                    k, c = key
                                    scale = 1.0 / den
                                    if k is not None:
                                        scale *= (-cp.expm1(-k * den)) ** 2
                                    pref = 1.0 if engine.restricted else 0.5
                                    if c == "tot":
                                        weight = -g * scale * pref if allowos else None
                                        if allowss:
                                            same = pref * ((ex - g) * scale)
                                            weight = same if weight is None else weight + same
                                            del same
                                    else:
                                        weight = (
                                            -g * scale * pref
                                            if c == "os"
                                            else pref * ((ex - g) * scale)
                                        )
                                    increments[key] = weight @ right
                                    del weight, scale
                                for key, value in increments.items():
                                    kernels[key][ii, aa] += value
                                del increments, value, g, den, ex, right
                                stats["pair_blocks"] += 1
                        del fi
                    del fj
            stats["factor_host_staging_peak_bytes"] = max(
                stats.get("factor_host_staging_peak_bytes", 0), staging.peak_bytes
            )
        # The factor staging buffer is released before any kernel .get output.
        energies = []
        for ii in range(count):
            row = {}
            for key in keys:
                kernel = kernels[key][ii]
                row[key] = float(host(cp.sum(fi_batch[ii] * kernel)))
                value = solve_triangular(chol_t, kernel.T, lower=False).T
                kstores[s][key].write(
                    i0 + ii, value if kstores[s][key].kind == "gpu" else value.get(blocking=True)
                )
                del value, kernel
            energies.append(row)
        del fi_batch, kernels
        return energies

    for s, f in enumerate(factors):
        no, nv, _ = f.shape
        if not no or not nv:
            continue
        partners = []
        for t, allowos, allowss in (
            [(s, engine.restricted, True)]
            if engine.restricted
            else [(s, False, True), (1 - s, True, False)]
        ):
            no2, nv2, _ = factors[t].shape
            wants = [
                key
                for key in keys
                if key[1] == "tot" or (key[1] == "os" and allowos) or (key[1] == "ss" and allowss)
            ]
            if not no2 or not nv2 or not wants:
                continue
            gap = float(host(ev[s].min() - eo[s].max() + ev[t].min() - eo[t].max()))
            if not np.isfinite(gap) or gap <= 1e-12:
                raise ValueError("Local MP2 requires positive occupied-virtual energy denominators")
            partners.append(
                dict(spin=t, no=no2, nv=nv2, same=t == s, os=allowos, ss=allowss, wants=wants)
            )
        i0 = 0
        fraction = 1.0
        while i0 < no:
            _clear(engine, pool)
            available = int(_remaining(pool, budget) * fraction)
            plan = disk_plan(
                no - i0,
                nv,
                q,
                len(keys),
                partners,
                available,
                arena.remaining,
                kernels_resident=all(k.kind == "gpu" for k in kstores[s].values()),
                start=i0,
                cap=pair_cap,
            )
            count = plan["occupied_block"]
            stats["observation_plans"].append(dict(spin=s, occupied_start=i0, **plan))
            if progress:
                done = stats["disk_observation_batches"]
                print(
                    f'Strategy 1 observation batch {done+1}/{done+plan["observation_batches"]}: spin {s+1}, occupied {i0+1}-{i0+count}/{no}',
                    flush=True,
                )
            try:
                energies = block(s, i0, count, plan, partners)
            except cp.cuda.memory.OutOfMemoryError:
                stats["retries"] += 1
                import gc

                gc.collect()
                _clear(engine, pool)
                fraction *= 0.5
                if fraction < 1 / 1024:
                    raise MemoryError(
                        "MP2 minimum observation/partner tile cannot fit the GPU budget"
                    )
                if progress:
                    print(
                        "Strategy 1 allocation retry: smaller uncommitted observation/partner batch",
                        flush=True,
                    )
                continue
            # Preserve the original order of the analytical scalar summation.
            for row in energies:
                for key, value in row.items():
                    analytic[key] += value
            i0 += count
            stats["disk_observation_batches"] += 1
            fraction = 1.0
    _clear(engine, pool)
    return kstores, analytic


# Streamed strat1 evaluation
# ----------------------------------------------------------------------------


def _device_storage_fits(required, available):
    return required < available


def _remaining(pool, budget):
    return int(budget - pool.used_bytes())


def _clear(engine, pool):
    engine.sync()
    pool.free_all_blocks()


def _build_factors(engine, aux, co, cv, arena, pool, budget, scratch_mib, progress):
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular
    from gpu4pyscf.df.int3c2e_bdiv import Int3c2eOpt
    from gpu4pyscf.mp import dfmp2_addons as addons

    q = aux.nao
    qc = aux.nao_cart()
    nc = engine.mol.nao_cart()
    if 3 * q * q * 8 + 32 * MIB >= _remaining(pool, budget):
        raise MemoryError(
            "MP2 budget cannot fit the unchanged full Coulomb metric and Cholesky workspace"
        )
    engine.sync()
    counts = {
        "factor_retries": 0,
        "integral_calls": 0,
        "whitening_blocks": 0,
        "raw_host_bytes": 0,
        "auxiliary_batch": None,
        "ov_transform_batch": None,
        "factor_output_blocks": 0,
    }
    with cp.cuda.Stream.null:
        opt = Int3c2eOpt(engine.mol, aux).build()
        metric = addons.get_j2c_bdiv(opt)
        chol = cp.linalg.cholesky(metric)
        if not bool(cp.all(cp.isfinite(chol))):
            raise np.linalg.LinAlgError(
                "DF Coulomb metric is not positive definite; no automatic rank truncation"
            )
        del metric
        chol = cp.asfortranarray(chol)
        _clear(engine, pool)
        shapes = [(c.shape[1], v.shape[1], q) for c, v in zip(co, cv)]
        cart_shapes = [(*shape[:2], qc) for shape in shapes]
        active = [s for s, shape in enumerate(shapes) if shape[0] and shape[1]]
        cart_bytes = sum(np.prod(cart_shapes[s], dtype=np.int64) * 8 for s in active)
        reserve = 32 * MIB + 32 * nc * nc + sum(c.size + v.size for c, v in zip(co, cv)) * 16
        caches = 2 if len(active) == 1 else 3
        # Native shells require at least their largest Cartesian degeneracy.
        min_aux = max(
            (
                (aux.bas_angular(i) + 1) * (aux.bas_angular(i) + 2) // 2 * aux.bas_nctr(i)
                for i in range(aux.nbas)
            ),
            default=1,
        )
        resident = _device_storage_fits(
            cart_bytes + reserve + caches * nc * nc * 8 * min_aux, _remaining(pool, budget)
        )
        stores = []
        raw_stores = []
        targets = []
        for s, shape in enumerate(shapes):
            if s not in active:
                stores.append(DeviceStore(cp.empty(shape), "factor_empty"))
                continue
            if resident:
                store = DeviceStore(cp.empty(cart_shapes[s]), f"factor_{s}")
                stores.append(store)
                targets.append(store.array)
            else:
                stores.append(arena.allocate(shape, f"factor_{s}", force="disk"))
                raw = arena.allocate((qc, shape[0] * shape[1]), f"raw_{s}", force="disk")
                raw_stores.append(raw)
                targets.append(AuxiliarySink(raw, *cart_shapes[s]))
        if active:
            fraction = 1.0
            while True:
                try:
                    _clear(engine, pool)
                    scratch = max(0, int((_remaining(pool, budget) - reserve) * fraction))
                    if scratch_mib is not None:
                        scratch = min(scratch, int(scratch_mib * MIB))
                    ab = largest_rows(qc, caches * nc * nc * 8, scratch)
                    if not resident:
                        # Native .get tile and the sorted auxiliary-major host tile.
                        ab = min(
                            ab,
                            largest_rows(
                                qc,
                                3 * max(t.shape[0] * t.shape[1] for t in targets) * 8,
                                arena.remaining,
                            ),
                        )
                    if ab < min_aux:
                        raise MemoryError(
                            "MP2 budgets cannot fit one unchanged auxiliary shell output"
                        )
                    counts["auxiliary_batch"] = ab
                    if progress:
                        print(
                            f"DF factors: auxiliary target {ab}/{qc}; "
                            + ("GPU resident" if resident else "bounded host tiles to disk"),
                            flush=True,
                        )
                    actual_blocks = host_cart_factors(
                        opt, [co[s] for s in active], [cv[s] for s in active], targets, ab
                    )
                    counts["integral_calls"] += 1
                    counts["factor_output_blocks"] = actual_blocks
                    _clear(engine, pool)
                    # Cartesian construction and the final spherical factors
                    # have different lifetimes and sizes. A lossless raw disk
                    # build need not force all later pair work through disk.
                    # Retain the complete compact factors when they fit along
                    # with minimum pair/solve workspaces; otherwise preserve
                    # the existing fully bounded disk path.
                    if not resident and any(stores[s].kind != "gpu" for s in active):
                        compact_bytes = sum(np.prod(shapes[s], dtype=np.int64) * 8 for s in active)
                        maxv = max(shapes[s][1] for s in active)
                        minimum_work = (
                            32 * MIB
                            + maxv * maxv * 64
                            + 8 * maxv * q * 8
                            + max(2 * qc + q, 3 * q) * 8
                        )
                        if _device_storage_fits(
                            compact_bytes + minimum_work, _remaining(pool, budget)
                        ):
                            promoted = []
                            try:
                                for s in active:
                                    promoted.append(
                                        (s, DeviceStore(cp.empty(shapes[s]), f"factor_{s}"))
                                    )
                            except cp.cuda.memory.OutOfMemoryError:
                                for _, owner in promoted:
                                    owner.close()
                                promoted = []
                                _clear(engine, pool)
                            if len(promoted) == len(active):
                                for s, owner in promoted:
                                    stores[s].close()
                                    stores[s] = owner
                                counts["compact_factors_promoted_to_gpu"] = True
                                if progress:
                                    print(
                                        "DF factors: compact spherical factors fit on GPU after raw construction",
                                        flush=True,
                                    )
                            del promoted
                    if resident:
                        nov = max(t.shape[0] * t.shape[1] for t in targets)
                        rows = largest_rows(
                            nov,
                            (qc + 4 * q) * 8,
                            int(_remaining(pool, budget) * fraction),
                            fixed=16 * MIB,
                        )
                        result = host_spherical_factors(opt, targets, rows)
                        for s, array in zip(active, result):
                            stores[s].array = array
                            stores[s].shape = shapes[s]
                        del result, array
                    else:
                        # GPU F-to-C layout: both Cartesian owners plus spherical
                        # output coexist. The later solve has no Cartesian owner;
                        # reserve three spherical tiles for that phase.
                        rows = min(
                            largest_rows(
                                max(t.shape[0] * t.shape[1] for t in targets),
                                max(2 * qc + q, 3 * q) * 8,
                                int(_remaining(pool, budget) * fraction),
                                fixed=16 * MIB,
                            ),
                            largest_rows(
                                max(t.shape[0] * t.shape[1] for t in targets),
                                2 * (qc + q) * 8,
                                arena.remaining,
                                fixed=1024,
                            ),
                        )
                    counts["ov_transform_batch"] = rows
                    for j, s in enumerate(active):
                        no, nv, _ = shapes[s]
                        nov = no * nv
                        if progress:
                            print(f"DF metric solve: {ceil(nov/rows)} blocks", flush=True)
                        flat = stores[s].array.reshape(-1, q) if stores[s].kind == "gpu" else None
                        for start in range(0, nov, rows):
                            stop = min(nov, start + rows)
                            if resident:
                                x = cp.ascontiguousarray(flat[start:stop])
                            else:
                                _clear(engine, pool)
                                raw = raw_stores[j].read((slice(None), slice(start, stop))).T
                                # Preserve host layout for upload. The native
                                # C-order request then performs its copy on GPU.
                                device = cp.asarray(raw, order="F", blocking=True)
                                del raw
                                x = opt.auxmol.mat_dot_C(device)
                                del device
                            x = solve_triangular(chol, x.T, lower=True, overwrite_b=True).T
                            if flat is not None:
                                flat[start:stop] = x
                            else:
                                values = x.get(blocking=True)
                                # Write flattened ov rows without a full factor host array.
                                cursor = start
                                while cursor < stop:
                                    i, a = divmod(cursor, nv)
                                    end = min(stop, cursor + nv - a)
                                    stores[s].write(
                                        (i, slice(a, a + end - cursor), slice(None)),
                                        values[cursor - start : end - start],
                                    )
                                    cursor = end
                                del values
                            del x
                            counts["whitening_blocks"] += 1
                    if not resident:
                        for raw_store in raw_stores:
                            raw_store.close()
                    break
                except cp.cuda.memory.OutOfMemoryError:
                    counts["factor_retries"] += 1
                # Re-evaluate raw integrals before retrying an in-place conversion.
                import gc

                gc.collect()
                _clear(engine, pool)
                fraction *= 0.5
                if fraction < 1 / 1024:
                    raise MemoryError("MP2 minimum native factor tile cannot fit the GPU budget")
                if resident:
                    for s, t in zip(active, targets):
                        stores[s].array = t
                        stores[s].shape = cart_shapes[s]
                if progress:
                    print("DF factor retry with smaller temporary tiles", flush=True)
        del targets, raw_stores, opt
        cp.cuda.Stream.null.synchronize()
    return stores, chol, counts


def _kernels_on_device(factors, nkeys, pool, budget, pair_cap, host_available):
    q = factors[0].shape[-1]
    total_kernel_bytes = sum(np.prod(f.shape, dtype=np.int64) * 8 for f in factors) * nkeys
    maxv = max(f.shape[1] for f in factors)
    reserve = (2 * nkeys + 6) * maxv * q * 8 + maxv * maxv * 64 + q * q * 8 + 32 * MIB
    # Retain all K on device only if it does not force more pair tiles.
    all_device = all(f.kind == "gpu" for f in factors)

    def pair_count(remaining):
        try:
            return max(
                pair_plan(
                    f.shape[0],
                    maxv,
                    f.shape[1],
                    q,
                    remaining - reserve,
                    cap=pair_cap,
                    resident=all_device,
                    host_available=host_available,
                )["total_blocks_per_observation"]
                for f in factors
                if f.shape[0] and f.shape[1]
            )
        except (MemoryError, ValueError):
            return float("inf")

    available = _remaining(pool, budget)
    return _device_storage_fits(total_kernel_bytes + reserve, available) and pair_count(
        available - total_kernel_bytes
    ) == pair_count(available)


def _group_keys(requests, factors, arena, pool, budget, pair_cap=None):
    keys = []
    for family, k, channels in requests:
        if set(channels) == {"tot"}:
            keys.append((k, "tot"))
            continue
        for ch in ("os", "ss"):
            if ch in channels or "tot" in channels:
                keys.append((k, ch))
    keys = list(dict.fromkeys(keys))
    fbytes = sum(np.prod(f.shape, dtype=np.int64) * 8 for f in factors)
    largest = max((f.shape[1] * f.shape[2] * 8 for f in factors), default=1)
    # CPU scratch allowance is not tensor backing capacity. Keep the largest
    # group that can actually reside on the GPU or on the available disk.
    device = max(1, int((_remaining(pool, budget) - 32 * MIB) // max(1, largest)) - 5)
    width = max(1, min(len(keys), device))
    while width > 1:
        if fbytes * width <= arena.disk_available or _kernels_on_device(
            factors, width, pool, budget, pair_cap, arena.remaining
        ):
            break
        width -= 1
    return [keys[i : i + width] for i in range(0, len(keys), width)]


class _FactorUploader:
    """One bounded pinned host buffer, reused for exact factor transfers.

    Direct cudaHostAlloc avoids power-of-two pool rounding and never changes
    the caller's pinned allocator. HDF5 reads directly into this buffer, with
    no intermediate pageable tile. Upload completion precedes every reuse.
    """

    def __init__(self, limit):
        import cupy as cp

        # Native factor construction may leave large unused transfer buffers.
        # Release only cached blocks; live caller arrays and its allocator stay.
        cp.get_default_pinned_memory_pool().free_all_blocks()
        self.limit = int(limit)
        self.pointer = None
        self.buffer = None
        self.capacity_bytes = 0
        self.peak_bytes = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.buffer = None
        self.pointer = None
        self.capacity_bytes = 0

    def upload(self, store, index):
        import cupy as cp

        single = isinstance(index, (int, np.integer))
        if single:
            start = int(index)
            stop = start + 1
            if not 0 <= start < store.shape[0]:
                raise IndexError(index)
        else:
            start, stop, step = index.indices(store.shape[0])
            if step != 1:
                raise ValueError("Factor staging requires contiguous occupied slices")
        shape = (max(0, stop - start), *store.shape[1:])
        count = int(np.prod(shape))
        size = count * 8
        if size > self.limit:
            raise MemoryError("MP2 factor transfer exceeds the host scratch allowance")
        # Fail early on device pressure, before reading another large host tile.
        out = cp.empty(shape, dtype=cp.float64)
        if not size:
            return out
        self.upload_into(store, start, stop, out)
        return out[0] if single else out

    def upload_into(self, store, start, stop, out):
        """Fill an existing contiguous device slice with one bounded host owner."""
        import cupy as cp

        shape = (stop - start, *store.shape[1:])
        if not 0 <= start <= stop <= store.shape[0] or tuple(out.shape) != shape:
            raise ValueError("Tensor transfer slice and device shape must match")
        if out.dtype != cp.float64 or not out.flags.c_contiguous:
            raise ValueError("Tensor transfer requires a contiguous FP64 device slice")
        size = out.nbytes
        count = size // 8
        if size > self.limit:
            raise MemoryError("MP2 tensor transfer exceeds the host scratch allowance")
        if not size:
            return
        if size > self.capacity_bytes:
            self.close()  # Never keep old and enlarged host buffers together.
            try:
                self.pointer = cp.cuda.PinnedMemoryPointer(cp.cuda.PinnedMemory(size), 0)
            except cp.cuda.runtime.CUDARuntimeError as error:
                if error.status != 2:
                    raise
                raise MemoryError(
                    "MP2 pinned host buffer could not be allocated; lower mp2_host_mem_gb"
                ) from error
            self.buffer = np.frombuffer(self.pointer, dtype=np.float64, count=count)
            self.capacity_bytes = size
            self.peak_bytes = max(self.peak_bytes, size)
        target = self.buffer[:count].reshape(shape)
        if store.kind == "disk":
            store.array.read_direct(target, source_sel=np.s_[start:stop])
        else:
            np.copyto(target, store.array[start:stop])
        stream = cp.cuda.get_current_stream()
        out.set(target, stream=stream)
        stream.synchronize()


def _accumulate_group(
    engine, factors, co, cv, eo, ev, chol, keys, arena, pool, budget, pair_cap, progress, stats
):
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular

    # Do not split a freed factor-build workspace behind small persistent K
    # or metric-transpose allocations at the start of this phase.
    _clear(engine, pool)
    q = chol.shape[0]
    total_kernel_bytes = sum(np.prod(f.shape, dtype=np.int64) * 8 for f in factors) * len(keys)
    device_kernels = _kernels_on_device(factors, len(keys), pool, budget, pair_cap, arena.remaining)

    def allocate(shape, name):
        if device_kernels:
            return DeviceStore(cp.empty(shape), name)
        return arena.allocate(shape, name, force="disk")

    kstores = [
        {
            key: allocate(f.shape, f'kernel_{stats["groups_done"]}_{s}_{n}')
            for n, key in enumerate(keys)
        }
        for s, f in enumerate(factors)
    ]
    stats.setdefault("kernel_storage", []).append(
        {
            "group": stats["groups_done"] + 1,
            "bytes": int(total_kernel_bytes),
            "kind": "gpu" if device_kernels else "disk",
        }
    )
    analytic = {key: 0.0 for key in keys}
    fbytes = sum(np.prod(f.shape, dtype=np.int64) * 8 for f in factors)
    maxv = max(f.shape[1] for f in factors)
    reserve = (len(keys) + 5) * maxv * q * 8 + maxv * maxv * 64 + 32 * MIB
    cached = [f.array for f in factors] if all(f.kind == "gpu" for f in factors) else None
    chol_t = cp.asfortranarray(chol.T)
    if cached is None:

        return accumulate(
            engine,
            factors,
            eo,
            ev,
            chol_t,
            keys,
            arena,
            pool,
            budget,
            pair_cap,
            progress,
            stats,
            kstores,
        )
    plan_cache = {}
    reported_plans = set()

    def occupied(s, i, nv, f, fraction):
        # Previous pair buffers are now dead. Clear them before allocating
        # small per-orbital arrays that could strand their remaining space.
        _clear(engine, pool)
        local_energy = {}
        fi = cached[s][i] if cached is not None else staging.upload(f, i)
        kernels = {key: cp.zeros((nv, q), dtype=cp.float64) for key in keys}
        partners = [(s, engine.restricted, True)]
        if not engine.restricted:
            partners.append((1 - s, True, False))
        for t, allowos, allowss in partners:
            wants = [
                key
                for key in keys
                if key[1] == "tot" or (key[1] == "os" and allowos) or (key[1] == "ss" and allowss)
            ]
            if not wants:
                continue
            no2, nv2, _ = factors[t].shape
            if not no2 or not nv2:
                continue
            gap = float(host(ev[s].min() - eo[s].max() + ev[t].min() - eo[t].max()))
            if not np.isfinite(gap) or gap <= 1e-12:
                raise ValueError("Local MP2 requires positive occupied-virtual energy denominators")
            _clear(engine, pool)
            plan_available = int(
                (_remaining(pool, budget) - 16 * MIB - len(wants) * nv * q * 8) * fraction
            )
            cache_key = (s, t, no2, nv, nv2, plan_available, cached is not None, arena.remaining)
            if cache_key not in plan_cache:
                plan_cache[cache_key] = pair_plan(
                    no2,
                    nv,
                    nv2,
                    q,
                    plan_available,
                    cap=pair_cap,
                    resident=cached is not None,
                    host_available=arena.remaining,
                )
            plan = dict(plan_cache[cache_key])
            if cached is None:
                plan["j"] = min(
                    plan["j"], largest_rows(no2, nv2 * q * 8, arena.remaining, fixed=4096)
                )
                plan["estimated_bytes"] = 4096 + plan["j"] * plan["bytes_per_j"]
            signature = (s, t, plan["j"], plan["a"], plan["b"], plan["resident_factors"])
            if signature not in reported_plans:
                reported_plans.add(signature)
                stats["pair_plans"].append(
                    dict(
                        spin=s,
                        partner=t,
                        occupied_observation=i + 1,
                        host_factor_tile_bytes=(
                            0 if cached is not None else plan["j"] * nv2 * q * 8
                        ),
                        **plan,
                    )
                )
                if progress:
                    print(
                        f'Strategy 1 pair plan: {plan["total_blocks_per_observation"]} blocks per observation; j/a/b = {plan["j"]}/{plan["a"]}/{plan["b"]}',
                        flush=True,
                    )
            j0 = 0
            while j0 < no2:
                nj = min(plan["j"], no2 - j0)
                j1 = j0 + nj
                # A successful tile is committed only after all temporaries fit.
                try:
                    fj = (
                        cached[t][j0:j1]
                        if cached is not None
                        else staging.upload(factors[t], slice(j0, j1))
                    )
                    for a0 in range(0, nv, plan["a"]):
                        aa = slice(a0, min(nv, a0 + plan["a"]))
                        na = aa.stop - aa.start
                        for b0 in range(0, nv2, plan["b"]):
                            bb = slice(b0, min(nv2, b0 + plan["b"]))
                            nb = bb.stop - bb.start
                            right = cp.ascontiguousarray(fj[:, bb]).reshape(nj * nb, q)
                            g = fi[aa] @ right.T
                            den = (
                                ev[s][aa, None]
                                + cp.broadcast_to(ev[t][bb], (nj, nb)).reshape(1, -1)
                                - eo[s][i]
                                - cp.repeat(eo[t][j0:j1], nb)[None, :]
                            )
                            ex = None
                            if allowss and any(c in ("ss", "tot") for _, c in wants):
                                if na == nv and nb == nv2 and nv == nv2:
                                    ex = (
                                        g.reshape(nv, nj, nv)
                                        .transpose(2, 1, 0)
                                        .reshape(nv, nj * nv)
                                    )
                                else:
                                    ex = (
                                        (
                                            fi[bb]
                                            @ cp.ascontiguousarray(fj[:, aa]).reshape(nj * na, q).T
                                        )
                                        .reshape(nb, nj, na)
                                        .transpose(2, 1, 0)
                                        .reshape(na, nj * nb)
                                    )
                            # Stage increments first: allocation failure cannot partly update K.
                            increments = {}
                            for key in wants:
                                k, c = key
                                scale = 1.0 / den
                                if k is not None:
                                    scale *= (-cp.expm1(-k * den)) ** 2
                                pref = 1.0 if engine.restricted else 0.5
                                if c == "tot":
                                    # Same OS/SS contributions, combined before storing K
                                    # only when neither separate channel was requested.
                                    weight = -g * scale * pref if allowos else None
                                    if allowss:
                                        same = pref * ((ex - g) * scale)
                                        weight = same if weight is None else weight + same
                                        del same
                                else:
                                    weight = (
                                        -g * scale * pref
                                        if c == "os"
                                        else pref * ((ex - g) * scale)
                                    )
                                increments[key] = weight @ right
                                del weight, scale
                            for key, value in increments.items():
                                kernels[key][aa] += value
                            del increments, value, g, den, ex, right
                            stats["pair_blocks"] += 1
                    del fj
                except cp.cuda.memory.OutOfMemoryError:
                    # Restart the occupied observation; earlier j contributions are reset.
                    raise
                j0 = j1
        for key, kernel in kernels.items():
            local_energy[key] = float(host(cp.sum(fi * kernel)))
            value = solve_triangular(chol_t, kernel.T, lower=False).T
            kstores[s][key].write(
                i, value if kstores[s][key].kind == "gpu" else value.get(blocking=True)
            )
            del value
        del fi, kernels, kernel
        return local_energy

    staging = _FactorUploader(max(0, arena.remaining - 4096)) if cached is None else None
    try:
        for s, f in enumerate(factors):
            no, nv, _ = f.shape
            if not no or not nv:
                continue
            for i in range(no):
                if progress:
                    print(f"Strategy 1 kernels: spin {s+1}, occupied {i+1}/{no}", flush=True)
                fraction = 1.0
                while True:
                    try:
                        local = occupied(s, i, nv, f, fraction)
                        for key, value in local.items():
                            analytic[key] += value
                        break
                    except cp.cuda.memory.OutOfMemoryError:
                        stats["retries"] += 1
                    # Only this occupied observation is restarted. Stored earlier
                    # observations and their analytic sums remain committed once.
                    # GPU factor owners remain resident during a retry. Keep the
                    # correct residency model instead of budgeting fictitious copies.
                    import gc

                    gc.collect()
                    _clear(engine, pool)
                    fraction *= 0.5
                    if fraction < 1 / 1024:
                        raise MemoryError("MP2 minimum contraction tile cannot fit the GPU budget")
                    if progress:
                        print(
                            f"Strategy 1 allocation retry: occupied {i+1}, smaller pair blocks",
                            flush=True,
                        )
    finally:
        if staging is not None:
            stats["factor_host_staging_peak_bytes"] = max(
                stats.get("factor_host_staging_peak_bytes", 0), staging.peak_bytes
            )
            staging.close()
    del cached, chol_t
    _clear(engine, pool)
    return kstores, analytic


def _upload_kernel(store, start, stop, host_bytes):
    """Bound host staging independently of the selected resident GPU block."""
    import cupy as cp

    if store.kind == "gpu":
        return store.read(slice(start, stop)), 0
    nv, q = store.shape[1:]
    rows = stop - start
    tile = largest_rows(rows, nv * q * 8, host_bytes)
    out = cp.empty((rows, nv, q), dtype=cp.float64)
    count = 0
    with _FactorUploader(host_bytes) as staging:
        for offset in range(0, rows, tile):
            length = min(tile, rows - offset)
            staging.upload_into(
                store, start + offset, start + offset + length, out[offset : offset + length]
            )
            count += 1
    return out, count


def _grid_group(
    engine, aux, coords, co, cv, stores, keys, numerators, pool, budget, arena, progress, stats
):
    import cupy as cp

    q = aux.nao
    nc = engine.mol.nao_cart()
    layout = potential_layout(aux)
    stats["potential_backend"] = "cuda_unscreened_1e16"
    ao_key = id(engine.mol)
    if ao_key not in engine._ao_opts:
        engine._ao_opts[ao_key] = engine._opt_factory(engine.mol)
    sorted_nao = engine._ao_opts[ao_key]._sorted_mol.nao
    nao = engine.mol.nao
    fixed_gpu = 16 * MIB + 32 * nao * nao
    if layout:
        fixed_gpu += 32 * q * q

    # Peak lifetimes, not the sum of temporaries from disjoint stages:
    # upload/current potential; sorted AO + output/layout; field contraction.
    def workspace(no, nv):
        stages = {
            "potential_upload": 2 * q + 16,
            "ao_evaluation": q + 2 * sorted_nao + nao + 16,
            "field_contraction": q + nao + no + 3 * nv + len(keys) + 16,
        }
        if layout:
            stages["cuda_potential"] = (
                2 * layout["ncart"] + 4 * q + 32 + (12 * layout["nshells"] + 7) // 8
            )
        return max(stages.values()) * 8, {k: v * 8 for k, v in stages.items()}

    def evaluate(s, spin, i0, ik, per_point, planned_r):
        i1 = i0 + ik
        kernels = {}
        for key, store in spin.items():
            kernels[key], uploads = _upload_kernel(store, i0, i1, arena.remaining)
            stats["kernel_upload_blocks"] += uploads
        nv = cv[s].shape[1]
        rblock = planned_r
        stats["grid_plans"].append(
            {"spin": s, "occupied_start": i0, "occupied_count": ik, "grid_block": rblock}
        )
        if progress:
            print(
                f"Strategy 1 grid: occupied {i0+1}-{i1}/{co[s].shape[1]}; {ceil(len(coords)/rblock)} r blocks",
                flush=True,
            )
        # Commit this occupied block only after its complete field is available.
        local = {key: np.zeros(len(coords)) for key in keys}
        start = 0
        completed = 0
        while start < len(coords):
            restart = start
            try:
                for sl, points, potential in potential_blocks(
                    engine, aux, coords[restart:], stats, layout, planned_points=rblock
                ):
                    # The generator has now released its previous potential.
                    # Purge unused cache before AO allocations can split that
                    # large buffer and strand memory needed by the contraction.
                    _clear(engine, pool)
                    ao = engine.ao(engine.mol, points)
                    po = ao @ co[s][:, i0:i1]
                    pv = ao @ cv[s]
                    block_values = {key: cp.zeros(len(points), dtype=cp.float64) for key in keys}
                    for key, value in kernels.items():
                        if getattr(engine, "_mp2_batched_grid", True):
                            contract_field(
                                potential,
                                value,
                                po,
                                pv,
                                block_values[key],
                                _remaining(pool, budget),
                                stats,
                            )
                        else:
                            for i in range(ik):
                                block_values[key] += po[:, i] * cp.sum(
                                    pv * (potential @ value[i].T), axis=1
                                )
                    # No GPU allocations after conversion; all channels commit together.
                    copied = {key: value.get(blocking=True) for key, value in block_values.items()}
                    dest = slice(restart + sl.start, restart + sl.stop)
                    for key, value in copied.items():
                        local[key][dest] = value
                    start = dest.stop
                    completed += 1
                    stats["grid_blocks_evaluated"] += 1
                    if progress:
                        print(
                            f"Strategy 1 r block {completed}/{completed+ceil((len(coords)-start)/rblock)}; occupied {i0+1}-{i1}",
                            flush=True,
                        )
                    del ao, po, pv, block_values, copied, value, potential
            except cp.cuda.memory.OutOfMemoryError:
                stats["retries"] += 1
                if rblock == 1:
                    raise
                rblock = max(1, rblock // 2)
                ao = po = pv = block_values = copied = value = potential = None
                # Ended generator cancels/joins its one-block host prefetch.
                import gc

                gc.collect()
                _clear(engine, pool)
                if progress:
                    print(
                        f"Strategy 1 grid allocation retry at point {start}: r block {rblock}",
                        flush=True,
                    )
        return local

    for s, spin in enumerate(stores):
        no, nv, _ = next(iter(spin.values())).shape
        if not no or not nv:
            continue
        i0 = 0
        while i0 < no:
            _clear(engine, pool)
            per_point, stage_bytes = workspace(no, nv)
            # Count actual host potential/current-prefetch buffers, not GPU AO
            # arrays. Include the transactional host fields and coordinate copy.
            host_per_point = 8 * (3 * q + 32)
            if layout:
                host_per_point = 16 * layout["nshells"] + 512
            host_fixed = len(coords) * (len(keys) * 16 + 24) + 1024
            on_device = all(store.kind == "gpu" for store in spin.values())
            plan = grid_plan(
                no - i0,
                len(coords),
                0 if on_device else len(keys) * nv * q * 8,
                per_point,
                host_per_point,
                _remaining(pool, budget) - fixed_gpu,
                arena.remaining - host_fixed,
                max_points_per_block=None,
            )
            stats.setdefault("joint_grid_plans", []).append(
                dict(
                    spin=s,
                    occupied_start=i0,
                    gpu_available_bytes=_remaining(pool, budget) - fixed_gpu,
                    gpu_fixed_reserve_bytes=fixed_gpu,
                    gpu_per_point_by_stage=stage_bytes,
                    host_available_bytes=arena.remaining - host_fixed,
                    host_fixed_bytes=host_fixed,
                    kernel_already_resident=on_device,
                    **plan,
                )
            )
            ik = plan["occupied_block"]
            planned_r = plan["grid_block"]
            if layout:
                stats["native_potential_point_limit"] = layout["point_limit"]
            while True:
                try:
                    local = evaluate(s, spin, i0, ik, per_point, planned_r)
                    for key, value in local.items():
                        numerators[key] += value
                    del local, value
                    stats["kernel_blocks_committed"] += 1
                    break
                except cp.cuda.memory.OutOfMemoryError:
                    stats["retries"] += 1
                if ik == 1:
                    raise MemoryError("MP2 minimum grid and kernel tiles cannot fit the GPU budget")
                ik = max(1, ik // 2)
                import gc

                gc.collect()
                _clear(engine, pool)
                if progress:
                    print(f"Strategy 1 grid retry: occupied block {ik}", flush=True)
            i0 += ik
    _clear(engine, pool)


def streamed_fields(
    engine,
    coords,
    density,
    aux_basis,
    requests,
    ks,
    budget,
    scratch_mib,
    pair_cap,
    progress,
    pool,
    host_mem_gb=None,
):
    from pyscf import df

    if pair_cap is not None and (isinstance(pair_cap, bool) or int(pair_cap) < 1):
        raise ValueError("pair_block must be positive")
    engine.sync()
    start = perf_counter()
    aux = df.addons.make_auxmol(engine.mol, aux_basis)
    if max(aux.bas_angular(i) for i in range(aux.nbas)) > 6:
        raise NotImplementedError(
            "GPU auxiliary potentials support angular momentum through i (l <= 6)"
        )
    co, cv, eo, ev = _orbitals(engine)
    stats = {
        "pair_blocks": 0,
        "pair_plans": [],
        "grid_plans": [],
        "groups_done": 0,
        "retries": 0,
        "kernel_upload_blocks": 0,
        "grid_blocks_evaluated": 0,
        "kernel_blocks_committed": 0,
    }
    phases = {
        "factors": 0.0,
        "pair_contraction_and_compression": 0.0,
        "metric_backtransform": 0.0,
        "grid_and_transfers": 0.0,
    }
    values = {}
    analytic = {}
    with (
        HostArena() if host_mem_gb is None else HostArena(host_bytes=int(host_mem_gb * 1e9))
    ) as arena:
        factors, chol, fmeta = _build_factors(
            engine, aux, co, cv, arena, pool, budget, scratch_mib, progress
        )
        phases["factors"] = perf_counter() - start
        factor_bytes = int(sum(np.prod(f.shape, dtype=np.int64) * 8 for f in factors))
        groups = _group_keys(requests, factors, arena, pool, budget, pair_cap)
        stats["channel_groups"] = [
            [("mp2" if k is None else f"kappa={k}", c) for k, c in group] for group in groups
        ]
        storage = [{"shape": f.shape, "kind": f.kind, "allocated_bytes": f.nbytes} for f in factors]
        for group_index, group in enumerate(groups):
            if progress:
                print(
                    f'Strategy 1 channel group {stats["groups_done"]+1}/{len(groups)}: {group}',
                    flush=True,
                )
            t = perf_counter()
            stores, energy = _accumulate_group(
                engine,
                factors,
                co,
                cv,
                eo,
                ev,
                chol,
                group,
                arena,
                pool,
                budget,
                pair_cap,
                progress,
                stats,
            )
            phases["pair_contraction_and_compression"] += perf_counter() - t
            analytic.update(energy)
            # No later amplitudes need the factors after the last group. Free
            # their sole device/disk owner before the grid, not after it.
            if group_index == len(groups) - 1:
                for factor in factors:
                    factor.close()
                del chol
                _clear(engine, pool)
            numerators = {key: np.zeros(len(coords)) for key in group}
            t = perf_counter()
            _grid_group(
                engine,
                aux,
                coords,
                co,
                cv,
                stores,
                group,
                numerators,
                pool,
                budget,
                arena,
                progress,
                stats,
            )
            phases["grid_and_transfers"] += perf_counter() - t
            rho = host(density)
            safe = np.where(rho > 0, rho, 1.0)
            for key, val in numerators.items():
                values[key] = np.where(rho > 0, val / safe, 0.0)
            del numerators, val
            for spin in stores:
                for store in spin.values():
                    store.close()
            stats["groups_done"] += 1
        host_peak = arena.peak
        host_budget = arena.limit
        disk_peak = arena.disk_peak
        del factors
    out = {}
    energy = {}
    for family, k, channels in requests:
        for c in channels:
            key = field_key(family, c, k, family == "kmp2" and len(ks) > 1)
            out[key] = (
                values[k, "os"] + values[k, "ss"]
                if c == "tot" and (k, "tot") not in values
                else values[k, c]
            )
            energy[key] = (
                analytic[k, "os"] + analytic[k, "ss"]
                if c == "tot" and (k, "tot") not in analytic
                else analytic[k, c]
            )
    engine.sync()
    meta = dict(
        strategy="strat1",
        amplitude_integrals="Coulomb-metric DF",
        spatial_representation="original observation orbital product",
        precision="float64",
        metric_solver="Cholesky; no rank truncation",
        naux=aux.nao,
        factor_bytes=factor_bytes,
        orbital_kernel_bytes=factor_bytes * len(values),
        auxiliary_matrix_bytes_per_channel=0,
        gpu_budget_bytes=budget,
        integral_scratch_target_mib=scratch_mib,
        pair_block=pair_cap,
        pair_blocks=stats["pair_blocks"],
        analytic_energies=energy,
        phase_seconds=phases,
        budget_policy="Bounded GPU tiles; lossless host/HDF5 factor and kernel storage; native workspaces additional",
        streaming=stats,
        factor_storage=storage,
        host_storage_budget_bytes=host_budget,
        peak_host_storage_bytes=host_peak,
        peak_disk_storage_bytes=disk_peak,
        host_policy="Bounded host scratch (default 80% available); no full RAM factor/kernel copies; output arrays and caller state additional",
        **fmeta,
    )
    return out, meta
