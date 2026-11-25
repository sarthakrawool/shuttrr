"""Custom Metal kernel prototypes (mx.fast.metal_kernel) and the analytic FLOP model.

Each kernel module is self-contained (imports only mlx) and has `reference(...)` (the model's MLX ops, used
when Metal is unavailable), `kernel(...)` (Metal only), a dispatching function, and `selftest()`.
None of the kernels is bit-identical to the MLX path, so all are opt-in (see docs/KERNELS.md).
`flops` is pure Python and needs no MLX. `block_sparse` is not a Metal kernel: block-sparse 3D attention in MLX
ops (opt-in approximation, exact at sparsity 0), driven by `params.speed` (see docs/KERNELS.md).
"""

KERNEL_MODULES = ("adaln_mod", "norm_rope", "qlinear8", "flash_attn")


def selftest_all() -> list[dict]:
    import importlib

    return [importlib.import_module(f"{__name__}.{m}").selftest() for m in KERNEL_MODULES]
