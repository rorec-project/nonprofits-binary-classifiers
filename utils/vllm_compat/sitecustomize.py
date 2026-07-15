# Compatibility shim for serving deepseek-ai/DeepSeek-V4-Flash on this vLLM
# build. Activate it by prepending this directory to PYTHONPATH for the
# `vllm serve` invocation only — see docs/RUNNING_ON_UCLOUD.md's
# "DeepSeek-V4-Flash setup" section. Python auto-imports `sitecustomize` from
# any directory on sys.path at interpreter start, including in the TP worker
# subprocesses vLLM spawns (PYTHONPATH propagates across multiprocessing
# spawn), which is where the bugs below actually manifest.
#
# Root cause (one broken dependency, two fallout sites): the installed
# nvidia-cutlass-dsl wheel's cute/__init__.py imports a symbol
# (`increment_coord`) that cute/core.py does not define -- verified missing
# in every public PyPI release from 4.3.5 through 4.5.2 (the latest), so no
# version pin fixes it. `import cutlass.cute` raises ImportError.
#
# Order matters in this file: the has_cutedsl() patch (site 2) must run
# BEFORE anything imports vllm.models.deepseek_v4.common.ops.fused_indexer_q,
# because that module does `from vllm.utils.import_utils import has_cutedsl`
# -- a name binding evaluated at the moment its own import statement runs, not
# at call time. The compressor shim (site 1) transitively imports sibling
# vllm.models.deepseek_v4.* modules, which can pull fused_indexer_q in before
# its own patch would otherwise apply. Patching has_cutedsl() first, with no
# vllm.models.* import ahead of it, guarantees every later import of
# fused_indexer_q binds the patched version.
import sys
import types

# ---------------------------------------------------------------------------
# Site 1: vllm/models/deepseek_v4/common/ops/fused_indexer_q.py (Lightning
# Indexer). This one IS correctly gated behind `has_cutedsl()` -- but that
# helper only checks importlib.util.find_spec("cutlass"), i.e. whether the
# package is *installed*, not whether it actually imports. Since the wheel is
# present-but-broken, has_cutedsl() reports True and the indexer takes the
# broken `import cutlass.cute` path anyway. Patching has_cutedsl() to
# correctly report "no" routes it to its own already-complete inline triton
# kernel (the genuine `else` branch of its own `if has_cutedsl(): ... else:
# ...` -- not something this shim invents).
# ---------------------------------------------------------------------------
try:
    import vllm.utils.import_utils as _import_utils
except ImportError:
    pass
else:
    _import_utils.has_cutedsl = lambda: False

# ---------------------------------------------------------------------------
# Site 2: vllm/models/deepseek_v4/compressor.py picks a kernel for
# head_dim == 512 (DeepSeek-V4's MLA head size) by UNCONDITIONALLY importing
# `compress_norm_rope_store_cutedsl` from
# vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl -- no
# has_cutedsl() gate at all, so patching has_cutedsl() (site 1) does not cover
# it. vLLM already ships a complete, signature-compatible triton kernel for
# this exact case: compress_norm_rope_store_triton (in
# vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py) dispatches
# on head_dim itself and has a dedicated branch for head_dim == 512 -- it is
# not a degraded fallback, it's the purpose-built kernel for this model. The
# call site in compressor.py invokes whichever function is bound to
# `compress_norm_rope_store_fn` with an identical kwarg set regardless of
# which branch chose it, so the two functions are drop-in compatible.
#
# Pre-registering the cutedsl submodule in sys.modules (pointing its one
# function at the triton implementation) satisfies compressor.py's lazy
# `from .nvidia.ops.sparse_attn_compress_cutedsl import
# compress_norm_rope_store_cutedsl` without ever importing cutlass.
#
# Neither site modifies any installed vendor file, so both survive
# `uv sync` / package reinstalls.
# ---------------------------------------------------------------------------
_MODULE_NAME = "vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl"

if _MODULE_NAME not in sys.modules:
    try:
        from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
            compress_norm_rope_store_triton,
        )
    except ImportError:
        # vLLM not installed, or this module moved -- never block startup;
        # worst case the original (broken) cutedsl import path is hit again
        # and fails with its own clear error.
        pass
    else:
        _shim = types.ModuleType(_MODULE_NAME)
        _shim.compress_norm_rope_store_cutedsl = compress_norm_rope_store_triton
        sys.modules[_MODULE_NAME] = _shim
