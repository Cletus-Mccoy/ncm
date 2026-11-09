"""models.common must use a compiled flex_attention, and it must agree with the eager implementation.

Needs CUDA (flex attention kernels); skipped otherwise. Run: python tests/test_flex_compiled.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import models.common as common


def test_flex_attention_is_compiled_and_matches_eager():
    if not (torch.cuda.is_available() and common.HAS_FLEX_ATTENTION):
        print("skipped: needs CUDA and flex attention"); return
    from torch.nn.attention.flex_attention import flex_attention as eager, create_block_mask
    assert hasattr(common._flex_attention, "_torchdynamo_orig_callable"), "models.common._flex_attention is not compiled"
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 128, 16, device="cuda") for _ in range(3))
    mask = create_block_mask(lambda b, h, qi, ki: qi >= ki, B=None, H=None, Q_LEN=128, KV_LEN=128, device="cuda")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = eager(q, k, v, block_mask=mask)
    out = common._flex_attention(q, k, v, block_mask=mask)
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3), float((out - ref).abs().max())


if __name__ == "__main__":
    test_flex_attention_is_compiled_and_matches_eager()
    print("ok")
