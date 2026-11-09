"""The flex block mask must have the same length as the model's input (prefix + target positions), also when query_level0 is on.

Run: python tests/test_flex_block_mask.py   (or pytest tests/test_flex_block_mask.py)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from types import SimpleNamespace
import torch
from models.next_concept import NextConceptTransformer


def _stub(query_level0, levels=(1, 2, 4), prefix=3):
    """Just the attributes _get_flex_block_mask reads, built the way NextConceptTransformer.__init__ builds them."""
    lvl_first = 0 if query_level0 else 1
    lvl_full = torch.cat([torch.full((prefix,), -1, dtype=torch.long),
                          torch.cat([torch.full((l,), i, dtype=torch.long) for i, l in enumerate(levels[lvl_first:])])])
    return SimpleNamespace(prefix_len=prefix, L=sum(levels), levels=list(levels), query_level0=query_level0, flex_lvl_full=lvl_full,
                           config=SimpleNamespace(), _flex_block_masks={})


def _mask_len(stub):
    m = NextConceptTransformer._get_flex_block_mask(stub, "cpu")
    return tuple(m.shape[-2:])


def test_mask_matches_input_length_with_query_level0():
    stub = _stub(query_level0=True)
    n = stub.flex_lvl_full.shape[0]                      # prefix + all L target positions
    assert _mask_len(stub) == (n, n), f"mask {_mask_len(stub)} but the input has {n} positions"


def test_mask_matches_input_length_without_query_level0():
    stub = _stub(query_level0=False)
    n = stub.flex_lvl_full.shape[0]                      # prefix + (L - levels[0]) target positions
    assert _mask_len(stub) == (n, n)


if __name__ == "__main__":
    test_mask_matches_input_length_without_query_level0()
    test_mask_matches_input_length_with_query_level0()
    print("ok")
