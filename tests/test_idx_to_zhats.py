"""idx_to_zhats must return one distinct cumulative reconstruction per level.

Run: python tests/test_idx_to_zhats.py   (or pytest tests/test_idx_to_zhats.py)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from types import SimpleNamespace
from models.quant import MultiscaleResidualVectorQuantizer


def _vq():
    torch.manual_seed(0)
    return MultiscaleResidualVectorQuantizer(
        vqvae_vocab_size=[32, 32, 32], embed_dim=8, levels=SimpleNamespace(use_manual_levels=True, manual_levels=[1, 2, 4])).eval()


def test_per_level_reconstructions_are_distinct_and_cumulative():
    vq = _vq()
    idx = torch.randint(0, 32, (3, sum(vq.levels)))
    with torch.no_grad():
        z_hats, _ = vq.idx_to_zhats(idx)
        expected = torch.zeros_like(z_hats[-1])
        off = 0
        for k, n in enumerate(vq.levels):
            z = vq.embeddings[k](idx[:, off:off + n])
            z = vq._up_to_full(z, k, vq.levels[-1]) if k != len(vq.levels) - 1 else z
            expected = expected + vq._apply_quant_resi(z, k)
            off += n
            assert torch.allclose(z_hats[k], expected, atol=1e-6), f"level {k} is not the cumulative sum of levels 0..{k}"
    assert len({z.data_ptr() for z in z_hats}) == len(z_hats), "entries share storage (aliased)"
    assert not torch.allclose(z_hats[0], z_hats[-1]), "coarsest and finest reconstruction must differ"
    assert torch.allclose(z_hats[-1], vq.idx_to_z_q(idx), atol=1e-6)


if __name__ == "__main__":
    test_per_level_reconstructions_are_distinct_and_cumulative()
    print("ok")
