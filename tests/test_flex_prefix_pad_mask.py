"""prefix_pad_mask must be honoured by the flex attention path, as it is by the dense path.

Builds the small Sudoku architecture from config/eval/sudoku-hard.yaml with random weights on CPU. Padded prefix slots must not influence the
output, and flex and dense attention must agree. Run: python tests/test_flex_prefix_pad_mask.py   (or pytest)
"""
import os, sys, copy
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from models.next_concept import NextConceptTransformer

CFG = OmegaConf.load(os.path.join(ROOT, "config", "eval", "sudoku-hard.yaml"))


def _net(flex, vqvae, state=None):
    """The generator keeps its codec in a plain list (not in state_dict), so both models must be built on the same codec object."""
    cfg = copy.deepcopy(CFG.model); cfg.use_flex_attention = flex
    net = NextConceptTransformer(vqvae_local=vqvae, config=cfg).eval()
    if state is not None:
        net.load_state_dict(state)
    return net


def _logits(net, prefix_tokens, idx, mask):
    vq = net.vqvae_local[0] if isinstance(net.vqvae_local, (list, tuple)) else net.vqvae_local
    z = vq.encoder.wte(prefix_tokens)
    x = torch.cat([z, net.build_l0_query_input(z.shape[0], dtype=z.dtype), vq.vq.idx_to_ncm_input(idx)], dim=1)
    with torch.no_grad():
        return net(x, prefix_pad_mask=mask)


def test_flex_honours_prefix_pad_mask():
    torch.manual_seed(0)
    codec = VQVAE(**CFG.tokenizer.vqvae).eval()
    dense = _net(False, codec); flex = _net(True, codec, dense.state_dict())
    B, P = 2, dense.prefix_len
    tokens = torch.randint(1, 10, (B, P)); idx = torch.randint(0, 10, (B, sum(dense.levels)))
    mask = torch.zeros(B, P, dtype=torch.bool); mask[:, P - 30:] = True            # last 30 prefix slots are padding
    junk = tokens.clone(); junk[:, P - 30:] = torch.randint(1, 10, (B, 30))         # different content in the padded slots only
    a, b = _logits(flex, tokens, idx, mask), _logits(flex, junk, idx, mask)
    assert (a - b).abs().max() < 1e-4, f"padded slots change the flex output by {(a - b).abs().max():.2e}"
    d = _logits(dense, tokens, idx, mask)
    assert torch.allclose(a, d, atol=1e-3, rtol=1e-3), f"flex and dense disagree by {(a - d).abs().max():.2e}"


if __name__ == "__main__":
    test_flex_honours_prefix_pad_mask()
    print("ok")
