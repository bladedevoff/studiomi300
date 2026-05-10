# AITER FP8 Linear drop-in for Wan2.2 transformer. Static per-tensor weight scale,
# dynamic per-tensor activation quant, gemm_a8w8 -> bf16 out. Off by default
# because of the ROCm/aiter#2187 multi-shape crash (see incidents.md).
import logging

import torch
from torch import nn

log = logging.getLogger("studiomi300.aiter_linear")
FP8_DTYPE = torch.float8_e4m3fnuz   # ROCm uses fnuz variant
FP8_AMAX = 240.0                    # max representable in e4m3fnuz before overflow


class AiterFP8Linear(nn.Module):
    # nn.Linear replacement: FP8 (e4m3fnuz) weight + static weight scale,
    # dynamic per-tensor activation quant on every forward.

    def __init__(self, in_features, out_features, weight_fp8, w_scale, bias):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_fp8 = nn.Parameter(weight_fp8, requires_grad=False)
        self.w_scale = nn.Parameter(w_scale, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    @torch.no_grad()
    def forward(self, x):
        import aiter
        orig_shape = x.shape
        # AITER expects bf16 input range; some upstream layers send fp32 (timestep
        # embeddings). Always cast to bf16 first so the activation quant scale is
        # computed in the same range as the weight scale.
        x2d = x.reshape(-1, orig_shape[-1]).to(torch.bfloat16).contiguous()

        # dynamic per-tensor activation quant. aiter has a hand-tuned hip kernel.
        try:
            x_fp8, x_scale = aiter.per_tensor_quant(
                x2d, scale_dtype=torch.float32, quant_dtype=FP8_DTYPE,
            )
        except Exception:
            amax = x2d.abs().amax().clamp(min=1e-12)
            x_scale = (amax / FP8_AMAX).to(torch.float32).reshape(1)
            x_fp8 = (x2d / x_scale).to(FP8_DTYPE)

        # gemm_a8w8 (non-CK ASM path). On ROCm 7.2 the CK variant standalone
        # works on the cross-attn shape (M=512, K=4096, N=5120) but the same
        # call inside Wan2.2's full pipeline still crashes (torch.compile +
        # AITER state interaction, matches ROCm/aiter#2187). gemm_a8w8 (no CK)
        # is the ASM-tuned fallback that survives more pipeline shapes.
        try:
            out = aiter.gemm_a8w8(
                x_fp8, self.weight_fp8, x_scale, self.w_scale,
                bias=None, dtype=torch.bfloat16,
            )
        except Exception:
            # dequantize + BF16 matmul as last-resort safety net
            w_bf = self.weight_fp8.to(torch.bfloat16) * self.w_scale
            x_bf = x_fp8.to(torch.bfloat16) * x_scale
            out = x_bf @ w_bf.t()

        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], out.shape[-1])

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, fp8=e4m3fnuz"


def _convert_linear(lin):
    w = lin.weight.detach()                       # (out, in), bf16/fp16
    # static per-tensor weight scale: amax / 240
    amax = w.float().abs().amax().clamp(min=1e-12)
    w_scale = (amax / FP8_AMAX).to(torch.float32).reshape(1).to(w.device)
    # quantize weight to e4m3fnuz
    w_fp8 = (w / w_scale).to(FP8_DTYPE).contiguous()
    # AITER gemm_a8w8_CK requires bias dtype to match the OUTPUT dtype (bf16).
    # Wan2.2 happens to store some Linear biases in fp32, fail loudly otherwise.
    bias = lin.bias.detach().to(torch.bfloat16) if lin.bias is not None else None
    new_mod = AiterFP8Linear(lin.in_features, lin.out_features, w_fp8, w_scale, bias)
    new_mod.to(w.device)
    return new_mod


def patch_linears_to_fp8(module, *, skip_names=("proj_out",),
                         min_in_features=1024, min_out_features=1024):
    # skip proj_out (last layer FP8 hurts fidelity) and Linears smaller than
    # 1024x1024 (gemm_a8w8_CK crashes on (1, 256, 5120) in time_embedder, and
    # FP8 win is dominated by quant/dequant overhead anyway).
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name not in skip_names:
            if child.in_features >= min_in_features and child.out_features >= min_out_features:
                setattr(module, name, _convert_linear(child))
                n += 1
            else:
                log.debug(f"skip small linear {name}: {child.in_features}x{child.out_features}")
        else:
            n += patch_linears_to_fp8(
                child, skip_names=skip_names,
                min_in_features=min_in_features, min_out_features=min_out_features,
            )
    return n
