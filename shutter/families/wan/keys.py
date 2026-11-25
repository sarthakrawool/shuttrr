"""Tensor names: every layout a Wan checkpoint comes in, mapped onto the names the MLX model uses.

Wan checkpoints in the wild use four namings for the same DiT:

* official  (Wan-AI repos, most fine-tunes):  ``blocks.0.self_attn.q.weight``, ``patch_embedding.weight`` ...
* comfy     (ComfyUI repacks, Civitai):       the official names behind ``model.diffusion_model.`` (or ``diffusion_model.``)
* diffusers (``*-Diffusers`` repos):          ``blocks.0.attn1.to_q.weight``, ``condition_embedder.time_proj`` ...
* shuttrr   (our converted build):            the MLX names below, 8-bit

and the umT5 prompt encoder comes as official Wan names (``blocks.0.attn.q.weight``; also LongCat's copy) or
Hugging Face T5 names (``encoder.block.0.layer.0.SelfAttention.q.weight``; diffusers and ComfyUI's umt5_xxl).

Everything is mapped to official names first, then official -> MLX (as mlx-video's sanitizer does). Conversions that
change a tensor's shape (the patch-embedding Conv3d becomes a Linear) are returned as a reshape instruction, so the
converter can apply them while it reads one tensor at a time.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

# ----------------------------------------------------------------------------- DiT

COMFY_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "model.")

# diffusers -> official (block-level pieces are applied with the block prefix kept)
_DIFFUSERS_TOP = [
    ("condition_embedder.text_embedder.linear_1.", "text_embedding.0."),
    ("condition_embedder.text_embedder.linear_2.", "text_embedding.2."),
    ("condition_embedder.time_embedder.linear_1.", "time_embedding.0."),
    ("condition_embedder.time_embedder.linear_2.", "time_embedding.2."),
    ("condition_embedder.time_proj.", "time_projection.1."),
    ("condition_embedder.image_embedder.", "img_emb."),
    ("proj_out.", "head.head."),
]
_DIFFUSERS_BLOCK = [
    (".attn1.to_q.", ".self_attn.q."), (".attn1.to_k.", ".self_attn.k."), (".attn1.to_v.", ".self_attn.v."),
    (".attn1.to_out.0.", ".self_attn.o."), (".attn1.norm_q.", ".self_attn.norm_q."),
    (".attn1.norm_k.", ".self_attn.norm_k."),
    (".attn2.to_q.", ".cross_attn.q."), (".attn2.to_k.", ".cross_attn.k."), (".attn2.to_v.", ".cross_attn.v."),
    (".attn2.to_out.0.", ".cross_attn.o."), (".attn2.norm_q.", ".cross_attn.norm_q."),
    (".attn2.norm_k.", ".cross_attn.norm_k."),
    (".attn2.add_k_proj.", ".cross_attn.k_img."), (".attn2.add_v_proj.", ".cross_attn.v_img."),
    (".attn2.norm_added_k.", ".cross_attn.norm_k_img."),
    (".norm2.", ".norm3."),                     # diffusers' norm2 is the (affine) cross-attention norm
    (".ffn.net.0.proj.", ".ffn.0."), (".ffn.net.2.", ".ffn.2."),
]


def strip_comfy(name: str) -> str:
    for p in COMFY_PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def dit_layout(names: Iterable[str]) -> str:
    """"official", "comfy", "diffusers", "shuttrr" or "" (not a Wan DiT) from tensor names alone."""
    names = list(names)
    s = set(names)
    if any(n.startswith("patch_embedding_proj.") for n in s):
        return "shuttrr"
    if any(n.startswith(p + "patch_embedding.") for p in COMFY_PREFIXES for n in s):
        return "comfy"
    if "patch_embedding.weight" in s:
        if any(".attn1." in n for n in s) or any(n.startswith("condition_embedder.") for n in s):
            return "diffusers"
        if any(".self_attn." in n for n in s):
            return "official"
    return ""


def diffusers_to_official(name: str) -> Optional[str]:
    """Official name for a diffusers WanTransformer3DModel tensor; None for tensors with no counterpart."""
    if name == "scale_shift_table":
        return "head.modulation"
    m = re.match(r"^blocks\.(\d+)\.scale_shift_table$", name)
    if m:
        return f"blocks.{m.group(1)}.modulation"
    for a, b in _DIFFUSERS_TOP:
        if name.startswith(a):
            return b + name[len(a):]
    if name.startswith("blocks."):
        for a, b in _DIFFUSERS_BLOCK:
            if a in name:
                return name.replace(a, b, 1)
        return name
    if name.startswith("patch_embedding."):
        return name
    return None


def to_official(name: str, layout: str) -> Optional[str]:
    if layout == "comfy":
        return strip_comfy(name)
    if layout == "diffusers":
        return diffusers_to_official(name)
    return name


def official_to_mlx(name: str) -> tuple[Optional[str], Optional[str]]:
    """(MLX name, transform) for an official DiT tensor. transform is None or "flatten" (Conv3d weight
    [O, I, 1, 2, 2] -> Linear [O, I*4]). (None, None) means the tensor is not used (the RoPE `freqs` buffer)."""
    if name == "freqs":
        return None, None
    if name == "patch_embedding.weight":
        return "patch_embedding_proj.weight", "flatten"
    if name == "patch_embedding.bias":
        return "patch_embedding_proj.bias", None
    for a, b in (("text_embedding.0.", "text_embedding_0."), ("text_embedding.2.", "text_embedding_1."),
                 ("time_embedding.0.", "time_embedding_0."), ("time_embedding.2.", "time_embedding_1."),
                 ("time_projection.1.", "time_projection.")):
        if name.startswith(a):
            return b + name[len(a):], None
    return name.replace(".ffn.0.", ".ffn.fc1.").replace(".ffn.2.", ".ffn.fc2."), None


def dit_to_mlx(name: str, layout: str) -> tuple[Optional[str], Optional[str]]:
    if layout == "shuttrr":
        return name, None
    off = to_official(name, layout)
    if off is None:
        return None, None
    return official_to_mlx(off)


#: Linear layers inside a block that the 8-bit build quantizes (mlx-video's predicate; embeddings, norms, the head
#: and modulation stay in bf16/fp32 because they are small and precision-sensitive)
QUANT_SUFFIXES = (".self_attn.q", ".self_attn.k", ".self_attn.v", ".self_attn.o",
                  ".cross_attn.q", ".cross_attn.k", ".cross_attn.v", ".cross_attn.o",
                  ".ffn.fc1", ".ffn.fc2")


def is_quantized_linear(module_path: str) -> bool:
    return module_path.startswith("blocks.") and module_path.endswith(QUANT_SUFFIXES)


# Tensors that mark variants this family cannot run (yet). Matched against official names.
UNSUPPORTED_MARKERS = [
    (re.compile(r"^img_emb\.|\.cross_attn\.k_img\."),
     "it has a CLIP image branch (Wan 2.1 image-to-video, first-last-frame or a model built on them). "
     "Shuttrr runs Wan 2.2's image-to-video, which has no CLIP branch; the CLIP branch is not supported yet."),
    (re.compile(r"^vace_|\.vace_"), "it is a VACE model (video editing with control inputs), which is not supported yet."),
    (re.compile(r"^control_adapter\.|^ref_conv\.|^camera_"),
     "it is a Fun / control or camera model, which needs control inputs that are not supported yet."),
    (re.compile(r"^audio_|^casual_audio_encoder|^audio_injector|^motion_encoder|^pose_patch|^face_"),
     "it is a speech-to-video or animate model (Wan S2V / Animate / LongCat-style audio), which is a different "
     "family. LongCat avatar models run with Shuttrr's LongCat adapter."),
]


# ----------------------------------------------------------------------------- umT5

def t5_layout(names: Iterable[str]) -> str:
    """"wan" (official / LongCat naming), "hf" (Hugging Face T5), or ""."""
    s = set(names)
    if "token_embedding.weight" in s and any(n.startswith("blocks.0.attn.") for n in s):
        return "wan"
    if any(n.startswith("encoder.block.0.layer.0.SelfAttention.") for n in s):
        return "hf"
    return ""


def hf_t5_to_wan(name: str) -> Optional[str]:
    if name in ("shared.weight", "encoder.embed_tokens.weight"):
        return "token_embedding.weight"
    if name == "encoder.final_layer_norm.weight":
        return "norm.weight"
    m = re.match(r"^encoder\.block\.(\d+)\.layer\.(\d)\.(.+)$", name)
    if not m:
        return None
    i, layer, rest = m.group(1), m.group(2), m.group(3)
    table = {
        ("0", "SelfAttention.q.weight"): "attn.q.weight",
        ("0", "SelfAttention.k.weight"): "attn.k.weight",
        ("0", "SelfAttention.v.weight"): "attn.v.weight",
        ("0", "SelfAttention.o.weight"): "attn.o.weight",
        ("0", "SelfAttention.relative_attention_bias.weight"): "pos_embedding.embedding.weight",
        ("0", "layer_norm.weight"): "norm1.weight",
        ("1", "DenseReluDense.wi_0.weight"): "ffn.gate.0.weight",    # gated (GELU) branch
        ("1", "DenseReluDense.wi_1.weight"): "ffn.fc1.weight",
        ("1", "DenseReluDense.wo.weight"): "ffn.fc2.weight",
        ("1", "layer_norm.weight"): "norm2.weight",
    }
    tail = table.get((layer, rest))
    return f"blocks.{i}.{tail}" if tail else None


def t5_to_mlx(name: str, layout: str) -> Optional[str]:
    """MLX T5Encoder name (mlx-video naming) for a umT5 tensor."""
    if layout == "hf":
        name = hf_t5_to_wan(name)
        if name is None:
            return None
    return name.replace(".ffn.gate.0.", ".ffn.gate_proj.")


# ----------------------------------------------------------------------------- VAE

def vae_kind(names: Iterable[str], shapes: dict[str, tuple]) -> str:
    """"wan2.1" (16 latent channels), "wan2.2" (48), "diffusers", or ""."""
    s = set(names)
    if any(n.startswith("decoder.up_blocks.") for n in s) or "post_quant_conv.weight" in s:
        return "diffusers"
    conv2 = shapes.get("conv2.weight")
    if conv2 is None:
        return ""
    # conv2 is the 1x1x1 conv on the latent: [z_dim, z_dim, 1, 1, 1]
    z = conv2[1] if len(conv2) == 5 else conv2[-1]
    return {16: "wan2.1", 48: "wan2.2"}.get(int(z), "")


# ----------------------------------------------------------------------------- LoRA

_LORA_PAIRS = ((".lora_down.weight", ".lora_up.weight"), (".lora_A.weight", ".lora_B.weight"),
               (".lora.down.weight", ".lora.up.weight"))


def lora_target(name: str) -> Optional[tuple[str, str]]:
    """(official module path, "down" | "up" | "alpha") for a LoRA tensor in any common naming, else None.

    Handles ``diffusion_model.blocks.0.self_attn.q.lora_down.weight`` (official/comfy), the PEFT ``lora_A/lora_B``
    names (also with ``transformer.`` and diffusers module names) and kohya's ``lora_unet_blocks_0_self_attn_q``.
    """
    base, kind = None, None
    for down, up in _LORA_PAIRS:
        if name.endswith(down):
            base, kind = name[: -len(down)], "down"
        elif name.endswith(up):
            base, kind = name[: -len(up)], "up"
    if base is None and name.endswith(".alpha"):
        base, kind = name[: -len(".alpha")], "alpha"
    if base is None:
        return None
    if base.startswith("lora_unet_"):
        # kohya flattens dots to underscores; rebuild the known module names
        b = base[len("lora_unet_"):]
        m = re.match(r"^blocks_(\d+)_(self_attn|cross_attn)_(q|k|v|o)$", b)
        if m:
            return f"blocks.{m.group(1)}.{m.group(2)}.{m.group(3)}", kind
        m = re.match(r"^blocks_(\d+)_ffn_(0|2)$", b)
        if m:
            return f"blocks.{m.group(1)}.ffn.{m.group(2)}", kind
        return None
    for p in ("base_model.model.", "transformer.", "unet."):
        if base.startswith(p):
            base = base[len(p):]
    base = strip_comfy(base)
    if ".attn1." in base or ".attn2." in base or ".ffn.net." in base:
        off = diffusers_to_official(base + ".weight")
        base = off[: -len(".weight")] if off else base
    return base, kind


def lora_mlx_module(official_module: str) -> str:
    return official_module.replace(".ffn.0", ".ffn.fc1").replace(".ffn.2", ".ffn.fc2")
