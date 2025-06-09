# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

from torch import nn

from torchtune.models.llama3._model_utils import scale_hidden_dim_for_mlp

from torchtune.modules import (
    FeedForward,
    FrozenNF4Linear,
    MultiHeadAttention,
    RMSNorm,
    RotaryPositionalEmbeddings,
    TransformerDecoder,
    TransformerSelfAttentionLayer,
)

from torchtune.modules.common_utils import _register_reparametrize_state_dict_hooks

from torchtune.modules.peft import DoRALinear, LORA_ATTN_MODULES, LoRALinear
from torchtune.modules.peft._utils import resolve_lora_value

"""
Component builders for the Llama3 model and popular variants such as LoRA.

torchtune provides composable building blocks. Builder functions help
stitch these building blocks into higher-level components. This design has
two benefits:
- The building blocks themselves are very flexible. For example, ``MultiHeadAttention``
can take either nn.Linear or nn.LoRALinear for ``q_proj``.
- Builder functions expose a set of configurable params which keep the constructors of
the building blocks simple.
"""


# ------------------ Vanilla Llama3 ------------------


def llama3(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: int = 500_000,
    intermediate_dim: Optional[int] = None,
    norm_eps: float = 1e-5,
) -> TransformerDecoder:
    """
    Build the decoder associated with the Llama3 model. This includes:
    - Token embeddings
    - num_layers number of TransformerSelfAttentionLayer blocks
    - RMS Norm layer applied to the output of the transformer
    - Final projection into token space

    Args:
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        embed_dim (int): embedding dimension for self-attention
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        rope_base (int): base for the rotary positional embeddings. Default: 500_000
        intermediate_dim (Optional[int]): intermediate dimension for MLP. If not specified,
            this is computed using :func:`~torchtune.modules.scale_hidden_dim_for_mlp`
        norm_eps (float): epsilon in RMS norms.

    Returns:
        TransformerDecoder: Instantiation of Llama3 model.
    """
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    hidden_dim = (
        intermediate_dim if intermediate_dim else scale_hidden_dim_for_mlp(embed_dim)
    )
    rope = RotaryPositionalEmbeddings(
        dim=head_dim, max_seq_len=max_seq_len, base=rope_base
    )

    layers = nn.ModuleList()
    for _ in range(num_layers):
        self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )
        mlp = llama3_mlp(dim=embed_dim, hidden_dim=hidden_dim)
        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        )
        layers.append(layer)

    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )


def llama3_mlp(dim: int, hidden_dim: int, quantize_base: bool = False) -> FeedForward:
    """
    Build the MLP layer associated with the Llama model.
    """
    gate_proj = (
        nn.Linear(dim, hidden_dim, bias=False)
        if not quantize_base
        else FrozenNF4Linear(dim, hidden_dim, bias=False)
    )
    down_proj = (
        nn.Linear(hidden_dim, dim, bias=False)
        if not quantize_base
        else FrozenNF4Linear(hidden_dim, dim, bias=False)
    )
    up_proj = (
        nn.Linear(dim, hidden_dim, bias=False)
        if not quantize_base
        else FrozenNF4Linear(dim, hidden_dim, bias=False)
    )
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj)


# ------------------ LoRA Llama3 ------------------


def lora_llama3(
    lora_attn_modules: list[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    *,
    # llama3 args
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    intermediate_dim: Optional[int] = None,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-5,
    rope_base: int = 500_000,
    # LoRA args
    lora_rank: Union[int, dict[str, int]],
    lora_alpha: Union[float, dict[str, float]],
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    # Quantization args
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Return a version of Llama3 (an instance of :func:`~torchtune.modules.TransformerDecoder`)
    with LoRA applied based on the passed in configuration.

    Args:
        lora_attn_modules (list[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        embed_dim (int): embedding dimension for self-attention
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        intermediate_dim (Optional[int]): intermediate dimension for MLP. If not specified,
            this is computed using :func:`~torchtune.modules.scale_hidden_dim_for_mlp`
        norm_eps (float): epsilon in RMS norms.
        lora_rank (Union[int, dict[str, int]]): rank of each low-rank approximation. Can be a single
            int for uniform rank across all layers, or a dict mapping layer names to ranks for
            per-layer configuration. Example: ``{"layers.0.attn.q_proj": 8, "layers.1.attn.v_proj": 16}``.
        lora_alpha (Union[float, dict[str, float]]): scaling factor for the low-rank approximation.
            Can be a single float for uniform alpha across all layers, or a dict mapping layer names
            to alpha values for per-layer configuration. Example: ``{"layers.0.attn.q_proj": 16.0, "layers.1.attn.v_proj": 32.0}``.
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base: (bool): Whether to quantize base model weights or not. Only applied to base
            weights within linear layers LoRA is applied to. The final output linear projection is not
            supported for quantization currently.

    Returns:
        TransformerDecoder: Instantiation of Llama3 model with LoRA applied to
        a subset of the attention projections in each layer.

    """
    hidden_dim = (
        intermediate_dim if intermediate_dim else scale_hidden_dim_for_mlp(embed_dim)
    )

    # Extract default values for backward compatibility
    default_rank = None
    default_alpha = None
    if isinstance(lora_rank, dict):
        # Check for common fallback patterns - users may provide default_ prefixed values
        # This is just a convenience; resolve_lora_value will handle missing values gracefully
        pass
    if isinstance(lora_alpha, dict):
        pass

    layers = nn.ModuleList()
    for layer_idx in range(num_layers):
        # Build layer-specific names for per-layer LoRA configuration
        layer_prefix = f"layers.{layer_idx}"
        
        # Resolve layer-specific rank and alpha for attention modules
        attn_rank_resolver = lambda module_name: resolve_lora_value(
            lora_rank, f"{layer_prefix}.attn.{module_name}", default_value=8 if isinstance(lora_rank, dict) else None
        )
        attn_alpha_resolver = lambda module_name: resolve_lora_value(
            lora_alpha, f"{layer_prefix}.attn.{module_name}", default_value=16.0 if isinstance(lora_alpha, dict) else None
        )

        self_attn = lora_llama3_self_attention(
            lora_modules=lora_attn_modules,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
            rope_base=rope_base,
            lora_rank=lora_rank,  # Pass full config to self_attention for per-module resolution
            lora_alpha=lora_alpha,  # Pass full config to self_attention for per-module resolution
            lora_dropout=lora_dropout,
            quantize_base=quantize_base,
            use_dora=use_dora,
            layer_idx=layer_idx,  # Pass layer index for name resolution
        )

        if apply_lora_to_mlp:
            mlp = lora_llama3_mlp(
                dim=embed_dim,
                hidden_dim=hidden_dim,
                lora_rank=lora_rank,  # Pass full config to mlp for per-component resolution
                lora_alpha=lora_alpha,  # Pass full config to mlp for per-component resolution
                quantize_base=quantize_base,
                lora_dropout=lora_dropout,
                use_dora=use_dora,
                layer_idx=layer_idx,  # Pass layer index for name resolution
            )
        else:
            mlp = llama3_mlp(
                dim=embed_dim, hidden_dim=hidden_dim, quantize_base=quantize_base
            )

        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        )
        layers.append(layer)

    tok_embeddings = nn.Embedding(vocab_size, embed_dim)

    # Handle per-layer configuration for output projection
    if apply_lora_to_output:
        output_rank = resolve_lora_value(
            lora_rank, "output", default_value=8 if isinstance(lora_rank, dict) else None
        )
        output_alpha = resolve_lora_value(
            lora_alpha, "output", default_value=16.0 if isinstance(lora_alpha, dict) else None
        )
    
    # TODO: quantize_base is not applied to final output_proj currently.
    adapter_cls = DoRALinear if use_dora else LoRALinear
    output_proj = (
        adapter_cls(
            embed_dim,
            vocab_size,
            rank=output_rank,
            alpha=output_alpha,
            dropout=lora_dropout,
        )
        if apply_lora_to_output
        else nn.Linear(embed_dim, vocab_size, bias=False)
    )
    model = TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=(embed_dim // num_heads),
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )

    if quantize_base:
        # For QLoRA, we reparametrize 4-bit tensors to bf16, and offload to CPU on the fly
        # so as to not increase peak memory
        _register_reparametrize_state_dict_hooks(model)

    return model


def lora_llama3_self_attention(
    lora_modules: list[LORA_ATTN_MODULES],
    *,
    # MultiHeadAttention args
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: int = 500_000,
    # LoRA args
    lora_rank: Union[int, dict[str, int]],
    lora_alpha: Union[float, dict[str, float]],
    lora_dropout: float = 0.0,
    quantize_base: bool = False,
    use_dora: bool = False,
    layer_idx: Optional[int] = None,
) -> MultiHeadAttention:
    """
    Return an instance of :func:`~torchtune.modules.MultiHeadAttention` with LoRA
    applied to a subset of its linear layers

    Args:
        lora_modules (list[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to. Options are ``{"q_proj", "k_proj", "v_proj",
            "output_proj"}``.
        embed_dim (int): embedding dimension for self-attention
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        lora_rank (Union[int, dict[str, int]]): rank of each low-rank approximation. Can be a single
            int for uniform rank across all layers, or a dict mapping layer names to ranks for
            per-layer configuration.
        lora_alpha (Union[float, dict[str, float]]): scaling factor for the low-rank approximation.
            Can be a single float for uniform alpha across all layers, or a dict mapping layer names
            to alpha values for per-layer configuration.
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        quantize_base (bool): Whether to quantize base model parameters for linear layers
            LoRA is being applied to. Default is ``False``.
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        layer_idx (Optional[int]): Layer index for building layer-specific names when using
            per-layer LoRA configuration. Default: None.

    Returns:
        MultiHeadAttention: instantiation of self-attention module with LoRA
        applied to a subset of Q, K, V, output projections.

    Raises:
        ValueError: If lora_modules arg is an empty list
    """
    if not lora_modules:
        raise ValueError(
            f"Must pass one or more of {LORA_ATTN_MODULES} as lora_modules"
        )

    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    adapter_cls = DoRALinear if use_dora else LoRALinear
    
    # Helper function to resolve rank and alpha for each projection
    def get_projection_params(proj_name: str):
        if layer_idx is not None:
            layer_name = f"layers.{layer_idx}.attn.{proj_name}"
        else:
            layer_name = proj_name
            
        rank = resolve_lora_value(
            lora_rank, layer_name, default_value=8 if isinstance(lora_rank, dict) else None
        )
        alpha = resolve_lora_value(
            lora_alpha, layer_name, default_value=16.0 if isinstance(lora_alpha, dict) else None
        )
        return rank, alpha

    q_rank, q_alpha = get_projection_params("q_proj")
    q_proj = (
        adapter_cls(
            embed_dim,
            num_heads * head_dim,
            rank=q_rank,
            alpha=q_alpha,
            dropout=lora_dropout,
            quantize_base=quantize_base,
        )
        if "q_proj" in lora_modules
        else (
            nn.Linear(embed_dim, num_heads * head_dim, bias=False)
            if not quantize_base
            else FrozenNF4Linear(embed_dim, num_heads * head_dim, bias=False)
        )
    )
    
    k_rank, k_alpha = get_projection_params("k_proj")
    k_proj = (
        adapter_cls(
            embed_dim,
            num_kv_heads * head_dim,
            rank=k_rank,
            alpha=k_alpha,
            dropout=lora_dropout,
            quantize_base=quantize_base,
        )
        if "k_proj" in lora_modules
        else (
            nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
            if not quantize_base
            else FrozenNF4Linear(embed_dim, num_kv_heads * head_dim, bias=False)
        )
    )
    
    v_rank, v_alpha = get_projection_params("v_proj")
    v_proj = (
        adapter_cls(
            embed_dim,
            num_kv_heads * head_dim,
            rank=v_rank,
            alpha=v_alpha,
            dropout=lora_dropout,
            quantize_base=quantize_base,
        )
        if "v_proj" in lora_modules
        else (
            nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
            if not quantize_base
            else FrozenNF4Linear(embed_dim, num_kv_heads * head_dim, bias=False)
        )
    )
    
    output_rank, output_alpha = get_projection_params("output_proj")
    output_proj = (
        adapter_cls(
            embed_dim,
            embed_dim,
            rank=output_rank,
            alpha=output_alpha,
            dropout=lora_dropout,
            quantize_base=quantize_base,
        )
        if "output_proj" in lora_modules
        else (
            nn.Linear(embed_dim, embed_dim, bias=False)
            if not quantize_base
            else FrozenNF4Linear(embed_dim, embed_dim, bias=False)
        )
    )
    rope = RotaryPositionalEmbeddings(
        dim=head_dim, max_seq_len=max_seq_len, base=rope_base
    )
    self_attn = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        output_proj=output_proj,
        pos_embeddings=rope,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
    )
    return self_attn


def lora_llama3_mlp(
    *,
    dim: int,
    hidden_dim: int,
    lora_rank: Union[int, dict[str, int]],
    lora_alpha: Union[float, dict[str, float]],
    lora_dropout: float = 0.0,
    quantize_base: bool = False,
    use_dora: bool = False,
    layer_idx: Optional[int] = None,
) -> FeedForward:
    """
    Build a LoRA MLP layer with support for per-component rank and alpha configuration.
    
    Args:
        dim (int): Input and output dimension of the MLP.
        hidden_dim (int): Hidden dimension of the MLP.
        lora_rank (Union[int, dict[str, int]]): Rank for LoRA. Can be a single int for uniform rank
            across all components, or a dict mapping component names to ranks for per-component
            configuration.
        lora_alpha (Union[float, dict[str, float]]): Alpha for LoRA. Can be a single float for uniform
            alpha across all components, or a dict mapping component names to alpha values for
            per-component configuration.
        lora_dropout (float): LoRA dropout probability. Default: 0.0.
        quantize_base (bool): Whether to quantize base model parameters. Default: False.
        use_dora (bool): Whether to use DoRA instead of LoRA. Default: False.
        layer_idx (Optional[int]): Layer index for building layer-specific names when using
            per-component LoRA configuration. Default: None.
    
    Returns:
        FeedForward: MLP layer with LoRA applied to components.
    """
    from torchtune.modules.peft._utils import resolve_lora_value
    
    adapter_cls = DoRALinear if use_dora else LoRALinear
    
    # Helper function to resolve rank and alpha for each MLP component
    def get_component_params(component_name: str):
        if layer_idx is not None:
            layer_name = f"layers.{layer_idx}.mlp.{component_name}"
        else:
            layer_name = component_name
            
        rank = resolve_lora_value(
            lora_rank, layer_name, default_value=8 if isinstance(lora_rank, dict) else None
        )
        alpha = resolve_lora_value(
            lora_alpha, layer_name, default_value=16.0 if isinstance(lora_alpha, dict) else None
        )
        return rank, alpha

    # Gate projection (w1)
    w1_rank, w1_alpha = get_component_params("w1")
    gate_proj = adapter_cls(
        in_dim=dim,
        out_dim=hidden_dim,
        rank=w1_rank,
        alpha=w1_alpha,
        dropout=lora_dropout,
        quantize_base=quantize_base,
    )
    
    # Down projection (w2)
    w2_rank, w2_alpha = get_component_params("w2")
    down_proj = adapter_cls(
        in_dim=hidden_dim,
        out_dim=dim,
        rank=w2_rank,
        alpha=w2_alpha,
        dropout=lora_dropout,
        quantize_base=quantize_base,
    )
    
    # Up projection (w3)
    w3_rank, w3_alpha = get_component_params("w3")
    up_proj = adapter_cls(
        in_dim=dim,
        out_dim=hidden_dim,
        rank=w3_rank,
        alpha=w3_alpha,
        dropout=lora_dropout,
        quantize_base=quantize_base,
    )
    
    return FeedForward(
        gate_proj=gate_proj,
        down_proj=down_proj,
        up_proj=up_proj,
    )
