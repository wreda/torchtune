# Implementation Pattern for Per-Layer LoRA Support

This document outlines the pattern for adding per-layer LoRA support to additional model architectures in torchtune.

## Files Modified Per Architecture

For each model architecture (e.g., `llama2`, `phi3`, `mistral`), the following files need updates:

### 1. Model Builders (`_model_builders.py`)

Update function signatures to accept Union types:

```python
# Before
def lora_model_name(
    lora_rank: int = 8,
    lora_alpha: float = 16,
    ...
) -> TransformerDecoder:

# After  
def lora_model_name(
    lora_rank: Union[int, dict[str, int]] = 8,
    lora_alpha: Union[float, dict[str, float]] = 16,
    ...
) -> TransformerDecoder:
```

Update docstrings to document the new Union types:

```python
"""
Args:
    lora_rank (Union[int, dict[str, int]]): rank of each low-rank approximation. Can be a single
        int for uniform rank across all layers, or a dict mapping layer names to ranks for
        per-layer configuration. Default: 8. Example: ``{"layers.0.attn.q_proj": 8, "layers.1.attn.v_proj": 16}``.
    lora_alpha (Union[float, dict[str, float]]): scaling factor for the low-rank approximation.
        Can be a single float for uniform alpha across all layers, or a dict mapping layer names
        to alpha values for per-layer configuration. Default: 16. Example: ``{"layers.0.attn.q_proj": 16.0, "layers.1.attn.v_proj": 32.0}``.
"""
```

Add import for Union type:

```python
from typing import Optional, Union
```

### 2. Component Builders (`_component_builders.py`)

Update the main component builder function signature:

```python
# Before
def lora_model_component(
    lora_rank: int,
    lora_alpha: float,
    ...
) -> TransformerDecoder:

# After
def lora_model_component(
    lora_rank: Union[int, dict[str, int]],
    lora_alpha: Union[float, dict[str, float]],
    ...
) -> TransformerDecoder:
```

Add import for the resolve function:

```python
from torchtune.modules.peft._utils import resolve_lora_value
```

Update the layer building loop to resolve per-layer values:

```python
# Before
for _ in range(num_layers):
    self_attn = lora_model_self_attention(
        # ... other args ...
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        # ...
    )

# After
for layer_idx in range(num_layers):
    self_attn = lora_model_self_attention(
        # ... other args ...
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        layer_idx=layer_idx,  # Pass layer index
        # ...
    )
```

Update the self-attention function to resolve per-layer values:

```python
def lora_model_self_attention(
    lora_rank: Union[int, dict[str, int]],
    lora_alpha: Union[float, dict[str, float]],
    layer_idx: Optional[int] = None,
    ...
) -> MultiHeadAttention:
    
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

    # Use resolved values for each projection
    q_rank, q_alpha = get_projection_params("q_proj")
    k_rank, k_alpha = get_projection_params("k_proj")
    # ... etc for v_proj, output_proj
```

Update MLP handling:

```python
if apply_lora_to_mlp:
    mlp_rank = resolve_lora_value(
        lora_rank, f"layers.{layer_idx}.mlp", default_value=8 if isinstance(lora_rank, dict) else None
    )
    mlp_alpha = resolve_lora_value(
        lora_alpha, f"layers.{layer_idx}.mlp", default_value=16.0 if isinstance(lora_alpha, dict) else None
    )
    
    mlp = lora_model_mlp(
        # ... other args ...
        lora_rank=mlp_rank,
        lora_alpha=mlp_alpha,
        # ...
    )
```

Update output projection handling:

```python
if apply_lora_to_output:
    output_rank = resolve_lora_value(
        lora_rank, "output", default_value=8 if isinstance(lora_rank, dict) else None
    )
    output_alpha = resolve_lora_value(
        lora_alpha, "output", default_value=16.0 if isinstance(lora_alpha, dict) else None
    )

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
```

## Layer Naming Convention

All models should use consistent layer naming:
- Attention projections: `layers.{layer_idx}.attn.{projection_name}`
- MLP layers: `layers.{layer_idx}.mlp`  
- Output projection: `output`

## Testing

For each new model, add test cases to verify:
1. Backward compatibility with single values
2. Per-layer dictionary configuration works
3. Default fallback behavior functions correctly
4. Error handling for missing layers without defaults

## Example Implementation

See the complete implementation in:
- `torchtune/models/llama3/_component_builders.py`
- `torchtune/models/llama3/_model_builders.py`
- `torchtune/models/qwen2/_model_builders.py`

## Models to Update

Remaining model architectures that need this pattern applied:
- [ ] llama2 (partially started)
- [ ] phi3
- [ ] phi4
- [ ] mistral
- [ ] gemma
- [ ] gemma2
- [ ] llama3_1
- [ ] llama3_2
- [ ] llama3_3
- [ ] llama4
- [ ] qwen2_5
- [ ] qwen3

Each follows the same pattern with minor variations for architecture-specific details.