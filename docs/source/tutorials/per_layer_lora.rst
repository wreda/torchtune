# Per-Layer LoRA Configuration

This document explains the new per-layer LoRA rank and alpha configuration feature in torchtune.

## Overview

LoRA (Low-Rank Adaptation) fine-tuning now supports per-layer configuration of rank and alpha parameters, allowing for more fine-grained control over model adaptation. This feature maintains full backward compatibility with existing configurations.

## Usage

### Backward Compatible (Single Values)

The traditional single-value configuration continues to work as before:

```yaml
model:
  _component_: torchtune.models.llama3.lora_llama3_8b
  lora_attn_modules: ['q_proj', 'v_proj', 'output_proj']
  apply_lora_to_mlp: True
  lora_rank: 8          # Applied uniformly to all LoRA layers
  lora_alpha: 16        # Applied uniformly to all LoRA layers
```

### Per-Layer Configuration (New)

You can now specify different rank and alpha values for different layers and projections:

```yaml
model:
  _component_: torchtune.models.llama3.lora_llama3_8b
  lora_attn_modules: ['q_proj', 'v_proj', 'output_proj']
  apply_lora_to_mlp: True
  
  # Per-layer rank configuration
  lora_rank:
    layers.0.attn.q_proj: 16      # First layer q_proj gets rank 16
    layers.0.attn.v_proj: 8       # First layer v_proj gets rank 8
    layers.1.attn.q_proj: 12      # Second layer q_proj gets rank 12
    layers.2.mlp: 8               # MLP layer gets rank 8
    output: 16                    # Output projection gets rank 16
    
  # Per-layer alpha configuration  
  lora_alpha:
    layers.0.attn.q_proj: 32.0    # First layer q_proj gets alpha 32
    layers.0.attn.v_proj: 16.0    # First layer v_proj gets alpha 16
    layers.1.attn.q_proj: 24.0    # Second layer q_proj gets alpha 24
    layers.2.mlp: 16.0            # MLP layer gets alpha 16
    output: 32.0                  # Output projection gets alpha 32
```

### Layer Naming Convention

Layer names follow this pattern:
- Attention projections: `layers.{layer_idx}.attn.{projection_name}`
  - `projection_name` can be: `q_proj`, `k_proj`, `v_proj`, `output_proj`
- MLP layers: `layers.{layer_idx}.mlp`
- Output projection: `output`

Examples:
- `layers.0.attn.q_proj` - First layer, attention, query projection
- `layers.15.attn.v_proj` - 16th layer, attention, value projection  
- `layers.31.mlp` - 32nd layer, MLP
- `output` - Final output projection

### Fallback Behavior

When using per-layer configuration, you can specify values for only some layers. For layers not explicitly configured:

1. **With default values**: The system uses reasonable defaults (rank=8, alpha=16.0)
2. **Custom fallbacks**: You can specify fallback values in your configuration logic

## Supported Models

The per-layer LoRA configuration is currently supported for:

- ✅ Llama3 models (`lora_llama3_8b`, `lora_llama3_70b`)
- ✅ Qwen2 models (`lora_qwen2_7b`, `lora_qwen2_1_5b`, `lora_qwen2_0_5b`)
- 🚧 Additional models (Llama2, Phi3, etc.) - coming soon

## Use Cases

### 1. Layer-Depth Adaptation
Apply higher ranks to earlier layers and lower ranks to deeper layers:

```yaml
lora_rank:
  layers.0.attn.q_proj: 32    # Early layers get higher rank
  layers.10.attn.q_proj: 16   # Middle layers get medium rank  
  layers.20.attn.q_proj: 8    # Later layers get lower rank
```

### 2. Projection-Specific Tuning
Use different configurations for different projection types:

```yaml
lora_rank:
  layers.0.attn.q_proj: 16    # Query projection
  layers.0.attn.k_proj: 8     # Key projection (lower rank)
  layers.0.attn.v_proj: 16    # Value projection
  layers.0.attn.output_proj: 12  # Output projection
```

### 3. Critical Layer Enhancement
Apply higher rank/alpha to specific important layers:

```yaml
lora_rank:
  layers.0.attn.q_proj: 32    # First layer is critical
  layers.15.attn.q_proj: 32   # Mid-layer attention
  output: 24                  # Output layer is important
  
lora_alpha:
  layers.0.attn.q_proj: 64.0  # Higher scaling for critical layers
  layers.15.attn.q_proj: 64.0
  output: 48.0
```

## Implementation Details

The feature is implemented through:

1. **`resolve_lora_value()` utility**: Resolves rank/alpha values for specific layers
2. **Updated type hints**: Model builders now accept `Union[int, dict[str, int]]` for rank and `Union[float, dict[str, float]]` for alpha
3. **Backward compatibility**: Single values are handled as before
4. **Per-layer resolution**: Component builders resolve values for each layer during model construction

## Migration Guide

Existing configurations require no changes and will continue to work exactly as before. To use per-layer configuration:

1. Replace single `lora_rank`/`lora_alpha` values with dictionaries
2. Use the layer naming convention documented above
3. Specify values only for layers you want to customize
4. Test your configuration with the new functionality

## Examples

See the example configurations in:
- `recipes/configs/llama3/8B_per_layer_lora_example.yaml` - Per-layer configuration
- `recipes/configs/llama3/8B_backward_compatible_lora_example.yaml` - Backward compatibility demo