# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torchtune.modules.peft._utils import resolve_lora_value


class TestPerLayerLoRA:
    """Test cases for per-layer LoRA rank and alpha configuration."""

    def test_resolve_lora_value_single_value(self):
        """Test backward compatibility with single values."""
        # Test int
        result = resolve_lora_value(8, "layers.0.attn.q_proj")
        assert result == 8
        
        # Test float
        result = resolve_lora_value(16.0, "layers.0.attn.q_proj")
        assert result == 16.0

    def test_resolve_lora_value_dict_exact_match(self):
        """Test dict configuration with exact layer name match."""
        config = {
            "layers.0.attn.q_proj": 16,
            "layers.1.attn.v_proj": 32
        }
        
        result = resolve_lora_value(config, "layers.0.attn.q_proj")
        assert result == 16
        
        result = resolve_lora_value(config, "layers.1.attn.v_proj")
        assert result == 32

    def test_resolve_lora_value_dict_with_default(self):
        """Test dict configuration with default fallback."""
        config = {
            "layers.0.attn.q_proj": 16,
            "layers.1.attn.v_proj": 32
        }
        
        # Layer not in dict, should use default
        result = resolve_lora_value(config, "layers.2.attn.k_proj", default_value=8)
        assert result == 8
        
        # Float default
        result = resolve_lora_value(config, "layers.2.attn.k_proj", default_value=24.0)
        assert result == 24.0

    def test_resolve_lora_value_mlp_components(self):
        """Test dict configuration with MLP component-level granularity."""
        config = {
            "layers.0.mlp.w1": 16,
            "layers.0.mlp.w2": 8,
            "layers.0.mlp.w3": 12,
            "layers.1.mlp.w1": 24,
        }
        
        # Test individual MLP components
        result = resolve_lora_value(config, "layers.0.mlp.w1")
        assert result == 16
        
        result = resolve_lora_value(config, "layers.0.mlp.w2")
        assert result == 8
        
        result = resolve_lora_value(config, "layers.0.mlp.w3")
        assert result == 12
        
        result = resolve_lora_value(config, "layers.1.mlp.w1")
        assert result == 24
        
        # Test fallback for missing component
        result = resolve_lora_value(config, "layers.1.mlp.w2", default_value=10)
        assert result == 10

    def test_resolve_lora_value_dict_no_default_raises(self):
        """Test that missing layer without default raises ValueError."""
        config = {
            "layers.0.attn.q_proj": 16,
            "layers.1.attn.v_proj": 32
        }
        
        with pytest.raises(ValueError, match="Layer 'layers.2.attn.k_proj' not found"):
            resolve_lora_value(config, "layers.2.attn.k_proj")

    def test_resolve_lora_value_mixed_types(self):
        """Test dict with mixed int/float values."""
        config = {
            "layers.0.attn.q_proj": 16,         # int
            "layers.1.attn.v_proj": 32.5,       # float
            "layers.2.mlp": 8                   # int
        }
        
        assert resolve_lora_value(config, "layers.0.attn.q_proj") == 16
        assert resolve_lora_value(config, "layers.1.attn.v_proj") == 32.5
        assert resolve_lora_value(config, "layers.2.mlp") == 8

    def test_resolve_lora_value_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty dict with default
        result = resolve_lora_value({}, "any.layer", default_value=42)
        assert result == 42
        
        # Empty dict without default should raise
        with pytest.raises(ValueError):
            resolve_lora_value({}, "any.layer")
        
        # Zero values
        config = {"layer": 0}
        assert resolve_lora_value(config, "layer") == 0
        assert resolve_lora_value(0, "any.layer") == 0

    def test_layer_naming_patterns(self):
        """Test various layer naming patterns that might be used."""
        config = {
            "layers.0.attn.q_proj": 8,
            "layers.15.attn.v_proj": 16,
            "layers.31.mlp": 12,
            "output": 24
        }
        
        assert resolve_lora_value(config, "layers.0.attn.q_proj") == 8
        assert resolve_lora_value(config, "layers.15.attn.v_proj") == 16  
        assert resolve_lora_value(config, "layers.31.mlp") == 12
        assert resolve_lora_value(config, "output") == 24

    def test_real_world_config_examples(self):
        """Test realistic configuration examples."""
        # Example 1: Decreasing rank by layer depth
        rank_config = {
            "layers.0.attn.q_proj": 64,
            "layers.5.attn.q_proj": 32,
            "layers.10.attn.q_proj": 16,
            "layers.15.attn.q_proj": 8,
        }
        
        # Example 2: Higher alpha for important projections
        alpha_config = {
            "layers.0.attn.q_proj": 128.0,  # First layer gets higher alpha
            "layers.0.attn.k_proj": 64.0,
            "layers.0.attn.v_proj": 64.0,
            "output": 32.0,                  # Output projection gets special treatment
        }
        
        # Test rank resolution
        assert resolve_lora_value(rank_config, "layers.0.attn.q_proj") == 64
        assert resolve_lora_value(rank_config, "layers.15.attn.q_proj") == 8
        assert resolve_lora_value(rank_config, "layers.20.attn.q_proj", default_value=4) == 4
        
        # Test alpha resolution  
        assert resolve_lora_value(alpha_config, "layers.0.attn.q_proj") == 128.0
        assert resolve_lora_value(alpha_config, "output") == 32.0
        assert resolve_lora_value(alpha_config, "layers.1.attn.q_proj", default_value=16.0) == 16.0