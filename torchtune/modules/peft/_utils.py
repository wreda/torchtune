# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from typing import Any, Generator, Literal, Optional, Protocol, runtime_checkable, Union

import torch
from torch import nn
from torchtune.utils._logging import deprecate_parameter

# Modules from MultiHeadAttention that LoRA can be applied to
LORA_ATTN_MODULES = Literal["q_proj", "k_proj", "v_proj", "output_proj"]


def resolve_lora_value(
    value: Union[int, float, dict[str, Union[int, float]]],
    layer_name: str,
    default_value: Optional[Union[int, float]] = None,
) -> Union[int, float]:
    """
    Resolve a LoRA parameter value (rank or alpha) for a specific layer.

    This function supports both backward-compatible single values and the new
    per-layer dictionary configuration. If a dictionary is provided, it looks
    up the layer-specific value. If not found, it falls back to the default value.

    Args:
        value (Union[int, float, dict[str, Union[int, float]]]): The LoRA parameter value.
            Can be a single int/float (backward compatible) or a dict mapping layer names to values.
        layer_name (str): The name of the layer to resolve the value for.
        default_value (Optional[Union[int, float]]): Default value to use if layer_name
            is not found in the dictionary. If None and layer_name is not found,
            raises ValueError.

    Returns:
        Union[int, float]: The resolved value for the specified layer.

    Raises:
        ValueError: If value is a dict, layer_name is not found, and no default_value is provided.

    Example:
        >>> # Single value (backward compatible)
        >>> resolve_lora_value(8, "layers.0.attn.q_proj")
        8

        >>> # Per-layer dictionary with fallback
        >>> config = {"layers.0.attn.q_proj": 16, "layers.1.attn.v_proj": 32}
        >>> resolve_lora_value(config, "layers.0.attn.q_proj")
        16
        >>> resolve_lora_value(config, "layers.2.attn.q_proj", default_value=8)
        8
    """
    if isinstance(value, dict):
        # Try direct lookup first
        if layer_name in value:
            result = value[layer_name]
            # Convert DictConfig objects to Python primitives
            if hasattr(result, '__float__') and not isinstance(result, (int, float)):
                # This handles DictConfig values
                try:
                    return float(result) if '.' in str(result) else int(result)
                except (ValueError, TypeError):
                    return result
            return result
        
        # Handle MLP component name mapping: gate_proj/down_proj/up_proj <-> w1/w2/w3
        if ".mlp." in layer_name:
            # Map state dict names to config names
            alt_layer_name = layer_name
            if ".mlp.gate_proj" in layer_name:
                alt_layer_name = layer_name.replace(".mlp.gate_proj", ".mlp.w1")
            elif ".mlp.down_proj" in layer_name:
                alt_layer_name = layer_name.replace(".mlp.down_proj", ".mlp.w2") 
            elif ".mlp.up_proj" in layer_name:
                alt_layer_name = layer_name.replace(".mlp.up_proj", ".mlp.w3")
            # Also try the reverse mapping for completeness
            elif ".mlp.w1" in layer_name:
                alt_layer_name = layer_name.replace(".mlp.w1", ".mlp.gate_proj")
            elif ".mlp.w2" in layer_name:
                alt_layer_name = layer_name.replace(".mlp.w2", ".mlp.down_proj")
            elif ".mlp.w3" in layer_name:
                alt_layer_name = layer_name.replace(".mlp.w3", ".mlp.up_proj")
                
            # Try the alternative name
            if alt_layer_name in value and alt_layer_name != layer_name:
                result = value[alt_layer_name]
                # Convert DictConfig objects to Python primitives
                if hasattr(result, '__float__') and not isinstance(result, (int, float)):
                    try:
                        return float(result) if '.' in str(result) else int(result)
                    except (ValueError, TypeError):
                        return result
                return result
                
            # Also try fallback to whole MLP layer if component-specific config not found
            mlp_layer_name = layer_name
            if any(comp in layer_name for comp in [".gate_proj", ".down_proj", ".up_proj", ".w1", ".w2", ".w3"]):
                # Extract the layer part: layers.X.mlp 
                import re
                match = re.match(r"(layers\.\d+\.mlp)", layer_name)
                if match:
                    mlp_layer_name = match.group(1)
                    if mlp_layer_name in value:
                        result = value[mlp_layer_name]
                        # Convert DictConfig objects to Python primitives
                        if hasattr(result, '__float__') and not isinstance(result, (int, float)):
                            try:
                                return float(result) if '.' in str(result) else int(result)
                            except (ValueError, TypeError):
                                return result
                        return result
        
        # If still not found, use default value
        if default_value is not None:
            return default_value
        else:
            raise ValueError(
                f"Layer '{layer_name}' not found in per-layer LoRA configuration "
                f"and no default value provided. Available layers: {list(value.keys())}"
            )
    else:
        # Single value - backward compatible behavior
        # Also handle DictConfig for single values
        if hasattr(value, '__float__') and not isinstance(value, (int, float)):
            try:
                return float(value) if '.' in str(value) else int(value)
            except (ValueError, TypeError):
                return value
        return value


@runtime_checkable
class AdapterModule(Protocol):
    """
    Interface for an ``nn.Module`` containing adapter weights.
    Note that an adapter module does not have to explicitly implement this protocol,
    but it must define the ``adapter_params(self)`` method.
    """

    def adapter_params(self) -> list[str]:
        """
        Return a list of strings corresponding to the names of the ``nn.Parameter`` s in
        the model coming from the adapter.
        E.g. if an nn.Module has adapter ``self.proj = nn.Linear(in_dim, out_dim)``,
        then adapter_params should return ``['proj.weight', 'proj.bias']``.

        See LoRALinear's :func:`~torchtune.modules.peft.LoRALinear.adapter_params` for an example.
        """
        pass


def get_adapter_params(model: nn.Module) -> dict[str, nn.Parameter]:
    """
    Return the subset of parameters from a model that correspond to an adapter.
    Assumes that any adapter class has defined the
    :func:`~torchtune.modules.peft.AdapterModule.adapter_params` method.

    Args:
        model (nn.Module): Instance of model class containing some adapter params.

    Returns:
        dict[str, nn.Parameter]: the subset of model's state dict containing
        only adapter parameters.

    """
    adapter_params = {}
    for k, v in model.named_modules():
        if hasattr(v, "adapter_params") and callable(v.adapter_params):
            current_adapter_params = v.adapter_params()
            for n, p in v.named_parameters(recurse=True):
                if n in current_adapter_params:
                    full_key = f"{k}.{n}" if k else n
                    adapter_params.update({full_key: p})
                    current_adapter_params.remove(n)
            assert (
                current_adapter_params == []
            ), f"Adapter params {current_adapter_params} not converted"
    return adapter_params


def set_trainable_params(
    model: nn.Module, adapter_params: Union[dict[str, Any], set]
) -> None:
    """
    Set trainable parameters for an nn.Module based on a state dict of adapter parameters.

    Args:
        model (nn.Module): Instance of model class containing some adapter params.
        adapter_params (Union[dict[str, Any], set]): State dict mapping adapter key names to their
            respective nn.Parameters (i.e. outputs of :func:`~torchtune.modules.peft.get_adapter_params`.)

    Returns:
        None
    """
    for k, v in model.named_parameters():
        v.requires_grad_(k in adapter_params)


def get_lora_module_names(
    lora_attn_modules: list[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool,
    apply_lora_to_output: bool,
) -> list[str]:
    """
    Return a list of the names of modules in the model that have LoRA applied. Note that
    the names here are local to their modules and not the fully qualified names from the
    model state dict.


    Args:
        lora_attn_modules (list[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether LoRA is applied to each MLP linear.
        apply_lora_to_output (bool): whether LoRA is applied to the final output projection.

    Returns:
        list[str]: list of module names in the model that have LoRA applied.
    """
    lora_module_keys = lora_attn_modules
    if apply_lora_to_mlp:
        lora_module_keys = lora_module_keys + ["w1", "w2", "w3"]
    if apply_lora_to_output:
        lora_module_keys.append("output")
    return lora_module_keys


def get_adapter_state_dict(
    state_dict: dict[str, Any], device: Optional[str] = "cpu"
) -> dict[str, Any]:
    """
    Return the subset of the full state_dict from a model that correspond to an adapter.
    Assumes that "lora" and "magnitude" are unique names for adapter parameters, and
    that the state_dict is not sharded. All returned parameters are moved to CPU.

    Args:
        state_dict (dict[str, Any]): Full model state dict.
        device (Optional[str]): device to move adapter parameters to. Default: 'cpu'

    Returns:
        dict[str, Any]: the subset of model's state dict containing
        only adapter parameters.

    """
    adapter_key_filter = lambda x: "lora" in x or "magnitude" in x
    return {k: v.to(device) for k, v in state_dict.items() if adapter_key_filter(k)}


def _get_lora_modules(state_dict: dict[str, Any]) -> set[str]:
    """
    Get the keys from a state dict that correspond to LoRALinear modules.

    For example, if state_dict is the state dict of model and model.x.y.z is a
    LoRALinear, this method will return "model.x.y.z", not
    "model.x.y.z.lora_a.weight" or "model.x.y.z.lora_b.weight".

    Args:
        state_dict (dict[str, Any]): State dict from a model.

    Returns:
        set[str]: Set of keys in the state dict that correspond to LoRA modules.
    """
    lora_keys = [
        k
        for k in state_dict.keys()
        if ("lora" in k or "magnitude" in k) and ("experts" not in k)
    ]
    return set(
        [
            k.replace(".lora_a.weight", "")
            .replace(".lora_b.weight", "")
            .replace(".magnitude", "")
            for k in lora_keys
        ]
    )


def _get_lora_moe_modules(state_dict: dict[str, Any]) -> set[str]:
    """
    Get the keys from a state dict that correspond to LoRAGroupedExperts modules.

    For example, if state_dict is the state dict of model and model.x.y.z is a
    LoRAGroupedExperts, this method will return "model.x.y.z", not
    "model.x.y.z.lora_a.weight" or "model.x.y.z.lora_b.weight".

    Args:
        state_dict (dict[str, Any]): State dict from a model.

    Returns:
        set[str]: Set of keys in the state dict that correspond to LoRA MoE modules.
    """
    lora_keys = [k for k in state_dict.keys() if "lora" in k and "experts" in k]
    return set(
        [
            k.replace(".lora_gate_a", "")
            .replace(".lora_gate_b", "")
            .replace(".lora_down_a", "")
            .replace(".lora_down_b", "")
            .replace(".lora_up_a", "")
            .replace(".lora_up_b", "")
            for k in lora_keys
        ]
    )


@torch.no_grad
def get_merged_lora_ckpt(
    state_dict: dict[str, Any],
    rank: Union[int, dict[str, int]],
    alpha: Union[float, dict[str, float]],
) -> dict[str, Any]:
    """
    Merge LoRA weights into the base model format for efficient inference.
    NOTE: This function modifies state_dict inplace. If you do not want to do that,
    make a copy prior to calling this function.

    For every LoRA module in the state dict, this function will convert its
    base weight then delete the LoRA-specific parameters.

    Args:
        state_dict (dict[str, Any]): State dict from a model.
        rank (Union[int, dict[str, int]]): The rank of LoRA matrices. Can be a single
            int for uniform rank across all layers, or a dict mapping layer names to ranks
            for per-layer configuration.
        alpha (Union[float, dict[str, float]]): The alpha value used for scaling LoRA 
            decompositions. Can be a single float for uniform alpha across all layers, 
            or a dict mapping layer names to alpha values for per-layer configuration.

    Returns:
        dict[str, Any]: The merged state dict.
    """
    lora_modules = _get_lora_modules(state_dict)
    lora_moe_modules = _get_lora_moe_modules(state_dict)
    
    print(f"DEBUG: rank type: {type(rank)}, value: {rank}")
    print(f"DEBUG: alpha type: {type(alpha)}, value: {alpha}")
    print(f"DEBUG: lora_modules: {lora_modules}")
    print(f"DEBUG: lora_moe_modules: {lora_moe_modules}")
    
    for module in lora_modules.union(lora_moe_modules):
        print(f"DEBUG: Processing module: {module}")
        
        # Resolve rank and alpha for this specific module
        # Use default values for backward compatibility when dictionaries are provided
        module_rank = resolve_lora_value(
            rank, module, default_value=8 if isinstance(rank, dict) else None
        )
        module_alpha = resolve_lora_value(
            alpha, module, default_value=16.0 if isinstance(alpha, dict) else None
        )
        
        print(f"DEBUG: Resolved module_rank type: {type(module_rank)}, value: {module_rank}")
        print(f"DEBUG: Resolved module_alpha type: {type(module_alpha)}, value: {module_alpha}")

        # TODO: we don't currently support DoRA for MoE layers
        if "experts" in module:
            for param in ["gate", "up", "down"]:
                lora_a_weight = state_dict[f"{module}.lora_{param}_a"]
                lora_b_weight = state_dict[f"{module}.lora_{param}_b"]
                state_dict[f"{module}.{param}_proj"] += (
                    (module_alpha / module_rank)
                    * lora_b_weight.transpose(1, 2)
                    @ lora_a_weight.transpose(1, 2)
                ).transpose(1, 2)
                del state_dict[f"{module}.lora_{param}_a"]
                del state_dict[f"{module}.lora_{param}_b"]
            continue

        lora_a_weight = state_dict[f"{module}.lora_a.weight"]
        lora_b_weight = state_dict[f"{module}.lora_b.weight"]
        lora_magnitude = state_dict.get(f"{module}.magnitude", None)

        # If magnitude is present, calculate merged DoRA weight
        if lora_magnitude is not None:
            base_weight = state_dict[f"{module}.weight"].to(lora_a_weight.dtype)

            lora_weight = (module_alpha / module_rank) * lora_b_weight @ lora_a_weight
            merged_weight = base_weight + lora_weight
            weight_norm = torch.linalg.norm(base_weight + lora_weight, dim=1)
            mag_norm_scale = (lora_magnitude / weight_norm).view(-1, 1)
            merged_weight *= mag_norm_scale
            state_dict[f"{module}.weight"] = merged_weight
            del state_dict[f"{module}.magnitude"]

        # Otherwise it is just vanilla LoRA
        else:
            state_dict[f"{module}.weight"] += (
                (module_alpha / module_rank) * lora_b_weight @ lora_a_weight
            )

        del state_dict[f"{module}.lora_a.weight"]
        del state_dict[f"{module}.lora_b.weight"]

    return state_dict


@contextlib.contextmanager
def disable_adapter(model: nn.Module) -> Generator[None, None, None]:
    """
    Temporarily disable the adapters in a model. For example,
    this can be used in DPO for treating the LoRA adapters as the policy model
    and disabling it to treat the base model as the reference model.

    This context manager goes through all modules in the provided neural network model,
    and if a module has an ``adapter_params`` attribute that is callable and a ``disabled`` attribute,
    it sets ``disabled`` to True. Then, the control is given back to caller. When exiting the context manager,
    it sets ``disabled`` back to False for all modules that were temporarily disabled.

    Args:
        model (nn.Module): The model whose adapters are to be temporarily disabled.
    Yields:
        None: This function yields control back to the caller, with the adapters disabled.
    Example:
        >>> with disable_adapter(model):
        ...     # Perform operations with adapters disabled
        ...     pass

    """
    for _, module in model.named_modules():
        if (
            hasattr(module, "adapter_params")
            and callable(module.adapter_params)
            and hasattr(module, "disabled")
        ):
            module.disabled = True
    try:
        yield
    finally:
        for _, module in model.named_modules():
            if (
                hasattr(module, "adapter_params")
                and callable(module.adapter_params)
                and hasattr(module, "disabled")
            ):
                module.disabled = False


@deprecate_parameter(
    param_name="lora_attn_modules", msg="Please use state_dict_keys instead."
)
@deprecate_parameter(
    param_name="apply_lora_to_mlp", msg="Please use state_dict_keys instead."
)
@deprecate_parameter(
    param_name="apply_lora_to_output", msg="Please use state_dict_keys instead."
)
def validate_missing_and_unexpected_for_lora(
    lora_attn_modules: Optional[list[LORA_ATTN_MODULES]] = None,
    apply_lora_to_mlp: Optional[bool] = None,
    apply_lora_to_output: Optional[bool] = None,
    state_dict_keys: Optional[list[str]] = None,
    base_missing: Optional[list[str]] = None,
    base_unexpected: Optional[list[str]] = None,
    lora_missing: Optional[list[str]] = None,
    lora_unexpected: Optional[list[str]] = None,
) -> None:
    """
    This function checks that LoRA and/or base model weights are loaded into the full model correctly.
    via set comparison of the missing kets. This function relies only on the values of missing and
    unexpected as returned by the load_state_dict API with strict=False.

    Args:
        lora_attn_modules (Optional[list[LORA_ATTN_MODULES]]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
            DEPRECATED: use state_dict_keys instead.
        apply_lora_to_mlp (Optional[bool]): whether LoRA is applied to each MLP linear.
            DEPRECATED: use state_dict_keys instead.
        apply_lora_to_output (Optional[bool]): whether LoRA is applied to the final output projection.
            DEPRECATED: use state_dict_keys instead.
        state_dict_keys (Optional[list[str]]): ground truth model state dict we are validating against
        base_missing (Optional[list[str]]): list of missing keys when loading base model weights.
            Default: None
        base_unexpected (Optional[list[str]]): list of unexpected keys when loading base model weights.
            Default: None
        lora_missing (Optional[list[str]]): list of missing keys when loading LoRA weights.
            Default: None
        lora_unexpected (Optional[list[str]]): list of unexpected keys when loading LoRA weights.
            Default: None
    Returns:
        None
    Raises:
        RuntimeError:
            If base_missing contains any base model keys, **or**
            if base_unexpected is nonempty, **or**
            if lora_missing contains any LoRA keys, **or**
            if lora_unexpected is nonempty.
    """
    if state_dict_keys is not None:
        is_lora_key = lambda x: "lora" in x or "magnitude" in x
        base_state_dict = set(k for k in state_dict_keys if not is_lora_key(k))
        lora_state_dict = set(k for k in state_dict_keys if is_lora_key(k))
        base_missing_set = set(base_missing or [])
        lora_missing_set = set(lora_missing or [])
        # Base missing should have LoRA keys only, and LoRA missing should have base model keys only
        # If there is an overlap, check if the key is adapter or base model key and raise accordingly
        missing_keys = (
            (base_missing_set & lora_missing_set)
            | (lora_missing_set & lora_state_dict)
            | (base_missing_set & base_state_dict)
        )
        err_msgs = []
        for key in missing_keys:
            if key in base_state_dict:
                err_msgs.append(f"- Missing non-LoRA key {key} from base model dict")
            elif key in lora_state_dict:
                err_msgs.append(f"- Missing LoRA key {key} from adapter state dict")
            else:
                raise RuntimeError(f"Unexpected key found missing: {key}")
        if len(err_msgs) > 0:
            raise RuntimeError(
                "Missing keys when validating state dict: \n" + "\n".join(err_msgs)
            )
    else:
        assert lora_attn_modules is not None
        assert apply_lora_to_mlp is not None
        assert apply_lora_to_output is not None
        lora_modules = get_lora_module_names(
            lora_attn_modules, apply_lora_to_mlp, apply_lora_to_output
        )
        is_lora_param = lambda x: any(
            [
                ".".join([k, "lora"]) in x or ".".join([k, "magnitude"]) in x
                for k in lora_modules
            ]
        )

        if base_missing:
            for k in base_missing:
                if not is_lora_param(k):
                    raise RuntimeError(f"Missing non-LoRA key {k} from base model dict")
        if lora_missing:
            for k in lora_missing:
                if is_lora_param(k):
                    raise RuntimeError(f"Missing LoRA key {k} from adapter state dict")

    if base_unexpected:
        raise RuntimeError("Unexpected key loading base model")
    if lora_unexpected:
        raise RuntimeError("Unexpected key loading adapter")
