# `.\transformers\integrations\awq.py`

```py
# 导入所需模块
"AWQ (Activation aware Weight Quantization) integration file"
from ..activations import ACT2FN  # 导入激活函数映射
from ..modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ..utils import is_auto_awq_available, is_torch_available  # 导入工具函数和判断是否可用的函数
from ..utils.quantization_config import AwqBackendPackingMethod, AwqConfig, AWQLinearVersion  # 导入 AWQ 相关的配置类

# 检查是否导入了 PyTorch
if is_torch_available():
    import torch  # 导入 PyTorch
    import torch.nn as nn  # 导入 PyTorch 的神经网络模块

# 定义 AWQ_FUSED_MAPPINGS 字典，用于指定不同模型的层名称
AWQ_FUSED_MAPPINGS = {
    "mistral": {  # Mistral 模型的层名称映射
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],  # 注意力层的参数名称
        "mlp": ["gate_proj", "up_proj", "down_proj"],  # MLP 层的参数名称
        "layernorm": ["input_layernorm", "post_attention_layernorm", "norm"],  # LayerNorm 层的参数名称
        "use_alibi": False,  # 是否使用 Alibi
    },
    "mixtral": {  # MixTral 模型的层名称映射
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],  # 注意力层的参数名称
        "mlp": ["w1", "w3", "w2"],  # MLP 层的参数名称
        "layernorm": ["input_layernorm", "post_attention_layernorm", "norm"],  # LayerNorm 层的参数名称
        "use_alibi": False,  # 是否使用 Alibi
        "rope_theta": 1000000.0,  # rope_theta 参数值
    },
    "llama": {  # Llama 模型的层名称映射
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],  # 注意力层的参数名称
        "mlp": ["gate_proj", "up_proj", "down_proj"],  # MLP 层的参数名称
        "layernorm": ["input_layernorm", "post_attention_layernorm", "norm"],  # LayerNorm 层的参数名称
        "use_alibi": False,  # 是否使用 Alibi
    },
    "llava": {  # Llava 模型的层名称映射
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],  # 注意力层的参数名称
        "mlp": ["gate_proj", "up_proj", "down_proj"],  # MLP 层的参数名称
        "layernorm": ["input_layernorm", "post_attention_layernorm", "norm"],  # LayerNorm 层的参数名称
        "use_alibi": False,  # 是否使用 Alibi
    },
}

# 定义用于替换模型中 Linear 层为 AWQ 量化层的函数
def replace_with_awq_linear(
    model,
    modules_to_not_convert=None,
    quantization_config=None,
    current_key_name=None,
    has_been_replaced=False,
) -> bool:
    """
    Public method that recursively replaces the Linear layers of the given model with AWQ quantized layers.
    `accelerate` is needed to use this method. Returns the converted model and a boolean that indicates if the
    conversion has been successfull or not.

    During the module replacement, we also infer the backend to use through the `quantization_config` object.
    """
    Args:
        model (`torch.nn.Module`):
            要转换的模型，可以是任何 `torch.nn.Module` 实例。
        quantization_config (`AwqConfig`):
            包含量化参数的量化配置对象。
        modules_to_not_convert (`list`, *optional*):
            不转换的模块列表。如果模块名在列表中（例如 `lm_head`），则不会转换。
        current_key_name (`list`, *optional*):
            包含当前键名的列表。这用于递归，用户不应传递此参数。
        has_been_replaced (`bool`, *optional*):
            表示转换是否成功的布尔值。这用于递归，用户不应传递此参数。
    """
    # 如果没有传入不转换的模块列表，则默认为空列表
    if modules_to_not_convert is None:
        modules_to_not_convert = []

    # 获取量化配置的后端
    backend = quantization_config.backend

    # 检查是否自动 AWQ 可用
    if not is_auto_awq_available():
        # 如果不可用，则抛出错误提示用户安装或查看安装指南
        raise ValueError(
            "AWQ（`autoawq` 或 `llmawq`）不可用。请使用 `pip install autoawq` 安装或查看 https://github.com/mit-han-lab/llm-awq 中的安装指南"
        )

    # 根据后端导入相应的模块
    if backend == AwqBackendPackingMethod.AUTOAWQ:
        from awq.modules.linear import WQLinear_GEMM, WQLinear_GEMV
    elif backend == AwqBackendPackingMethod.LLMAWQ:
        from awq.quantize.qmodule import WQLinear

    # 根据后端和版本确定目标类
    if backend == AwqBackendPackingMethod.AUTOAWQ:
        target_cls = WQLinear_GEMM if quantization_config.version == AWQLinearVersion.GEMM else WQLinear_GEMV
    else:
        target_cls = WQLinear
    # 遍历模型的所有子模块，获取模块名和对应的模块对象
    for name, module in model.named_children():
        # 如果当前键名为 None，则将其初始化为空列表
        if current_key_name is None:
            current_key_name = []
        # 将当前模块名添加到键名列表中
        current_key_name.append(name)
    
        # 检查当前模块是否为线性层，并且其名称不在不转换的模块列表中
        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # 检查当前键名是否不在 `modules_to_not_convert` 中的任何键名中
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                # 获取当前线性层的输入特征数和输出特征数
                in_features = module.in_features
                out_features = module.out_features
    
                # 用新的量化线性层替换原始线性层
                model._modules[name] = target_cls(
                    w_bit=quantization_config.bits,  # 权重位数
                    group_size=quantization_config.group_size,  # 分组大小
                    in_features=in_features,  # 输入特征数
                    out_features=out_features,  # 输出特征数
                    bias=module.bias is not None,  # 是否包含偏置
                    dev=module.weight.device,  # 设备类型
                )
                # 标记模型已经被替换
                has_been_replaced = True
    
                # 强制将新的模块设置为不需要梯度，以避免意外错误
                model._modules[name].requires_grad_(False)
        # 如果当前模块还包含子模块
        if len(list(module.children())) > 0:
            # 递归替换子模块的线性层
            _, has_been_replaced = replace_with_awq_linear(
                module,
                modules_to_not_convert=modules_to_not_convert,  # 不转换的模块列表
                current_key_name=current_key_name,  # 当前键名列表
                quantization_config=quantization_config,  # 量化配置
                has_been_replaced=has_been_replaced,  # 是否已替换标志
            )
        # 移除用于递归的最后一个键名
        current_key_name.pop(-1)
    # 返回替换后的模型和替换标志
    return model, has_been_replaced
def get_modules_to_fuse(model, quantization_config):
    """
    Returns the fusing mapping given the quantization config and the model

    Args:
        model (`~PreTrainedModel`):
            The model to fuse - note this model should have been converted into AWQ format beforehand.
        quantization_config (`~transformers.quantization_config.AWQConfig`):
            The quantization configuration to use.
    """
    # 检查传入的 model 是否为 PreTrainedModel 的实例，如果不是则抛出数值错误
    if not isinstance(model, PreTrainedModel):
        raise ValueError(f"The model should be an instance of `PreTrainedModel`, got {model.__class__.__name__}")

    # 如果 quantization_config.modules_to_fuse 不为 None，则使用其作为当前融合映射
    if quantization_config.modules_to_fuse is not None:
        current_fused_mapping = quantization_config.modules_to_fuse
        current_fused_mapping["max_seq_len"] = quantization_config.fuse_max_seq_len
    # 如果 model.config.model_type 在 AWQ_FUSED_MAPPINGS 中，则使用对应的融合映射
    elif model.config.model_type in AWQ_FUSED_MAPPINGS:
        current_fused_mapping = AWQ_FUSED_MAPPINGS[model.config.model_type]

        # 处理多模态模型的情况（例如 Llava）
        if not hasattr(model.config, "text_config"):
            config = model.config
        else:
            config = model.config.text_config

        # 处理 hidden_size、num_attention_heads、num_key_value_heads
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)

        # 填充 current_fused_mapping 的预期值
        current_fused_mapping["hidden_size"] = hidden_size
        current_fused_mapping["num_attention_heads"] = num_attention_heads
        current_fused_mapping["num_key_value_heads"] = num_key_value_heads
        current_fused_mapping["max_seq_len"] = quantization_config.fuse_max_seq_len
    else:
        # 如果找不到融合映射，则抛出数值错误
        raise ValueError(
            "Fusing mapping not found either on the quantization config or the supported `AWQ_FUSED_MAPPINGS`. Please pass a `fused_mapping` argument"
            " in the `quantization_config` or raise an issue on transformers https://github.com/huggingface/transformers to add its support."
        )
    return current_fused_mapping


def fuse_awq_modules(model, quantization_config):
    """
    Optionally fuse some modules in the model to speedup inference.

    Args:
        model (`~PreTrainedModel`):
            The model to fuse - note this model should have been converted into AWQ format beforehand.
        quantization_config (`dict`):
            The quantization configuration to use.
    """
    # 将 quantization_config 转换为 AwqConfig 对象，以便获取 backend 等字段
    awq_config = AwqConfig.from_dict(quantization_config)
    backend = awq_config.backend

    # 获取需要融合的模块
    modules_to_fuse = get_modules_to_fuse(model, awq_config)
    # 从 awq_config 中获取不需要转换的模块列表，如果未指定则为 None
    modules_to_not_convert = getattr(awq_config, "modules_to_not_convert", None)

    # 如果后端为 AUTOAWQ，则导入相关模块
    if backend == AwqBackendPackingMethod.AUTOAWQ:
        # 导入自动 AWQ 后端的融合注意力模块
        from awq.modules.fused.attn import QuantAttentionFused
        # 导入自动 AWQ 后端的融合 MLP 模块
        from awq.modules.fused.mlp import QuantFusedMLP
        # 导入自动 AWQ 后端的融合规范化模块
        from awq.modules.fused.norm import FasterTransformerRMSNorm
    else:
        # 如果后端不是 AUTOAWQ，则抛出数值错误
        raise ValueError("Fusing is only supported for the AutoAWQ backend")

    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 如果存在不需要转换的模块列表
        if modules_to_not_convert is not None:
            # 检查当前模块是否在不需要转换的模块列表中，如果是则跳过当前模块
            if any(module_name_to_not_convert in name for module_name_to_not_convert in modules_to_not_convert):
                continue

        # 替换层归一化模块
        _fuse_awq_layernorm(modules_to_fuse["layernorm"], module, FasterTransformerRMSNorm)

        # 替换MLP层
        _fuse_awq_mlp(model, name, modules_to_fuse["mlp"], module, QuantFusedMLP)

        # 替换注意力层
        _fuse_awq_attention_layers(model, module, modules_to_fuse, name, QuantAttentionFused)
    # 返回修改后的模型
    return model
def _fuse_awq_layernorm(fuse_module_names, module, target_cls):
    """
    Fuse the LayerNorm layers into a target class using autoawq

    Args:
        fuse_module_names (`List[str]`):
            The list of module names to fuse
        module (`nn.Module`):
            The pytorch parent module that has layernorm modules to fuse
        target_cls (`~autoawq.FasterTransformerRMSNorm`):
            The `FasterTransformerRMSNorm` class as it only supports that class
            for now.
    """
    # 遍历需要融合的模块名列表
    for module_name in fuse_module_names:
        # 检查父模块是否包含指定模块名
        if hasattr(module, module_name):
            # 获取旧模块
            old_module = getattr(module, module_name)
            # 创建新的目标类实例，并替换原有模块
            module._modules[module_name] = target_cls(
                old_module.weight,
                old_module.variance_epsilon,
            ).to(old_module.weight.device)
            # 删除旧模块
            del old_module


def _fuse_awq_mlp(model, current_module_name, fuse_module_names, module, target_cls):
    """
    Fuse the MLP layers into a target class using autoawq

    Args:
        model (`~PreTrainedModel`):
            The input pretrained model
        current_module_name (`str`):
            The current submodule name
        fuse_module_names (`List[str]`):
            The list of module names to fuse. For the MLP layers it has to be an array
            of length 3 that consists of the 3 MLP layers in the order (gate (dense layer post-attention) / up / down layers)
        module (`nn.Module`):
            The pytorch parent module that has layernorm modules to fuse
        target_cls (`~autoawq.QuantFusedMLP`):
            The `QuantFusedMLP` class as it only supports that class
            for now.
    """
    # 如果没有需要融合的模块名，则直接返回
    if len(fuse_module_names) == 0:
        return

    # 检查父模块是否包含指定的融合模块名
    if hasattr(module, fuse_module_names[0]):
        # 获取门控、上采样和下采样模块
        gate_proj = getattr(module, fuse_module_names[0])
        up_proj = getattr(module, fuse_module_names[1])
        down_proj = getattr(module, fuse_module_names[2])

        # 记录门控模块的设备信息
        previous_device = gate_proj.qweight.device

        # 处理模型具有`text_config`属性的情况
        hidden_act = (
            model.config.hidden_act
            if not hasattr(model.config, "text_config")
            else model.config.text_config.hidden_act
        )
        # 获取激活函数
        activation_fn = ACT2FN[hidden_act]
        # 创建新的融合模块实例
        new_module = target_cls(gate_proj, down_proj, up_proj, activation_fn)

        # 获取父模块和子模块名称
        parent_name, child_name = current_module_name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        # 替换子模块为新的融合模块
        setattr(parent, child_name, new_module.to(previous_device))

        # 删除门控、上采样和下采样模块
        del gate_proj, up_proj, down_proj


def _fuse_awq_attention_layers(model, module, modules_to_fuse, current_module_name, target_cls):
    """
    Fuse the Attention layers into a target class using autoawq
    """
    Args:
        model (`~PreTrainedModel`):
            预训练模型的输入
        module (`nn.Module`):
            包含要融合的 layernorm 模块的 PyTorch 父模块
        modules_to_fuse (`List[str]`):
            融合模块的映射。字典必须包含一个名为 `attention` 的字段，其中包含正确顺序的注意力模块名称：q、k、v、o 层
        current_module_name (`str`):
            当前子模块名称
        target_cls (`~autoawq.QuantAttentionFused`):
            作为目标的 `QuantAttentionFused` 类，因为现在仅支持该类。
    """
    from awq.modules.linear import WQLinear_GEMM, WQLinear_GEMV

    # 如果要融合的注意力模块列表为空，则返回
    if len(modules_to_fuse["attention"]) == 0:
        return
    # 检查模块是否具有指定的注意力模块属性
    if hasattr(module, modules_to_fuse["attention"][0]):
        # 首先，将 QKV 层组合在一起
        q_proj = getattr(module, modules_to_fuse["attention"][0])

        # 检查 q_proj 类型，确定线性层的类和连接维度
        if isinstance(q_proj, WQLinear_GEMV):
            linear_target_cls = WQLinear_GEMV
            cat_dim = 0
        elif isinstance(q_proj, WQLinear_GEMM):
            linear_target_cls = WQLinear_GEMM
            cat_dim = 1
        else:
            raise ValueError(f"Unsupported q_proj type: {type(q_proj)}")

        # 保存之前的设备信息
        previous_device = q_proj.qweight.device

        # 获取注意力机制的其他参数
        k_proj = getattr(module, modules_to_fuse["attention"][1])
        v_proj = getattr(module, modules_to_fuse["attention"][2])
        o_proj = getattr(module, modules_to_fuse["attention"][3])

        # 合并偏置项
        bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0) if q_proj.bias is not None else None

        # 创建合并后的 QKV 层
        qkv_layer = linear_target_cls(
            q_proj.w_bit,
            q_proj.group_size,
            q_proj.in_features,
            q_proj.out_features + k_proj.out_features + v_proj.out_features,
            q_proj.bias is not None,
            next(iter(module.state_dict().values())).device,
        )

        # 合并权重和量化零点
        qkv_layer.qweight = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=cat_dim)
        qkv_layer.qzeros = torch.cat([q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=cat_dim)
        qkv_layer.scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=cat_dim)

        # 如果是 WQLinear_GEMV 类型，则设置 split_k_iters
        if isinstance(qkv_layer, WQLinear_GEMV):
            qkv_layer.split_k_iters = q_proj.split_k_iters

        # 设置合并后的偏置项
        qkv_layer.bias = bias

        # 创建融合后的注意力层
        fused_attention_layer = target_cls(
            modules_to_fuse["hidden_size"],
            modules_to_fuse["num_attention_heads"],
            modules_to_fuse["num_key_value_heads"],
            qkv_layer,
            o_proj,
            previous_device,
            modules_to_fuse["max_seq_len"],
            use_alibi=modules_to_fuse["use_alibi"],
            # autoawq 中的默认值设定为 10000.0
            rope_theta=modules_to_fuse.get("rope_theta", 10000.0),
        )

        # 标记融合后的注意力层为 HF Transformers
        fused_attention_layer.is_hf_transformers = True

        # 分离当前模块的父模块和子模块名
        parent_name, child_name = current_module_name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        # 将融合后的注意力层设置为父模块的子模块
        setattr(parent, child_name, fused_attention_layer.to(previous_device))

        # 删除临时变量以释放内存
        del q_proj, k_proj, v_proj, o_proj
```