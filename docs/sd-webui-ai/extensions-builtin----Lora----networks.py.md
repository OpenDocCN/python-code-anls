# `stable-diffusion-webui\extensions-builtin\Lora\networks.py`

```py
# 导入日志模块
import logging
# 导入操作系统模块
import os
# 导入正则表达式模块
import re

# 导入 lora_patches 模块
import lora_patches
# 导入 network 模块
import network
# 导入 network_lora 模块
import network_lora
# 导入 network_glora 模块
import network_glora
# 导入 network_hada 模块
import network_hada
# 导入 network_ia3 模块
import network_ia3
# 导入 network_lokr 模块
import network_lokr
# 导入 network_full 模块
import network_full
# 导入 network_norm 模块
import network_norm
# 导入 network_oft 模块
import network_oft

# 导入 torch 模块
import torch
# 导入 Union 类型
from typing import Union

# 导入 shared, devices, sd_models, errors, scripts, sd_hijack 模块
from modules import shared, devices, sd_models, errors, scripts, sd_hijack
# 导入 textual_inversion 模块
import modules.textual_inversion.textual_inversion as textual_inversion

# 导入 logger 模块
from lora_logger import logger

# 定义模块类型列表
module_types = [
    network_lora.ModuleTypeLora(),
    network_hada.ModuleTypeHada(),
    network_ia3.ModuleTypeIa3(),
    network_lokr.ModuleTypeLokr(),
    network_full.ModuleTypeFull(),
    network_norm.ModuleTypeNorm(),
    network_glora.ModuleTypeGLora(),
    network_oft.ModuleTypeOFT(),
]

# 编译匹配数字的正则表达式
re_digits = re.compile(r"\d+")
# 编译匹配 x_proj 的正则表达式
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")
# 编译后的正则表达式存储字典
re_compiled = {}

# 后缀转换字典
suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "norm1": "in_layers_0",
        "norm2": "out_layers_0",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    }
}

# 定义函数，将 diffusers 名称转换为 compvis 名称
def convert_diffusers_name_to_compvis(key, is_sd2):
    # 定义匹配函数
    def match(match_list, regex_text):
        # 获取编译后的正则表达式
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex

        # 匹配正则表达式
        r = re.match(regex, key)
        if not r:
            return False

        # 清空匹配列表，将匹配结果存入列表
        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    # 匹配 lora_unet_conv_in
    if match(m, r"lora_unet_conv_in(.*)"):
        return f'diffusion_model_input_blocks_0_0{m[0]}'

    # 匹配 lora_unet_conv_out
    if match(m, r"lora_unet_conv_out(.*)"):
        return f'diffusion_model_out_2{m[0]}'

    # 匹配 lora_unet_time_embedding_linear
    if match(m, r"lora_unet_time_embedding_linear_(\d+)(.*)"):
        return f"diffusion_model_time_embed_{m[0] * 2 - 2}{m[1]}"
    # 如果匹配到 lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+) 格式的字符串
    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        # 获取后缀
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        # 返回对应的字符串
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    # 如果匹配到 lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+) 格式的字符串
    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        # 获取后缀
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        # 返回对应的字符串
        return f"diffusion_model_middle_block_{1 if m[0] == 'attentions' else m[1] * 2}_{suffix}"

    # 如果匹配到 lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+) 格式的字符串
    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        # 获取后缀
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        # 返回对应的字符串
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    # 如果匹配到 lora_unet_down_blocks_(\d+)_downsamplers_0_conv 格式的字符串
    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        # 返回对应的字符串
        return f"diffusion_model_input_blocks_{3 + m[0] * 3}_0_op"

    # 如果匹配到 lora_unet_up_blocks_(\d+)_upsamplers_0_conv 格式的字符串
    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        # 返回对应的字符串
        return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"

    # 如果匹配到 lora_te_text_model_encoder_layers_(\d+)_(.+) 格式的字符串
    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        # 如果是 sd2
        if is_sd2:
            # 根据条件返回不同的字符串
            if 'mlp_fc1' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
            elif 'mlp_fc2' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
            else:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"
        # 如果不是 sd2
        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"
    # 如果匹配到指定格式的字符串
    if match(m, r"lora_te2_text_model_encoder_layers_(\d+)_(.+)"):
        # 如果字符串中包含'mlp_fc1'
        if 'mlp_fc1' in m[1]:
            # 返回替换后的字符串
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
        # 如果字符串中包含'mlp_fc2'
        elif 'mlp_fc2' in m[1]:
            # 返回替换后的字符串
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
        else:
            # 返回替换后的字符串
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"
    
    # 如果未匹配到指定格式的字符串，则返回原始key
    return key
# 为神经网络模块分配网络名称
def assign_network_names_to_compvis_modules(sd_model):
    # 创建一个空字典，用于存储网络层名称和对应的模块
    network_layer_mapping = {}

    # 如果模型是 SDXL 模型
    if shared.sd_model.is_sdxl:
        # 遍历条件器中的嵌入器
        for i, embedder in enumerate(shared.sd_model.conditioner.embedders):
            # 如果嵌入器没有 'wrapped' 属性，则跳过
            if not hasattr(embedder, 'wrapped'):
                continue

            # 遍历嵌入器中的模块
            for name, module in embedder.wrapped.named_modules():
                # 生成网络名称，格式为索引_模块名称（将点替换为下划线）
                network_name = f'{i}_{name.replace(".", "_")}'
                network_layer_mapping[network_name] = module
                module.network_layer_name = network_name
    else:
        # 遍历条件阶段模型中的模块
        for name, module in shared.sd_model.cond_stage_model.wrapped.named_modules():
            # 生成网络名称，将点替换为下划线
            network_name = name.replace(".", "_")
            network_layer_mapping[network_name] = module
            module.network_layer_name = network_name

    # 遍历模型中的模块
    for name, module in shared.sd_model.model.named_modules():
        # 生成网络名称，将点替换为下划线
        network_name = name.replace(".", "_")
        network_layer_mapping[network_name] = module
        module.network_layer_name = network_name

    # 将网络层名称映射存储到模型中
    sd_model.network_layer_mapping = network_layer_mapping


# 加载网络
def load_network(name, network_on_disk):
    # 创建网络对象
    net = network.Network(name, network_on_disk)
    # 获取网络文件的修改时间
    net.mtime = os.path.getmtime(network_on_disk.filename)

    # 读取网络状态字典
    sd = sd_models.read_state_dict(network_on_disk.filename)

    # 如果共享模型中没有 'network_layer_mapping' 属性，则为其分配网络名称
    if not hasattr(shared.sd_model, 'network_layer_mapping'):
        assign_network_names_to_compvis_modules(shared.sd_model)

    # 初始化一个空字典，用于存储匹配失败的键
    keys_failed_to_match = {}
    # 检查是否为 SD2 模型
    is_sd2 = 'model_transformer_resblocks' in shared.sd_model.network_layer_mapping

    # 初始化匹配网络字典和捆绑嵌入
    matched_networks = {}
    bundle_embeddings = {}
    # 遍历匹配网络的字典，其中包含键和权重
    for key, weights in matched_networks.items():
        # 初始化网络模块
        net_module = None
        # 遍历模块类型列表
        for nettype in module_types:
            # 根据网络类型创建模块
            net_module = nettype.create_module(net, weights)
            # 如果成功创建模块，则跳出循环
            if net_module is not None:
                break

        # 如果未成功创建模块，则抛出断言错误
        if net_module is None:
            raise AssertionError(f"Could not find a module type (out of {', '.join([x.__class__.__name__ for x in module_types])}) that would accept those keys: {', '.join(weights.w)}")

        # 将模块添加到网络的模块字典中
        net.modules[key] = net_module

    # 初始化嵌入字典
    embeddings = {}
    # 遍历捆绑嵌入的字典
    for emb_name, data in bundle_embeddings.items():
        # 根据数据创建嵌入对象
        embedding = textual_inversion.create_embedding_from_data(data, emb_name, filename=network_on_disk.filename + "/" + emb_name)
        # 将加载状态设置为 None
        embedding.loaded = None
        # 将嵌入对象添加到嵌入字典中
        embeddings[emb_name] = embedding

    # 将嵌入字典赋值给网络的捆绑嵌入属性
    net.bundle_embeddings = embeddings

    # 如果存在未匹配的键，则记录调试信息
    if keys_failed_to_match:
        logging.debug(f"Network {network_on_disk.filename} didn't match keys: {keys_failed_to_match}")

    # 返回网络对象
    return net
# 从内存中清除网络，直到网络数量小于指定限制或者内存中没有网络为止
def purge_networks_from_memory():
    while len(networks_in_memory) > shared.opts.lora_in_memory_limit and len(networks_in_memory) > 0:
        # 获取内存中第一个网络的名称
        name = next(iter(networks_in_memory))
        # 从内存中移除指定名称的网络
        networks_in_memory.pop(name, None)

    # 执行设备的 Torch 垃圾回收
    devices.torch_gc()


# 加载网络模型
def load_networks(names, te_multipliers=None, unet_multipliers=None, dyn_dims=None):
    # 获取嵌入数据库
    emb_db = sd_hijack.model_hijack.embedding_db
    # 已加载的网络字典
    already_loaded = {}

    # 遍历已加载的网络
    for net in loaded_networks:
        # 如果网络名称在指定的名称列表中
        if net.name in names:
            # 将已加载的网络添加到已加载字典中
            already_loaded[net.name] = net
        # 遍历网络的嵌入字典
        for emb_name, embedding in net.bundle_embeddings.items():
            # 如果嵌入已加载
            if embedding.loaded:
                # 在嵌入数据库中注册嵌入
                emb_db.register_embedding_by_name(None, shared.sd_model, emb_name)

    # 清空已加载网络列表
    loaded_networks.clear()

    # 获取磁盘上的网络模型
    networks_on_disk = [available_network_aliases.get(name, None) for name in names]
    # 如果有网络未找到
    if any(x is None for x in networks_on_disk):
        # 列出可用的网络
        list_available_networks()

        # 重新获取磁盘上的网络模型
        networks_on_disk = [available_network_aliases.get(name, None) for name in names]

    # 加载失败的网络列表
    failed_to_load_networks = []
    # 遍历网络列表和名称列表，同时获取索引和网络对象
    for i, (network_on_disk, name) in enumerate(zip(networks_on_disk, names)):
        # 获取已加载的网络对象，如果不存在则为 None
        net = already_loaded.get(name, None)

        # 如果磁盘上存在网络对象
        if network_on_disk is not None:
            # 如果已加载的网络对象为空
            if net is None:
                # 尝试从内存中获取网络对象
                net = networks_in_memory.get(name)

            # 如果网络对象为空或者磁盘上的网络文件修改时间比内存中的网络对象更新
            if net is None or os.path.getmtime(network_on_disk.filename) > net.mtime:
                try:
                    # 加载网络对象
                    net = load_network(name, network_on_disk)

                    # 从内存中移除旧的网络对象，将新的网络对象存入内存
                    networks_in_memory.pop(name, None)
                    networks_in_memory[name] = net
                except Exception as e:
                    # 显示加载网络对象时的错误信息
                    errors.display(e, f"loading network {network_on_disk.filename}")
                    continue

            # 设置网络对象的名称
            net.mentioned_name = name

            # 读取磁盘上网络对象的哈希值
            network_on_disk.read_hash()

        # 如果网络对象为空
        if net is None:
            # 将加载失败的网络名称添加到列表中
            failed_to_load_networks.append(name)
            # 记录日志，显示找不到网络对象的信息
            logging.info(f"Couldn't find network with name {name}")
            continue

        # 设置网络对象的 TE 倍增器、UNET 倍增器和动态维度
        net.te_multiplier = te_multipliers[i] if te_multipliers else 1.0
        net.unet_multiplier = unet_multipliers[i] if unet_multipliers else 1.0
        net.dyn_dim = dyn_dims[i] if dyn_dims else 1.0
        # 将加载成功的网络对象添加到列表中
        loaded_networks.append(net)

        # 遍历网络对象的捆绑嵌入，检查是否已加载
        for emb_name, embedding in net.bundle_embeddings.items():
            # 如果嵌入未加载且存在于嵌入数据库中
            if embedding.loaded is None and emb_name in emb_db.word_embeddings:
                # 记录警告信息，跳过加载
                logger.warning(
                    f'Skip bundle embedding: "{emb_name}"'
                    ' as it was already loaded from embeddings folder'
                )
                continue

            # 将嵌入标记为未加载
            embedding.loaded = False
            # 如果嵌入的形状符合预期或者预期形状为-1
            if emb_db.expected_shape == -1 or emb_db.expected_shape == embedding.shape:
                # 标记嵌入为已加载，并注册到嵌入数据库中
                embedding.loaded = True
                emb_db.register_embedding(embedding, shared.sd_model)
            else:
                # 将跳过的嵌入添加到跳过的嵌入字典中
                emb_db.skipped_embeddings[name] = embedding

    # 如果存在加载失败的网络
    if failed_to_load_networks:
        # 将未找到的网络名称添加到 hijack 模型的注释中
        sd_hijack.model_hijack.comments.append("Networks not found: " + ", ".join(failed_to_load_networks))
    # 从内存中清除网络数据
    purge_networks_from_memory()
# 从备份中恢复网络层的权重和偏置
def network_restore_weights_from_backup(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention]):
    # 获取网络层的权重备份和偏置备份
    weights_backup = getattr(self, "network_weights_backup", None)
    bias_backup = getattr(self, "network_bias_backup", None)

    # 如果权重备份和偏置备份都为None，则直接返回
    if weights_backup is None and bias_backup is None:
        return

    # 如果权重备份不为None
    if weights_backup is not None:
        # 如果是多头注意力机制网络层
        if isinstance(self, torch.nn.MultiheadAttention):
            # 恢复权重备份到in_proj_weight和out_proj.weight
            self.in_proj_weight.copy_(weights_backup[0])
            self.out_proj.weight.copy_(weights_backup[1])
        else:
            # 恢复权重备份到weight
            self.weight.copy_(weights_backup)

    # 如果偏置备份不为None
    if bias_backup is not None:
        # 如果是多头注意力机制网络层
        if isinstance(self, torch.nn.MultiheadAttention):
            # 恢复偏置备份到out_proj.bias
            self.out_proj.bias.copy_(bias_backup)
        else:
            # 恢复偏置备份到bias
            self.bias.copy_(bias_backup)
    else:
        # 如果是多头注意力机制网络层
        if isinstance(self, torch.nn.MultiheadAttention):
            # 将out_proj.bias设置为None
            self.out_proj.bias = None
        else:
            # 将bias设置为None
            self.bias = None


# 将当前选择的网络应用到torch层的权重
def network_apply_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention]):
    """
    Applies the currently selected set of networks to the weights of torch layer self.
    If weights already have this particular set of networks applied, does nothing.
    If not, restores orginal weights from backup and alters weights according to networks.
    """

    # 获取网络层的名称
    network_layer_name = getattr(self, 'network_layer_name', None)
    # 如果网络层名称为None，则直接返回
    if network_layer_name is None:
        return

    # 获取当前网络层的名称、te_multiplier、unet_multiplier和dyn_dim的元组
    current_names = getattr(self, "network_current_names", ())
    # 获取加载的网络的名称、te_multiplier、unet_multiplier和dyn_dim的元组
    wanted_names = tuple((x.name, x.te_multiplier, x.unet_multiplier, x.dyn_dim) for x in loaded_networks)

    # 获取网络层的权重备份
    weights_backup = getattr(self, "network_weights_backup", None)
    # 如果没有备份权重并且想要的权重不为空
    if weights_backup is None and wanted_names != ():
        # 如果当前权重不为空
        if current_names != ():
            # 抛出运行时错误，表示没有找到备份权重并且当前权重已经改变
            raise RuntimeError("no backup weights found and current weights are not unchanged")

        # 如果是多头注意力模型
        if isinstance(self, torch.nn.MultiheadAttention):
            # 备份输入投影权重和输出投影权重到 CPU
            weights_backup = (self.in_proj_weight.to(devices.cpu, copy=True), self.out_proj.weight.to(devices.cpu, copy=True))
        else:
            # 备份权重到 CPU
            weights_backup = self.weight.to(devices.cpu, copy=True)

        # 将备份的权重保存到网络中
        self.network_weights_backup = weights_backup

    # 获取网络中的偏置备份
    bias_backup = getattr(self, "network_bias_backup", None)
    # 如果偏置备份为空
    if bias_backup is None:
        # 如果是多头注意力模型并且输出投影有偏置
        if isinstance(self, torch.nn.MultiheadAttention) and self.out_proj.bias is not None:
            # 备份输出投影的偏置到 CPU
            bias_backup = self.out_proj.bias.to(devices.cpu, copy=True)
        # 如果存在偏置
        elif getattr(self, 'bias', None) is not None:
            # 备份偏置到 CPU
            bias_backup = self.bias.to(devices.cpu, copy=True)
        else:
            # 否则偏置备份为空
            bias_backup = None
        # 将备份的偏置保存到网络中
        self.network_bias_backup = bias_backup
# 定义一个函数，用于在神经网络前向传播时应用 Lora，以及执行层的前向操作
def network_forward(module, input, original_forward):
    """
    Old way of applying Lora by executing operations during layer's forward.
    Stacking many loras this way results in big performance degradation.
    """

    # 如果加载的网络为空，则直接调用原始的前向传播函数
    if len(loaded_networks) == 0:
        return original_forward(module, input)

    # 对输入进行类型转换
    input = devices.cond_cast_unet(input)

    # 恢复模块的权重备份
    network_restore_weights_from_backup(module)
    # 重置模块的缓存权重
    network_reset_cached_weight(module)

    # 执行原始的前向传播函数
    y = original_forward(module, input)

    # 获取模块的网络层名称
    network_layer_name = getattr(module, 'network_layer_name', None)
    # 遍历加载的网络
    for lora in loaded_networks:
        # 获取当前网络层的模块
        module = lora.modules.get(network_layer_name, None)
        if module is None:
            continue

        # 对输入和输出进行前向传播
        y = module.forward(input, y)

    return y


# 重置模块的缓存权重
def network_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    self.network_current_names = ()
    self.network_weights_backup = None
    self.network_bias_backup = None


# 对线性层进行前向传播
def network_Linear_forward(self, input):
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.Linear_forward)

    # 应用权重
    network_apply_weights(self)

    return originals.Linear_forward(self, input)


# 加载线性层的状态字典
def network_Linear_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.Linear_load_state_dict(self, *args, **kwargs)


# 对卷积层进行前向传播
def network_Conv2d_forward(self, input):
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.Conv2d_forward)

    # 应用权重
    network_apply_weights(self)

    return originals.Conv2d_forward(self, input)


# 加载卷积层的状态字典
def network_Conv2d_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.Conv2d_load_state_dict(self, *args, **kwargs)


# 对 GroupNorm 层进行前向传播
def network_GroupNorm_forward(self, input):
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.GroupNorm_forward)

    # 应用权重
    network_apply_weights(self)

    return originals.GroupNorm_forward(self, input)
# 重置网络的缓存权重
def network_GroupNorm_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    # 调用原始的 GroupNorm_load_state_dict 函数，并返回结果
    return originals.GroupNorm_load_state_dict(self, *args, **kwargs)


# LayerNorm 前向传播函数
def network_LayerNorm_forward(self, input):
    # 如果启用了 lora_functional，则调用 network_forward 函数并返回结果
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.LayerNorm_forward)

    # 应用网络的权重
    network_apply_weights(self)

    # 调用原始的 LayerNorm_forward 函数，并返回结果
    return originals.LayerNorm_forward(self, input)


# 重置网络的缓存权重
def network_LayerNorm_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    # 调用原始的 LayerNorm_load_state_dict 函数，并返回结果
    return originals.LayerNorm_load_state_dict(self, *args, **kwargs)


# MultiheadAttention 前向传播函数
def network_MultiheadAttention_forward(self, *args, **kwargs):
    # 应用网络的权重
    network_apply_weights(self)

    # 调用原始的 MultiheadAttention_forward 函数，并返回结果
    return originals.MultiheadAttention_forward(self, *args, **kwargs)


# 重置网络的缓存权重
def network_MultiheadAttention_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    # 调用原始的 MultiheadAttention_load_state_dict 函数，并返回结果
    return originals.MultiheadAttention_load_state_dict(self, *args, **kwargs)


# 列出可用的网络
def list_available_networks():
    # 清空可用网络列表、网络别名列表、禁止的网络别名列表和网络哈希查找表
    available_networks.clear()
    available_network_aliases.clear()
    forbidden_network_aliases.clear()
    available_network_hash_lookup.clear()
    # 将 "none" 和 "Addams" 添加到禁止的网络别名列表中
    forbidden_network_aliases.update({"none": 1, "Addams": 1})

    # 创建目录 shared.cmd_opts.lora_dir，如果目录不存在则创建
    os.makedirs(shared.cmd_opts.lora_dir, exist_ok=True)

    # 获取 shared.cmd_opts.lora_dir 目录下的文件列表，允许的扩展名为 [".pt", ".ckpt", ".safetensors"]
    candidates = list(shared.walk_files(shared.cmd_opts.lora_dir, allowed_extensions=[".pt", ".ckpt", ".safetensors"]))
    # 获取 shared.cmd_opts.lyco_dir_backcompat 目录下的文件列表，允许的扩展名为 [".pt", ".ckpt", ".safetensors"]
    candidates += list(shared.walk_files(shared.cmd_opts.lyco_dir_backcompat, allowed_extensions=[".pt", ".ckpt", ".safetensors"]))
    # 遍历候选文件列表
    for filename in candidates:
        # 如果文件是目录，则跳过
        if os.path.isdir(filename):
            continue

        # 获取文件名的基本名称（不包含扩展名）
        name = os.path.splitext(os.path.basename(filename))[0]
        
        # 尝试创建 NetworkOnDisk 对象，如果出现 OSError 异常则捕获并处理
        try:
            entry = network.NetworkOnDisk(name, filename)
        except OSError:  # 应该捕获 FileNotFoundError 和 PermissionError 等异常
            # 报告加载网络失败的错误信息
            errors.report(f"Failed to load network {name} from {filename}", exc_info=True)
            continue

        # 将网络对象添加到可用网络字典中
        available_networks[name] = entry

        # 如果网络别名已经存在于可用网络别名中，则将其添加到禁止网络别名字典中
        if entry.alias in available_network_aliases:
            forbidden_network_aliases[entry.alias.lower()] = 1

        # 将网络对象添加到可用网络别名字典中
        available_network_aliases[name] = entry
        available_network_aliases[entry.alias] = entry
# 编译正则表达式，用于匹配网络名称
re_network_name = re.compile(r"(.*)\s*\([0-9a-fA-F]+\)")


# 处理粘贴的信息文本，根据参数生成新的信息文本
def infotext_pasted(infotext, params):
    # 如果其他扩展已激活，则由其他扩展处理字段，无需执行任何操作
    if "AddNet Module 1" in [x[1] for x in scripts.scripts_txt2img.infotext_fields]:
        return

    added = []

    # 遍历参数
    for k in params:
        # 如果参数不以"AddNet Model "开头，则跳过
        if not k.startswith("AddNet Model "):
            continue

        num = k[13:]

        # 如果对应的"AddNet Module "参数不是"LoRA"，则跳过
        if params.get("AddNet Module " + num) != "LoRA":
            continue

        name = params.get("AddNet Model " + num)
        if name is None:
            continue

        # 匹配网络名称中的正则表达式
        m = re_network_name.match(name)
        if m:
            name = m.group(1)

        multiplier = params.get("AddNet Weight A " + num, "1.0")

        # 将生成的字符串添加到列表中
        added.append(f"<lora:{name}:{multiplier}>")

    # 如果有添加的内容，则将其添加到参数的"Prompt"字段中
    if added:
        params["Prompt"] += "\n" + "".join(added)


# 初始化变量
originals: lora_patches.LoraPatches = None
extra_network_lora = None
available_networks = {}
available_network_aliases = {}
loaded_networks = []
loaded_bundle_embeddings = {}
networks_in_memory = {}
available_network_hash_lookup = {}
forbidden_network_aliases = {}

# 调用函数列出可用的网络
list_available_networks()
```