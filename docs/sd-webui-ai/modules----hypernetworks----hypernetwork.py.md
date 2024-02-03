# `stable-diffusion-webui\modules\hypernetworks\hypernetwork.py`

```py
# 导入必要的库
import datetime
import glob
import html
import os
import inspect
from contextlib import closing

# 导入自定义模块
import modules.textual_inversion.dataset
import torch
import tqdm
from einops import rearrange, repeat
from ldm.util import default
from modules import devices, sd_models, shared, sd_samplers, hashes, sd_hijack_checkpoint, errors
from modules.textual_inversion import textual_inversion, logging
from modules.textual_inversion.learn_schedule import LearnRateScheduler
from torch import einsum
from torch.nn.init import normal_, xavier_normal_, xavier_uniform_, kaiming_normal_, kaiming_uniform_, zeros_

# 导入额外的库
from collections import deque
from statistics import stdev, mean

# 创建优化器字典，包含所有 torch.optim 模块中的优化器类
optimizer_dict = {optim_name : cls_obj for optim_name, cls_obj in inspect.getmembers(torch.optim, inspect.isclass) if optim_name != "Optimizer"}

# 定义超网络模块类
class HypernetworkModule(torch.nn.Module):
    # 激活函数字典，包含常见激活函数类
    activation_dict = {
        "linear": torch.nn.Identity,
        "relu": torch.nn.ReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
        "swish": torch.nn.Hardswish,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
    }
    # 更新激活函数字典，包含 torch.nn.modules.activation 模块中的激活函数类
    activation_dict.update({cls_name.lower(): cls_obj for cls_name, cls_obj in inspect.getmembers(torch.nn.modules.activation) if inspect.isclass(cls_obj) and cls_obj.__module__ == 'torch.nn.modules.activation'})

    # 修复旧状态字典的方法
    def fix_old_state_dict(self, state_dict):
        # 定义状态字典中需要更改的键值对
        changes = {
            'linear1.bias': 'linear.0.bias',
            'linear1.weight': 'linear.0.weight',
            'linear2.bias': 'linear.1.bias',
            'linear2.weight': 'linear.1.weight',
        }

        # 遍历需要更改的键值对
        for fr, to in changes.items():
            # 获取旧键对应的值
            x = state_dict.get(fr, None)
            # 如果值不存在，则跳过
            if x is None:
                continue

            # 删除旧键，并将值赋给新键
            del state_dict[fr]
            state_dict[to] = x

    # 前向传播方法
    def forward(self, x):
        # 返回输入加上线性层的输出乘以倍增器（如果不处于训练状态，则乘以1）
        return x + self.linear(x) * (self.multiplier if not self.training else 1)
    # 定义一个方法用于获取模型中可训练参数的结构
    def trainables(self):
        # 初始化一个空列表用于存储层结构
        layer_structure = []
        # 遍历模型中的每一层
        for layer in self.linear:
            # 检查当前层是否为线性层或 LayerNorm 层
            if type(layer) == torch.nn.Linear or type(layer) == torch.nn.LayerNorm:
                # 如果是线性层或 LayerNorm 层，则将其权重和偏置参数添加到层结构列表中
                layer_structure += [layer.weight, layer.bias]
        # 返回包含权重和偏置参数的层结构列表
        return layer_structure
# 解析dropout结构，根据给定的层结构、是否使用dropout和最后一层是否使用dropout来确定每一层的dropout值
def parse_dropout_structure(layer_structure, use_dropout, last_layer_dropout):
    # 如果层结构为空，则默认为[1, 2, 1]
    if layer_structure is None:
        layer_structure = [1, 2, 1]
    # 如果不使用dropout，则返回全零列表
    if not use_dropout:
        return [0] * len(layer_structure)
    # 初始化dropout值列表，第一个元素为0
    dropout_values = [0]
    # 从第二个元素开始，设置为0.3，直到倒数第三个元素
    dropout_values.extend([0.3] * (len(layer_structure) - 3))
    # 如果最后一层使用dropout，则在列表末尾添加0.3，否则添加0
    if last_layer_dropout:
        dropout_values.append(0.3)
    else:
        dropout_values.append(0)
    # 在列表末尾再添加一个0
    dropout_values.append(0)
    # 返回dropout值列表
    return dropout_values

# 定义一个超网络类
class Hypernetwork:
    # 初始化文件名和名称为None
    filename = None
    name = None
    # 初始化神经网络对象，设置各种参数和属性
    def __init__(self, name=None, enable_sizes=None, layer_structure=None, activation_func=None, weight_init=None, add_layer_norm=False, use_dropout=False, activate_output=False, **kwargs):
        # 初始化文件名为空
        self.filename = None
        # 设置网络名称
        self.name = name
        # 初始化网络层字典
        self.layers = {}
        # 初始化步数
        self.step = 0
        # 初始化检查点
        self.sd_checkpoint = None
        # 初始化检查点名称
        self.sd_checkpoint_name = None
        # 设置网络结构
        self.layer_structure = layer_structure
        # 设置激活函数
        self.activation_func = activation_func
        # 设置权重初始化方法
        self.weight_init = weight_init
        # 是否添加层归一化
        self.add_layer_norm = add_layer_norm
        # 是否使用 dropout
        self.use_dropout = use_dropout
        # 是否激活输出
        self.activate_output = activate_output
        # 获取最后一层是否使用 dropout
        self.last_layer_dropout = kwargs.get('last_layer_dropout', True)
        # 获取 dropout 结构
        self.dropout_structure = kwargs.get('dropout_structure', None)
        # 如果没有指定 dropout 结构，则根据网络结构和是否使用 dropout 设置默认 dropout 结构
        if self.dropout_structure is None:
            self.dropout_structure = parse_dropout_structure(self.layer_structure, self.use_dropout, self.last_layer_dropout)
        # 初始化优化器名称
        self.optimizer_name = None
        # 初始化优化器状态字典
        self.optimizer_state_dict = None
        # 初始化可选信息
        self.optional_info = None

        # 遍历每个启用的尺寸
        for size in enable_sizes or []:
            # 为每个尺寸创建两个超网络模块，并存储在网络层字典中
            self.layers[size] = (
                HypernetworkModule(size, None, self.layer_structure, self.activation_func, self.weight_init,
                                   self.add_layer_norm, self.activate_output, dropout_structure=self.dropout_structure),
                HypernetworkModule(size, None, self.layer_structure, self.activation_func, self.weight_init,
                                   self.add_layer_norm, self.activate_output, dropout_structure=self.dropout_structure),
            )
        # 将网络设置为评估模式
        self.eval()

    # 获取网络中所有层的权重参数
    def weights(self):
        # 初始化结果列表
        res = []
        # 遍历每个尺寸对应的两个超网络模块
        for layers in self.layers.values():
            # 遍历每个超网络模块
            for layer in layers:
                # 将每个超网络模块的参数添加到结果列表中
                res += layer.parameters()
        # 返回所有参数列表
        return res
    # 设置神经网络处于训练模式还是评估模式
    def train(self, mode=True):
        # 遍历神经网络的所有层
        for layers in self.layers.values():
            for layer in layers:
                # 设置每一层的训练模式
                layer.train(mode=mode)
                # 设置每一层参数是否需要梯度
                for param in layer.parameters():
                    param.requires_grad = mode

    # 将神经网络移动到指定设备
    def to(self, device):
        # 遍历神经网络的所有层
        for layers in self.layers.values():
            for layer in layers:
                # 将每一层移动到指定设备
                layer.to(device)

        return self

    # 设置神经网络的乘数
    def set_multiplier(self, multiplier):
        # 遍历神经网络的所有层
        for layers in self.layers.values():
            for layer in layers:
                # 设置每一层的乘数
                layer.multiplier = multiplier

        return self

    # 将神经网络设置为评估模式
    def eval(self):
        # 遍历神经网络的所有层
        for layers in self.layers.values():
            for layer in layers:
                # 设置每一层为评估模式
                layer.eval()
                # 设置每一层参数不需要梯度
                for param in layer.parameters():
                    param.requires_grad = False
    # 保存神经网络模型的状态到文件中
    def save(self, filename):
        # 初始化状态字典和优化器状态字典
        state_dict = {}
        optimizer_saved_dict = {}

        # 遍历神经网络的每一层，保存每一层的状态到状态字典中
        for k, v in self.layers.items():
            state_dict[k] = (v[0].state_dict(), v[1].state_dict())

        # 保存神经网络的其他状态信息到状态字典中
        state_dict['step'] = self.step
        state_dict['name'] = self.name
        state_dict['layer_structure'] = self.layer_structure
        state_dict['activation_func'] = self.activation_func
        state_dict['is_layer_norm'] = self.add_layer_norm
        state_dict['weight_initialization'] = self.weight_init
        state_dict['sd_checkpoint'] = self.sd_checkpoint
        state_dict['sd_checkpoint_name'] = self.sd_checkpoint_name
        state_dict['activate_output'] = self.activate_output
        state_dict['use_dropout'] = self.use_dropout
        state_dict['dropout_structure'] = self.dropout_structure
        state_dict['last_layer_dropout'] = (self.dropout_structure[-2] != 0) if self.dropout_structure is not None else self.last_layer_dropout
        state_dict['optional_info'] = self.optional_info if self.optional_info else None

        # 如果存在优化器名称，则保存到优化器状态字典中
        if self.optimizer_name is not None:
            optimizer_saved_dict['optimizer_name'] = self.optimizer_name

        # 保存神经网络状态字典到文件中
        torch.save(state_dict, filename)
        # 如果需要保存优化器状态并且存在优化器状态字典，则保存到文件中
        if shared.opts.save_optimizer_state and self.optimizer_state_dict:
            optimizer_saved_dict['hash'] = self.shorthash()
            optimizer_saved_dict['optimizer_state_dict'] = self.optimizer_state_dict
            torch.save(optimizer_saved_dict, filename + '.optim')

    # 计算文件的哈希值
    def shorthash(self):
        # 使用 SHA256 算法计算文件的哈希值
        sha256 = hashes.sha256(self.filename, f'hypernet/{self.name}')

        # 返回哈希值的前10位作为简短哈希值，如果不存在则返回 None
        return sha256[0:10] if sha256 else None
# 列出指定路径下所有以 .pt 结尾的文件，并将它们的文件名和路径存储在字典中返回
def list_hypernetworks(path):
    res = {}
    # 遍历指定路径下所有以 .pt 结尾的文件
    for filename in sorted(glob.iglob(os.path.join(path, '**/*.pt'), recursive=True), key=str.lower):
        # 获取文件名（不包括扩展名）
        name = os.path.splitext(os.path.basename(filename))[0]
        # 防止一个假设的 "None.pt" 被列出
        if name != "None":
            res[name] = filename
    return res


# 加载指定名称的超网络
def load_hypernetwork(name):
    # 获取指定名称的超网络路径
    path = shared.hypernetworks.get(name, None)

    if path is None:
        return None

    try:
        # 创建超网络对象并加载超网络
        hypernetwork = Hypernetwork()
        hypernetwork.load(path)
        return hypernetwork
    except Exception:
        # 报告加载超网络时的错误
        errors.report(f"Error loading hypernetwork {path}", exc_info=True)
        return None


# 加载多个超网络
def load_hypernetworks(names, multipliers=None):
    already_loaded = {}

    # 检查已加载的超网络是否在指定的名称列表中
    for hypernetwork in shared.loaded_hypernetworks:
        if hypernetwork.name in names:
            already_loaded[hypernetwork.name] = hypernetwork

    # 清空已加载的超网络列表
    shared.loaded_hypernetworks.clear()

    # 遍历指定的名称列表
    for i, name in enumerate(names):
        # 如果超网络已经加载过，则直接使用已加载的超网络
        hypernetwork = already_loaded.get(name, None)
        if hypernetwork is None:
            hypernetwork = load_hypernetwork(name)

        if hypernetwork is None:
            continue

        # 设置超网络的乘数
        hypernetwork.set_multiplier(multipliers[i] if multipliers else 1.0)
        shared.loaded_hypernetworks.append(hypernetwork)


# 应用单个超网络到给定的上下文键和值
def apply_single_hypernetwork(hypernetwork, context_k, context_v, layer=None):
    # 获取超网络的层
    hypernetwork_layers = (hypernetwork.layers if hypernetwork is not None else {}).get(context_k.shape[2], None)

    if hypernetwork_layers is None:
        return context_k, context_v

    if layer is not None:
        # 设置层的超网络键和值
        layer.hyper_k = hypernetwork_layers[0]
        layer.hyper_v = hypernetwork_layers[1]

    # 对上下文键和值应用超网络
    context_k = devices.cond_cast_unet(hypernetwork_layers[0](devices.cond_cast_float(context_k)))
    context_v = devices.cond_cast_unet(hypernetwork_layers[1](devices.cond_cast_float(context_v)))
    return context_k, context_v
# 应用超网络到给定上下文中的每个层，返回更新后的上下文键和值
def apply_hypernetworks(hypernetworks, context, layer=None):
    # 初始化上下文键和值
    context_k = context
    context_v = context
    # 遍历超网络列表，应用单个超网络到上下文键和值
    for hypernetwork in hypernetworks:
        context_k, context_v = apply_single_hypernetwork(hypernetwork, context_k, context_v, layer)

    return context_k, context_v


# 实现注意力机制的前向传播
def attention_CrossAttention_forward(self, x, context=None, mask=None, **kwargs):
    h = self.heads

    # 将输入 x 转换为查询向量 q
    q = self.to_q(x)
    # 如果没有指定上下文，则使用输入 x 作为上下文
    context = default(context, x)

    # 应用超网络到共享的加载的超网络列表中的上下文，得到上下文键和值
    context_k, context_v = apply_hypernetworks(shared.loaded_hypernetworks, context, self)
    # 将上下文键和值转换为键向量 k 和值向量 v
    k = self.to_k(context_k)
    v = self.to_v(context_v)

    # 重排查询、键、值向量的维度
    q, k, v = (rearrange(t, 'b n (h d) -> (b h) n d', h=h) for t in (q, k, v))

    # 计算查询向量和键向量之间的相似度
    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    # 如果存在掩码，则进行掩码操作
    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # 计算注意力权重
    attn = sim.softmax(dim=-1)

    # 计算输出
    out = einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)


# 将条件列表堆叠成张量
def stack_conds(conds):
    if len(conds) == 1:
        return torch.stack(conds)

    # 计算条件列表中最大的 token 数量
    token_count = max([x.shape[0] for x in conds])
    # 对每个条件进行处理，使其 token 数量一致
    for i in range(len(conds)):
        if conds[i].shape[0] != token_count:
            last_vector = conds[i][-1:]
            last_vector_repeated = last_vector.repeat([token_count - conds[i].shape[0], 1])
            conds[i] = torch.vstack([conds[i], last_vector_repeated])

    return torch.stack(conds)


# 计算数据的统计信息
def statistics(data):
    if len(data) < 2:
        std = 0
    else:
        std = stdev(data)
    total_information = f"loss:{mean(data):.3f}" + u"\u00B1" + f"({std/ (len(data) ** 0.5):.3f})"
    recent_data = data[-32:]
    if len(recent_data) < 2:
        std = 0
    else:
        std = stdev(recent_data)
    # 创建包含最近32次损失的信息字符串，包括平均值和标准差
    recent_information = f"recent 32 loss:{mean(recent_data):.3f}" + u"\u00B1" + f"({std / (len(recent_data) ** 0.5):.3f})"
    # 返回总体信息和最近信息
    return total_information, recent_information
# 创建超网络，用于生成神经网络的参数
def create_hypernetwork(name, enable_sizes, overwrite_old, layer_structure=None, activation_func=None, weight_init=None, add_layer_norm=False, use_dropout=False, dropout_structure=None):
    # 从名称中删除非法字符
    name = "".join( x for x in name if (x.isalnum() or x in "._- "))
    # 断言名称不为空
    assert name, "Name cannot be empty!"

    # 构建文件路径
    fn = os.path.join(shared.cmd_opts.hypernetwork_dir, f"{name}.pt")
    # 如果不覆盖旧文件
    if not overwrite_old:
        assert not os.path.exists(fn), f"file {fn} already exists"

    # 将层结构字符串转换为列表
    if type(layer_structure) == str:
        layer_structure = [float(x.strip()) for x in layer_structure.split(",")]

    # 如果使用 dropout 并且有 dropout 结构
    if use_dropout and dropout_structure and type(dropout_structure) == str:
        dropout_structure = [float(x.strip()) for x in dropout_structure.split(",")]
    else:
        dropout_structure = [0] * len(layer_structure)

    # 创建超网络对象
    hypernet = modules.hypernetworks.hypernetwork.Hypernetwork(
        name=name,
        enable_sizes=[int(x) for x in enable_sizes],
        layer_structure=layer_structure,
        activation_func=activation_func,
        weight_init=weight_init,
        add_layer_norm=add_layer_norm,
        use_dropout=use_dropout,
        dropout_structure=dropout_structure
    )
    # 保存超网络对象到文件
    hypernet.save(fn)

    # 重新加载超网络对象
    shared.reload_hypernetworks()


# 训练超网络
def train_hypernetwork(id_task, hypernetwork_name: str, learn_rate: float, batch_size: int, gradient_step: int, data_root: str, log_directory: str, training_width: int, training_height: int, varsize: bool, steps: int, clip_grad_mode: str, clip_grad_value: float, shuffle_tags: bool, tag_drop_out: bool, latent_sampling_method: str, use_weight: bool, create_image_every: int, save_hypernetwork_every: int, template_filename: str, preview_from_txt2img: bool, preview_prompt: str, preview_negative_prompt: str, preview_steps: int, preview_sampler_name: str, preview_cfg_scale: float, preview_seed: int, preview_width: int, preview_height: int):
    # 导入模块
    from modules import images, processing
    # 如果 save_hypernetwork_every 为假值，则设为 0
    save_hypernetwork_every = save_hypernetwork_every or 0
    # 如果 create_image_every 为假值，则设为 0
    create_image_every = create_image_every or 0
    # 从 textual_inversion_templates 字典中获取指定模板文件名对应的模板文件，若不存在则为 None
    template_file = textual_inversion.textual_inversion_templates.get(template_filename, None)
    # 验证训练输入参数的有效性
    textual_inversion.validate_train_inputs(hypernetwork_name, learn_rate, batch_size, gradient_step, data_root, template_file, template_filename, steps, save_hypernetwork_every, create_image_every, log_directory, name="hypernetwork")
    # 获取模板文件的路径
    template_file = template_file.path

    # 获取指定超网络名称对应的路径
    path = shared.hypernetworks.get(hypernetwork_name, None)
    # 创建超网络对象
    hypernetwork = Hypernetwork()
    # 加载超网络
    hypernetwork.load(path)
    # 将加载的超网络存入 loaded_hypernetworks 列表中
    shared.loaded_hypernetworks = [hypernetwork]

    # 设置状态为训练超网络
    shared.state.job = "train-hypernetwork"
    shared.state.textinfo = "Initializing hypernetwork training..."
    shared.state.job_count = steps

    # 从超网络名称中移除括号及其后的内容
    hypernetwork_name = hypernetwork_name.rsplit('(', 1)[0]
    # 构建超网络文件路径
    filename = os.path.join(shared.cmd_opts.hypernetwork_dir, f'{hypernetwork_name}.pt')

    # 构建日志目录
    log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), hypernetwork_name)
    # 根据设置确定是否在训练时卸载模型
    unload = shared.opts.unload_models_when_training

    # 如果 save_hypernetwork_every 大于 0，则创建超网络目录
    if save_hypernetwork_every > 0:
        hypernetwork_dir = os.path.join(log_directory, "hypernetworks")
        os.makedirs(hypernetwork_dir, exist_ok=True)
    else:
        hypernetwork_dir = None

    # 如果 create_image_every 大于 0，则创建图片目录
    if create_image_every > 0:
        images_dir = os.path.join(log_directory, "images")
        os.makedirs(images_dir, exist_ok=True)
    else:
        images_dir = None

    # 选择检查点
    checkpoint = sd_models.select_checkpoint()

    # 获取初始步数
    initial_step = hypernetwork.step or 0
    # 如果初始步数大于等于总步数，则返回已经训练超过指定最大步数的信息
    if initial_step >= steps:
        shared.state.textinfo = "Model has already been trained beyond specified max steps"
        return hypernetwork, filename

    # 创建学习率调度器
    scheduler = LearnRateScheduler(learn_rate, steps, initial_step)

    # 根据 clip_grad_mode 设置梯度裁剪函数
    clip_grad = torch.nn.utils.clip_grad_value_ if clip_grad_mode == "value" else torch.nn.utils.clip_grad_norm_ if clip_grad_mode == "norm" else None
    # 如果需要对梯度进行裁剪，则创建一个裁剪梯度的学习率调度器
    if clip_grad:
        clip_grad_sched = LearnRateScheduler(clip_grad_value, steps, initial_step, verbose=False)

    # 如果启用了 TensorBoard，则设置 TensorBoard 写入器
    if shared.opts.training_enable_tensorboard:
        tensorboard_writer = textual_inversion.tensorboard_setup(log_directory)

    # 数据集加载可能需要一段时间，因此应在此之前进行输入验证和提前返回
    shared.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."

    # 设置是否将数据加载到固定内存中
    pin_memory = shared.opts.pin_memory

    # 创建数据集对象
    ds = modules.textual_inversion.dataset.PersonalizedBase(data_root=data_root, width=training_width, height=training_height, repeats=shared.opts.training_image_repeats_per_epoch, placeholder_token=hypernetwork_name, model=shared.sd_model, cond_model=shared.sd_model.cond_stage_model, device=devices.device, template_file=template_file, include_cond=True, batch_size=batch_size, gradient_step=gradient_step, shuffle_tags=shuffle_tags, tag_drop_out=tag_drop_out, latent_sampling_method=latent_sampling_method, varsize=varsize, use_weight=use_weight)

    # 如果需要将训练设置保存到文本文件中
    if shared.opts.save_training_settings_to_txt:
        # 保存训练设置到文件中
        saved_params = dict(
            model_name=checkpoint.model_name, model_hash=checkpoint.shorthash, num_of_dataset_images=len(ds),
            **{field: getattr(hypernetwork, field) for field in ['layer_structure', 'activation_func', 'weight_init', 'add_layer_norm', 'use_dropout', ]}
        )
        logging.save_settings_to_file(log_directory, {**saved_params, **locals()})

    # 获取数据集的潜在采样方法
    latent_sampling_method = ds.latent_sampling_method

    # 创建数据加载器对象
    dl = modules.textual_inversion.dataset.PersonalizedDataLoader(ds, latent_sampling_method=latent_sampling_method, batch_size=ds.batch_size, pin_memory=pin_memory)

    # 保存旧的并行处理允许状态
    old_parallel_processing_allowed = shared.parallel_processing_allowed

    # 如果需要卸载模型
    if unload:
        # 禁用并行处理
        shared.parallel_processing_allowed = False
        # 将条件模型和第一阶段模型移动到 CPU
        shared.sd_model.cond_stage_model.to(devices.cpu)
        shared.sd_model.first_stage_model.to(devices.cpu)

    # 获取超网络的权重
    weights = hypernetwork.weights()
    # 训练超网络
    hypernetwork.train()

    # 使用保存的 HN 中的优化器，或者可以作为用户界面选项指定
    if hypernetwork.optimizer_name in optimizer_dict:
        optimizer = optimizer_dict[hypernetwork.optimizer_name](params=weights, lr=scheduler.learn_rate)
        optimizer_name = hypernetwork.optimizer_name
    else:
        # 如果指定的优化器类型未定义，则使用默认的 AdamW 优化器
        print(f"Optimizer type {hypernetwork.optimizer_name} is not defined!")
        optimizer = torch.optim.AdamW(params=weights, lr=scheduler.learn_rate)
        optimizer_name = 'AdamW'

    # 如果存在保存的优化器状态字典，则加载该状态
    if hypernetwork.optimizer_state_dict:  # 如果 Optimizer 类型可能与保存的优化器不同，则必须更改此行。
        try:
            optimizer.load_state_dict(hypernetwork.optimizer_state_dict)
        except RuntimeError as e:
            print("Cannot resume from saved optimizer!")
            print(e)

    # 创建梯度缩放器
    scaler = torch.cuda.amp.GradScaler()

    # 获取批处理大小和梯度步数
    batch_size = ds.batch_size
    gradient_step = ds.gradient_step
    # 计算每个 epoch 的步数
    steps_per_epoch = len(ds) // batch_size // gradient_step
    max_steps_per_epoch = len(ds) // batch_size - (len(ds) // batch_size) % gradient_step
    loss_step = 0
    _loss_step = 0 # 内部变量
    # 初始化损失日志队列
    loss_logging = deque(maxlen=len(ds) * 3)  # 这应该是可配置的参数，这里是 3 * epoch(数据集大小)
    
    steps_without_grad = 0

    last_saved_file = "<none>"
    last_saved_image = "<none>"
    forced_filename = "<none>"

    # 创建进度条
    pbar = tqdm.tqdm(total=steps - initial_step)
# 输出损失值、步数、上一个提示、上一个保存的超网络、上一个保存的图像
<p>
Loss: {loss_step:.7f}<br/>
Step: {steps_done}<br/>
Last prompt: {html.escape(batch.cond_text[0])}<br/>
Last saved hypernetwork: {html.escape(last_saved_file)}<br/>
Last saved image: {html.escape(last_saved_image)}<br/>
</p>
"""
# 捕获异常并报告异常信息
except Exception:
    errors.report("Exception in training hypernetwork", exc_info=True)
finally:
    # 设置进度条的 leave 属性为 False，关闭进度条
    pbar.leave = False
    pbar.close()
    # 将超网络设置为评估模式
    hypernetwork.eval()
    # 移除 hijack 检查点
    sd_hijack_checkpoint.remove()

# 设置文件名为超网络目录下的超网络名称加上 '.pt' 后缀
filename = os.path.join(shared.cmd_opts.hypernetwork_dir, f'{hypernetwork_name}.pt')
# 设置超网络的优化器名称为给定的优化器名称
hypernetwork.optimizer_name = optimizer_name
# 如果设置了保存优化器状态的选项，则保存优化器状态字典
if shared.opts.save_optimizer_state:
    hypernetwork.optimizer_state_dict = optimizer.state_dict()
# 保存超网络
save_hypernetwork(hypernetwork, checkpoint, hypernetwork_name, filename)

# 删除优化器
del optimizer
# 在保存后将超网络的优化器状态字典设置为 None，释放内存
hypernetwork.optimizer_state_dict = None
# 将模型转移到指定设备
shared.sd_model.cond_stage_model.to(devices.device)
shared.sd_model.first_stage_model.to(devices.device)
# 恢复并设置并行处理允许状态
shared.parallel_processing_allowed = old_parallel_processing_allowed

# 返回超网络和文件名
return hypernetwork, filename

# 保存超网络
def save_hypernetwork(hypernetwork, checkpoint, hypernetwork_name, filename):
    # 保存旧的超网络名称、检查点、检查点名称
    old_hypernetwork_name = hypernetwork.name
    old_sd_checkpoint = hypernetwork.sd_checkpoint if hasattr(hypernetwork, "sd_checkpoint") else None
    old_sd_checkpoint_name = hypernetwork.sd_checkpoint_name if hasattr(hypernetwork, "sd_checkpoint_name") else None
    try:
        # 设置超网络的检查点为给定检查点的短哈希值
        hypernetwork.sd_checkpoint = checkpoint.shorthash
        # 设置超网络的检查点名称为给定检查点的模型名称
        hypernetwork.sd_checkpoint_name = checkpoint.model_name
        # 设置超网络名称为给定的超网络名称
        hypernetwork.name = hypernetwork_name
        # 保存超网络到指定文件名
        hypernetwork.save(filename)
    except:
        # 如果保存出错，则恢复旧的超网络名称、检查点、检查点名称
        hypernetwork.sd_checkpoint = old_sd_checkpoint
        hypernetwork.sd_checkpoint_name = old_sd_checkpoint_name
        hypernetwork.name = old_hypernetwork_name
        # 抛出异常
        raise
```