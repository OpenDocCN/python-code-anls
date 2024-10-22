# `.\cogvideo-finetune\sat\sgm\util.py`

```py
# 导入 functools 模块以支持高阶函数功能
import functools
# 导入 importlib 模块以动态导入模块
import importlib
# 导入 os 模块以支持操作系统相关功能
import os
# 从 functools 导入 partial 函数以便于函数部分应用
from functools import partial
# 从 inspect 导入 isfunction 函数用于检查对象是否为函数
from inspect import isfunction

# 导入 fsspec 库以支持文件系统规范
import fsspec
# 导入 numpy 库用于科学计算
import numpy as np
# 导入 torch 库以支持深度学习功能
import torch
# 从 PIL 导入 Image、ImageDraw 和 ImageFont 以支持图像处理
from PIL import Image, ImageDraw, ImageFont
# 从 safetensors.torch 导入 load_file 函数以加载安全张量
from safetensors.torch import load_file as load_safetensors
# 导入 torch.distributed 模块以支持分布式训练
import torch.distributed

# 定义全局变量以存储并行组的上下文
_CONTEXT_PARALLEL_GROUP = None
# 定义全局变量以存储并行组的大小
_CONTEXT_PARALLEL_SIZE = None


# 定义函数以检查上下文并行是否已初始化
def is_context_parallel_initialized():
    # 检查上下文并行组是否为 None
    if _CONTEXT_PARALLEL_GROUP is None:
        return False
    else:
        return True


# 定义函数以设置上下文并行组和大小
def set_context_parallel_group(size, group):
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_SIZE
    # 设置上下文并行组
    _CONTEXT_PARALLEL_GROUP = group
    # 设置上下文并行大小
    _CONTEXT_PARALLEL_SIZE = size


# 定义函数以初始化上下文并行
def initialize_context_parallel(context_parallel_size):
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_SIZE

    # 断言上下文并行组未被初始化
    assert _CONTEXT_PARALLEL_GROUP is None, "context parallel group is already initialized"
    # 设置上下文并行大小
    _CONTEXT_PARALLEL_SIZE = context_parallel_size

    # 获取当前进程的 rank
    rank = torch.distributed.get_rank()
    # 获取全局进程的数量
    world_size = torch.distributed.get_world_size()

    # 按上下文并行大小遍历所有进程以创建新的并行组
    for i in range(0, world_size, context_parallel_size):
        ranks = range(i, i + context_parallel_size)
        # 创建新的分组
        group = torch.distributed.new_group(ranks)
        # 如果当前 rank 在创建的 ranks 中，则设置上下文并行组
        if rank in ranks:
            _CONTEXT_PARALLEL_GROUP = group
            break


# 定义函数以获取当前上下文并行组
def get_context_parallel_group():
    # 断言上下文并行组已初始化
    assert _CONTEXT_PARALLEL_GROUP is not None, "context parallel group is not initialized"

    return _CONTEXT_PARALLEL_GROUP


# 定义函数以获取当前上下文并行的世界大小
def get_context_parallel_world_size():
    # 断言上下文并行大小已初始化
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"

    return _CONTEXT_PARALLEL_SIZE


# 定义函数以获取当前上下文并行的 rank
def get_context_parallel_rank():
    # 断言上下文并行大小已初始化
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"

    # 获取当前进程的 rank
    rank = torch.distributed.get_rank()
    # 计算当前上下文并行的 rank
    cp_rank = rank % _CONTEXT_PARALLEL_SIZE
    return cp_rank


# 定义函数以获取当前上下文并行组的 rank
def get_context_parallel_group_rank():
    # 断言上下文并行大小已初始化
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"

    # 获取当前进程的 rank
    rank = torch.distributed.get_rank()
    # 计算当前上下文并行组的 rank
    cp_group_rank = rank // _CONTEXT_PARALLEL_SIZE

    return cp_group_rank


# 定义 SafeConv3d 类，继承自 torch.nn.Conv3d
class SafeConv3d(torch.nn.Conv3d):
    # 定义前向传播函数，接收输入数据
    def forward(self, input):
        # 计算输入数据的内存占用（以 GB 为单位），乘以 2 是因为需要考虑反向传播的内存
        memory_count = torch.prod(torch.tensor(input.shape)).item() * 2 / 1024**3
        # 如果内存占用超过 2GB，则进行内存优化处理
        if memory_count > 2:
            # kernel_size 取自实例属性，表示卷积核的大小
            kernel_size = self.kernel_size[0]
            # 计算需要将输入分成的部分数量，以控制内存占用
            part_num = int(memory_count / 2) + 1
            # 将输入数据按时间维度进行分块处理
            input_chunks = torch.chunk(input, part_num, dim=2)  # NCTHW
            # 如果卷积核大小大于 1，则需要处理相邻块的拼接
            if kernel_size > 1:
                # 将第一个块保留，后续块与前一个块拼接
                input_chunks = [input_chunks[0]] + [
                    torch.cat((input_chunks[i - 1][:, :, -kernel_size + 1 :], input_chunks[i]), dim=2)
                    for i in range(1, len(input_chunks))
                ]

            # 初始化输出块列表
            output_chunks = []
            # 对每个输入块进行前向传播，并将结果存储到输出块列表中
            for input_chunk in input_chunks:
                output_chunks.append(super(SafeConv3d, self).forward(input_chunk))
            # 将所有输出块在时间维度上拼接成最终输出
            output = torch.cat(output_chunks, dim=2)
            # 返回拼接后的输出
            return output
        else:
            # 如果内存占用不超过 2GB，直接调用父类的前向传播
            return super(SafeConv3d, self).forward(input)
# 禁用训练模式，确保模型的训练/评估模式不会再更改
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self  # 返回当前对象，保持状态不变


# 从元组字符串中提取字符串
def get_string_from_tuple(s):
    try:
        # 检查字符串是否以括号开始和结束
        if s[0] == "(" and s[-1] == ")":
            # 将字符串转换为元组
            t = eval(s)
            # 检查 t 的类型是否为元组
            if type(t) == tuple:
                return t[0]  # 返回元组的第一个元素
            else:
                pass  # 如果不是元组则不做处理
    except:
        pass  # 捕获异常，防止程序崩溃
    return s  # 返回原始字符串


# 检查一个数是否是 2 的幂
def is_power_of_two(n):
    """
    chat.openai.com/chat
    Return True if n is a power of 2, otherwise return False.

    The function is_power_of_two takes an integer n as input and returns True if n is a power of 2, otherwise it returns False.
    The function works by first checking if n is less than or equal to 0. If n is less than or equal to 0, it can't be a power of 2, so the function returns False.
    If n is greater than 0, the function checks whether n is a power of 2 by using a bitwise AND operation between n and n-1. If n is a power of 2, then it will have only one bit set to 1 in its binary representation. When we subtract 1 from a power of 2, all the bits to the right of that bit become 1, and the bit itself becomes 0. So, when we perform a bitwise AND between n and n-1, we get 0 if n is a power of 2, and a non-zero value otherwise.
    Thus, if the result of the bitwise AND operation is 0, then n is a power of 2 and the function returns True. Otherwise, the function returns False.

    """
    if n <= 0:
        return False  # 如果 n 小于或等于 0，返回 False
    return (n & (n - 1)) == 0  # 返回 n 是否是 2 的幂


# 自动混合精度处理函数
def autocast(f, enabled=True):
    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(
            enabled=enabled,  # 启用或禁用自动混合精度
            dtype=torch.get_autocast_gpu_dtype(),  # 获取 GPU 的自动混合精度数据类型
            cache_enabled=torch.is_autocast_cache_enabled(),  # 检查缓存是否启用
        ):
            return f(*args, **kwargs)  # 调用原始函数并返回结果

    return do_autocast  # 返回自动混合精度处理的封装函数


# 从配置加载部分对象
def load_partial_from_config(config):
    return partial(get_obj_from_str(config["target"]), **config.get("params", dict()))  # 返回部分应用的对象


# 将文本作为图像进行记录
def log_txt_as_img(wh, xc, size=10):
    # wh 是一个包含 (宽度, 高度) 的元组
    # xc 是要绘制的标题列表
    b = len(xc)  # 获取标题列表的长度
    txts = list()  # 初始化文本图像列表
    for bi in range(b):  # 遍历每个标题
        txt = Image.new("RGB", wh, color="white")  # 创建一个白色背景的图像
        draw = ImageDraw.Draw(txt)  # 创建可绘制对象
        font = ImageFont.truetype("data/DejaVuSans.ttf", size=size)  # 加载指定字体
        nc = int(40 * (wh[0] / 256))  # 计算每行最多字符数
        if isinstance(xc[bi], list):  # 如果标题是列表
            text_seq = xc[bi][0]  # 获取第一个元素
        else:
            text_seq = xc[bi]  # 否则直接使用标题
        lines = "\n".join(text_seq[start : start + nc] for start in range(0, len(text_seq), nc))  # 将文本分行

        try:
            draw.text((0, 0), lines, fill="black", font=font)  # 绘制文本
        except UnicodeEncodeError:  # 如果编码错误
            print("Cant encode string for logging. Skipping.")  # 打印错误信息并跳过

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0  # 转换图像为数组并归一化
        txts.append(txt)  # 添加到列表中
    txts = np.stack(txts)  # 堆叠所有文本图像
    txts = torch.tensor(txts)  # 转换为 PyTorch 张量
    # 返回包含文本数据的列表或字符串
        return txts
# 定义一个部分类，用于传递参数到初始化方法
def partialclass(cls, *args, **kwargs):
    # 创建一个新类，继承自原始类
    class NewCls(cls):
        # 使用 functools.partialmethod 将原始类的初始化方法与参数绑定
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    # 返回新创建的类
    return NewCls


# 将给定路径转换为绝对路径
def make_path_absolute(path):
    # 解析路径并获取文件系统和路径
    fs, p = fsspec.core.url_to_fs(path)
    # 如果文件系统协议为文件，则返回绝对路径
    if fs.protocol == "file":
        return os.path.abspath(p)
    # 否则返回原始路径
    return path


# 判断输入是否为地图类型的张量
def ismap(x):
    # 如果输入不是张量，则返回 False
    if not isinstance(x, torch.Tensor):
        return False
    # 检查张量是否为四维且第二维大于3
    return (len(x.shape) == 4) and (x.shape[1] > 3)


# 判断输入是否为图像类型的张量
def isimage(x):
    # 如果输入不是张量，则返回 False
    if not isinstance(x, torch.Tensor):
        return False
    # 检查张量是否为四维且第二维为3或1
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


# 判断输入是否为热图类型的张量
def isheatmap(x):
    # 如果输入不是张量，则返回 False
    if not isinstance(x, torch.Tensor):
        return False
    # 检查张量是否为二维
    return x.ndim == 2


# 判断输入是否为邻接类型的张量
def isneighbors(x):
    # 如果输入不是张量，则返回 False
    if not isinstance(x, torch.Tensor):
        return False
    # 检查张量是否为五维且第三维为3或1
    return x.ndim == 5 and (x.shape[2] == 3 or x.shape[2] == 1)


# 检查输入是否存在
def exists(x):
    # 判断输入是否不为 None
    return x is not None


# 扩展张量的维度，使其与目标张量的维度相同
def expand_dims_like(x, y):
    # 当 x 的维度不等于 y 的维度时，持续扩展 x 的维度
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    # 返回扩展后的 x
    return x


# 返回存在的值或默认值
def default(val, d):
    # 如果 val 存在，则返回 val
    if exists(val):
        return val
    # 返回默认值，如果 d 是函数则调用它
    return d() if isfunction(d) else d


# 计算张量的平坦均值
def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    对所有非批次维度进行均值计算。
    """
    # 计算张量在所有非批次维度上的均值
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


# 计算模型的参数总数
def count_params(model, verbose=False):
    # 计算模型所有参数的总数量
    total_params = sum(p.numel() for p in model.parameters())
    # 如果 verbose 为真，则打印参数数量
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    # 返回参数总数
    return total_params


# 根据配置实例化对象
def instantiate_from_config(config, **extra_kwargs):
    # 检查配置中是否包含目标键
    if not "target" in config:
        # 如果配置为 "__is_first_stage__"，返回 None
        if config == "__is_first_stage__":
            return None
        # 如果配置为 "__is_unconditional__"，返回 None
        elif config == "__is_unconditional__":
            return None
        # 抛出缺少目标键的异常
        raise KeyError("Expected key `target` to instantiate.")
    # 根据目标字符串实例化对象，并传递参数
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **extra_kwargs)


# 从字符串获取对象
def get_obj_from_str(string, reload=False, invalidate_cache=True):
    # 分割模块和类名
    module, cls = string.rsplit(".", 1)
    # 如果需要，失效缓存
    if invalidate_cache:
        importlib.invalidate_caches()
    # 如果需要重载模块，导入并重载
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    # 返回模块中的类对象
    return getattr(importlib.import_module(module, package=None), cls)


# 在张量末尾添加零
def append_zero(x):
    # 将零张量与输入张量拼接
    return torch.cat([x, x.new_zeros([1])])


# 添加维度到张量以达到目标维度
def append_dims(x, target_dims):
    """将维度添加到张量末尾，直到其具有目标维度。"""
    # 计算需要添加的维度数量
    dims_to_append = target_dims - x.ndim
    # 如果输入维度大于目标维度，抛出异常
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    # 返回添加了维度的张量
    return x[(...,) + (None,) * dims_to_append]


# 从配置加载模型
def load_model_from_config(config, ckpt, verbose=True, freeze=True):
    # 打印加载模型的检查点路径
    print(f"Loading model from {ckpt}")
    # 检查检查点文件名是否以 "ckpt" 结尾
    if ckpt.endswith("ckpt"):
        # 从检查点文件加载状态字典到 CPU
        pl_sd = torch.load(ckpt, map_location="cpu")
        # 如果状态字典中包含 "global_step" 键
        if "global_step" in pl_sd:
            # 打印全局步数
            print(f"Global Step: {pl_sd['global_step']}")
        # 提取状态字典
        sd = pl_sd["state_dict"]
    # 检查检查点文件名是否以 "safetensors" 结尾
    elif ckpt.endswith("safetensors"):
        # 从 safetensors 文件加载状态字典
        sd = load_safetensors(ckpt)
    # 如果文件名不符合上述格式，则抛出未实现的错误
    else:
        raise NotImplementedError

    # 根据配置实例化模型
    model = instantiate_from_config(config.model)

    # 加载模型状态字典，允许非严格匹配
    m, u = model.load_state_dict(sd, strict=False)

    # 如果有缺失的键且详细模式开启
    if len(m) > 0 and verbose:
        # 打印缺失的键
        print("missing keys:")
        print(m)
    # 如果有意外的键且详细模式开启
    if len(u) > 0 and verbose:
        # 打印意外的键
        print("unexpected keys:")
        print(u)

    # 如果冻结标志为真
    if freeze:
        # 遍历模型参数
        for param in model.parameters():
            # 设置参数为不需要梯度
            param.requires_grad = False

    # 将模型设置为评估模式
    model.eval()
    # 返回已加载的模型
    return model
# 定义一个获取配置路径的函数，返回字符串类型
def get_configs_path() -> str:
    # 函数文档说明：获取 `configs` 目录的位置
    this_dir = os.path.dirname(__file__)  # 获取当前文件所在目录的路径
    # 创建候选路径，包含当前目录下的 configs 目录和上级目录下的 configs 目录
    candidates = (
        os.path.join(this_dir, "configs"),
        os.path.join(this_dir, "..", "configs"),
    )
    # 遍历每个候选路径
    for candidate in candidates:
        candidate = os.path.abspath(candidate)  # 将候选路径转换为绝对路径
        if os.path.isdir(candidate):  # 检查该路径是否为一个目录
            return candidate  # 如果是，返回该路径
    # 如果没有找到有效的 configs 目录，抛出文件未找到错误
    raise FileNotFoundError(f"Could not find SGM configs in {candidates}")


# 定义一个获取嵌套属性的函数
def get_nested_attribute(obj, attribute_path, depth=None, return_key=False):
    # 函数文档说明：递归获取对象的嵌套属性
    attributes = attribute_path.split(".")  # 根据 '.' 分割属性路径
    if depth is not None and depth > 0:  # 如果指定深度且大于零
        attributes = attributes[:depth]  # 限制属性列表到指定深度
    assert len(attributes) > 0, "At least one attribute should be selected"  # 确保至少有一个属性
    current_attribute = obj  # 初始化当前属性为对象
    current_key = None  # 初始化当前键
    # 遍历每个属性
    for level, attribute in enumerate(attributes):
        current_key = ".".join(attributes[: level + 1])  # 生成当前键的字符串
        try:
            id_ = int(attribute)  # 尝试将属性转换为整数
            current_attribute = current_attribute[id_]  # 使用索引访问属性
        except ValueError:  # 如果转换失败
            current_attribute = getattr(current_attribute, attribute)  # 使用 getattr 获取属性

    # 返回当前属性和当前键的元组，或者只返回当前属性
    return (current_attribute, current_key) if return_key else current_attribute


# 从 math 模块导入平方根函数
from math import sqrt


# 定义一个 SeededNoise 类
class SeededNoise:
    # 初始化方法，接受种子和权重
    def __init__(self, seeds, weights):
        self.seeds = seeds  # 存储种子
        self.weights = weights  # 存储权重
        weight_square_sum = 0  # 初始化权重平方和
        # 遍历每个权重
        for weight in weights:
            weight_square_sum += weight**2  # 计算权重的平方和
        self.weight_square_sum_sqrt = sqrt(weight_square_sum)  # 计算权重平方和的平方根
        self.cnt = 0  # 初始化计数器

    # 定义可调用方法
    def __call__(self, x):
        self.cnt += 1  # 计数器加一
        randn_combined = torch.zeros_like(x)  # 创建与 x 同形状的零张量
        # 遍历种子和权重
        for seed, weight in zip(self.seeds, self.weights):
            randn = np.random.RandomState(seed + self.cnt).randn(*x.shape)  # 生成正态分布随机数
            randn = torch.from_numpy(randn, dtype=x.dtype, device=x.device)  # 将随机数转换为张量
            randn_combined += randn * weight  # 将加权随机数累加到组合中
        randn_combined /= self.weight_square_sum_sqrt  # 将组合随机数归一化
        return randn_combined  # 返回最终的组合随机数
```