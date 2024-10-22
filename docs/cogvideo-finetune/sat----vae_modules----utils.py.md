# `.\cogvideo-finetune\sat\vae_modules\utils.py`

```py
# 导入所需的模块
import functools  # 引入 functools 模块，提供高阶函数
import importlib  # 引入 importlib 模块，用于动态导入模块
import os  # 引入 os 模块，提供与操作系统交互的功能
from functools import partial  # 从 functools 导入 partial，用于部分函数应用
from inspect import isfunction  # 从 inspect 导入 isfunction，用于检查对象是否为函数

import fsspec  # 导入 fsspec 库，用于文件系统规范
import numpy as np  # 导入 numpy 库并简化为 np，提供数组和数值计算功能
import torch  # 导入 PyTorch 库，提供深度学习功能
from PIL import Image, ImageDraw, ImageFont  # 从 PIL 导入图像处理相关模块
from safetensors.torch import load_file as load_safetensors  # 导入 safetensors 加载函数并重命名
import torch.distributed  # 导入 PyTorch 的分布式模块

# 初始化上下文并行组的变量
_CONTEXT_PARALLEL_GROUP = None  # 用于存储上下文并行组
_CONTEXT_PARALLEL_SIZE = None  # 用于存储上下文并行大小


def is_context_parallel_initialized():
    # 检查上下文并行组是否已初始化
    if _CONTEXT_PARALLEL_GROUP is None:
        return False  # 如果未初始化，返回 False
    else:
        return True  # 否则返回 True


def initialize_context_parallel(context_parallel_size):
    # 初始化上下文并行组
    global _CONTEXT_PARALLEL_GROUP  # 声明全局变量
    global _CONTEXT_PARALLEL_SIZE  # 声明全局变量

    # 确保上下文并行组尚未初始化
    assert _CONTEXT_PARALLEL_GROUP is None, "context parallel group is already initialized"
    _CONTEXT_PARALLEL_SIZE = context_parallel_size  # 设置上下文并行大小

    rank = torch.distributed.get_rank()  # 获取当前进程的排名
    world_size = torch.distributed.get_world_size()  # 获取所有进程的总数

    # 根据上下文并行大小创建新的分组
    for i in range(0, world_size, context_parallel_size):
        ranks = range(i, i + context_parallel_size)  # 获取当前分组的排名
        group = torch.distributed.new_group(ranks)  # 创建新的分组
        if rank in ranks:  # 如果当前排名在分组中
            _CONTEXT_PARALLEL_GROUP = group  # 设置全局上下文并行组
            break  # 退出循环


def get_context_parallel_group():
    # 获取当前上下文并行组
    assert _CONTEXT_PARALLEL_GROUP is not None, "context parallel group is not initialized"  # 确保已初始化

    return _CONTEXT_PARALLEL_GROUP  # 返回上下文并行组


def get_context_parallel_world_size():
    # 获取上下文并行的世界大小
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"  # 确保已初始化

    return _CONTEXT_PARALLEL_SIZE  # 返回上下文并行大小


def get_context_parallel_rank():
    # 获取当前上下文并行组的排名
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"  # 确保已初始化

    rank = torch.distributed.get_rank()  # 获取当前进程的排名
    cp_rank = rank % _CONTEXT_PARALLEL_SIZE  # 计算上下文并行组排名
    return cp_rank  # 返回上下文并行组排名


def get_context_parallel_group_rank():
    # 获取当前上下文并行组的组排名
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"  # 确保已初始化

    rank = torch.distributed.get_rank()  # 获取当前进程的排名
    cp_group_rank = rank // _CONTEXT_PARALLEL_SIZE  # 计算组排名

    return cp_group_rank  # 返回组排名


class SafeConv3d(torch.nn.Conv3d):
    # 自定义 3D 卷积类，继承自 torch.nn.Conv3d
    def forward(self, input):
        # 前向传播方法
        memory_count = torch.prod(torch.tensor(input.shape)).item() * 2 / 1024**3  # 计算输入所需内存
        if memory_count > 2:  # 如果内存需求超过 2 GB
            kernel_size = self.kernel_size[0]  # 获取卷积核大小
            part_num = int(memory_count / 2) + 1  # 计算分块数量
            input_chunks = torch.chunk(input, part_num, dim=2)  # 将输入按维度 2 分块，格式为 NCTHW
            if kernel_size > 1:  # 如果卷积核大于 1
                input_chunks = [input_chunks[0]] + [  # 将第一块加入结果
                    torch.cat((input_chunks[i - 1][:, :, -kernel_size + 1 :], input_chunks[i]), dim=2)  # 拼接相邻块
                    for i in range(1, len(input_chunks))
                ]

            output_chunks = []  # 初始化输出块列表
            for input_chunk in input_chunks:  # 遍历每个输入块
                output_chunks.append(super(SafeConv3d, self).forward(input_chunk))  # 调用父类的 forward 方法
            output = torch.cat(output_chunks, dim=2)  # 将输出块拼接
            return output  # 返回拼接后的输出
        else:
            return super(SafeConv3d, self).forward(input)  # 否则直接调用父类的 forward 方法


def disabled_train(self, mode=True):
    # 重写模型的 train 方法，确保训练/评估模式不再改变
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self  # 返回当前对象
# 定义一个函数，通过元组字符串返回其第一个元素
def get_string_from_tuple(s):
    # 尝试执行下面的代码块，处理可能出现的异常
    try:
        # 检查字符串是否以括号开头和结尾
        if s[0] == "(" and s[-1] == ")":
            # 将字符串转换为元组
            t = eval(s)
            # 检查 t 的类型是否为元组
            if type(t) == tuple:
                # 返回元组的第一个元素
                return t[0]
            else:
                pass
    # 捕获所有异常，防止程序崩溃
    except:
        pass
    # 如果不满足条件，返回原始字符串
    return s


# 定义一个函数，检查一个整数是否是 2 的幂
def is_power_of_two(n):
    """
    chat.openai.com/chat
    如果 n 是 2 的幂，返回 True；否则返回 False。

    函数 is_power_of_two 接受一个整数 n 作为输入，如果 n 是 2 的幂，则返回 True；否则返回 False。
    该函数首先检查 n 是否小于或等于 0。如果 n 小于或等于 0，则它不能是 2 的幂，因此函数返回 False。
    如果 n 大于 0，函数通过使用 n 和 n-1 之间的按位与操作来检查 n 是否是 2 的幂。
    如果 n 是 2 的幂，它的二进制表示中只有一个位被设置为 1。当我们从 2 的幂中减去 1 时，所有右侧的位变为 1，该位本身变为 0。
    因此，当我们对 n 和 n-1 进行按位与操作时，如果 n 是 2 的幂，结果为 0，否则为非零值。
    因此，如果按位与操作的结果为 0，则 n 是 2 的幂，函数返回 True；否则返回 False。
    """
    # 检查 n 是否小于等于 0
    if n <= 0:
        # 如果小于等于 0，返回 False
        return False
    # 返回 n 和 n-1 进行按位与操作的结果是否为 0
    return (n & (n - 1)) == 0


# 定义一个函数，实现自动类型转换的功能
def autocast(f, enabled=True):
    # 定义一个内部函数，执行自动类型转换
    def do_autocast(*args, **kwargs):
        # 使用自动类型转换上下文管理器
        with torch.cuda.amp.autocast(
            enabled=enabled,  # 是否启用自动类型转换
            dtype=torch.get_autocast_gpu_dtype(),  # 获取 GPU 的自动类型转换数据类型
            cache_enabled=torch.is_autocast_cache_enabled(),  # 检查是否启用缓存
        ):
            # 执行传入的函数 f，并返回结果
            return f(*args, **kwargs)

    # 返回内部函数
    return do_autocast


# 定义一个函数，从配置中加载部分对象
def load_partial_from_config(config):
    # 使用部分应用函数返回目标对象，带有指定的参数
    return partial(get_obj_from_str(config["target"]), **config.get("params", dict()))


# 定义一个函数，将文本以图像形式记录
def log_txt_as_img(wh, xc, size=10):
    # wh 是一个包含 (宽度, 高度) 的元组
    # xc 是要绘制的字幕列表
    b = len(xc)  # 获取字幕列表的长度
    txts = list()  # 初始化一个空列表，用于存储图像数据
    # 遍历每个字幕
    for bi in range(b):
        # 创建一个新的白色背景图像
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)  # 创建可用于绘图的对象
        font = ImageFont.truetype("data/DejaVuSans.ttf", size=size)  # 加载指定字体
        nc = int(40 * (wh[0] / 256))  # 根据图像宽度计算每行的字符数
        # 检查当前字幕是否为列表
        if isinstance(xc[bi], list):
            text_seq = xc[bi][0]  # 获取列表中的第一个元素
        else:
            text_seq = xc[bi]  # 直接使用字幕

        # 将文本序列分行，每行不超过 nc 个字符
        lines = "\n".join(text_seq[start : start + nc] for start in range(0, len(text_seq), nc))

        try:
            # 在图像上绘制文本
            draw.text((0, 0), lines, fill="black", font=font)
        # 捕获文本编码错误
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")  # 输出错误信息

        # 将图像数据转换为 NumPy 数组并进行归一化
        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)  # 将处理后的图像数据添加到列表
    # 将列表中的图像数据堆叠成一个 NumPy 数组
    txts = np.stack(txts)
    # 将 NumPy 数组转换为 PyTorch 张量
    txts = torch.tensor(txts)
    # 返回最终的张量
    return txts


# 定义一个部分类，允许使用部分应用创建新类
def partialclass(cls, *args, **kwargs):
    # 定义一个新类，继承自原始类
    class NewCls(cls):
        # 使用部分应用替换原始类的初始化方法
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    # 返回新类
    return NewCls
# 根据给定路径生成绝对路径
def make_path_absolute(path):
    # 解析路径并获取文件系统和路径
    fs, p = fsspec.core.url_to_fs(path)
    # 如果协议是文件，则返回绝对路径
    if fs.protocol == "file":
        return os.path.abspath(p)
    # 否则返回原始路径
    return path


# 判断输入是否为一个特定形状的张量
def ismap(x):
    # 检查 x 是否为张量
    if not isinstance(x, torch.Tensor):
        return False
    # 检查张量的维度和通道数
    return (len(x.shape) == 4) and (x.shape[1] > 3)


# 判断输入是否为图像张量
def isimage(x):
    # 检查 x 是否为张量
    if not isinstance(x, torch.Tensor):
        return False
    # 检查张量的维度和通道数
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


# 判断输入是否为热图张量
def isheatmap(x):
    # 检查 x 是否为张量
    if not isinstance(x, torch.Tensor):
        return False
    # 检查张量的维度
    return x.ndim == 2


# 判断输入是否为邻居张量
def isneighbors(x):
    # 检查 x 是否为张量
    if not isinstance(x, torch.Tensor):
        return False
    # 检查张量的维度和通道数
    return x.ndim == 5 and (x.shape[2] == 3 or x.shape[2] == 1)


# 检查输入是否存在
def exists(x):
    # 返回 x 是否不为 None
    return x is not None


# 将张量的维度扩展到与另一个张量相同
def expand_dims_like(x, y):
    # 在 x 的维度与 y 不同的情况下循环扩展
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    # 返回扩展后的张量
    return x


# 返回给定值或默认值
def default(val, d):
    # 如果 val 存在，返回 val
    if exists(val):
        return val
    # 如果 d 是函数，则调用并返回其结果，否则返回 d
    return d() if isfunction(d) else d


# 计算张量的扁平化均值
def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    对所有非批处理维度进行均值计算。
    """
    # 计算并返回指定维度的均值
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


# 统计模型参数数量
def count_params(model, verbose=False):
    # 计算模型所有参数的总数
    total_params = sum(p.numel() for p in model.parameters())
    # 如果 verbose 为真，打印参数数量
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    # 返回总参数数量
    return total_params


# 根据配置实例化对象
def instantiate_from_config(config):
    # 检查配置中是否包含 'target' 键
    if not "target" in config:
        # 返回 None，如果配置是特定字符串
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        # 否则抛出异常
        raise KeyError("Expected key `target` to instantiate.")
    # 从配置中获取目标对象并返回实例化结果
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


# 从字符串获取对象
def get_obj_from_str(string, reload=False, invalidate_cache=True):
    # 将字符串分解为模块和类
    module, cls = string.rsplit(".", 1)
    # 如果需要，失效缓存
    if invalidate_cache:
        importlib.invalidate_caches()
    # 如果需要重新加载模块
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    # 返回模块中的类对象
    return getattr(importlib.import_module(module, package=None), cls)


# 在张量末尾追加一个零
def append_zero(x):
    # 将 x 与一个新零张量连接
    return torch.cat([x, x.new_zeros([1])])


# 将张量扩展到目标维度
def append_dims(x, target_dims):
    """将维度附加到张量末尾，直到它具有 target_dims 维度。"""
    # 计算需要附加的维度数量
    dims_to_append = target_dims - x.ndim
    # 如果目标维度小于当前维度，抛出异常
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    # 返回附加维度后的张量
    return x[(...,) + (None,) * dims_to_append]


# 从配置加载模型
def load_model_from_config(config, ckpt, verbose=True, freeze=True):
    # 打印加载模型的检查点信息
    print(f"Loading model from {ckpt}")
    # 如果检查点是以 'ckpt' 结尾，加载相应的数据
    if ckpt.endswith("ckpt"):
        pl_sd = torch.load(ckpt, map_location="cpu")
        # 如果包含全局步骤信息，打印
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
    # 如果检查点是以 'safetensors' 结尾，加载相应的数据
    elif ckpt.endswith("safetensors"):
        sd = load_safetensors(ckpt)
    # 否则抛出未实现异常
    else:
        raise NotImplementedError
    # 根据配置文件实例化模型
        model = instantiate_from_config(config.model)
    
        # 加载状态字典（权重）到模型中，返回缺失和意外的键
        m, u = model.load_state_dict(sd, strict=False)
    
        # 如果有缺失的键且需要详细输出，则打印缺失的键
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        # 如果有意外的键且需要详细输出，则打印意外的键
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
    
        # 如果需要冻结模型参数，禁止其更新
        if freeze:
            for param in model.parameters():
                # 设置参数的 requires_grad 属性为 False，停止梯度计算
                param.requires_grad = False
    
        # 将模型设置为评估模式，禁用训练时的特性（如 Dropout）
        model.eval()
        # 返回实例化后的模型
        return model
# 获取配置文件的路径
def get_configs_path() -> str:
    """
    获取 `configs` 目录。
    对于工作副本，这是存储库根目录下的目录，
    而对于已安装副本，则在 `sgm` 包中（见 pyproject.toml）。
    """
    # 获取当前文件所在目录的路径
    this_dir = os.path.dirname(__file__)
    # 定义可能的配置目录路径
    candidates = (
        os.path.join(this_dir, "configs"),  # 当前目录下的 configs
        os.path.join(this_dir, "..", "configs"),  # 上级目录下的 configs
    )
    # 遍历候选目录
    for candidate in candidates:
        # 将候选路径转换为绝对路径
        candidate = os.path.abspath(candidate)
        # 检查该路径是否为目录
        if os.path.isdir(candidate):
            # 如果是目录，返回该路径
            return candidate
    # 如果没有找到任何目录，抛出文件未找到异常
    raise FileNotFoundError(f"Could not find SGM configs in {candidates}")

# 获取嵌套属性的函数
def get_nested_attribute(obj, attribute_path, depth=None, return_key=False):
    """
    将返回递归获取属性调用的结果。
    例如：
        a.b.c
        = getattr(getattr(a, "b"), "c")
        = get_nested_attribute(a, "b.c")
    如果属性调用的任何部分是整数 x，并且当前 obj 为 a，将
    尝试首先调用 a[x] 而不是 a.x。
    """
    # 将属性路径以 "." 分割为列表
    attributes = attribute_path.split(".")
    # 如果设置了深度，截取属性列表
    if depth is not None and depth > 0:
        attributes = attributes[:depth]
    # 确保至少选择了一个属性
    assert len(attributes) > 0, "At least one attribute should be selected"
    # 初始化当前属性为传入对象
    current_attribute = obj
    current_key = None
    # 遍历每一层的属性
    for level, attribute in enumerate(attributes):
        # 生成当前的属性路径字符串
        current_key = ".".join(attributes[: level + 1])
        try:
            # 尝试将属性转换为整数
            id_ = int(attribute)
            # 如果成功，将当前属性设置为索引访问的结果
            current_attribute = current_attribute[id_]
        except ValueError:
            # 否则使用 getattr 获取属性
            current_attribute = getattr(current_attribute, attribute)

    # 返回当前属性和当前键，或仅返回当前属性
    return (current_attribute, current_key) if return_key else current_attribute

# 定义检查点函数
def checkpoint(func, inputs, params, flag):
    """
    在不缓存中间激活的情况下评估函数，从而减少内存，
    代价是反向传播时增加额外计算。
    :param func: 要评估的函数。
    :param inputs: 要传递给 `func` 的参数序列。
    :param params: `func` 依赖的参数序列，但不作为参数显式传递。
    :param flag: 如果为 False，则禁用梯度检查点。
    """
    # 如果启用标志
    if flag:
        # 将输入和参数组合成元组
        args = tuple(inputs) + tuple(params)
        # 使用 CheckpointFunction 应用函数并返回结果
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        # 否则直接调用函数并返回结果
        return func(*inputs)

# 定义检查点功能的类
class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    # 前向计算方法
    def forward(ctx, run_function, length, *args):
        # 保存要运行的函数
        ctx.run_function = run_function
        # 保存输入张量
        ctx.input_tensors = list(args[:length])
        # 保存输入参数
        ctx.input_params = list(args[length:])
        # 保存 GPU 自动混合精度的设置
        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
        # 在无梯度上下文中执行函数
        with torch.no_grad():
            # 获取输出张量
            output_tensors = ctx.run_function(*ctx.input_tensors)
        # 返回输出张量
        return output_tensors

    @staticmethod
    # 定义反向传播函数，接收上下文和输出梯度作为参数
    def backward(ctx, *output_grads):
        # 对输入张量进行detach操作，并设置requires_grad为True，以便计算梯度
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        # 启用梯度计算，并在自动混合精度模式下执行代码
        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # 修复一个bug：第一个操作会修改Tensor存储，这对已detach的Tensors不允许
            # 创建输入张量的浅拷贝，确保原始张量不被修改
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            # 调用运行函数，传入浅拷贝的输入张量，获得输出张量
            output_tensors = ctx.run_function(*shallow_copies)
        # 计算输入张量和参数的梯度，output_grads作为输出梯度
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        # 删除上下文中的输入张量以释放内存
        del ctx.input_tensors
        # 删除上下文中的输入参数以释放内存
        del ctx.input_params
        # 删除输出张量以释放内存
        del output_tensors
        # 返回None和输入梯度的元组
        return (None, None) + input_grads
```