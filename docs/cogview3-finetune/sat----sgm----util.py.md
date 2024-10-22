# `.\cogview3-finetune\sat\sgm\util.py`

```py
import functools  # 导入 functools 模块以使用高阶函数
import importlib  # 导入 importlib 模块以动态导入模块
import os  # 导入 os 模块以进行操作系统相关的功能
from functools import partial  # 从 functools 导入 partial，用于创建偏函数
from inspect import isfunction  # 从 inspect 导入 isfunction，以检查对象是否为函数

import fsspec  # 导入 fsspec 模块，用于文件系统规范化和操作
import numpy as np  # 导入 numpy 并重命名为 np，进行数值计算
import torch  # 导入 PyTorch 库进行深度学习
from PIL import Image, ImageDraw, ImageFont  # 从 PIL 导入图像处理相关的类
from safetensors.torch import load_file as load_safetensors  # 从 safetensors 导入 load_file，并重命名

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""  # 文档字符串，说明该函数用于覆盖模型的 train 方法
    return self  # 直接返回当前对象，忽略训练模式的变化

def get_string_from_tuple(s):
    try:
        # Check if the string starts and ends with parentheses  # 检查字符串是否以括号开头和结尾
        if s[0] == "(" and s[-1] == ")":
            # Convert the string to a tuple  # 将字符串转换为元组
            t = eval(s)  # 使用 eval 函数评估字符串
            # Check if the type of t is tuple  # 检查 t 的类型是否为元组
            if type(t) == tuple:
                return t[0]  # 返回元组的第一个元素
            else:
                pass  # 如果不是元组，则不做任何操作
    except:  # 捕获所有异常
        pass  # 如果发生异常，则不做任何操作
    return s  # 如果条件不满足，则返回原始字符串

def is_power_of_two(n):
    """
    chat.openai.com/chat
    Return True if n is a power of 2, otherwise return False.  # 文档字符串，说明该函数的作用
    ...
    """
    if n <= 0:  # 如果 n 小于或等于 0
        return False  # 返回 False，因为负数和零不是 2 的幂
    return (n & (n - 1)) == 0  # 使用位运算检查 n 是否为 2 的幂

def autocast(f, enabled=True):
    def do_autocast(*args, **kwargs):  # 定义内部函数，接受任意位置和关键字参数
        with torch.cuda.amp.autocast(  # 使用自动混合精度上下文
            enabled=enabled,  # 根据 enabled 参数决定是否启用
            dtype=torch.get_autocast_gpu_dtype(),  # 获取自动混合精度的 GPU 数据类型
            cache_enabled=torch.is_autocast_cache_enabled(),  # 检查缓存是否启用
        ):
            return f(*args, **kwargs)  # 调用原函数并返回结果

    return do_autocast  # 返回内部函数

def load_partial_from_config(config):
    return partial(get_obj_from_str(config["target"]), **config.get("params", dict()))  # 从配置中加载部分参数并返回偏函数

def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)  # wh 是一个包含宽度和高度的元组
    # xc a list of captions to plot  # xc 是一个包含要绘制的标题的列表
    b = len(xc)  # 获取标题列表的长度
    txts = list()  # 初始化一个空列表，用于存储文本
    # 遍历给定的 bi 范围，执行 b 次循环
    for bi in range(b):
        # 创建一个白色背景的 RGB 图像，尺寸为 wh
        txt = Image.new("RGB", wh, color="white")
        # 为图像创建可绘制对象
        draw = ImageDraw.Draw(txt)
        # 加载指定字体和大小
        font = ImageFont.truetype("data/DejaVuSans.ttf", size=size)
        # 计算每行可以容纳的字符数
        nc = int(40 * (wh[0] / 256))
        # 如果 xc[bi] 是列表，取第一个元素，否则直接使用 xc[bi]
        if isinstance(xc[bi], list):
            text_seq = xc[bi][0]
        else:
            text_seq = xc[bi]
        # 将文本序列分割成多行
        lines = "\n".join(
            text_seq[start : start + nc] for start in range(0, len(text_seq), nc)
        )
    
        try:
            # 在图像上绘制文本
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            # 捕捉编码错误并输出跳过提示
            print("Cant encode string for logging. Skipping.")
    
        # 将图像转换为 NumPy 数组并进行标准化处理
        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        # 将处理后的图像添加到列表
        txts.append(txt)
    # 将所有图像堆叠成一个数组
    txts = np.stack(txts)
    # 转换为 PyTorch 张量
    txts = torch.tensor(txts)
    # 返回最终的张量
    return txts
# 定义一个部分类，用于包装原始类并接受附加参数
def partialclass(cls, *args, **kwargs):
    # 创建一个新类，该类继承自给定的类，并重定义其初始化方法
    class NewCls(cls):
        # 使用 functools.partialmethod 将原始初始化方法与给定参数结合
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    # 返回新创建的类
    return NewCls


# 将给定路径转换为绝对路径
def make_path_absolute(path):
    # 使用 fsspec 库将路径转换为文件系统和路径
    fs, p = fsspec.core.url_to_fs(path)
    # 如果协议是文件，则返回绝对路径
    if fs.protocol == "file":
        return os.path.abspath(p)
    # 否则，返回原路径
    return path


# 检查输入是否为四维张量，且通道数大于3
def ismap(x):
    # 如果输入不是 torch.Tensor 类型，返回 False
    if not isinstance(x, torch.Tensor):
        return False
    # 返回是否为四维且通道数大于3
    return (len(x.shape) == 4) and (x.shape[1] > 3)


# 检查输入是否为图像张量，通道数为3或1
def isimage(x):
    # 如果输入不是 torch.Tensor 类型，返回 False
    if not isinstance(x, torch.Tensor):
        return False
    # 返回是否为四维且通道数为3或1
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


# 检查输入是否为二维热图张量
def isheatmap(x):
    # 如果输入不是 torch.Tensor 类型，返回 False
    if not isinstance(x, torch.Tensor):
        return False
    # 返回是否为二维张量
    return x.ndim == 2


# 检查输入是否为五维邻接张量，且通道数为3或1
def isneighbors(x):
    # 如果输入不是 torch.Tensor 类型，返回 False
    if not isinstance(x, torch.Tensor):
        return False
    # 返回是否为五维且第三维通道数为3或1
    return x.ndim == 5 and (x.shape[2] == 3 or x.shape[2] == 1)


# 检查输入是否存在
def exists(x):
    # 返回输入是否不为 None
    return x is not None


# 扩展张量的维度，直到其维度与目标张量相同
def expand_dims_like(x, y):
    # 当 x 的维度不等于 y 时，逐步扩展 x 的最后一维
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    # 返回扩展后的张量
    return x


# 返回给定值或默认值，若默认值是函数则调用它
def default(val, d):
    # 如果 val 存在，则返回它
    if exists(val):
        return val
    # 返回默认值，调用函数或直接返回
    return d() if isfunction(d) else d


# 计算张量的平均值，跨越所有非批次维度
def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    计算所有非批次维度的平均值。
    """
    # 返回在指定维度上计算的平均值
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


# 计算模型的参数总数，支持可选的详细输出
def count_params(model, verbose=False):
    # 计算模型所有参数的总数量
    total_params = sum(p.numel() for p in model.parameters())
    # 如果需要详细信息，则打印模型参数数量
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    # 返回参数总数
    return total_params


# 根据配置实例化对象，并接受额外的关键字参数
def instantiate_from_config(config, **extra_kwargs):
    # 检查配置中是否包含 'target' 键
    if not "target" in config:
        # 返回 None，表示无条件
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        # 如果没有 'target' 键，则抛出错误
        raise KeyError("Expected key `target` to instantiate.")
    # 返回从字符串获取的对象，传递参数
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **extra_kwargs)


# 从字符串获取模块和类，并可选地重新加载模块
def get_obj_from_str(string, reload=False, invalidate_cache=True):
    # 分割字符串，提取模块名和类名
    module, cls = string.rsplit(".", 1)
    # 如果需要无效化缓存，则执行
    if invalidate_cache:
        importlib.invalidate_caches()
    # 如果需要重新加载模块，则执行
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    # 返回指定模块中的类
    return getattr(importlib.import_module(module, package=None), cls)


# 在张量末尾追加一个零
def append_zero(x):
    # 将一个零张量与输入张量连接
    return torch.cat([x, x.new_zeros([1])])


# 将张量的维度扩展到目标维度
def append_dims(x, target_dims):
    """将维度追加到张量的末尾，直到达到目标维度。"""
    # 计算需要追加的维度数量
    dims_to_append = target_dims - x.ndim
    # 如果目标维度小于输入维度，则抛出错误
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    # 返回扩展后的张量
    return x[(...,) + (None,) * dims_to_append]


# 从配置加载模型，并支持冻结和详细输出
def load_model_from_config(config, ckpt, verbose=True, freeze=True):
    # 打印加载模型的信息
    print(f"Loading model from {ckpt}")
    # 检查检查点文件是否以 "ckpt" 结尾
        if ckpt.endswith("ckpt"):
            # 从检查点加载模型状态字典到 CPU
            pl_sd = torch.load(ckpt, map_location="cpu")
            # 如果状态字典中包含全局步数，打印其值
            if "global_step" in pl_sd:
                print(f"Global Step: {pl_sd['global_step']}")
            # 获取模型的状态字典
            sd = pl_sd["state_dict"]
        # 检查点文件以 "safetensors" 结尾
        elif ckpt.endswith("safetensors"):
            # 从 safetensors 文件加载模型状态字典
            sd = load_safetensors(ckpt)
        # 如果文件名不匹配，抛出未实现错误
        else:
            raise NotImplementedError
    
        # 从配置中实例化模型
        model = instantiate_from_config(config.model)
    
        # 加载模型状态字典，允许非严格匹配
        m, u = model.load_state_dict(sd, strict=False)
    
        # 如果有缺失的键且详细模式开启，打印缺失的键
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        # 如果有意外的键且详细模式开启，打印意外的键
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
    
        # 如果冻结参数为真，禁用模型参数的梯度计算
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
    
        # 将模型设置为评估模式
        model.eval()
        # 返回已配置的模型
        return model
# 获取 `configs` 目录的路径
def get_configs_path() -> str:
    # 文档字符串，说明函数的作用
    """
    Get the `configs` directory.
    For a working copy, this is the one in the root of the repository,
    but for an installed copy, it's in the `sgm` package (see pyproject.toml).
    """
    # 获取当前文件所在目录的路径
    this_dir = os.path.dirname(__file__)
    # 定义候选路径，可能的 `configs` 目录位置
    candidates = (
        os.path.join(this_dir, "configs"),  # 当前目录下的 configs
        os.path.join(this_dir, "..", "configs"),  # 上一级目录下的 configs
    )
    # 遍历每一个候选路径
    for candidate in candidates:
        # 将候选路径转换为绝对路径
        candidate = os.path.abspath(candidate)
        # 检查该路径是否为目录
        if os.path.isdir(candidate):
            # 如果是目录，则返回该路径
            return candidate
    # 如果未找到任何有效目录，抛出文件未找到错误
    raise FileNotFoundError(f"Could not find SGM configs in {candidates}")
```