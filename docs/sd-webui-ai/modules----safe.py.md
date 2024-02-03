# `stable-diffusion-webui\modules\safe.py`

```
# 导入 pickle、collections、torch、numpy、_codecs、zipfile、re 模块
import pickle
import collections
import torch
import numpy
import _codecs
import zipfile
import re

# 导入 errors 模块中的所有内容
from modules import errors

# 如果 torch.storage 模块中有 TypedStorage 类，则将其赋值给 TypedStorage，否则将 torch.storage._TypedStorage 赋值给 TypedStorage
TypedStorage = torch.storage.TypedStorage if hasattr(torch.storage, 'TypedStorage') else torch.storage._TypedStorage

# 定义一个函数 encode，接受任意数量的参数，调用 _codecs.encode 函数并返回结果
def encode(*args):
    out = _codecs.encode(*args)
    return out

# 定义一个 RestrictedUnpickler 类，继承自 pickle.Unpickler 类
class RestrictedUnpickler(pickle.Unpickler):
    # 定义一个类属性 extra_handler，初始值为 None
    extra_handler = None

    # 定义一个 persistent_load 方法，接受 self 和 saved_id 作为参数
    def persistent_load(self, saved_id):
        # 断言 saved_id 的第一个元素为 'storage'
        assert saved_id[0] == 'storage'

        # 尝试创建一个 TypedStorage 对象，如果出现 TypeError 则使用不带 _internal 参数的 TypedStorage 对象
        try:
            return TypedStorage(_internal=True)
        except TypeError:
            return TypedStorage()  # PyTorch before 2.0 does not have the _internal argument
    # 在给定模块和名称的情况下查找类
    def find_class(self, module, name):
        # 如果存在额外的处理程序，则调用额外处理程序来查找类
        if self.extra_handler is not None:
            res = self.extra_handler(module, name)
            if res is not None:
                return res

        # 如果模块为'collections'且名称为'OrderedDict'，则返回collections模块中的OrderedDict类
        if module == 'collections' and name == 'OrderedDict':
            return getattr(collections, name)
        # 如果模块为'torch._utils'且名称在指定列表中，则返回torch._utils模块中对应的类
        if module == 'torch._utils' and name in ['_rebuild_tensor_v2', '_rebuild_parameter', '_rebuild_device_tensor_from_numpy']:
            return getattr(torch._utils, name)
        # 如果模块为'torch'且名称在指定列表中，则返回torch模块中对应的类
        if module == 'torch' and name in ['FloatStorage', 'HalfStorage', 'IntStorage', 'LongStorage', 'DoubleStorage', 'ByteStorage', 'float32', 'BFloat16Storage']:
            return getattr(torch, name)
        # 如果模块为'torch.nn.modules.container'且名称为'ParameterDict'，则返回torch.nn.modules.container模块中的ParameterDict类
        if module == 'torch.nn.modules.container' and name in ['ParameterDict']:
            return getattr(torch.nn.modules.container, name)
        # 如果模块为'numpy.core.multiarray'且名称在指定列表中，则返回numpy.core.multiarray模块中对应的类
        if module == 'numpy.core.multiarray' and name in ['scalar', '_reconstruct']:
            return getattr(numpy.core.multiarray, name)
        # 如果模块为'numpy'且名称在指定列表中，则返回numpy模块中对应的类
        if module == 'numpy' and name in ['dtype', 'ndarray']:
            return getattr(numpy, name)
        # 如果模块为'_codecs'且名称为'encode'，则返回encode函数
        if module == '_codecs' and name == 'encode':
            return encode
        # 如果模块为'pytorch_lightning.callbacks'且名称为'model_checkpoint'，则返回pytorch_lightning.callbacks模块中的model_checkpoint类
        if module == "pytorch_lightning.callbacks" and name == 'model_checkpoint':
            import pytorch_lightning.callbacks
            return pytorch_lightning.callbacks.model_checkpoint
        # 如果模块为'pytorch_lightning.callbacks.model_checkpoint'且名称为'ModelCheckpoint'，则返回pytorch_lightning.callbacks.model_checkpoint模块中的ModelCheckpoint类
        if module == "pytorch_lightning.callbacks.model_checkpoint" and name == 'ModelCheckpoint':
            import pytorch_lightning.callbacks.model_checkpoint
            return pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint
        # 如果模块为'__builtin__'且名称为'set'，则返回set类
        if module == "__builtin__" and name == 'set':
            return set

        # 禁止其他情况的访问，抛出异常
        raise Exception(f"global '{module}/{name}' is forbidden")
# 定义一个正则表达式，用于匹配 'dirname/version'、'dirname/data.pkl' 和 'dirname/data/<number>' 这样的文件名格式
allowed_zip_names_re = re.compile(r"^([^/]+)/((data/\d+)|version|(data\.pkl))$")
# 定义一个正则表达式，用于匹配 'dirname/data.pkl' 这样的文件名格式
data_pkl_re = re.compile(r"^([^/]+)/data\.pkl$")

# 检查给定文件名列表中的文件名是否符合规定的格式
def check_zip_filenames(filename, names):
    for name in names:
        # 如果文件名符合规定的格式，则继续下一个文件名的检查
        if allowed_zip_names_re.match(name):
            continue
        # 如果文件名不符合规定的格式，则抛出异常
        raise Exception(f"bad file inside {filename}: {name}")

# 检查 PyTorch 模型文件是否符合规定的格式
def check_pt(filename, extra_handler):
    try:
        # 如果是新的 PyTorch 格式，文件是一个 ZIP 文件
        with zipfile.ZipFile(filename) as z:
            # 检查 ZIP 文件中的文件名是否符合规定的格式
            check_zip_filenames(filename, z.namelist())

            # 在 ZIP 文件中查找 'data.pkl' 文件的文件名：'<directory name>/data.pkl'
            data_pkl_filenames = [f for f in z.namelist() if data_pkl_re.match(f)]
            # 如果找不到 'data.pkl' 文件，则抛出异常
            if len(data_pkl_filenames) == 0:
                raise Exception(f"data.pkl not found in {filename}")
            # 如果找到多个 'data.pkl' 文件，则抛出异常
            if len(data_pkl_filenames) > 1:
                raise Exception(f"Multiple data.pkl found in {filename}")
            # 打开 'data.pkl' 文件并使用 RestrictedUnpickler 加载数据
            with z.open(data_pkl_filenames[0]) as file:
                unpickler = RestrictedUnpickler(file)
                unpickler.extra_handler = extra_handler
                unpickler.load()

    except zipfile.BadZipfile:
        # 如果不是 ZIP 文件，那么是旧的 PyTorch 格式，其中有五个对象被写入 pickle 文件
        with open(filename, "rb") as file:
            unpickler = RestrictedUnpickler(file)
            unpickler.extra_handler = extra_handler
            # 使用 RestrictedUnpickler 加载五次数据
            for _ in range(5):
                unpickler.load()

# 加载模型文件，使用全局的额外处理程序 global_extra_handler
def load(filename, *args, **kwargs):
    return load_with_extra(filename, *args, extra_handler=global_extra_handler, **kwargs)

# 加载模型文件，可以指定额外的处理程序
def load_with_extra(filename, extra_handler=None, *args, **kwargs):
    """
    this function is intended to be used by extensions that want to load models with
    some extra classes in them that the usual unpickler would find suspicious.
    """
    # 使用 extra_handler 参数指定一个函数，该函数接受模块和字段名作为文本，并返回该字段的值
    def extra(module, name):
        # 如果模块为 'collections' 并且字段名为 'OrderedDict'，则返回 collections.OrderedDict
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict

        return None

    # 使用 safe.load_with_extra('model.pt', extra_handler=extra) 加载模型文件，并使用额外处理函数 extra
    safe.load_with_extra('model.pt', extra_handler=extra)

    # 另一种选择是使用 safe.unsafe_torch_load('model.pt')，如其名称所示，这是绝对不安全的
    """

    # 导入 shared 模块
    from modules import shared

    try:
        # 如果未禁用安全反序列化，则检查文件
        if not shared.cmd_opts.disable_safe_unpickle:
            check_pt(filename, extra_handler)

    except pickle.UnpicklingError:
        # 报告反序列化错误
        errors.report(
            f"Error verifying pickled file from {filename}\n"
            "-----> !!!! The file is most likely corrupted !!!! <-----\n"
            "You can skip this check with --disable-safe-unpickle commandline argument, but that is not going to help you.\n\n",
            exc_info=True,
        )
        return None
    except Exception:
        # 报告异常情况
        errors.report(
            f"Error verifying pickled file from {filename}\n"
            f"The file may be malicious, so the program is not going to read it.\n"
            f"You can skip this check with --disable-safe-unpickle commandline argument.\n\n",
            exc_info=True,
        )
        return None

    # 返回使用 unsafe_torch_load 加载的结果
    return unsafe_torch_load(filename, *args, **kwargs)
# 定义一个类 Extra，用于临时设置全局处理程序，当无法显式调用 load_with_extra 时使用
class Extra:
    """
    A class for temporarily setting the global handler for when you can't explicitly call load_with_extra
    (because it's not your code making the torch.load call). The intended use is like this:


import torch
from modules import safe

def handler(module, name):
    if module == 'torch' and name in ['float64', 'float16']:
        return getattr(torch, name)

    return None

with safe.Extra(handler):
    x = torch.load('model.pt')

    """

    # 初始化方法，接收一个处理程序作为参数
    def __init__(self, handler):
        self.handler = handler

    # 进入上下文时调用的方法
    def __enter__(self):
        global global_extra_handler

        # 断言当前没有在 Extra() 块内部，避免重复进入
        assert global_extra_handler is None, 'already inside an Extra() block'
        global_extra_handler = self.handler

    # 退出上下文时调用的方法
    def __exit__(self, exc_type, exc_val, exc_tb):
        global global_extra_handler

        # 退出上下文后将全局处理程序设置为 None
        global_extra_handler = None


# 将 torch.load 的原始实现保存到 unsafe_torch_load 中
unsafe_torch_load = torch.load
# 将 torch.load 替换为 load 函数
torch.load = load
# 初始化全局处理程序为 None
global_extra_handler = None
```