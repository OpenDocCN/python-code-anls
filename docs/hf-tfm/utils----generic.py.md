# `.\transformers\utils\generic.py`

```py
# 版权声明和许可信息
# 版权声明和许可信息，指定了代码的版权和许可信息
# 详细信息可在 http://www.apache.org/licenses/LICENSE-2.0 获取

"""
通用工具
"""

# 导入所需的模块和库
import inspect
import tempfile
from collections import OrderedDict, UserDict
from collections.abc import MutableMapping
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, ContextManager, Iterable, List, Tuple

import numpy as np

# 导入自定义的模块
from .import_utils import is_flax_available, is_tf_available, is_torch_available, is_torch_fx_proxy

# 如果可用，导入 JAX 库
if is_flax_available():
    import jax.numpy as jnp

# 自定义的缓存属性装饰器
class cached_property(property):
    """
    Descriptor that mimics @property but caches output in member variable.

    From tensorflow_datasets

    Built-in in functools from Python 3.8.
    """

    def __get__(self, obj, objtype=None):
        # See docs.python.org/3/howto/descriptor.html#properties
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        attr = "__cached_" + self.fget.__name__
        cached = getattr(obj, attr, None)
        if cached is None:
            cached = self.fget(obj)
            setattr(obj, attr, cached)
        return cached


# 从 distutils.util 中导入的函数
def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values are 'n', 'no', 'f', 'false', 'off', and '0'.
    Raises ValueError if 'val' is anything else.
    """
    val = val.lower()
    if val in {"y", "yes", "t", "true", "on", "1"}:
        return 1
    if val in {"n", "no", "f", "false", "off", "0"}:
        return 0
    raise ValueError(f"invalid truth value {val!r}")


def infer_framework_from_repr(x):
    """
    Tries to guess the framework of an object `x` from its repr (brittle but will help in `is_tensor` to try the
    frameworks in a smart order, without the need to import the frameworks).
    """
    representation = str(type(x))
    if representation.startswith("<class 'torch."):
        return "pt"
    elif representation.startswith("<class 'tensorflow."):
        return "tf"
    elif representation.startswith("<class 'jax"):
        return "jax"
    elif representation.startswith("<class 'numpy."):
        return "np"


def _get_frameworks_and_test_func(x):
    """
    Returns an (ordered since we are in Python 3.7+) dictionary framework to test function, which places the framework
    we can guess from the repr first, then Numpy, then the others.
    """
    # 定义一个字典，用于存储不同框架的测试函数，按照指定顺序排列
    framework_to_test = {
        "pt": is_torch_tensor,
        "tf": is_tf_tensor,
        "jax": is_jax_tensor,
        "np": is_numpy_array,
    }
    # 推断输入数据的首选框架
    preferred_framework = infer_framework_from_repr(x)
    # 创建一个框架列表，按照优先顺序排列
    frameworks = [] if preferred_framework is None else [preferred_framework]
    # 如果首选框架不是"np"，则将"np"添加到列表中
    if preferred_framework != "np":
        frameworks.append("np")
    # 将除首选框架和"np"之外的其他框架添加到列表中
    frameworks.extend([f for f in framework_to_test if f not in [preferred_framework, "np"]])
    # 返回按照顺序排列的框架及其对应的测试函数
    return {f: framework_to_test[f] for f in frameworks}
def is_tensor(x):
    """
    Tests if `x` is a `torch.Tensor`, `tf.Tensor`, `jaxlib.xla_extension.DeviceArray` or `np.ndarray` in the order
    defined by `infer_framework_from_repr`
    """
    # 获取按照智能顺序测试框架的函数
    framework_to_test_func = _get_frameworks_and_test_func(x)
    # 遍历测试函数，如果有一个返回 True，则返回 True
    for test_func in framework_to_test_func.values():
        if test_func(x):
            return True

    # Tracers
    # 如果是 Torch FX 代理，则返回 True
    if is_torch_fx_proxy(x):
        return True

    # 如果 Flax 可用，则检查是否为 Jax 的 Tracer 类型
    if is_flax_available():
        from jax.core import Tracer

        if isinstance(x, Tracer):
            return True

    # 如果以上条件都不满足，则返回 False
    return False


def _is_numpy(x):
    # 检查是否为 numpy 数组
    return isinstance(x, np.ndarray)


def is_numpy_array(x):
    """
    Tests if `x` is a numpy array or not.
    """
    # 调用内部函数检查是否为 numpy 数组
    return _is_numpy(x)


def _is_torch(x):
    import torch

    # 检查是否为 Torch 的 Tensor 类型
    return isinstance(x, torch.Tensor)


def is_torch_tensor(x):
    """
    Tests if `x` is a torch tensor or not. Safe to call even if torch is not installed.
    """
    # 如果 Torch 可用，则调用内部函数检查是否为 Torch 的 Tensor 类型
    return False if not is_torch_available() else _is_torch(x)


def _is_torch_device(x):
    import torch

    # 检查是否为 Torch 的 Device 类型
    return isinstance(x, torch.device)


def is_torch_device(x):
    """
    Tests if `x` is a torch device or not. Safe to call even if torch is not installed.
    """
    # 如果 Torch 可用，则调用内部函数检查是否为 Torch 的 Device 类型
    return False if not is_torch_available() else _is_torch_device(x)


def _is_torch_dtype(x):
    import torch

    if isinstance(x, str):
        if hasattr(torch, x):
            x = getattr(torch, x)
        else:
            return False
    # 检查是否为 Torch 的 Dtype 类型
    return isinstance(x, torch.dtype)


def is_torch_dtype(x):
    """
    Tests if `x` is a torch dtype or not. Safe to call even if torch is not installed.
    """
    # 如果 Torch 可用，则调用内部函数检查是否为 Torch 的 Dtype 类型
    return False if not is_torch_available() else _is_torch_dtype(x)


def _is_tensorflow(x):
    import tensorflow as tf

    # 检查是否为 TensorFlow 的 Tensor 类型
    return isinstance(x, tf.Tensor)


def is_tf_tensor(x):
    """
    Tests if `x` is a tensorflow tensor or not. Safe to call even if tensorflow is not installed.
    """
    # 如果 TensorFlow 可用，则调用内部函数检查是否为 TensorFlow 的 Tensor 类型
    return False if not is_tf_available() else _is_tensorflow(x)


def _is_tf_symbolic_tensor(x):
    import tensorflow as tf

    # 检查是否为 TensorFlow 的 Symbolic Tensor 类型
    if hasattr(tf, "is_symbolic_tensor"):
        return tf.is_symbolic_tensor(x)
    return type(x) == tf.Tensor


def is_tf_symbolic_tensor(x):
    """
    Tests if `x` is a tensorflow symbolic tensor or not (ie. not eager). Safe to call even if tensorflow is not
    installed.
    """
    # 如果 TensorFlow 可用，则调用内部函数检查是否为 TensorFlow 的 Symbolic Tensor 类型
    return False if not is_tf_available() else _is_tf_symbolic_tensor(x)


def _is_jax(x):
    import jax.numpy as jnp  # noqa: F811

    # 检查是否为 Jax 的 ndarray 类型
    return isinstance(x, jnp.ndarray)


def is_jax_tensor(x):
    """
    Tests if `x` is a Jax tensor or not. Safe to call even if jax is not installed.
    """
    # 如果 Flax 可用，则调用内部函数检查是否为 Jax 的 Tensor 类型
    return False if not is_flax_available() else _is_jax(x)


def to_py_obj(obj):
    """
    Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a python list.
    """
    # 定义一个字典，将不同的框架映射到将对象转换为 Python 对象的函数
    framework_to_py_obj = {
        "pt": lambda obj: obj.detach().cpu().tolist(),
        "tf": lambda obj: obj.numpy().tolist(),
        "jax": lambda obj: np.asarray(obj).tolist(),
        "np": lambda obj: obj.tolist(),
    }

    # 如果对象是字典或 UserDict 类型，则递归调用 to_py_obj 函数转换其中的值为 Python 对象
    if isinstance(obj, (dict, UserDict)):
        return {k: to_py_obj(v) for k, v in obj.items()}
    # 如果对象是列表或元组类型，则递归调用 to_py_obj 函数转换其中的元素为 Python 对象
    elif isinstance(obj, (list, tuple)):
        return [to_py_obj(o) for o in obj]

    # 根据对象的特性选择合适的框架进行转换
    framework_to_test_func = _get_frameworks_and_test_func(obj)
    for framework, test_func in framework_to_test_func.items():
        # 根据测试函数判断对象是否适用于当前框架，如果是则使用对应的函数转换为 Python 对象
        if test_func(obj):
            return framework_to_py_obj[framework](obj)

    # 如果对象是 numpy 数字类型，则调用 tolist 方法转换为列表
    # 对于 0 维的 numpy 数组，也可以使用 tolist 方法
    if isinstance(obj, np.number):
        return obj.tolist()
    else:
        return obj
def to_numpy(obj):
    """
    Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a Numpy array.
    """

    # 定义不同框架到Numpy数组的转换函数
    framework_to_numpy = {
        "pt": lambda obj: obj.detach().cpu().numpy(),
        "tf": lambda obj: obj.numpy(),
        "jax": lambda obj: np.asarray(obj),
        "np": lambda obj: obj,
    }

    # 如果输入对象是字典或UserDict，则递归转换每个值为Numpy数组
    if isinstance(obj, (dict, UserDict)):
        return {k: to_numpy(v) for k, v in obj.items()}
    # 如果输入对象是列表或元组，则转换为Numpy数组
    elif isinstance(obj, (list, tuple)):
        return np.array(obj)

    # 获取用于测试框架的函数和对应的框架，按照智能顺序测试框架
    framework_to_test_func = _get_frameworks_and_test_func(obj)
    for framework, test_func in framework_to_test_func.items():
        if test_func(obj):
            return framework_to_numpy[framework](obj)

    return obj


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __init_subclass__(cls) -> None:
        """Register subclasses as pytree nodes.

        This is necessary to synchronize gradients when using `torch.nn.parallel.DistributedDataParallel` with
        `static_graph=True` with modules that output `ModelOutput` subclasses.
        """
        # 如果使用torch可用，则注册子类为pytree节点，用于同步梯度
        if is_torch_available():
            torch_pytree_register_pytree_node(
                cls,
                _model_output_flatten,
                _model_output_unflatten,
            )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 子类必须使用@dataclass装饰器，检查是否符合要求
        is_modeloutput_subclass = self.__class__ != ModelOutput
        if is_modeloutput_subclass and not is_dataclass(self):
            raise TypeError(
                f"{self.__module__}.{self.__class__.__name__} is not a dataclasss."
                " This is a subclass of ModelOutput and so must use the @dataclass decorator."
            )
    # 在数据类初始化之后执行的方法，用于检查 ModelOutput 数据类
    def __post_init__(self):
        """Check the ModelOutput dataclass.

        Only occurs if @dataclass decorator has been used.
        """
        # 获取数据类的所有字段
        class_fields = fields(self)

        # 安全性和一致性检查
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

        # 获取第一个字段的值
        first_field = getattr(self, class_fields[0].name)
        # 检查其他字段是否都为 None
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        # 如果其他字段都为 None 并且第一个字段不是张量
        if other_fields_are_none and not is_tensor(first_field):
            # 如果第一个字段是字典
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # 如果我们提供了一个迭代器作为第一个字段，并且迭代器是 (key, value) 迭代器
            # 设置相关字段
            if first_field_iterator:
                for idx, element in enumerate(iterator):
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        if idx == 0:
                            # 如果我们没有键/值迭代器，将其设置为属性
                            self[class_fields[0].name] = first_field
                        else:
                            # 如果有混合迭代器，抛出错误
                            raise ValueError(
                                f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            # 如果其他字段不都为 None，则将它们设置为属性
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    # 禁止使用 __delitem__ 方法
    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    # 禁止使用 setdefault 方法
    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    # 禁止使用 pop 方法
    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")
    # 更新方法，抛出异常，禁止在实例上使用 update 方法
    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    # 获取指定键的值，如果键是字符串，则返回内部字典中对应键的值，否则返回转换为元组后的第 k 个元素
    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    # 设置属性值，如果属性在键集合中且值不为空，则调用父类的 __setitem__ 方法设置值，否则直接设置属性值
    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # 避免递归错误，不调用 self.__setitem__
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    # 设置键值对，调用父类的 __setitem__ 方法设置键值对，避免递归错误，不调用 self.__setattr__
    def __setitem__(self, key, value):
        # 如果需要，会抛出 KeyException
        super().__setitem__(key, value)
        # 避免递归错误，不调用 self.__setattr__
        super().__setattr__(key, value)

    # 序列化对象，如果不是数据类，则调用父类的 __reduce__ 方法，否则获取所有属性值组成元组返回
    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return callable, args, *remaining

    # 将对象转换为包含所有非空属性/键的元组
    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())
# 如果 torch 可用，则导入 torch.utils._pytree 模块
if is_torch_available():
    import torch.utils._pytree as _torch_pytree

    # 定义将模型输出展平的函数，返回值为包含所有值的列表和上下文
    def _model_output_flatten(output: ModelOutput) -> Tuple[List[Any], "_torch_pytree.Context"]:
        return list(output.values()), (type(output), list(output.keys()))

    # 定义将模型输出还原的函数，接受值的迭代器和上下文作为参数，返回模型输出
    def _model_output_unflatten(values: Iterable[Any], context: "_torch_pytree.Context") -> ModelOutput:
        output_type, keys = context
        return output_type(**dict(zip(keys, values)))

    # 如果 _torch_pytree 模块中存在 register_pytree_node 方法，则使用该方法，否则使用 _register_pytree_node 方法
    if hasattr(_torch_pytree, "register_pytree_node"):
        torch_pytree_register_pytree_node = _torch_pytree.register_pytree_node
    else:
        torch_pytree_register_pytree_node = _torch_pytree._register_pytree_node
    # 注册 ModelOutput 类型到展平和还原函数
    torch_pytree_register_pytree_node(
        ModelOutput,
        _model_output_flatten,
        _model_output_unflatten,
    )

# 定义一个更明确的枚举类，用于处理缺失值的情况
class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )

# 定义填充策略的枚举类
class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the `padding` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for tab-completion in an
    IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"

# 定义张量类型的枚举类
class TensorType(ExplicitEnum):
    """
    Possible values for the `return_tensors` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for
    tab-completion in an IDE.
    """

    PYTORCH = "pt"
    TENSORFLOW = "tf"
    NUMPY = "np"
    JAX = "jax"

# 定义上下文管理器类
class ContextManagers:
    """
    Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
    in the `fastcore` library.
    """

    def __init__(self, context_managers: List[ContextManager]):
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        for context_manager in self.context_managers:
            self.stack.enter_context(context_manager)

    def __exit__(self, *args, **kwargs):
        self.stack.__exit__(*args, **kwargs)

# 检查给定模型是否可以返回损失值
def can_return_loss(model_class):
    """
    Check if a given model can return loss.

    Args:
        model_class (`type`): The class of the model.
    """
    # 推断模型所属框架
    framework = infer_framework(model_class)
    # 根据框架不同获取模型方法的参数签名
    if framework == "tf":
        signature = inspect.signature(model_class.call)  # TensorFlow models
    elif framework == "pt":
        signature = inspect.signature(model_class.forward)  # PyTorch models
    else:
        signature = inspect.signature(model_class.__call__)  # Flax models

    # 遍历参数签名，检查是否存在 return_loss 参数且默认值为 True
    for p in signature.parameters:
        if p == "return_loss" and signature.parameters[p].default is True:
            return True

    return False

# 查找给定模型使用的标签
def find_labels(model_class):
    """
    Find the labels used by a given model.

    Args:
        model_class (`type`): The class of the model.
    """
    """
    # 获取模型类的名称
    model_name = model_class.__name__
    # 推断模型所属的框架
    framework = infer_framework(model_class)
    # 根据框架不同获取模型的签名信息
    if framework == "tf":
        # 对于 TensorFlow 模型，获取 call 方法的签名
        signature = inspect.signature(model_class.call)
    elif framework == "pt":
        # 对于 PyTorch 模型，获取 forward 方法的签名
        signature = inspect.signature(model_class.forward)
    else:
        # 对于 Flax 模型，获取 __call__ 方法的签名
        signature = inspect.signature(model_class.__call__)

    # 如果模型名称中包含"QuestionAnswering"
    if "QuestionAnswering" in model_name:
        # 返回包含"label"关键字或者"start_positions"、"end_positions"的参数列表
        return [p for p in signature.parameters if "label" in p or p in ("start_positions", "end_positions")]
    else:
        # 返回包含"label"关键字的参数列表
        return [p for p in signature.parameters if "label" in p]
# 将嵌套字典展平为单层字典
def flatten_dict(d: MutableMapping, parent_key: str = "", delimiter: str = "."):
    """Flatten a nested dict into a single level dict."""

    def _flatten_dict(d, parent_key="", delimiter="."):
        # 遍历字典中的键值对
        for k, v in d.items():
            # 构建新的键
            key = str(parent_key) + delimiter + str(k) if parent_key else k
            # 如果值是字典且不为空，则递归展平
            if v and isinstance(v, MutableMapping):
                yield from flatten_dict(v, key, delimiter=delimiter).items()
            else:
                yield key, v

    return dict(_flatten_dict(d, parent_key, delimiter))


# 上下文管理器，用于在工作目录或临时目录中工作
@contextmanager
def working_or_temp_dir(working_dir, use_temp_dir: bool = False):
    # 如果使用临时目录
    if use_temp_dir:
        # 使用临时目录
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    else:
        # 使用工作目录
        yield working_dir


# 转置数组的函数，适用于多种框架的数组
def transpose(array, axes=None):
    """
    Framework-agnostic version of `numpy.transpose` that will work on torch/TensorFlow/Jax tensors as well as NumPy
    arrays.
    """
    # 如果是 NumPy 数组
    if is_numpy_array(array):
        return np.transpose(array, axes=axes)
    # 如果是 PyTorch 张量
    elif is_torch_tensor(array):
        return array.T if axes is None else array.permute(*axes)
    # 如果是 TensorFlow 张量
    elif is_tf_tensor(array):
        import tensorflow as tf

        return tf.transpose(array, perm=axes)
    # 如果是 Jax 张量
    elif is_jax_tensor(array):
        return jnp.transpose(array, axes=axes)
    else:
        raise ValueError(f"Type not supported for transpose: {type(array)}."


# 重塑数组的函数，适用于多种框架的数组
def reshape(array, newshape):
    """
    Framework-agnostic version of `numpy.reshape` that will work on torch/TensorFlow/Jax tensors as well as NumPy
    arrays.
    """
    # 如果是 NumPy 数组
    if is_numpy_array(array):
        return np.reshape(array, newshape)
    # 如果是 PyTorch 张量
    elif is_torch_tensor(array):
        return array.reshape(*newshape)
    # 如果是 TensorFlow 张量
    elif is_tf_tensor(array):
        import tensorflow as tf

        return tf.reshape(array, newshape)
    # 如果是 Jax 张量
    elif is_jax_tensor(array):
        return jnp.reshape(array, newshape)
    else:
        raise ValueError(f"Type not supported for reshape: {type(array)}."


# 压缩数组的函数，适用于多种框架的数组
def squeeze(array, axis=None):
    """
    Framework-agnostic version of `numpy.squeeze` that will work on torch/TensorFlow/Jax tensors as well as NumPy
    arrays.
    """
    # 如果是 NumPy 数组
    if is_numpy_array(array):
        return np.squeeze(array, axis=axis)
    # 如果是 PyTorch 张量
    elif is_torch_tensor(array):
        return array.squeeze() if axis is None else array.squeeze(dim=axis)
    # 如果是 TensorFlow 张量
    elif is_tf_tensor(array):
        import tensorflow as tf

        return tf.squeeze(array, axis=axis)
    # 如果是 Jax 张量
    elif is_jax_tensor(array):
        return jnp.squeeze(array, axis=axis)
    else:
        raise ValueError(f"Type not supported for squeeze: {type(array)}."


# 增加维度的函数，适用于多种框架的数组
def expand_dims(array, axis):
    """
    Framework-agnostic version of `numpy.expand_dims` that will work on torch/TensorFlow/Jax tensors as well as NumPy
    arrays.
    """
    # 如果是 NumPy 数组
    if is_numpy_array(array):
        return np.expand_dims(array, axis)
    # 如果是 PyTorch 张量
    elif is_torch_tensor(array):
        return array.unsqueeze(dim=axis)
    # 如果输入的数组是 TensorFlow 的张量
    elif is_tf_tensor(array):
        # 导入 TensorFlow 模块
        import tensorflow as tf
        # 使用 TensorFlow 的 expand_dims 函数在指定轴上增加维度
        return tf.expand_dims(array, axis=axis)
    # 如果输入的数组是 JAX 的张量
    elif is_jax_tensor(array):
        # 使用 JAX 的 expand_dims 函数在指定轴上增加维度
        return jnp.expand_dims(array, axis=axis)
    # 如果输入的数组不是 TensorFlow 或 JAX 的张量
    else:
        # 抛出数值错误，指示不支持的类型
        raise ValueError(f"Type not supported for expand_dims: {type(array)}.")
# 计算给定数组的大小，适用于 torch/TensorFlow/Jax 张量以及 NumPy 数组
def tensor_size(array):
    # 如果是 NumPy 数组，则返回其大小
    if is_numpy_array(array):
        return np.size(array)
    # 如果是 torch 张量，则返回其元素个数
    elif is_torch_tensor(array):
        return array.numel()
    # 如果是 TensorFlow 张量，则返回其大小
    elif is_tf_tensor(array):
        import tensorflow as tf

        return tf.size(array)
    # 如果是 Jax 张量，则返回其大小
    elif is_jax_tensor(array):
        return array.size
    # 如果不是以上类型，则抛出数值错误
    else:
        raise ValueError(f"Type not supported for expand_dims: {type(array)}.")


# 将 repo_id 的信息添加到给定的自动映射中
def add_model_info_to_auto_map(auto_map, repo_id):
    # 遍历自动映射的键值对
    for key, value in auto_map.items():
        # 如果值是元组或列表，则对每个元素添加 repo_id 前缀
        if isinstance(value, (tuple, list)):
            auto_map[key] = [f"{repo_id}--{v}" if (v is not None and "--" not in v) else v for v in value]
        # 如果值不为空且不包含 "--"，则添加 repo_id 前缀
        elif value is not None and "--" not in value:
            auto_map[key] = f"{repo_id}--{value}"

    return auto_map


# 推断给定模型的框架，不使用 isinstance()，因为无法保证相关类是否已导入或可用
def infer_framework(model_class):
    # 遍历模型类的基类
    for base_class in inspect.getmro(model_class):
        module = base_class.__module__
        name = base_class.__name__
        # 如果模块以 "tensorflow" 或 "keras" 开头，或者类名为 "TFPreTrainedModel"，则返回 "tf"
        if module.startswith("tensorflow") or module.startswith("keras") or name == "TFPreTrainedModel":
            return "tf"
        # 如果模块以 "torch" 开头，或者类名为 "PreTrainedModel"，则返回 "pt"
        elif module.startswith("torch") or name == "PreTrainedModel":
            return "pt"
        # 如果模块以 "flax" 或 "jax" 开头，或者类名为 "FlaxPreTrainedModel"，则返回 "flax"
        elif module.startswith("flax") or module.startswith("jax") or name == "FlaxPreTrainedModel":
            return "flax"
    # 如果无法推断框架，则抛出类型错误
    else:
        raise TypeError(f"Could not infer framework from class {model_class}.")
```