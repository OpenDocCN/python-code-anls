# `.\pytorch\torch\utils\_config_module.py`

```py
# mypy: allow-untyped-defs
# 导入需要的模块和函数
import contextlib  # 上下文管理工具
import copy  # 复制对象
import hashlib  # 哈希算法
import inspect  # 获取对象信息
import io  # 文件流操作
import pickle  # 序列化和反序列化
import tokenize  # Python 源码解析
import unittest  # 单元测试框架
from types import FunctionType, ModuleType  # 引入特定类型
from typing import Any, Dict, Optional, Set, Union  # 引入类型提示
from typing_extensions import deprecated  # 引入特定类型扩展
from unittest import mock  # 单元测试模拟对象

# 定义可以在配置中保存和加载的类型
CONFIG_TYPES = (int, float, bool, type(None), str, list, set, tuple, dict)


def install_config_module(module):
    """
    Converts a module-level config into a `ConfigModule()`.

    See _config_typing.pyi for instructions on how to get the converted module to typecheck.
    """

    class ConfigModuleInstance(ConfigModule):
        _bypass_keys = set({"_is_dirty", "_hash_digest"})

    def visit(source, dest, prefix):
        """Walk the module structure and move everything to module._config"""
        for key, value in list(source.__dict__.items()):
            # 跳过双下划线开头的私有属性、模块和函数类型，以及属于 'typing' 模块的对象
            if (
                key.startswith("__")
                or isinstance(value, (ModuleType, FunctionType))
                or (hasattr(value, "__module__") and value.__module__ == "typing")
            ):
                continue

            name = f"{prefix}{key}"
            if isinstance(value, CONFIG_TYPES):
                # 将符合配置类型的值存入配置字典和默认字典
                config[name] = value
                default[name] = value
                if dest is module:
                    delattr(module, key)
            elif isinstance(value, type):
                # 如果值是类类型，则使用子配置代理对象递归访问其属性
                assert value.__module__ == module.__name__
                proxy = SubConfigProxy(module, f"{name}.")
                visit(value, proxy, f"{name}.")
                setattr(dest, key, proxy)
            else:
                raise AssertionError(f"Unhandled config {key}={value} ({type(value)})")

    config: Dict[str, Any] = dict()
    default: Dict[str, Any] = dict()

    # 获取带有 @compile_ignored 注释的赋值语句
    compile_ignored_keys = get_assignments_with_compile_ignored_comments(module)

    # 访问模块结构并将其移动到 module._config 中
    visit(module, module, "")
    module._config = config
    module._default = default
    module._allowed_keys = set(config.keys())
    module._compile_ignored_keys = compile_ignored_keys
    module.__class__ = ConfigModuleInstance
    module._is_dirty = True
    module._hash_digest = None


COMPILE_IGNORED_MARKER = "@compile_ignored"


# 获取所有带有 @compile_ignored 注释的赋值语句的键
def get_assignments_with_compile_ignored_comments(module):
    source_code = inspect.getsource(module)
    assignments = set()

    # 使用 tokenize 模块解析源代码以获取注释
    tokens = tokenize.tokenize(io.BytesIO(source_code.encode("utf-8")).readline)
    current_comment = "", -1
    prev_name = ""
    # 遍历tokens中的每个token
    for token in tokens:
        # 如果当前token是注释类型
        if token.type == tokenize.COMMENT:
            # 初始化前一个名称为空字符串
            prev_name = ""
            # 获取当前注释的字符串，并去除首尾空白字符
            maybe_current = token.string.strip()
            # 如果当前注释包含COMPILE_IGNORED_MARKER
            if COMPILE_IGNORED_MARKER in maybe_current:
                # 确保当前注释与行号在current_comment中未被使用
                assert current_comment == (
                    "",
                    -1,
                ), f"unconsumed {COMPILE_IGNORED_MARKER}"
                # 更新current_comment为当前注释内容和当前行号
                current_comment = maybe_current, token.start[0]
        # 如果当前token是名称类型
        elif token.type == tokenize.NAME:
            # 只接受第一个名称token，用于处理类似于 foo: Bar = ... 的情况
            if not prev_name:
                prev_name = token.string
        # 如果当前token是操作符并且是等号("=")
        elif token.type == tokenize.OP and token.string == "=":
            # 检查当前赋值操作是否跟在带有COMPILE_IGNORED_MARKER的注释之后
            if (
                COMPILE_IGNORED_MARKER in current_comment[0]
                and current_comment[1] == token.start[0] - 1
            ):
                # 将前一个名称添加到assignments集合中
                assignments.add(prev_name)
                # 重置current_comment为空字符串和-1
                current_comment = "", -1  # reset
            # 重置prev_name为空字符串
            prev_name = ""
    # 最后确保current_comment被完全消耗，即为空字符串和-1
    assert current_comment == ("", -1), f"unconsumed {COMPILE_IGNORED_MARKER}"
    # 返回最终的assignments集合
    return assignments
class ConfigModule(ModuleType):
    # NOTE: This should be kept in sync with _config_typing.pyi.
    # 模块用于管理配置，继承自 ModuleType

    # The default values of the configuration settings.  This can be used to
    # determine if the config has been changed or not.
    # 配置设置的默认值，用于检测配置是否被修改过
    _default: Dict[str, Any]

    # The actual configuration settings.  E.g., torch._dynamo.config.debug
    # would live as "debug" in the key, and torch._inductor.config.triton.cudagraphs
    # maps as "triton.cudagraphs"
    # 实际的配置设置，以字典形式存储各个配置项及其对应的值

    _config: Dict[str, Any]

    # A set of keys that are allowed to be set through __setattr__.
    # 允许通过 __setattr__ 设置的键的集合
    _allowed_keys: Set[str]

    # Keys that should bypass the normal __setattr__ mechanism.
    # 应绕过正常 __setattr__ 机制的键的集合
    _bypass_keys: Set[str]

    # Keys that should be ignored during configuration compilation.
    # 在配置编译过程中应忽略的键的集合
    _compile_ignored_keys: Set[str]

    # Flag indicating whether the configuration has been modified.
    # 表示配置是否已被修改的标志
    _is_dirty: bool

    # Optional hash digest used for comparison or verification.
    # 可选的哈希摘要，用于比较或验证

    _hash_digest: Optional[bytes]

    def __init__(self):
        # Prevent direct instantiation of ConfigModule.
        # 防止直接实例化 ConfigModule
        raise NotImplementedError(
            f"use {__name__}.install_config_module(sys.modules[__name__])"
        )

    def __setattr__(self, name, value):
        # Override setattr to control setting of attributes dynamically.
        # 重写 setattr 控制动态属性的设置
        if name in self._bypass_keys:
            super().__setattr__(name, value)
        elif name not in self._allowed_keys:
            raise AttributeError(f"{self.__name__}.{name} does not exist")
        else:
            self._config[name] = value

    def __getattr__(self, name):
        # Override getattr to retrieve attribute values.
        # 重写 getattr 获取属性值
        try:
            return self._config[name]
        except KeyError as e:
            # Ensure proper error handling for hasattr()
            raise AttributeError(f"{self.__name__}.{name} does not exist") from e

    def __delattr__(self, name):
        # Override delattr to support attribute deletion.
        # 重写 delattr 支持属性删除
        del self._config[name]

    def save_config(self) -> bytes:
        """Convert config to a pickled blob"""
        # Serialize the configuration into a pickled byte stream.
        # 将配置序列化为 pickle 字节流
        config = dict(self._config)
        for key in config.get("_save_config_ignore", ()):
            config.pop(key)
        return pickle.dumps(config, protocol=2)

    def save_config_portable(self) -> Dict[str, Any]:
        """Convert config to portable format"""
        # Convert configuration into a portable dictionary format.
        # 将配置转换为可移植的字典格式
        config: Dict[str, Any] = {}
        for key in sorted(self._config):
            if key.startswith("_"):
                continue
            if any(
                key.startswith(e) for e in self._config["_cache_config_ignore_prefix"]
            ):
                continue
            config[key] = self._config[key]
        return config

    def codegen_config(self) -> str:
        """Convert config to Python statements that replicate current config.
        This does NOT include config settings that are at default values.
        """
        # Generate Python code statements that represent the current configuration.
        # 生成表示当前配置的 Python 语句，不包括默认值的配置设置
        lines = []
        mod = self.__name__
        for k, v in self._config.items():
            if k in self._config.get("_save_config_ignore", ()):
                continue
            if v == self._default[k]:
                continue
            lines.append(f"{mod}.{k} = {v!r}")
        return "\n".join(lines)
    # 返回配置对象的哈希值，仅对未被标记为 `_compile_ignored_keys` 的配置进行哈希
    def get_hash(self) -> bytes:
        """Hashes the configs that are not compile_ignored"""
        # 如果配置对象被修改过或者哈希摘要为空，则重新计算哈希
        if self._is_dirty or self._hash_digest is None:
            # 创建一个包含未被标记为 `_compile_ignored_keys` 的配置项的字典
            dict_to_hash = {
                k: v
                for k, v in self._config.items()
                if k not in self._compile_ignored_keys
            }
            # 对字典项按键排序并生成字符串进行哈希
            string_to_hash = repr(sorted(dict_to_hash.items()))
            self._hash_digest = hashlib.md5(string_to_hash.encode("utf-8")).digest()
            self._is_dirty = False
        # 返回计算好的哈希摘要
        return self._hash_digest

    # 标记为废弃函数，提供替代方法建议
    @deprecated(
        "`config.to_dict()` has been deprecated. It may no longer change the underlying config."
        " use `config.shallow_copy_dict()` or `config.get_config_copy()` instead",
        category=FutureWarning,
    )
    # 返回当前配置的浅拷贝字典
    def to_dict(self) -> Dict[str, Any]:
        return self.shallow_copy_dict()

    # 返回当前配置的浅拷贝字典
    def shallow_copy_dict(self) -> Dict[str, Any]:
        return {**self._config}

    # 根据给定的序列化配置或者字典，加载到当前配置中
    def load_config(self, maybe_pickled_config: Union[bytes, Dict[str, Any]]) -> None:
        """Restore from a prior call to save_config() or shallow_copy_dict()"""
        # 如果输入是序列化的字节流，则解序列化为字典
        if not isinstance(maybe_pickled_config, dict):
            config = pickle.loads(maybe_pickled_config)
        else:
            config = maybe_pickled_config
        # 更新当前配置对象
        self._config.update(config)

    # 返回当前配置的深拷贝字典
    def get_config_copy(self) -> Dict[str, Any]:
        return copy.deepcopy(self._config)

    # 对当前配置进行部分更新或补丁，可以接受可选的字符串或字典作为参数，以及额外的关键字参数
    def patch(
        self,
        arg1: Optional[Union[str, Dict[str, Any]]] = None,
        arg2: Any = None,
        **kwargs,
    ):
        """
        Decorator and/or context manager to make temporary changes to a config.

        As a decorator:

            @config.patch("name", val)
            @config.patch(name1=val1, name2=val2)
            @config.patch({"name1": val1, "name2", val2})
            def foo(...):
                ...

        As a context manager:

            with config.patch("name", val):
                ...
        """
        changes: Dict[str, Any]
        if arg1 is not None:
            if arg2 is not None:
                assert isinstance(arg1, str)
                # patch("key", True) syntax
                changes = {arg1: arg2}
            else:
                assert isinstance(arg1, dict)
                # patch({"key": True}) syntax
                changes = arg1
            assert not kwargs
        else:
            # patch(key=True) syntax
            changes = kwargs
            assert arg2 is None
        assert isinstance(changes, dict), f"expected `dict` got {type(changes)}"
        prior: Dict[str, Any] = {}
        config = self
        dirty = False

        class ConfigPatch(ContextDecorator):
            def __enter__(self):
                assert not prior
                nonlocal dirty
                for key in changes.keys():
                    # KeyError on invalid entry
                    prior[key] = config._config[key]
                    dirty = key not in config._compile_ignored_keys
                config._config.update(changes)
                config._is_dirty = dirty

            def __exit__(self, exc_type, exc_val, exc_tb):
                nonlocal dirty
                config._config.update(prior)
                config._is_dirty = dirty
                prior.clear()

        return ConfigPatch()

    def _make_closure_patcher(self, **changes):
        """
        A lower-overhead version of patch() for things on the critical path.

        Usage:

            # do this off the critical path
            change_fn = config.make_closure_patcher(foo=True)

            ...

            revert = change_fn()
            try:
              ...
            finally:
                revert()

        """
        config = self._config

        def change():
            prior = {k: config[k] for k in changes}
            config.update(changes)

            def revert():
                config.update(prior)

            return revert

        return change
class ContextDecorator(contextlib.ContextDecorator):
    """
    Same as contextlib.ContextDecorator, but with support for
    `unittest.TestCase`
    """

    # 进入上下文管理器时抛出未实现错误
    def __enter__(self):
        raise NotImplementedError("NYI")

    # 退出上下文管理器时抛出未实现错误
    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError("NYI")

    # 装饰函数，支持 `unittest.TestCase` 的装饰
    def __call__(self, func):
        # 如果 func 是类且是 unittest.TestCase 的子类
        if isinstance(func, type) and issubclass(func, unittest.TestCase):

            class _TestCase(func):  # type: ignore[valid-type, misc]
                # 设置类的准备工作，调用 __enter__ 进入上下文
                @classmethod
                def setUpClass(cls):
                    self.__enter__()
                    try:
                        super().setUpClass()  # 调用父类的 setUpClass 方法
                    except Exception:
                        self.__exit__(None, None, None)  # 出错时调用 __exit__ 退出上下文
                        raise

                # 设置类的清理工作
                @classmethod
                def tearDownClass(cls):
                    try:
                        super().tearDownClass()  # 调用父类的 tearDownClass 方法
                    finally:
                        self.__exit__(None, None, None)  # 调用 __exit__ 退出上下文

            _TestCase.__name__ = func.__name__  # 设置 _TestCase 的名称
            _TestCase.__qualname__ = func.__qualname__  # 设置 _TestCase 的限定名称
            _TestCase.__module__ = func.__module__  # 设置 _TestCase 的模块

            return _TestCase  # 返回包装后的 TestCase 类

        return super().__call__(func)  # 调用父类的 __call__ 方法，返回装饰后的函数


class SubConfigProxy:
    """
    Shim to redirect to main config.
    `config.triton.cudagraphs` maps to _config["triton.cudagraphs"]
    """

    # 初始化方法，设置 _config 和 _prefix 属性
    def __init__(self, config, prefix):
        super().__setattr__("_config", config)  # 绕过自定义 __setattr__ 设置 _config
        super().__setattr__("_prefix", prefix)  # 设置 _prefix

    # 自定义 __setattr__ 方法，设置属性到 _config 中
    def __setattr__(self, name, value):
        return self._config.__setattr__(self._prefix + name, value)

    # 自定义 __getattr__ 方法，从 _config 中获取属性
    def __getattr__(self, name):
        return self._config.__getattr__(self._prefix + name)

    # 自定义 __delattr__ 方法，从 _config 中删除属性
    def __delattr__(self, name):
        return self._config.__delattr__(self._prefix + name)


def patch_object(obj, name, value):
    """
    Workaround `mock.patch.object` issue with ConfigModule
    """
    # 如果 obj 是 ConfigModule 类的实例，调用其 patch 方法
    if isinstance(obj, ConfigModule):
        return obj.patch(name, value)
    # 否则调用 mock.patch.object 对象的方法
    return mock.patch.object(obj, name, value)
```