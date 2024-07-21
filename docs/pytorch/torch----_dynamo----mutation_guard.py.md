# `.\pytorch\torch\_dynamo\mutation_guard.py`

```py
# 引入必要的模块和类
# mypy: allow-untyped-defs
# mypy: disable-error-code="method-assign"
import functools  # 导入 functools 模块，用于创建包装函数
import weakref  # 导入 weakref 模块，用于创建弱引用对象

import torch.nn  # 导入 PyTorch 的神经网络模块
from torch.nn import Module  # 从 torch.nn 中导入 Module 类
from . import config  # 从当前包中导入 config 模块

from .utils import ExactWeakKeyDictionary, is_lazy_module, nn_module_has_global_hooks  # 从当前包中的 utils 模块导入特定函数和类

# 记录未修补的 torch.nn.Module.__init__ 方法
unpatched_nn_module_init = torch.nn.Module.__init__


class MutationTracker:
    # 使用 ExactWeakKeyDictionary 创建的类变量 db，用于跟踪对象和 MutationTracker 实例之间的映射关系
    db = ExactWeakKeyDictionary()

    def __init__(self):
        # 初始化 MutationTracker 实例，设置变异计数器和观察者列表
        self.mutation_count = 0  # 变异计数器
        self.watchers = []  # 观察者列表

    def on_mutation(self, name):
        # 当对象发生变异时调用，增加变异计数器，并使观察者列表失效
        self.mutation_count += 1  # 增加变异计数器
        tmp = self.watchers  # 临时存储观察者列表
        self.watchers = []  # 清空观察者列表
        for ref in tmp:
            guarded = ref()  # 获取弱引用指向的对象
            if guarded is not None:
                guarded.invalidate(ref)  # 调用 guarded 对象的 invalidate 方法，使其失效

    def track(self, guarded_code):
        # 跟踪受保护的代码，将其添加到观察者列表中
        self.watchers.append(weakref.ref(guarded_code))  # 将 guarded_code 的弱引用添加到观察者列表中


def watch(obj, guarded_code):
    """在 obj 发生变异时使 guarded_code 失效"""
    ensure_patched(type(obj))  # 确保 obj 的类型已经被修补

    if obj not in MutationTracker.db:
        MutationTracker.db[obj] = MutationTracker()  # 如果 obj 不在 db 中，创建一个新的 MutationTracker 实例
    tracker = MutationTracker.db[obj]  # 获取 obj 对应的 MutationTracker 实例
    tracker.track(guarded_code)  # 跟踪 guarded_code 的变异情况


def ensure_patched(cls):
    # 确保给定的类已经被修补，以便追踪其实例的变异
    if getattr(cls, "___needs_mutation_patch", True):
        cls.___needs_mutation_patch = False  # 标记该类已被修补
        original_setattr = cls.__setattr__  # 获取原始的 __setattr__ 方法

        @functools.wraps(original_setattr)
        def custom_setattr(self, key, value):
            try:
                MutationTracker.db[self].on_mutation(key)  # 在设置属性时调用 MutationTracker 实例的 on_mutation 方法
            except KeyError:
                pass
            return original_setattr(self, key, value)  # 调用原始的 setattr 方法设置属性值

        cls.__setattr__ = custom_setattr  # 替换类的 setattr 方法为自定义的 custom_setattr 函数


class GenerationTracker:
    generation = 0  # 当前代数计数器
    dynamic_classes = ExactWeakKeyDictionary()  # 弱引用字典，用于存储动态生成的类
    generation_values = ExactWeakKeyDictionary()  # 弱引用字典，用于存储对象的代数标签

    @classmethod
    def tag(cls, obj):
        # 为给定对象设置代数标签
        cls.generation_values[obj] = cls.generation  # 将对象与当前代数关联起来

    @staticmethod
    def mark_class_dynamic(cls):
        # 标记指定类为动态生成的类
        assert issubclass(cls, torch.nn.Module)  # 断言 cls 是 torch.nn.Module 的子类
        GenerationTracker.dynamic_classes[cls] = True  # 在 dynamic_classes 中标记该类为动态类

    @classmethod
    def get_generation_value(cls, obj):
        # 获取给定对象的代数标签值，如果不存在则返回 -1
        if obj not in cls.generation_values:
            return -1
        return cls.generation_values[obj]

    @classmethod
    def check(cls, obj):
        # 检查给定对象是否与当前代数关联
        return (
            obj in cls.generation_values
            and cls.generation_values[obj] == cls.generation
        )  # 返回对象是否与当前代数关联的布尔值

    @classmethod
    def clear(cls):
        # 清除所有代数相关的信息
        cls.generation = 0  # 重置当前代数计数器
        cls.dynamic_classes = ExactWeakKeyDictionary()  # 清空动态类字典
        cls.generation_values = ExactWeakKeyDictionary()  # 清空代数标签字典


def is_dynamic_nn_module(obj, is_export):
    """检查是否动态创建或变异的 nn.Module 对象"""
    if isinstance(obj, torch.nn.Module) and "forward" in obj.__dict__:
        # 如果 obj 是 torch.nn.Module 的实例并且具有 "forward" 属性，表示可能是动态创建或变异的模块
        return True
    if hasattr(obj, "torchdynamo_force_dynamic"):
        return obj.torchdynamo_force_dynamic  # 检查是否有强制动态标志
    if is_lazy_module(obj):
        return False  # 检查是否是惰性模块
    # 对于导出情况，需要修复输入签名问题，因为参数被提升为输入
    # 检查对象是否为 torch.nn.Module 类型，并且满足以下条件：
    # 1) 配置选项中允许内联内置的 nn 模块
    # 2) 不是导出操作
    # 如果以上条件都成立，则返回 True
    if (
        isinstance(obj, torch.nn.Module)
        and config.inline_inbuilt_nn_modules
        and not is_export
    ):
        return True

    # 检查对象是否为 torch.nn.Module 类型，并且具有全局钩子函数
    # 如果条件成立，则返回 True
    if isinstance(obj, torch.nn.Module) and nn_module_has_global_hooks():
        return True

    # 检查对象类型是否在动态类追踪器的动态类中，或者通过生成追踪器对该对象进行检查
    dyn = GenerationTracker.dynamic_classes.get(type(obj)) or GenerationTracker.check(
        obj
    )

    # 返回动态类追踪结果
    return dyn
def install_generation_tagging_init():
    """
    Monkey patch torch.nn.Module.__init__ and torch.nn.Module.__setstate__
    so we can detect nn.Module instances created dynamically inside forward methods.
    """

    # 检查是否需要对 Module 类进行补丁
    if getattr(Module, "___needs_generation_tag_patch", True):
        # 保存原始的 __init__ 方法
        init = Module.__init__

        # 定义补丁后的 __init__ 方法
        def patched_init(self, *args, **kwargs):
            # 调用原始的 __init__ 方法
            init(self, *args, **kwargs)
            # 对当前实例进行生成追踪标记
            GenerationTracker.tag(self)

        # 将补丁后的 __init__ 方法赋给 Module 类
        Module.__init__ = patched_init

        # 保存原始的 __setstate__ 方法
        setstate = Module.__setstate__

        # 定义补丁后的 __setstate__ 方法
        def patched_setstate(self, state):
            # 调用原始的 __setstate__ 方法
            setstate(self, state)
            # 对当前实例进行生成追踪标记
            GenerationTracker.tag(self)

        # 将补丁后的 __setstate__ 方法赋给 Module 类
        Module.__setstate__ = patched_setstate

        # 将 ___needs_generation_tag_patch 标记设置为 False，表示补丁已应用
        Module.___needs_generation_tag_patch = False  # type: ignore[attr-defined]

    # 增加 GenerationTracker 的 generation 计数
    GenerationTracker.generation += 1
```