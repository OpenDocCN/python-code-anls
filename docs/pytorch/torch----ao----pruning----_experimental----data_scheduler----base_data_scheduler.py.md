# `.\pytorch\torch\ao\pruning\_experimental\data_scheduler\base_data_scheduler.py`

```
# 导入必要的模块和函数
from functools import wraps  # 导入 wraps 装饰器，用于保留函数的元数据
import weakref  # 导入 weakref 模块，用于创建弱引用
import abc  # 导入 abc 模块，支持抽象基类
import warnings  # 导入 warnings 模块，用于警告处理

from ..data_sparsifier import BaseDataSparsifier  # 导入 BaseDataSparsifier 类，来自父级包中的 data_sparsifier 模块

__all__ = ['BaseDataScheduler']  # 在使用 `from module import *` 时可导出的符号列表

class BaseDataScheduler:
    r"""
    BaseDataScheduler 是为 BaseDataSparsifier 类设计的抽象调度器类。这个类控制着稀疏化类的一个特定超参数，
    并在训练过程中（或随时间变化）对其进行调整。

    Args:
        data_sparsifier (instance of BaseDataSparsifier)
            已实现的数据稀疏化类，在其中实现了 update_mask 方法
        schedule_param (str)
            需要调度/变化的传递给稀疏化器的特定超参数
        last_epoch (int, default=-1)
            当需要从特定点恢复训练时，传递的上一个 epoch 数
        verbose (bool, default=False)
            BaseDataScheduler 的详细程度

    The *get_hyperparam()* function needs to be implemented by the user.
    """
    def __init__(self, data_sparsifier, schedule_param: str, last_epoch=-1, verbose=False):
        # Attach sparsifier
        if not isinstance(data_sparsifier, BaseDataSparsifier):
            # 如果传入的 data_sparsifier 不是 BaseDataSparsifier 的实例，则抛出类型错误
            raise TypeError(f'{type(data_sparsifier).__name__} is not an instance of torch.ao.pruning.BaseDataSparsifier')
        self.data_sparsifier = data_sparsifier
        self.schedule_param = schedule_param

        # Initialize epoch and base hyper-params
        # 使用 schedule_param 作为配置名称，从 data_sparsifier 的 data_groups 中获取对应的配置值
        self.base_param = {
            name: config.get(schedule_param, None)
            for name, config in self.data_sparsifier.data_groups.items()
        }

        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `scheduler.step()` is called after
        # `sparsifier.step()`
        # 定义一个装饰器函数，确保在调用 sparsifier.step() 后再调用 scheduler.step()
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # 如果方法已经被替换过，直接返回原方法
                return method

            # 防止循环引用，使用弱引用保存 sparsifier 实例
            instance_ref = weakref.ref(method.__self__)
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1  # 类型提示忽略 _step_count 属性
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # 标记方法已经被替换过
            wrapper._with_counter = True  # 类型提示忽略 attr-defined 属性
            return wrapper

        # 将 data_sparsifier.step 方法替换为带有计数器的装饰器版本
        self.data_sparsifier.step = with_counter(self.data_sparsifier.step)  # 类型提示忽略 assignment
        # 初始化 sparsifier 的步数计数器
        self.data_sparsifier._step_count = 0  # 类型提示忽略 attr-defined
        # 初始化本类的步数计数器
        self._step_count: int = 0
        self.verbose = verbose

        # Housekeeping
        # 标记是否在 step 方法中调用了 get_sp 方法（sp 是 schedule parameter 的缩写）
        self._get_sp_called_within_step: bool = False
        # 调用当前类的 step 方法进行初始化
        self.step()

    @abc.abstractmethod
    # 抽象方法，子类需要实现该方法以返回调度参数的名称到值的字典
    # 当调度器的 step() 函数被调用时，返回的值将更新到稀疏器 sparsifier 中
    # 例如:
    # >>> def get_schedule_param(self):
    # ...     new_param = {}
    # ...     for name in self.sparsifier.data_groups.keys():
    # ...         new_param[name] = self.sparsifier.data_groups[name][self.schedule_param] * 0.5
    # ...     return new_param
    def get_schedule_param(self):
        raise NotImplementedError

    # 返回对象的字符串表示形式，包括类名、数据稀疏器的状态和调度参数的基础值
    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        format_string += '\n'
        format_string += f'Data Sparsifier {self.data_sparsifier}\n'
        format_string += f'    {self.schedule_param}: {self.base_param}\n'
        format_string += ')'
        return format_string

    # 返回调度器的状态作为一个字典
    # 字典包含 self.__dict__ 中每个变量的条目，但不包括 data_sparsifier
    # 注意: 调度器类不跟踪 data_sparsifier 的状态，在存储调度器的状态之前，请确保存储 data_sparsifier 的状态
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'data_sparsifier'}

    # 加载调度器的状态
    # 注意: 在恢复调度器之前，请记得恢复 data_sparsifier 的状态
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    # 返回最后一个参数的值
    def get_last_param(self):
        return self._last_param
    def step(self):
        # 如果在稀疏化处理之前调用调度器步骤，则发出警告。
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            # 检查是否在稀疏度调度器初始化后覆盖了 `data_sparsifier.step()`。
            if not hasattr(self.data_sparsifier.step, "_with_counter"):
                warnings.warn("Seems like `data_sparsifier.step()` has been overridden after sparsity scheduler "
                              "initialization. Please, make sure to call `data_sparsifier.step()` before "
                              "`scheduler.step()`.", UserWarning)

            # 检查是否在调用 `data_sparsifier.step()` 之前已调用了两次 `scheduler.step()`
            elif self.data_sparsifier._step_count < 1:  # type: ignore[attr-defined]
                warnings.warn("Detected call of `scheduler.step()` before `data_sparsifier.step()`. "
                              "You have to make sure you run the data_sparsifier.step() BEFORE any "
                              "calls to the scheduler.step().", UserWarning)
        
        # 增加步骤计数
        self._step_count += 1

        # 定义一个内部类 `_enable_get_sp_call` 用于管理获取稀疏度调度器调用的状态
        class _enable_get_sp_call:

            def __init__(self, o):
                self.o = o

            # 进入上下文管理器时设置 `_get_sp_called_within_step` 为 True
            def __enter__(self):
                self.o._get_sp_called_within_step = True
                return self

            # 退出上下文管理器时设置 `_get_sp_called_within_step` 为 False
            def __exit__(self, type, value, traceback):
                self.o._get_sp_called_within_step = False

        # 使用 `_enable_get_sp_call` 上下文管理器设置 `_get_sp_called_within_step` 为 True
        with _enable_get_sp_call(self):
            # 增加最后一个周期计数
            self.last_epoch += 1
            # 获取更新后的调度器参数
            updated_scheduler_params = self.get_schedule_param()

        # 更新数据组的调度参数
        for name, param in updated_scheduler_params.items():
            self.data_sparsifier.data_groups[name][self.schedule_param] = param
            # 如果启用详细输出，则打印调整后的参数信息
            if self.verbose:
                print(f"Adjusting {self.schedule_param} for group {name} to {param}")

        # 记录最后一个参数设置
        self._last_param = {
            name: config.get(self.schedule_param, None)
            for name, config in self.data_sparsifier.data_groups.items()
        }
        # 启用掩码更新
        self.data_sparsifier.enable_mask_update = True
```