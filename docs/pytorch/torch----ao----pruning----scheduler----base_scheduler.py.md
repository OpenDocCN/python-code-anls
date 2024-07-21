# `.\pytorch\torch\ao\pruning\scheduler\base_scheduler.py`

```py
# mypy: allow-untyped-defs

# 从 torch.ao.pruning 模块导入 BaseSparsifier 类
from torch.ao.pruning import BaseSparsifier

# 从 functools 模块导入 wraps 函数
from functools import wraps
# 导入 warnings 模块
import warnings
# 导入 weakref 模块
import weakref

# __all__ 列表定义了模块中对外公开的类和函数
__all__ = ["BaseScheduler"]

# 定义 BaseScheduler 类
class BaseScheduler:

    def __init__(self, sparsifier, last_epoch=-1, verbose=False):

        # Attach sparsifier
        # 检查传入的 sparsifier 是否为 BaseSparsifier 的实例，否则抛出异常
        if not isinstance(sparsifier, BaseSparsifier):
            raise TypeError(f'{type(sparsifier).__name__} is not an instance of torch.ao.pruning.BaseSparsifier')
        self.sparsifier = sparsifier

        # Initialize epoch and base sparsity levels

        # 初始化基础稀疏度水平列表为每个组的 sparsity_level 值
        self.base_sl = [group['sparsity_level'] for group in sparsifier.groups]
        # 设置初始的 epoch 为 last_epoch
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `scheduler.step()` is called after
        # `sparsifier.step()`

        # 定义一个装饰器函数 with_counter，用于确保在调用 `scheduler.step()` 前会先调用 `sparsifier.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # 如果已经替换过 `sparsifier.step()`，直接返回原方法
                return method

            # 使用弱引用避免循环引用问题
            instance_ref = weakref.ref(method.__self__)
            func = method.__func__
            cls = instance_ref().__class__
            del method

            # 定义一个包装函数 wrapper，用于在调用前增加计数
            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1  # type: ignore[union-attr]
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # 标记该函数已经被装饰过
            wrapper._with_counter = True  # type: ignore[attr-defined]
            return wrapper

        # 使用 with_counter 装饰 sparsifier.step 方法，确保在调用 sparsifier.step 时会增加计数
        self.sparsifier.step = with_counter(self.sparsifier.step)  # type: ignore[assignment]
        # 初始化 sparsifier 的计数器为 0
        self.sparsifier._step_count = 0  # type: ignore[attr-defined]
        # 初始化 scheduler 的计数器为 0
        self._step_count: int = 0
        # 设置是否详细输出信息的标志
        self.verbose = verbose

        # Housekeeping
        # 维护一个标志，指示是否在 step 方法内部调用了 get_last_sl 方法
        self._get_sl_called_within_step: bool = False

        # 初始化时立即执行一次 step 方法
        self.step()

    # 定义 state_dict 方法，返回 scheduler 的状态字典，不包含 sparsifier
    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the sparsifier.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'sparsifier'}

    # 定义 load_state_dict 方法，加载 scheduler 的状态字典
    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    # 定义 get_last_sl 方法，返回当前 scheduler 最后计算的稀疏度水平
    def get_last_sl(self):
        """ Return last computed sparsity level by current scheduler.
        """
        return self._last_sl
    # 计算稀疏水平使用调度程序的链式形式
    # 注意：此方法不应直接调用，仅由“.step”方法使用。请改用.get_last_sl()。
    def get_sl(self):
        if not self._get_sl_called_within_step:
            # 如果未在step方法内调用get_sl，则发出警告
            warnings.warn(
                "To get the last sparsity level computed by the scheduler, "
                "please use `get_last_sl()`.")

        # 抛出未实现错误，表示该方法需要在子类中实现
        raise NotImplementedError

    def print_sl(self, is_verbose, group, sl, epoch=None):
        """显示当前稀疏水平。
        """
        if is_verbose:
            if epoch is None:
                # 如果epoch为None，则显示调整后的稀疏水平信息
                print(f'Adjusting sparsity level of group {group} to {sl:.4e}.')
            else:
                # 否则，在指定epoch时显示稀疏水平信息
                print(f'Epoch {epoch:5d}: adjusting sparsity level of group {group} to {sl:.4e}.')

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        format_string += '\n'
        format_string += f'Sparsifier {self.sparsifier}\n'
        format_string += f'    base_sl: {self.base_sl}\n'
        format_string += ')'
        return format_string

    def step(self, epoch=None):
        # 如果在sparsifier之前调用scheduler的step方法，则发出警告
        if self._step_count == 1:
            if not hasattr(self.sparsifier.step, "_with_counter"):
                warnings.warn("Seems like `sparsifier.step()` has been overridden after sparsity scheduler "
                              "initialization. Please, make sure to call `sparsifier.step()` before "
                              "`scheduler.step()`.", UserWarning)
            # 检查在sparsifier.step()之前是否已经调用了两次scheduler.step()
            elif self.sparsifier._step_count < 1:  # type: ignore[attr-defined]
                warnings.warn("Detected call of `scheduler.step()` before `sparsifier.step()`. "
                              "You have to make sure you run the sparsifier.step() BEFORE any "
                              "calls to the scheduler.step().", UserWarning)
        
        # 增加步数计数
        self._step_count += 1

        # 启用_get_sl_called_within_step上下文管理器，确保在step方法中调用get_sl
        class _enable_get_sl_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_sl_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_sl_called_within_step = False

        with _enable_get_sl_call(self):
            # 增加最后的epoch计数，并获取sparsity level的值
            self.last_epoch += 1
            values = self.get_sl()

        # 更新sparsifier的各组参数组的稀疏水平值，并打印相应信息
        for i, data in enumerate(zip(self.sparsifier.groups, values)):
            param_group, sl = data
            param_group['sparsity_level'] = sl
            self.print_sl(self.verbose, i, sl, epoch)

        # 更新_last_sl列表为当前各组的稀疏水平值
        self._last_sl = [group['sparsity_level'] for group in self.sparsifier.groups]
        # 启用mask更新
        self.sparsifier.enable_mask_update = True
    # 定义一个方法 `_make_sure_a_list`，确保变量与 `self.sparsifier.groups` 的长度相同，并将其转换为列表形式
    def _make_sure_a_list(self, var):
        # 获取 `self.sparsifier.groups` 的长度
        n = len(self.sparsifier.groups)
        # 如果 `var` 不是列表或元组类型，则将其复制成长度为 `n` 的列表
        if not isinstance(var, (list, tuple)):
            return [var] * n
        else:
            # 如果 `var` 是列表或元组类型，但长度不等于 `n`，则抛出数值错误异常
            if len(var) != n:
                raise ValueError(f"Expected variable of length {n}, but got {len(var)}")
            # 将 `var` 转换为列表形式，并返回结果
            return list(var)  # 我们希望结果是列表，而不是元组
```