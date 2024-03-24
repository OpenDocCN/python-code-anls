# `.\lucidrains\pytorch-custom-utils\pytorch_custom_utils\accelerate_utils.py`

```py
# 导入必要的模块
from functools import partial, wraps
from typing import Optional, Callable
from contextlib import nullcontext, contextmanager

from torch.nn import Module

from accelerate import Accelerator
from accelerate.tracking import WandBTracker

# 辅助函数

# 检查变量是否存在
def exists(v):
    return v is not None

# 创建一个结合两个上下文管理器的上下文管理器
@contextmanager
def combine_contexts(a, b):
    with a() as c1, b() as c2:
        yield (c1, c2)

# 在数组中查找第一个满足条件的元素
def find_first(cond: Callable, arr):
    for el in arr:
        if cond(el):
            return el

    return None

# 添加一个用于 wandb 跟踪的上下文管理器，具有特定的项目和实验名称

def add_wandb_tracker_contextmanager(
    accelerator_instance_name = 'accelerator',
    tracker_hps_instance_name = 'tracker_hps'
):
    def decorator(klass):

        @contextmanager
        def wandb_tracking(
            self,
            project: str,
            run: Optional[str] = None,
            hps: Optional[dict] = None
        ):
            maybe_accelerator = getattr(self, accelerator_instance_name, None)

            assert exists(maybe_accelerator) and isinstance(maybe_accelerator, Accelerator), f'Accelerator instance not found at self.{accelerator_instance_name}'

            hps = getattr(self, tracker_hps_instance_name, hps)

            maybe_accelerator.init_trackers(project, config = hps)

            wandb_tracker = find_first(lambda el: isinstance(el, WandBTracker), maybe_accelerator.trackers)

            assert exists(wandb_tracker), 'wandb tracking was not enabled. you need to set `log_with = "wandb"` on your accelerate kwargs'

            if exists(run):
                assert exists(wandb_tracker)
                wandb_tracker.run.name = run

            yield

            maybe_accelerator.end_training() 

        if not hasattr(klass, 'wandb_tracking'):
            klass.wandb_tracking = wandb_tracking

        return klass

    return decorator

# 当在可能的 DDP 包装的主模型上找不到属性时，自动取消包装模型

class ForwardingWrapper:
  def __init__(self, parent, child):
    self.parent = parent
    self.child = child

  def __getattr__(self, key):
    if hasattr(self.parent, key):
      return getattr(self.parent, key)

    return getattr(self.child, key)

  def __call__(self, *args, **kwargs):
    call_fn = self.__getattr__('__call__')
    return call_fn(*args, **kwargs)

def auto_unwrap_model(
    accelerator_instance_name = 'accelerator',
    model_instance_name = 'model'
):
    def decorator(klass):
        _orig_init = klass.__init__

        @wraps(_orig_init)
        def __init__(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            model = getattr(self, model_instance_name)
            accelerator = getattr(self, accelerator_instance_name)

            assert isinstance(accelerator, Accelerator)
            forward_wrapped_model = ForwardingWrapper(model, accelerator.unwrap_model(model))
            setattr(self, model_instance_name, forward_wrapped_model)

        klass.__init__ = __init__
        return klass

    return decorator

# 梯度累积上下文管理器
# 对除最后一次迭代外的所有迭代应用 no_sync 上下文

def model_forward_contexts(
    accelerator: Accelerator,
    model: Module,
    grad_accum_steps: int = 1
):
    for i in range(grad_accum_steps):
        is_last_step = i == grad_accum_steps - 1

        maybe_no_sync = partial(accelerator.no_sync, model) if not is_last_step else nullcontext

        yield partial(combine_contexts, accelerator.autocast, maybe_no_sync)
```