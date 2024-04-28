# `.\transformers\optimization_tf.py`

```
# 导入 re 模块，用于正则表达式操作
import re
# 导入 typing 模块中的 Callable、List、Optional、Union 类型
from typing import Callable, List, Optional, Union
# 导入 TensorFlow 模块
import tensorflow as tf

# 尝试导入 TensorFlow 1.x 版本的 Adam 优化器，如果导入失败则导入 TensorFlow 2.x 版本的 Adam 优化器
try:
    from tensorflow.keras.optimizers.legacy import Adam
except ImportError:
    from tensorflow.keras.optimizers import Adam

# 定义一个类 WarmUp，继承自 TensorFlow 的 LearningRateSchedule 类
class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Applies a warmup schedule on a given learning rate decay schedule.

    Args:
        initial_learning_rate (`float`):
            The initial learning rate for the schedule after the warmup (so this will be the learning rate at the end
            of the warmup).
        decay_schedule_fn (`Callable`):
            The schedule function to apply after the warmup for the rest of training.
        warmup_steps (`int`):
            The number of steps for the warmup part of training.
        power (`float`, *optional*, defaults to 1.0):
            The power to use for the polynomial warmup (defaults is a linear warmup).
        name (`str`, *optional*):
            Optional name prefix for the returned tensors during the schedule.
    """
    # 定义初始化方法，接受初始学习率、衰减函数、热身步数、幂次、名称作为参数
    def __init__(
        self,
        initial_learning_rate: float,
        decay_schedule_fn: Callable,
        warmup_steps: int,
        power: float = 1.0,
        name: str = None,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置初始学习率
        self.initial_learning_rate = initial_learning_rate
        # 设置热身步数
        self.warmup_steps = warmup_steps
        # 设置幂次
        self.power = power
        # 设置衰减函数
        self.decay_schedule_fn = decay_schedule_fn
        # 设置名称
        self.name = name
```  
    # 定义一个调用函数，根据步数调整学习率
    def __call__(self, step):
        # 使用指定的名称范围，如果没有指定名称则使用默认名称"WarmUp"
        with tf.name_scope(self.name or "WarmUp") as name:
            # 将步数转换为浮点数
            global_step_float = tf.cast(step, tf.float32)
            # 将预热步数转换为浮点数
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            # 计算预热百分比
            warmup_percent_done = global_step_float / warmup_steps_float
            # 计算预热学习率
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)
            # 根据条件选择返回预热学习率或者调度函数计算的学习率
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step - self.warmup_steps),
                name=name,
            )

    # 获取配置信息
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }
# 创建一个优化器，使用一个包含预热阶段和线性衰减的学习率调度
def create_optimizer(
    init_lr: float,
    num_train_steps: int,
    num_warmup_steps: int,
    min_lr_ratio: float = 0.0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    adam_clipnorm: Optional[float] = None,
    adam_global_clipnorm: Optional[float] = None,
    weight_decay_rate: float = 0.0,
    power: float = 1.0,
    include_in_weight_decay: Optional[List[str]] = None,
):
    """
    创建一个优化器，使用一个包含预热阶段和线性衰减的学习率调度

    Args:
        init_lr (`float`):
            预热阶段结束时的期望学习率
        num_train_steps (`int`):
            总的训练步数
        num_warmup_steps (`int`):
            预热步数
        min_lr_ratio (`float`, *可选*, 默认为 0):
            线性衰减结束时的最终学习率将为 `init_lr * min_lr_ratio`
        adam_beta1 (`float`, *可选*, 默认为 0.9):
            Adam 中使用的 beta1
        adam_beta2 (`float`, *可选*, 默认为 0.999):
            Adam 中使用的 beta2
        adam_epsilon (`float`, *可选*, 默认为 1e-8):
            Adam 中使用的 epsilon
        adam_clipnorm (`float`, *可选*, 默认为 `None`):
            如果不为 `None`，则将每个权重张量的梯度范数剪裁为该值
        adam_global_clipnorm (`float`, *可选*, 默认为 `None`)
            如果不为 `None`，则将梯度范数剪裁为该值。使用此参数时，范数是在所有权重张量上计算的，就好像它们被连接成一个单一向量。
        weight_decay_rate (`float`, *可选*, 默认为 0):
            使用的权重衰减
        power (`float`, *可选*, 默认为 1.0):
            用于 PolynomialDecay 的幂
        include_in_weight_decay (`List[str]`, *可选*):
            应用权重衰减的参数名称（或正则表达式）列表。如果未传递任何内容，则将权重衰减应用于所有参数，除了偏置和层归一化参数。
    """
    # 实现学习率的线性衰减
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=init_lr,
        decay_steps=num_train_steps - num_warmup_steps,
        end_learning_rate=init_lr * min_lr_ratio,
        power=power,
    )
    # 如果存在预热步数，则使用预热函数
    if num_warmup_steps:
        lr_schedule = WarmUp(
            initial_learning_rate=init_lr,
            decay_schedule_fn=lr_schedule,
            warmup_steps=num_warmup_steps,
        )
    # 如果权重衰减率大于0，则使用带有权重衰减的Adam优化器
    if weight_decay_rate > 0.0:
        optimizer = AdamWeightDecay(
            learning_rate=lr_schedule,
            weight_decay_rate=weight_decay_rate,
            beta_1=adam_beta1,
            beta_2=adam_beta2,
            epsilon=adam_epsilon,
            clipnorm=adam_clipnorm,
            global_clipnorm=adam_global_clipnorm,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
            include_in_weight_decay=include_in_weight_decay,
        )
    # 如果权重衰减率为0，则使用普通的Adam优化器
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=adam_beta1,
            beta_2=adam_beta2,
            epsilon=adam_epsilon,
            clipnorm=adam_clipnorm,
            global_clipnorm=adam_global_clipnorm,
        )
    # 为了更好地跟踪学习率的演变，我们返回优化器和学习率调度器
    return optimizer, lr_schedule
class AdamWeightDecay(Adam):
    """
    Adam enables L2 weight decay and clip_by_global_norm on gradients. Just adding the square of the weights to the
    loss function is *not* the correct way of using L2 regularization/weight decay with Adam, since that will interact
    with the m and v parameters in strange ways as shown in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Instead we want to decay the weights in a manner that doesn't interact with the m/v parameters. This is equivalent
    to adding the square of the weights to the loss with plain (non-momentum) SGD.

    Args:
        learning_rate (`Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]`, *optional*, defaults to 0.001):
            The learning rate to use or a schedule.
        beta_1 (`float`, *optional*, defaults to 0.9):
            The beta1 parameter in Adam, which is the exponential decay rate for the 1st momentum estimates.
        beta_2 (`float`, *optional*, defaults to 0.999):
            The beta2 parameter in Adam, which is the exponential decay rate for the 2nd momentum estimates.
        epsilon (`float`, *optional*, defaults to 1e-07):
            The epsilon parameter in Adam, which is a small constant for numerical stability.
        amsgrad (`bool`, *optional*, defaults to `False`):
            Whether to apply AMSGrad variant of this algorithm or not, see [On the Convergence of Adam and
            Beyond](https://arxiv.org/abs/1904.09237).
        weight_decay_rate (`float`, *optional*, defaults to 0.0):
            The weight decay to apply.
        include_in_weight_decay (`List[str]`, *optional*):
            List of the parameter names (or re patterns) to apply weight decay to. If none is passed, weight decay is
            applied to all parameters by default (unless they are in `exclude_from_weight_decay`).
        exclude_from_weight_decay (`List[str]`, *optional*):
            List of the parameter names (or re patterns) to exclude from applying weight decay to. If a
            `include_in_weight_decay` is passed, the names in it will supersede this list.
        name (`str`, *optional*, defaults to `"AdamWeightDecay"`):
            Optional name for the operations created when applying gradients.
        kwargs (`Dict[str, Any]`, *optional*):
            Keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
            norm; `clipvalue` is clip gradients by value, `decay` is included for backward compatibility to allow time
            inverse decay of learning rate. `lr` is included for backward compatibility, recommended to use
            `learning_rate` instead.
    """
    # 初始化 AdamWeightDecay 类
    def __init__(
        self,
        learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule] = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        weight_decay_rate: float = 0.0,
        include_in_weight_decay: Optional[List[str]] = None,
        exclude_from_weight_decay: Optional[List[str]] = None,
        name: str = "AdamWeightDecay",
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs)
        # 设置权重衰减率
        self.weight_decay_rate = weight_decay_rate
        # 设置需要包含在权重衰减中的参数
        self._include_in_weight_decay = include_in_weight_decay
        # 设置不需要包含在权重衰减中的参数
        self._exclude_from_weight_decay = exclude_from_weight_decay

    @classmethod
    def from_config(cls, config):
        """从配置中创建带有 WarmUp 自定义对象的优化器。"""
        custom_objects = {"WarmUp": WarmUp}
        return super(AdamWeightDecay, cls).from_config(config, custom_objects=custom_objects)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        # 调用父类的 _prepare_local 方法
        super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype, apply_state)
        # 设置权重衰减率为常量
        apply_state[(var_device, var_dtype)]["weight_decay_rate"] = tf.constant(
            self.weight_decay_rate, name="adam_weight_decay_rate"
        )

    def _decay_weights_op(self, var, learning_rate, apply_state):
        # 检查是否需要对权重进行衰减
        do_decay = self._do_use_weight_decay(var.name)
        if do_decay:
            # 对权重进行衰减操作
            return var.assign_sub(
                learning_rate * var * apply_state[(var.device, var.dtype.base_dtype)]["weight_decay_rate"],
                use_locking=self._use_locking,
            )
        return tf.no_op()

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        # 将梯度和变量分开，然后调用父类的 apply_gradients 方法
        grads, tvars = list(zip(*grads_and_vars))
        return super(AdamWeightDecay, self).apply_gradients(zip(grads, tvars), name=name, **kwargs)

    def _get_lr(self, var_device, var_dtype, apply_state):
        """根据给定状态获取学习率。"""
        if apply_state is None:
            return self._decayed_lr_t[var_dtype], {}

        apply_state = apply_state or {}
        coefficients = apply_state.get((var_device, var_dtype))
        if coefficients is None:
            coefficients = self._fallback_apply_state(var_device, var_dtype)
            apply_state[(var_device, var_dtype)] = coefficients

        return coefficients["lr_t"], {"apply_state": apply_state}

    def _resource_apply_dense(self, grad, var, apply_state=None):
        # 获取学习率和参数，然后进行权重衰减操作
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay, self)._resource_apply_dense(grad, var, **kwargs)
    # 重写父类的稀疏梯度更新方法，应用Adam优化器带权重衰减
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # 获取当前变量的学习率和参数
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        # 计算权重衰减项
        decay = self._decay_weights_op(var, lr_t, apply_state)
        # 确保权重衰减操作在更新梯度之前进行
        with tf.control_dependencies([decay]):
            # 调用父类的稀疏梯度更新方法
            return super(AdamWeightDecay, self)._resource_apply_sparse(grad, var, indices, **kwargs)

    # 获取当前优化器的配置信息
    def get_config(self):
        # 调用父类的配置信息方法
        config = super().get_config()
        # 更新配置信息中的权重衰减率参数
        config.update({"weight_decay_rate": self.weight_decay_rate})
        return config

    # 判断是否对参数进行权重衰减
    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        # 如果权重衰减率为0，则不应用权重衰减
        if self.weight_decay_rate == 0:
            return False

        # 如果参数应该被包含在权重衰减中，则返回True
        if self._include_in_weight_decay:
            for r in self._include_in_weight_decay:
                if re.search(r, param_name) is not None:
                    return True

        # 如果参数应该被排除在权重衰减之外，则返回False
        if self._exclude_from_weight_decay:
            for r in self._exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        # 默认情况下应用权重衰减
        return True
# 定义梯度累加器类，用于在训练中累积梯度以实现梯度累积更新
class GradientAccumulator(object):
    """
    Gradient accumulation utility. When used with a distribution strategy, the accumulator should be called in a
    replica context. Gradients will be accumulated locally on each replica and without synchronization. Users should
    then call `.gradients`, scale the gradients if required, and pass the result to `apply_gradients`.
    """

    # 我们使用 ON_READ 同步策略，这样在赋值时不进行同步。要获取值，我们调用 .value()，它会在当前副本上返回值而不进行同步。

    def __init__(self):
        """Initializes the accumulator."""
        # 存储梯度的列表
        self._gradients = []
        # 累积步数变量
        self._accum_steps = None

    @property
    def step(self):
        """Number of accumulated steps."""
        # 如果累积步数尚未初始化，则初始化
        if self._accum_steps is None:
            self._accum_steps = tf.Variable(
                tf.constant(0, dtype=tf.int64),
                trainable=False,
                synchronization=tf.VariableSynchronization.ON_READ,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            )

        return self._accum_steps.value()

    @property
    def gradients(self):
        """The accumulated gradients on the current replica."""
        # 如果梯度尚未初始化，则抛出异常
        if not self._gradients:
            raise ValueError("The accumulator should be called first to initialize the gradients")
        # 返回当前副本上累积的梯度
        return [gradient.value() if gradient is not None else gradient for gradient in self._gradients]

    def __call__(self, gradients):
        """Accumulates `gradients` on the current replica."""
        # 如果梯度尚未初始化，则创建梯度变量
        if not self._gradients:
            _ = self.step  # Create the step variable.
            self._gradients.extend(
                [
                    tf.Variable(
                        tf.zeros_like(gradient),
                        trainable=False,
                        synchronization=tf.VariableSynchronization.ON_READ,
                        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                    )
                    if gradient is not None
                    else gradient
                    for gradient in gradients
                ]
            )
        # 检查梯度长度是否一致
        if len(gradients) != len(self._gradients):
            raise ValueError(f"Expected {len(self._gradients)} gradients, but got {len(gradients)}")

        # 逐个累积梯度
        for accum_gradient, gradient in zip(self._gradients, gradients):
            if accum_gradient is not None and gradient is not None:
                accum_gradient.assign_add(gradient)

        # 累积步数加一
        self._accum_steps.assign_add(1)
    # 重置当前副本上累积的梯度
    def reset(self):
        # 如果没有累积的梯度，则直接返回
        if not self._gradients:
            return
        # 将累积步数重置为0
        self._accum_steps.assign(0)
        # 遍历所有梯度，并将其重置为与其形状相同的零张量
        for gradient in self._gradients:
            if gradient is not None:
                gradient.assign(tf.zeros_like(gradient))
```