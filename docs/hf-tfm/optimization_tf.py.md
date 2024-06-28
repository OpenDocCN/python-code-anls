# `.\optimization_tf.py`

```py
# 版权声明和许可信息，指明使用许可证 Apache License, Version 2.0
#
# 引入正则表达式模块和类型提示模块
import re
from typing import Callable, List, Optional, Union

# 引入 TensorFlow 库
import tensorflow as tf

# 尝试从 tf_keras.optimizers.legacy 或 tensorflow.keras.optimizers.legacy 导入 Adam 优化器
try:
    from tf_keras.optimizers.legacy import Adam
except (ImportError, ModuleNotFoundError):
    from tensorflow.keras.optimizers.legacy import Adam

# 从 modeling_tf_utils 模块中导入 keras 对象
from .modeling_tf_utils import keras

# 根据 Keras 的随机移动模块位置问题进行条件分支，选择正确的学习率调度模块
if hasattr(keras.optimizers.schedules, "learning_rate_schedule"):
    # 如果存在 learning_rate_schedule，则使用其作为调度模块
    schedules = keras.optimizers.schedules.learning_rate_schedule
else:
    # 否则使用 modeling_tf_utils 模块中的 keras.optimizers.schedules
    schedules = keras.optimizers.schedules

# 定义一个 WarmUp 类，继承自 LearningRateSchedule 类
class WarmUp(schedules.LearningRateSchedule):
    """
    应用于给定学习率衰减计划的热身（warmup）计划。

    Args:
        initial_learning_rate (`float`):
            热身结束后计划的初始学习率（这将是热身结束时的学习率）。
        decay_schedule_fn (`Callable`):
            热身结束后应用于剩余训练的衰减计划函数。
        warmup_steps (`int`):
            训练过程中热身部分的步数。
        power (`float`, *optional*, defaults to 1.0):
            用于多项式热身的幂次数（默认为线性热身）。
        name (`str`, *optional*):
            计划期间返回张量的可选名称前缀。
    """

    def __init__(
        self,
        initial_learning_rate: float,
        decay_schedule_fn: Callable,
        warmup_steps: int,
        power: float = 1.0,
        name: str = None,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name
    # 定义一个调用对象的方法，用于实现学习率的WarmUp策略
    def __call__(self, step):
        # 使用命名空间，如果未提供名称，则使用默认名称"WarmUp"
        with tf.name_scope(self.name or "WarmUp") as name:
            # 将全局步骤转换为浮点数
            global_step_float = tf.cast(step, tf.float32)
            # 将WarmUp步骤数转换为浮点数
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            # 计算当前WarmUp进度的百分比
            warmup_percent_done = global_step_float / warmup_steps_float
            # 根据WarmUp进度百分比计算当前学习率
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)
            # 根据全局步骤是否小于WarmUp步骤数决定返回的学习率
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step - self.warmup_steps),
                name=name,
            )

    # 返回当前对象的配置信息，以字典形式
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }
# 创建一个优化器，其中包含使用热身阶段后的线性衰减学习率计划。

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
    创建一个优化器，并使用热身阶段后的线性衰减学习率计划。

    Args:
        init_lr (`float`):
            热身阶段结束时的初始学习率。
        num_train_steps (`int`):
            总训练步数。
        num_warmup_steps (`int`):
            热身步数。
        min_lr_ratio (`float`, *optional*, defaults to 0):
            线性衰减结束时的最终学习率将为 `init_lr * min_lr_ratio`。
        adam_beta1 (`float`, *optional*, defaults to 0.9):
            Adam优化器中的beta1参数。
        adam_beta2 (`float`, *optional*, defaults to 0.999):
            Adam优化器中的beta2参数。
        adam_epsilon (`float`, *optional*, defaults to 1e-8):
            Adam优化器中的epsilon参数。
        adam_clipnorm (`float`, *optional*, defaults to `None`):
            如果不为`None`，则对每个权重张量的梯度范数进行裁剪。
        adam_global_clipnorm (`float`, *optional*, defaults to `None`):
            如果不为`None`，则将梯度范数裁剪到此值。使用此参数时，梯度范数计算为所有权重张量的向量化结果。
        weight_decay_rate (`float`, *optional*, defaults to 0):
            使用的权重衰减率。
        power (`float`, *optional*, defaults to 1.0):
            PolynomialDecay中使用的幂次数。
        include_in_weight_decay (`List[str]`, *optional*):
            要应用权重衰减的参数名称列表（或正则表达式模式）。如果未传入，则权重衰减将应用于除偏置和层归一化参数之外的所有参数。
    """
    # 实现学习率的线性衰减。
    lr_schedule = schedules.PolynomialDecay(
        initial_learning_rate=init_lr,
        decay_steps=num_train_steps - num_warmup_steps,
        end_learning_rate=init_lr * min_lr_ratio,
        power=power,
    )
    # 如果存在热身步数，则将学习率计划包装在WarmUp对象中。
    if num_warmup_steps:
        lr_schedule = WarmUp(
            initial_learning_rate=init_lr,
            decay_schedule_fn=lr_schedule,
            warmup_steps=num_warmup_steps,
        )
    # 如果权重衰减率大于0，则使用带权重衰减的Adam优化器
    if weight_decay_rate > 0.0:
        optimizer = AdamWeightDecay(
            learning_rate=lr_schedule,                 # 学习率调度器
            weight_decay_rate=weight_decay_rate,       # 权重衰减率
            beta_1=adam_beta1,                         # Adam优化器的beta_1参数
            beta_2=adam_beta2,                         # Adam优化器的beta_2参数
            epsilon=adam_epsilon,                      # Adam优化器的epsilon参数
            clipnorm=adam_clipnorm,                    # Adam优化器的梯度范数裁剪参数
            global_clipnorm=adam_global_clipnorm,      # Adam优化器的全局梯度范数裁剪参数
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],  # 不进行权重衰减的参数列表
            include_in_weight_decay=include_in_weight_decay,  # 需要进行权重衰减的参数列表
        )
    else:
        # 如果权重衰减率为0，则使用普通的Adam优化器
        optimizer = keras.optimizers.Adam(
            learning_rate=lr_schedule,                 # 学习率调度器
            beta_1=adam_beta1,                         # Adam优化器的beta_1参数
            beta_2=adam_beta2,                         # Adam优化器的beta_2参数
            epsilon=adam_epsilon,                      # Adam优化器的epsilon参数
            clipnorm=adam_clipnorm,                    # Adam优化器的梯度范数裁剪参数
            global_clipnorm=adam_global_clipnorm,      # Adam优化器的全局梯度范数裁剪参数
        )
    
    # 我们返回优化器和学习率调度器，以便更好地独立追踪学习率的变化
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
        learning_rate (`Union[float, LearningRateSchedule]`, *optional*, defaults to 0.001):
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

    # 继承自 Adam 优化器的扩展类，支持在梯度上应用 L2 权重衰减和全局梯度裁剪
    def __init__(
        self,
        learning_rate: Union[float, LearningRateSchedule] = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-07,
        amsgrad: bool = False,
        weight_decay_rate: float = 0.0,
        include_in_weight_decay: Optional[List[str]] = None,
        exclude_from_weight_decay: Optional[List[str]] = None,
        name: str = "AdamWeightDecay",
        **kwargs: Dict[str, Any]
    ):
        # 调用父类 Adam 的构造函数
        super().__init__(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
            **kwargs
        )
        # 设置权重衰减率
        self.weight_decay_rate = weight_decay_rate
        # 设置应用权重衰减的参数列表
        self.include_in_weight_decay = include_in_weight_decay
        # 设置不应用权重衰减的参数列表
        self.exclude_from_weight_decay = exclude_from_weight_decay

    # 重写父类的 `apply_gradients` 方法以支持权重衰减
    def apply_gradients(self, grads_and_vars: List[Tuple[tf.Tensor, tf.Variable]], name: Optional[str] = None):
        # 如果设置了权重衰减率，应用权重衰减
        if self.weight_decay_rate > 0.0:
            # 获取当前优化器的学习率
            lr = self._get_hyper("learning_rate")
            # 获取应用了权重衰减的参数列表
            apply_decay = self._should_apply_weight_decay()
            # 遍历梯度和变量的元组
            for grad, var in grads_and_vars:
                if apply_decay and self._do_use_weight_decay(var.name):
                    # 对梯度应用权重衰减
                    grad += self.weight_decay_rate * var
                # 应用梯度到变量上
                self._resource_apply_dense(grad, var, apply_state=True)
        
        # 调用父类的 `apply_gradients` 方法应用梯度
        return super().apply_gradients(grads_and_vars, name=name)

    # 检查是否应用权重衰减到特定参数上
    def _do_use_weight_decay(self, param_name: str) -> bool:
        if self.include_in_weight_decay:
            for pattern in self.include_in_weight_decay:
                if re.search(pattern, param_name):
                    return True
        if self.exclude_from_weight_decay:
            for pattern in self.exclude_from_weight_decay:
                if re.search(pattern, param_name):
                    return False
        return True

    # 检查是否应该应用权重衰减
    def _should_apply_weight_decay(self) -> bool:
        return self.weight_decay_rate > 0.0
    def __init__(
        self,
        learning_rate: Union[float, schedules.LearningRateSchedule] = 0.001,
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
        # 调用父类的构造方法，初始化优化器的参数
        super().__init__(learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs)
        # 设置权重衰减率
        self.weight_decay_rate = weight_decay_rate
        # 指定需要参与权重衰减的变量名列表
        self._include_in_weight_decay = include_in_weight_decay
        # 指定不需要参与权重衰减的变量名列表
        self._exclude_from_weight_decay = exclude_from_weight_decay

    @classmethod
    def from_config(cls, config):
        """从配置中创建优化器，并添加WarmUp自定义对象。"""
        custom_objects = {"WarmUp": WarmUp}
        return super(AdamWeightDecay, cls).from_config(config, custom_objects=custom_objects)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        # 调用父类方法，准备本地变量
        super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype, apply_state)
        # 将权重衰减率作为常量添加到应用状态中
        apply_state[(var_device, var_dtype)]["weight_decay_rate"] = tf.constant(
            self.weight_decay_rate, name="adam_weight_decay_rate"
        )

    def _decay_weights_op(self, var, learning_rate, apply_state):
        # 检查当前变量是否需要进行权重衰减
        do_decay = self._do_use_weight_decay(var.name)
        if do_decay:
            # 如果需要进行权重衰减，则计算并应用权重更新操作
            return var.assign_sub(
                learning_rate * var * apply_state[(var.device, var.dtype.base_dtype)]["weight_decay_rate"],
                use_locking=self._use_locking,
            )
        # 如果不需要进行权重衰减，则返回空操作
        return tf.no_op()

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        # 将梯度和变量分离，然后调用父类的应用梯度方法
        grads, tvars = list(zip(*grads_and_vars))
        return super(AdamWeightDecay, self).apply_gradients(zip(grads, tvars), name=name, **kwargs)

    def _get_lr(self, var_device, var_dtype, apply_state):
        """从状态中获取给定变量的学习率。"""
        if apply_state is None:
            return self._decayed_lr_t[var_dtype], {}

        apply_state = apply_state or {}
        coefficients = apply_state.get((var_device, var_dtype))
        if coefficients is None:
            coefficients = self._fallback_apply_state(var_device, var_dtype)
            apply_state[(var_device, var_dtype)] = coefficients

        return coefficients["lr_t"], {"apply_state": apply_state}

    def _resource_apply_dense(self, grad, var, apply_state=None):
        # 获取学习率和参数，然后执行权重衰减操作
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay, self)._resource_apply_dense(grad, var, **kwargs)
    # 用于在稀疏梯度情况下应用优化器更新，继承自父类的方法
    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # 获取学习率和额外参数配置
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        # 计算权重衰减操作
        decay = self._decay_weights_op(var, lr_t, apply_state)
        # 确保权重衰减操作完成后再执行优化器更新
        with tf.control_dependencies([decay]):
            # 调用父类的稀疏梯度更新方法
            return super(AdamWeightDecay, self)._resource_apply_sparse(grad, var, indices, **kwargs)

    # 获取当前优化器配置信息
    def get_config(self):
        # 调用父类方法获取基础配置
        config = super().get_config()
        # 更新配置信息，添加权重衰减率
        config.update({"weight_decay_rate": self.weight_decay_rate})
        return config

    # 判断是否对给定参数名使用 L2 权重衰减
    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        # 如果权重衰减率为零，不使用权重衰减
        if self.weight_decay_rate == 0:
            return False

        # 如果参数名符合包含在权重衰减中的正则表达式规则，则使用权重衰减
        if self._include_in_weight_decay:
            for r in self._include_in_weight_decay:
                if re.search(r, param_name) is not None:
                    return True

        # 如果参数名符合排除在外的正则表达式规则，则不使用权重衰减
        if self._exclude_from_weight_decay:
            for r in self._exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        # 默认使用权重衰减
        return True
# 从 https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/optimizers/utils.py 中提取

class GradientAccumulator:
    """
    梯度累积工具类。当与分布策略一起使用时，累加器应在副本上下文中调用。
    梯度将在每个副本上本地累积，且不进行同步。用户应调用 `.gradients` 方法获取梯度，
    如果需要，对梯度进行缩放，并将结果传递给 `apply_gradients` 方法。
    """

    # 我们使用 ON_READ 同步策略，这样在赋值时不进行同步。要获取值，我们调用 .value() 方法，
    # 在当前副本上返回值，而不进行同步。

    def __init__(self):
        """初始化累加器。"""
        self._gradients = []  # 存储累积的梯度列表
        self._accum_steps = None  # 累积步数

    @property
    def step(self):
        """累积步数的属性。"""
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
        """当前副本上累积的梯度列表。"""
        if not self._gradients:
            raise ValueError("需要先调用累加器以初始化梯度")
        return [gradient.value() if gradient is not None else gradient for gradient in self._gradients]

    def __call__(self, gradients):
        """在当前副本上累积 `gradients`。"""
        if not self._gradients:
            _ = self.step  # 创建步数变量。
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
        if len(gradients) != len(self._gradients):
            raise ValueError(f"期望 {len(self._gradients)} 个梯度，但实际得到 {len(gradients)} 个")

        for accum_gradient, gradient in zip(self._gradients, gradients):
            if accum_gradient is not None and gradient is not None:
                accum_gradient.assign_add(gradient)

        self._accum_steps.assign_add(1)

    def reset(self):
        """重置当前副本上累积的梯度。"""
        if not self._gradients:
            return
        self._accum_steps.assign(0)
        for gradient in self._gradients:
            if gradient is not None:
                gradient.assign(tf.zeros_like(gradient))
```