# `numpy-ml\numpy_ml\neural_nets\initializers\initializers.py`

```py
# 导入所需的模块和类
import re
from functools import partial
from ast import literal_eval as _eval
import numpy as np

# 导入优化器类
from ..optimizers import OptimizerBase, SGD, AdaGrad, RMSProp, Adam
# 导入激活函数类
from ..activations import (
    ELU,
    GELU,
    SELU,
    ReLU,
    Tanh,
    Affine,
    Sigmoid,
    Identity,
    SoftPlus,
    LeakyReLU,
    Exponential,
    HardSigmoid,
    ActivationBase,
)
# 导入调度器类
from ..schedulers import (
    SchedulerBase,
    ConstantScheduler,
    ExponentialScheduler,
    NoamScheduler,
    KingScheduler,
)

# 导入初始化函数
from ..utils import (
    he_normal,
    he_uniform,
    glorot_normal,
    glorot_uniform,
    truncated_normal,
)

# 定义激活函数初始化器类
class ActivationInitializer(object):
    def __init__(self, param=None):
        """
        A class for initializing activation functions. Valid `param` values
        are:
            (a) ``__str__`` representations of an `ActivationBase` instance
            (b) `ActivationBase` instance

        If `param` is `None`, return the identity function: f(X) = X
        """
        # 初始化激活函数初始化器类，接受一个参数param
        self.param = param

    def __call__(self):
        """Initialize activation function"""
        # 获取参数param
        param = self.param
        # 如果param为None，则返回Identity激活函数
        if param is None:
            act = Identity()
        # 如果param是ActivationBase的实例，则直接使用该实例
        elif isinstance(param, ActivationBase):
            act = param
        # 如果param是字符串，则根据字符串初始化激活函数
        elif isinstance(param, str):
            act = self.init_from_str(param)
        # 如果param不是以上类型，则抛出异常
        else:
            raise ValueError("Unknown activation: {}".format(param))
        return act
    def init_from_str(self, act_str):
        """从字符串`param`初始化激活函数"""
        # 将输入字符串转换为小写
        act_str = act_str.lower()
        # 根据不同的激活函数字符串选择对应的激活函数对象
        if act_str == "relu":
            act_fn = ReLU()
        elif act_str == "tanh":
            act_fn = Tanh()
        elif act_str == "selu":
            act_fn = SELU()
        elif act_str == "sigmoid":
            act_fn = Sigmoid()
        elif act_str == "identity":
            act_fn = Identity()
        elif act_str == "hardsigmoid":
            act_fn = HardSigmoid()
        elif act_str == "softplus":
            act_fn = SoftPlus()
        elif act_str == "exponential":
            act_fn = Exponential()
        # 如果字符串中包含"affine"，则解析出斜率和截距，创建 Affine 激活函数对象
        elif "affine" in act_str:
            r = r"affine\(slope=(.*), intercept=(.*)\)"
            slope, intercept = re.match(r, act_str).groups()
            act_fn = Affine(float(slope), float(intercept))
        # 如果字符串中包含"leaky relu"，则解析出 alpha，创建 LeakyReLU 激活函数对象
        elif "leaky relu" in act_str:
            r = r"leaky relu\(alpha=(.*)\)"
            alpha = re.match(r, act_str).groups()[0]
            act_fn = LeakyReLU(float(alpha))
        # 如果字符串中包含"gelu"，则解析出是否近似，创建 GELU 激活函数对象
        elif "gelu" in act_str:
            r = r"gelu\(approximate=(.*)\)"
            approx = re.match(r, act_str).groups()[0] == "true"
            act_fn = GELU(approximation=approx)
        # 如果字符串中包含"elu"，则解析出 alpha，创建 ELU 激活函数对象
        elif "elu" in act_str:
            r = r"elu\(alpha=(.*)\)"
            approx = re.match(r, act_str).groups()[0]
            act_fn = ELU(alpha=float(alpha))
        else:
            # 如果未识别出激活函数，则抛出异常
            raise ValueError("Unknown activation: {}".format(act_str))
        # 返回选择的激活函数对象
        return act_fn
class SchedulerInitializer(object):
    # 初始化学习率调度器的类。有效的 `param` 值包括：
    # (a) `SchedulerBase` 实例的 `__str__` 表示
    # (b) `SchedulerBase` 实例
    # (c) 参数字典（例如，通过 `LayerBase` 实例中的 `summary` 方法生成）

    # 如果 `param` 为 `None`，返回学习率为 `lr` 的 ConstantScheduler
    def __init__(self, param=None, lr=None):
        if all([lr is None, param is None]):
            raise ValueError("lr and param cannot both be `None`")

        # 初始化学习率和参数
        self.lr = lr
        self.param = param

    # 初始化调度器
    def __call__(self):
        param = self.param
        if param is None:
            scheduler = ConstantScheduler(self.lr)
        elif isinstance(param, SchedulerBase):
            scheduler = param
        elif isinstance(param, str):
            scheduler = self.init_from_str()
        elif isinstance(param, dict):
            scheduler = self.init_from_dict()
        return scheduler

    # 从字符串参数初始化调度器
    def init_from_str(self):
        r = r"([a-zA-Z]*)=([^,)]*)"
        sch_str = self.param.lower()
        kwargs = {i: _eval(j) for i, j in re.findall(r, sch_str)}

        if "constant" in sch_str:
            scheduler = ConstantScheduler(**kwargs)
        elif "exponential" in sch_str:
            scheduler = ExponentialScheduler(**kwargs)
        elif "noam" in sch_str:
            scheduler = NoamScheduler(**kwargs)
        elif "king" in sch_str:
            scheduler = KingScheduler(**kwargs)
        else:
            raise NotImplementedError("{}".format(sch_str))
        return scheduler
    # 从参数字典中初始化调度器
    def init_from_dict(self):
        """Initialize scheduler from the param dictionary"""
        # 获取参数字典
        S = self.param
        # 获取超参数字典
        sc = S["hyperparameters"] if "hyperparameters" in S else None

        # 如果超参数字典为空，则抛出数值错误异常
        if sc is None:
            raise ValueError("Must have `hyperparameters` key: {}".format(S))

        # 根据超参数字典中的 id 创建不同类型的调度器对象
        if sc and sc["id"] == "ConstantScheduler":
            scheduler = ConstantScheduler()
        elif sc and sc["id"] == "ExponentialScheduler":
            scheduler = ExponentialScheduler()
        elif sc and sc["id"] == "NoamScheduler":
            scheduler = NoamScheduler()
        # 如果超参数字典中的 id 不在已知类型中，则抛出未实现错误异常
        elif sc:
            raise NotImplementedError("{}".format(sc["id"]))
        
        # 设置调度器的参数
        scheduler.set_params(sc)
        # 返回初始化后的调度器对象
        return scheduler
class OptimizerInitializer(object):
    # 定义一个初始化优化器的类
    def __init__(self, param=None):
        """
        A class for initializing optimizers. Valid `param` values are:
            (a) __str__ representations of `OptimizerBase` instances
            (b) `OptimizerBase` instances
            (c) Parameter dicts (e.g., as produced via the `summary` method in
                `LayerBase` instances)

        If `param` is `None`, return the SGD optimizer with default parameters.
        """
        # 初始化方法，接受一个参数param，可以是字符串、OptimizerBase实例或参数字典
        self.param = param

    def __call__(self):
        """Initialize the optimizer"""
        # 调用实例时初始化优化器
        param = self.param
        if param is None:
            opt = SGD()
        elif isinstance(param, OptimizerBase):
            opt = param
        elif isinstance(param, str):
            opt = self.init_from_str()
        elif isinstance(param, dict):
            opt = self.init_from_dict()
        return opt

    def init_from_str(self):
        """Initialize optimizer from the `param` string"""
        # 从字符串param初始化优化器
        r = r"([a-zA-Z]*)=([^,)]*)"
        opt_str = self.param.lower()
        kwargs = {i: _eval(j) for i, j in re.findall(r, opt_str)}
        if "sgd" in opt_str:
            optimizer = SGD(**kwargs)
        elif "adagrad" in opt_str:
            optimizer = AdaGrad(**kwargs)
        elif "rmsprop" in opt_str:
            optimizer = RMSProp(**kwargs)
        elif "adam" in opt_str:
            optimizer = Adam(**kwargs)
        else:
            raise NotImplementedError("{}".format(opt_str))
        return optimizer
    # 从参数字典中初始化优化器
    def init_from_dict(self):
        """Initialize optimizer from the `param` dictonary"""
        # 获取参数字典
        D = self.param
        # 如果参数字典中包含`cache`键，则将其赋值给cc，否则为None
        cc = D["cache"] if "cache" in D else None
        # 如果参数字典中包含`hyperparameters`键，则将其赋值给op，否则为None
        op = D["hyperparameters"] if "hyperparameters" in D else None

        # 如果op为None，则抛出数值错误异常
        if op is None:
            raise ValueError("`param` dictionary has no `hyperparemeters` key")

        # 根据op中的"id"字段选择相应的优化器
        if op and op["id"] == "SGD":
            optimizer = SGD()
        elif op and op["id"] == "RMSProp":
            optimizer = RMSProp()
        elif op and op["id"] == "AdaGrad":
            optimizer = AdaGrad()
        elif op and op["id"] == "Adam":
            optimizer = Adam()
        # 如果op存在但未匹配到任何优化器，则抛出未实现错误
        elif op:
            raise NotImplementedError("{}".format(op["id"]))
        # 设置优化器的参数
        optimizer.set_params(op, cc)
        # 返回初始化后的优化器
        return optimizer
class WeightInitializer(object):
    # 定义权重初始化器类
    def __init__(self, act_fn_str, mode="glorot_uniform"):
        """
        A factory for weight initializers.

        Parameters
        ----------
        act_fn_str : str
            The string representation for the layer activation function
        mode : str (default: 'glorot_uniform')
            The weight initialization strategy. Valid entries are {"he_normal",
            "he_uniform", "glorot_normal", glorot_uniform", "std_normal",
            "trunc_normal"}
        """
        # 初始化函数，接受激活函数字符串和初始化模式作为参数
        if mode not in [
            "he_normal",
            "he_uniform",
            "glorot_normal",
            "glorot_uniform",
            "std_normal",
            "trunc_normal",
        ]:
            # 如果初始化模式不在预定义的列表中，则抛出异常
            raise ValueError("Unrecognize initialization mode: {}".format(mode))

        self.mode = mode
        self.act_fn = act_fn_str

        if mode == "glorot_uniform":
            self._fn = glorot_uniform
        elif mode == "glorot_normal":
            self._fn = glorot_normal
        elif mode == "he_uniform":
            self._fn = he_uniform
        elif mode == "he_normal":
            self._fn = he_normal
        elif mode == "std_normal":
            self._fn = np.random.randn
        elif mode == "trunc_normal":
            self._fn = partial(truncated_normal, mean=0, std=1)

    def __call__(self, weight_shape):
        """Initialize weights according to the specified strategy"""
        # 根据指定的策略初始化权重
        if "glorot" in self.mode:
            gain = self._calc_glorot_gain()
            W = self._fn(weight_shape, gain)
        elif self.mode == "std_normal":
            W = self._fn(*weight_shape)
        else:
            W = self._fn(weight_shape)
        return W
    # 计算 Glorot 初始化的增益值
    def _calc_glorot_gain(self):
        """
        从以下链接获取数值:
        https://pytorch.org/docs/stable/nn.html?#torch.nn.init.calculate_gain
        """
        # 初始化增益值为 1.0
        gain = 1.0
        # 获取激活函数的字符串表示并转换为小写
        act_str = self.act_fn.lower()
        # 如果激活函数是 tanh，则设置增益值为 5/3
        if act_str == "tanh":
            gain = 5.0 / 3.0
        # 如果激活函数是 relu，则设置增益值为根号2
        elif act_str == "relu":
            gain = np.sqrt(2)
        # 如果激活函数是 leaky relu，则根据 alpha 计算增益值
        elif "leaky relu" in act_str:
            # 使用正则表达式提取 alpha 值
            r = r"leaky relu\(alpha=(.*)\)"
            alpha = re.match(r, act_str).groups()[0]
            # 根据 alpha 计算增益值
            gain = np.sqrt(2 / 1 + float(alpha) ** 2)
        # 返回计算得到的增益值
        return gain
```