# `numpy-ml\numpy_ml\neural_nets\wrappers\wrappers.py`

```
"""
A collection of objects thats can wrap / otherwise modify arbitrary neural
network layers.
"""

# 导入必要的库
from abc import ABC, abstractmethod
import numpy as np

# 定义一个抽象基类 WrapperBase
class WrapperBase(ABC):
    def __init__(self, wrapped_layer):
        """An abstract base class for all Wrapper instances"""
        # 初始化函数，接受一个 wrapped_layer 参数
        self._base_layer = wrapped_layer
        # 如果 wrapped_layer 中有 _base_layer 属性，则将其赋值给 self._base_layer
        if hasattr(wrapped_layer, "_base_layer"):
            self._base_layer = wrapped_layer._base_layer
        super().__init__()

    @abstractmethod
    def _init_wrapper_params(self):
        # 抽象方法，用于初始化包装器参数
        raise NotImplementedError

    @abstractmethod
    def forward(self, z, **kwargs):
        """Overwritten by inherited class"""
        # 抽象方法，用于前向传播，由子类实现
        raise NotImplementedError

    @abstractmethod
    def backward(self, out, **kwargs):
        """Overwritten by inherited class"""
        # 抽象方法，用于反向传播，由子类实现
        raise NotImplementedError

    @property
    def trainable(self):
        """Whether the base layer is frozen"""
        # 返回基础层是否可训练
        return self._base_layer.trainable

    @property
    def parameters(self):
        """A dictionary of the base layer parameters"""
        # 返回基础层的参数字典
        return self._base_layer.parameters

    @property
    def hyperparameters(self):
        """A dictionary of the base layer's hyperparameters"""
        # 返回基础层的超参数字典
        hp = self._base_layer.hyperparameters
        hpw = self._wrapper_hyperparameters
        if "wrappers" in hp:
            hp["wrappers"].append(hpw)
        else:
            hp["wrappers"] = [hpw]
        return hp

    @property
    def derived_variables(self):
        """
        A dictionary of the intermediate values computed during layer
        training.
        """
        # 返回在层训练期间计算的中间值的字典
        dv = self._base_layer.derived_variables.copy()
        if "wrappers" in dv:
            dv["wrappers"].append(self._wrapper_derived_variables)
        else:
            dv["wrappers"] = [self._wrapper_derived_variables]
        return dv

    @property
    def gradients(self):
        """A dictionary of the current layer parameter gradients."""
        # 返回当前层参数梯度的字典
        return self._base_layer.gradients

    @property
    def act_fn(self):
        """The activation function for the base layer."""
        # 返回基础层的激活函数
        return self._base_layer.act_fn

    @property
    def X(self):
        """The collection of layer inputs."""
        # 返回层输入的集合
        return self._base_layer.X

    def _init_params(self):
        # 初始化参数
        hp = self._wrapper_hyperparameters
        # 如果基础层的超参数中包含"wrappers"，则将当前超参数追加到列表中
        if "wrappers" in self._base_layer.hyperparameters:
            self._base_layer.hyperparameters["wrappers"].append(hp)
        else:
            self._base_layer.hyperparameters["wrappers"] = [hp]

    def freeze(self):
        """
        Freeze the base layer's parameters at their current values so they can
        no longer be updated.
        """
        # 冻结基础层的参数，使其无法再更新
        self._base_layer.freeze()

    def unfreeze(self):
        """Unfreeze the base layer's parameters so they can be updated."""
        # 解冻基础层的参数，使其可以更新
        self._base_layer.freeze()

    def flush_gradients(self):
        """Erase all the wrapper and base layer's derived variables and gradients."""
        # 清除所有包装器和基础层的派生变量和梯度
        assert self.trainable, "Layer is frozen"
        self._base_layer.flush_gradients()

        for k, v in self._wrapper_derived_variables.items():
            self._wrapper_derived_variables[k] = []

    def update(self, lr):
        """
        Update the base layer's parameters using the accrued gradients and
        layer optimizer. Flush all gradients once the update is complete.
        """
        # 使用累积的梯度和层优化器更新基础层的参数。更新完成后清除所有梯度
        assert self.trainable, "Layer is frozen"
        self._base_layer.update(lr)
        self.flush_gradients()

    def _set_wrapper_params(self, pdict):
        # 设置包装器参数
        for k, v in pdict.items():
            if k in self._wrapper_hyperparameters:
                self._wrapper_hyperparameters[k] = v
        return self
    def set_params(self, summary_dict):
        """
        从一个值字典中设置基础层参数。

        Parameters
        ----------
        summary_dict : dict
            一个包含层参数和超参数的字典。如果在 `summary_dict` 中没有包含所需的参数或超参数，
            这个方法将使用当前层的 :meth:`summary` 方法中的值。

        Returns
        -------
        layer : :doc:`Layer <numpy_ml.neural_nets.layers>` object
            新初始化的层。
        """
        return self._base_layer.set_params(summary_dict)

    def summary(self):
        """返回一个包含层参数、超参数和 ID 的字典。"""
        return {
            "layer": self.hyperparameters["layer"],
            "layer_wrappers": [i["wrapper"] for i in self.hyperparameters["wrappers"]],
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters,
        }
class Dropout(WrapperBase):
    def __init__(self, wrapped_layer, p):
        """
        A dropout regularization wrapper.

        Notes
        -----
        During training, a dropout layer zeroes each element of the layer input
        with probability `p` and scales the activation by `1 / (1 - p)` (to reflect
        the fact that on average only `(1 - p) * N` units are active on any
        training pass). At test time, does not adjust elements of the input at
        all (ie., simply computes the identity function).

        Parameters
        ----------
        wrapped_layer : :doc:`Layer <numpy_ml.neural_nets.layers>` instance
            The layer to apply dropout to.
        p : float in [0, 1)
            The dropout propbability during training
        """
        # 调用父类的构造函数
        super().__init__(wrapped_layer)
        # 初始化 dropout 概率
        self.p = p
        # 初始化包装器参数
        self._init_wrapper_params()
        # 初始化参数
        self._init_params()

    def _init_wrapper_params(self):
        # 初始化包装器的派生变量，dropout_mask 用于存储 dropout 掩码
        self._wrapper_derived_variables = {"dropout_mask": []}
        # 初始化包装器的超参数，包括包装器类型和 dropout 概率
        self._wrapper_hyperparameters = {"wrapper": "Dropout", "p": self.p}
    # 计算带有 dropout 的单个 minibatch 的层输出
    def forward(self, X, retain_derived=True):
        """
        Compute the layer output with dropout for a single minibatch.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            Layer input, representing the `n_in`-dimensional features for a
            minibatch of `n_ex` examples.
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through wrt. this input. Default
            is True.

        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)`
            Layer output for each of the `n_ex` examples.
        """
        # 初始化缩放因子为1.0，掩码为全为True的数组
        scaler, mask = 1.0, np.ones(X.shape).astype(bool)
        # 如果该层可训练
        if self.trainable:
            # 计算缩放因子
            scaler = 1.0 / (1.0 - self.p)
            # 生成与输入形状相同的随机掩码
            mask = np.random.rand(*X.shape) >= self.p
            # 对输入应用掩码
            X = mask * X

        # 如果需要保留派生变量
        if retain_derived:
            # 将 dropout 掩码添加到派生变量字典中
            self._wrapper_derived_variables["dropout_mask"].append(mask)

        # 返回经过缩放的输入经过基础层前向传播的结果
        return scaler * self._base_layer.forward(X, retain_derived)
    # 反向传播，从基础层的输出到输入
    def backward(self, dLdy, retain_grads=True):
        """
        Backprop from the base layer's outputs to inputs.

        Parameters
        ----------
        dLdy : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)` or list of arrays
            The gradient(s) of the loss wrt. the layer output(s).
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.

        Returns
        -------
        dLdX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)` or list of arrays
            The gradient of the loss wrt. the layer input(s) `X`.
        """  # noqa: E501
        # 检查层是否可训练，如果不可训练则抛出异常
        assert self.trainable, "Layer is frozen"
        # 将梯度乘以 1/(1-p)，其中 p 是 dropout 概率
        dLdy *= 1.0 / (1.0 - self.p)
        # 调用基础层的反向传播方法，传入梯度和是否保留中间参数梯度
        return self._base_layer.backward(dLdy, retain_grads)
# 初始化层包装器并返回一个包装后的层对象
def init_wrappers(layer, wrappers_list):
    """
    Initialize the layer wrappers in `wrapper_list` and return a wrapped
    `layer` object.

    Parameters
    ----------
    layer : :doc:`Layer <numpy_ml.neural_nets.layers>` instance
        The base layer object to apply the wrappers to.
    wrappers : list of dicts
        A list of parameter dictionaries for a the wrapper objects. The
        wrappers are initialized and applied to the the layer sequentially.

    Returns
    -------
    wrapped_layer : :class:`WrapperBase` instance
        The wrapped layer object
    """
    # 遍历包装器列表
    for wr in wrappers_list:
        # 如果包装器是 "Dropout"
        if wr["wrapper"] == "Dropout":
            # 创建一个 Dropout 包装器对象并应用到层上
            layer = Dropout(layer, 1)._set_wrapper_params(wr)
        else:
            # 如果包装器不是 "Dropout"，则抛出未实现的错误
            raise NotImplementedError("{}".format(wr["wrapper"]))
    # 返回包装后的层对象
    return layer
```