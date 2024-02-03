# `numpy-ml\numpy_ml\neural_nets\layers\layers.py`

```
"""A collection of composable layer objects for building neural networks"""
# 导入必要的库和模块
from abc import ABC, abstractmethod

import numpy as np

# 导入自定义的包装器和初始化器
from ..wrappers import init_wrappers, Dropout

from ..initializers import (
    WeightInitializer,
    OptimizerInitializer,
    ActivationInitializer,
)

from ..utils import (
    pad1D,
    pad2D,
    conv1D,
    conv2D,
    im2col,
    col2im,
    dilate,
    deconv2D_naive,
    calc_pad_dims_2D,
)

# 定义抽象基类 LayerBase，所有神经网络层都继承自该类
class LayerBase(ABC):
    def __init__(self, optimizer=None):
        """An abstract base class inherited by all neural network layers"""
        # 初始化实例变量
        self.X = []
        self.act_fn = None
        self.trainable = True
        self.optimizer = OptimizerInitializer(optimizer)()

        self.gradients = {}
        self.parameters = {}
        self.derived_variables = {}

        super().__init__()

    @abstractmethod
    def _init_params(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, z, **kwargs):
        """Perform a forward pass through the layer"""
        raise NotImplementedError

    @abstractmethod
    def backward(self, out, **kwargs):
        """Perform a backward pass through the layer"""
        raise NotImplementedError

    def freeze(self):
        """
        Freeze the layer parameters at their current values so they can no
        longer be updated.
        """
        # 冻结层的参数，使其不可更新
        self.trainable = False

    def unfreeze(self):
        """Unfreeze the layer parameters so they can be updated."""
        # 解冻层的参数，使其可以更新
        self.trainable = True

    def flush_gradients(self):
        """Erase all the layer's derived variables and gradients."""
        # 清空层的所有派生变量和梯度
        assert self.trainable, "Layer is frozen"
        self.X = []
        for k, v in self.derived_variables.items():
            self.derived_variables[k] = []

        for k, v in self.gradients.items():
            self.gradients[k] = np.zeros_like(v)
    def update(self, cur_loss=None):
        """
        Update the layer parameters using the accrued gradients and layer
        optimizer. Flush all gradients once the update is complete.
        """
        # 确保该层可训练，如果不可训练则抛出异常
        assert self.trainable, "Layer is frozen"
        # 使用优化器更新层参数
        self.optimizer.step()
        # 遍历梯度字典，更新参数
        for k, v in self.gradients.items():
            if k in self.parameters:
                self.parameters[k] = self.optimizer(self.parameters[k], v, k, cur_loss)
        # 清空梯度
        self.flush_gradients()

    def set_params(self, summary_dict):
        """
        Set the layer parameters from a dictionary of values.

        Parameters
        ----------
        summary_dict : dict
            A dictionary of layer parameters and hyperparameters. If a required
            parameter or hyperparameter is not included within `summary_dict`,
            this method will use the value in the current layer's
            :meth:`summary` method.

        Returns
        -------
        layer : :doc:`Layer <numpy_ml.neural_nets.layers>` object
            The newly-initialized layer.
        """
        # 将当前层和参数字典赋值给变量
        layer, sd = self, summary_dict

        # 将`parameters`和`hyperparameters`嵌套字典合并为一个字典
        flatten_keys = ["parameters", "hyperparameters"]
        for k in flatten_keys:
            if k in sd:
                entry = sd[k]
                sd.update(entry)
                del sd[k]

        # 遍历参数字典，根据键值更新层参数
        for k, v in sd.items():
            if k in self.parameters:
                layer.parameters[k] = v
            if k in self.hyperparameters:
                if k == "act_fn":
                    # 初始化激活函数
                    layer.act_fn = ActivationInitializer(v)()
                elif k == "optimizer":
                    # 初始化优化器
                    layer.optimizer = OptimizerInitializer(sd[k])()
                elif k == "wrappers":
                    # 初始化包装器
                    layer = init_wrappers(layer, sd[k])
                elif k not in ["wrappers", "optimizer"]:
                    # 设置其他属性
                    setattr(layer, k, v)
        return layer
    # 定义一个方法用于返回包含层参数、超参数和ID的字典
    def summary(self):
        """Return a dict of the layer parameters, hyperparameters, and ID."""
        # 返回包含层、参数、超参数的字典
        return {
            "layer": self.hyperparameters["layer"],
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters,
        }
class DotProductAttention(LayerBase):
    # 定义 DotProductAttention 类，继承自 LayerBase 类

    def _init_params(self):
        # 初始化参数方法
        self.softmax = Dropout(Softmax(), self.dropout_p)
        # 创建 Softmax 层，并添加 Dropout，存储到 self.softmax
        smdv = self.softmax.derived_variables
        # 获取 softmax 层的派生变量

        self.gradients = {}
        # 初始化梯度字典
        self.parameters = {}
        # 初始化参数字典
        self.derived_variables = {
            "attention_weights": [],
            "dropout_mask": smdv["wrappers"][0]["dropout_mask"],
        }
        # 初始化派生变量字典

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        # 返回包含层超参数的字典
        return {
            "layer": "DotProductAttention",
            "init": self.init,
            "scale": self.scale,
            "dropout_p": self.dropout_p,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }
        # 返回超参数字典

    def freeze(self):
        """
        Freeze the layer parameters at their current values so they can no
        longer be updated.
        """
        # 冻结层参数，使其不能再更新
        self.trainable = False
        # 设置 trainable 属性为 False
        self.softmax.freeze()
        # 冻结 softmax 层

    def unfreeze(self):
        """Unfreeze the layer parameters so they can be updated."""
        # 解冻层参数，使其可以更新
        self.trainable = True
        # 设置 trainable 属性为 True
        self.softmax.unfreeze()
        # 解冻 softmax 层

    def _fwd(self, Q, K, V):
        """Actual computation of forward pass"""
        # 实际的前向传播计算
        scale = 1 / np.sqrt(Q.shape[-1]) if self.scale else 1
        # 计算缩放因子
        scores = Q @ K.swapaxes(-2, -1) * scale  # attention scores
        # 计算注意力分数
        weights = self.softmax.forward(scores)  # attention weights
        # 计算注意力权重
        Y = weights @ V
        # 计算加权后的输出
        return Y, weights
        # 返回输出和注意力权重
    def backward(self, dLdy, retain_grads=True):
        r"""
        Backprop from layer outputs to inputs.

        Parameters
        ----------
        dLdY : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, *, d_v)`
            The gradient of the loss wrt. the layer output `Y`
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.

        Returns
        -------
        dQ : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, *, d_k)` or list of arrays
            The gradient of the loss wrt. the layer query matrix/matrices `Q`.
        dK : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, *, d_k)` or list of arrays
            The gradient of the loss wrt. the layer key matrix/matrices `K`.
        dV : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, *, d_v)` or list of arrays
            The gradient of the loss wrt. the layer value matrix/matrices `V`.
        """  # noqa: E501
        assert self.trainable, "Layer is frozen"
        # 如果输入的梯度不是列表形式，则转换为列表
        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        # 初始化存储梯度的列表
        dQ, dK, dV = [], [], []
        # 获取注意力权重
        weights = self.derived_variables["attention_weights"]
        # 遍历每个输入梯度、查询矩阵、键矩阵和值矩阵以及注意力权重
        for dy, (q, k, v), w in zip(dLdy, self.X, weights):
            # 调用内部方法计算反向传播梯度
            dq, dk, dv = self._bwd(dy, q, k, v, w)
            # 将计算得到的梯度添加到对应的列表中
            dQ.append(dq)
            dK.append(dk)
            dV.append(dv)

        # 如果只有一个输入，则将梯度列表转换为单个数组
        if len(self.X) == 1:
            dQ, dK, dV = dQ[0], dK[0], dV[0]

        # 返回计算得到的梯度
        return dQ, dK, dV
    # 计算损失对 q、k、v 的梯度
    def _bwd(self, dy, q, k, v, weights):
        # 获取 k 的维度
        d_k = k.shape[-1]
        # 如果开启了缩放，计算缩放因子
        scale = 1 / np.sqrt(d_k) if self.scale else 1

        # 计算对 v 的梯度
        dV = weights.swapaxes(-2, -1) @ dy
        # 计算对权重的梯度
        dWeights = dy @ v.swapaxes(-2, -1)
        # 计算对分数的梯度
        dScores = self.softmax.backward(dWeights)
        # 计算对 q 的梯度
        dQ = dScores @ k * scale
        # 计算对 k 的梯度
        dK = dScores.swapaxes(-2, -1) @ q * scale
        # 返回 q、k、v 的梯度
        return dQ, dK, dV
class RBM(LayerBase):
    def __init__(self, n_out, K=1, init="glorot_uniform", optimizer=None):
        """
        A Restricted Boltzmann machine with Bernoulli visible and hidden units.

        Parameters
        ----------
        n_out : int
            The number of output dimensions/units.
        K : int
            The number of contrastive divergence steps to run before computing
            a single gradient update. Default is 1.
        init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
            The weight initialization strategy. Default is `'glorot_uniform'`.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the :class:`SGD
            <numpy_ml.neural_nets.optimizers.SGD>` optimizer with
            default parameters. Default is None.

        Attributes
        ----------
        X : list
            Unused
        gradients : dict
            Dictionary of loss gradients with regard to the layer parameters
        parameters : dict
            Dictionary of layer parameters
        hyperparameters : dict
            Dictionary of layer hyperparameters
        derived_variables : dict
            Dictionary of any intermediate values computed during
            forward/backward propagation.
        """  # noqa: E501
        # 调用父类的构造函数，初始化优化器
        super().__init__(optimizer)

        # 设置 CD-K的值
        self.K = K  # CD-K
        # 设置权重初始化策略
        self.init = init
        self.n_in = None
        # 设置输出维度
        self.n_out = n_out
        # 初始化标志
        self.is_initialized = False
        # 初始化可见单元的激活函数为Sigmoid
        self.act_fn_V = ActivationInitializer("Sigmoid")()
        # 初始化隐藏单元的激活函数为Sigmoid
        self.act_fn_H = ActivationInitializer("Sigmoid")()
        # 初始化参数字典
        self.parameters = {"W": None, "b_in": None, "b_out": None}

        # 初始化参数
        self._init_params()
    # 初始化参数，包括权重、偏置和梯度
    def _init_params(self):
        # 使用指定的激活函数和初始化方式初始化权重
        init_weights = WeightInitializer(str(self.act_fn_V), mode=self.init)

        # 初始化输入层偏置
        b_in = np.zeros((1, self.n_in))
        # 初始化输出层偏置
        b_out = np.zeros((1, self.n_out))
        # 初始化权重矩阵
        W = init_weights((self.n_in, self.n_out))

        # 存储参数的字典
        self.parameters = {"W": W, "b_in": b_in, "b_out": b_out}

        # 存储梯度的字典
        self.gradients = {
            "W": np.zeros_like(W),
            "b_in": np.zeros_like(b_in),
            "b_out": np.zeros_like(b_out),
        }

        # 存储派生变量的字典
        self.derived_variables = {
            "V": None,
            "p_H": None,
            "p_V_prime": None,
            "p_H_prime": None,
            "positive_grad": None,
            "negative_grad": None,
        }
        # 标记参数是否已经初始化
        self.is_initialized = True

    @property
    def hyperparameters(self):
        """返回包含层超参数的字典。"""
        return {
            "layer": "RBM",
            "K": self.K,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "init": self.init,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameterse,
            },
        }

    def CD_update(self, X):
        """
        使用可见输入 `X` 作为 Gibbs 采样的起点，执行单次对比散度-`k`训练更新。

        参数
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            层输入，表示 `n_ex` 个示例的 `n_in` 维特征的小批量数据。X 中的每个特征理想情况下应为二值，尽管也可以在 (0, 1) 范围内训练实值特征（例如，灰度图像）。
        """
        # 前向传播
        self.forward(X)
        # 反向传播
        self.backward()
    # 定义反向传播函数，用于通过对比散度方程对层参数执行梯度更新
    def backward(self, retain_grads=True, *args):
        """
        Perform a gradient update on the layer parameters via the contrastive
        divergence equations.

        Parameters
        ----------
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.
        """
        # 从派生变量中获取需要的变量
        V = self.derived_variables["V"]
        p_H = self.derived_variables["p_H"]
        p_V_prime = self.derived_variables["p_V_prime"]
        p_H_prime = self.derived_variables["p_H_prime"]
        positive_grad = self.derived_variables["positive_grad"]
        negative_grad = self.derived_variables["negative_grad"]

        # 如果需要保留梯度
        if retain_grads:
            # 计算输入层偏置的梯度
            self.gradients["b_in"] = V - p_V_prime
            # 计算输出层偏置的梯度
            self.gradients["b_out"] = p_H - p_H_prime
            # 计算权重矩阵的梯度
            self.gradients["W"] = positive_grad - negative_grad
    # 通过运行经过训练的 Gibbs 采样器进行 CD-k 的 `n_steps` 次重构输入 `X`
    def reconstruct(self, X, n_steps=10, return_prob=False):
        """
        Reconstruct an input `X` by running the trained Gibbs sampler for
        `n_steps`-worth of CD-`k`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            Layer input, representing the `n_in`-dimensional features for a
            minibatch of `n_ex` examples. Each feature in `X` should ideally be
            binary-valued, although it is possible to also train on real-valued
            features ranging between (0, 1) (e.g., grayscale images). If `X` has
            missing values, it may be sufficient to mark them with random
            entries and allow the reconstruction to impute them.
        n_steps : int
            The number of Gibbs sampling steps to perform when generating the
            reconstruction. Default is 10.
        return_prob : bool
            Whether to return the real-valued feature probabilities for the
            reconstruction or the binary samples. Default is False.

        Returns
        -------
        V : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_ch)`
            The reconstruction (or feature probabilities if `return_prob` is
            true) of the visual input `X` after running the Gibbs sampler for
            `n_steps`.
        """
        # 运行前向传播，使用 `n_steps` 次采样
        self.forward(X, K=n_steps)
        # 获取派生变量中的 `p_V_prime`
        p_V_prime = self.derived_variables["p_V_prime"]

        # 忽略在这次重构期间产生的梯度
        self.flush_gradients()

        # 如果 `return_prob` 为 False，则对 V_prime 重构进行采样
        V = p_V_prime
        if not return_prob:
            # 生成与 `p_V_prime` 形状相同的随机数组，根据概率 `p_V_prime` 进行二值化
            V = (np.random.rand(*p_V_prime.shape) <= p_V_prime).astype(float)
        return V
# 定义一个名为 Add 的层，用于计算输入的和，并可选择应用非线性激活函数
class Add(LayerBase):
    def __init__(self, act_fn=None, optimizer=None):
        """
        初始化 Add 层对象

        Parameters
        ----------
        act_fn : str, :doc:`Activation <numpy_ml.neural_nets.activations>` object, or None
            用于计算最终输出的逐元素非线性激活函数。如果为 None，则使用恒等函数 :math:`f(x) = x`。默认为 None。
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            在执行梯度更新时使用的优化策略。如果为 None，则使用默认参数的 :class:`SGD <numpy_ml.neural_nets.optimizers.SGD>` 优化器。默认为 None。

        Attributes
        ----------
        X : list
            自上次调用 :meth:`update <numpy_ml.neural_nets.LayerBase.update>` 方法以来传递给 :meth:`forward <numpy_ml.neural_nets.LayerBase.forward>` 方法的输入的运行列表。仅在设置 `retain_derived` 参数为 True 时更新。
        gradients : dict
            未使用
        parameters : dict
            未使用
        hyperparameters : dict
            层超参数的字典
        derived_variables : dict
            在前向/反向传播期间计算的任何中间值的字典
        """  # noqa: E501
        # 调用父类的初始化方法
        super().__init__(optimizer)
        # 初始化激活函数
        self.act_fn = ActivationInitializer(act_fn)()
        # 初始化参数
        self._init_params()

    def _init_params(self):
        # 初始化梯度字典
        self.gradients = {}
        # 初始化参数字典
        self.parameters = {}
        # 初始化派生变量字典
        self.derived_variables = {"sum": []}

    @property
    # 返回包含层超参数的字典
    def hyperparameters(self):
        return {
            "layer": "Sum",  # 层类型为Sum
            "act_fn": str(self.act_fn),  # 激活函数
            "optimizer": {
                "cache": self.optimizer.cache,  # 优化器缓存
                "hyperparameters": self.optimizer.hyperparameters,  # 优化器超参数
            },
        }

    # 计算单个小批量的层输出
    def forward(self, X, retain_derived=True):
        r"""
        Compute the layer output on a single minibatch.

        Parameters
        ----------
        X : list of length `n_inputs`
            A list of tensors, all of the same shape.
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through wrt. this input. Default
            is True.

        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, *)`
            The sum over the `n_ex` examples.
        """
        # 复制第一个输入张量作为初始输出
        out = X[0].copy()
        # 对所有输入张量求和
        for i in range(1, len(X)):
            out += X[i]
        # 如果需要保留派生变量，则将输入张量和求和结果添加到相应列表中
        if retain_derived:
            self.X.append(X)
            self.derived_variables["sum"].append(out)
        # 返回激活函数应用于求和结果的输出
        return self.act_fn(out)
    # 从层输出向输入反向传播，计算损失对输入的梯度
    def backward(self, dLdY, retain_grads=True):
        r"""
        Backprop from layer outputs to inputs.

        Parameters
        ----------
        dLdY : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, *)`
            The gradient of the loss wrt. the layer output `Y`.
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.

        Returns
        -------
        dX : list of length `n_inputs`
            The gradient of the loss wrt. each input in `X`.
        """
        # 如果梯度不是列表，则转换为列表
        if not isinstance(dLdY, list):
            dLdY = [dLdY]

        # 获取输入 X 和派生变量 sum
        X = self.X
        _sum = self.derived_variables["sum"]
        # 计算每个输入的梯度
        grads = [self._bwd(dy, x, ss) for dy, x, ss in zip(dLdY, X, _sum)]
        # 如果输入只有一个，则返回第一个梯度，否则返回所有梯度
        return grads[0] if len(X) == 1 else grads

    # 实际计算损失对每个输入的梯度
    def _bwd(self, dLdY, X, _sum):
        """Actual computation of gradient of the loss wrt. each input"""
        # 计算每个输入的梯度
        grads = [dLdY * self.act_fn.grad(_sum) for _ in X]
        return grads
class Multiply(LayerBase):
    def __init__(self, act_fn=None, optimizer=None):
        """
        A multiplication layer that returns the *elementwise* product of its
        inputs, passed through an optional nonlinearity.

        Parameters
        ----------
        act_fn : str, :doc:`Activation <numpy_ml.neural_nets.activations>` object, or None
            The element-wise output nonlinearity used in computing the final
            output. If None, use the identity function :math:`f(x) = x`.
            Default is None.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the :class:`SGD
            <numpy_ml.neural_nets.optimizers.SGD>` optimizer with
            default parameters. Default is None.

        Attributes
        ----------
        X : list
            Running list of inputs to the :meth:`forward <numpy_ml.neural_nets.LayerBase.forward>` method since the last call to :meth:`update <numpy_ml.neural_nets.LayerBase.update>`. Only updated if the `retain_derived` argument was set to True.
        gradients : dict
            Unused
        parameters : dict
            Unused
        hyperparameters : dict
            Dictionary of layer hyperparameters
        derived_variables : dict
            Dictionary of any intermediate values computed during
            forward/backward propagation.
        """  # noqa: E501
        # 调用父类的初始化方法，传入优化器参数
        super().__init__(optimizer)
        # 初始化激活函数为传入的参数act_fn
        self.act_fn = ActivationInitializer(act_fn)()
        # 初始化参数
        self._init_params()

    def _init_params(self):
        # 初始化梯度为空字典
        self.gradients = {}
        # 初始化参数为空字典
        self.parameters = {}
        # 初始化派生变量为包含"product"键的空列表
        self.derived_variables = {"product": []}

    @property
    # 返回包含层超参数的字典
    def hyperparameters(self):
        return {
            "layer": "Multiply",  # 层类型为 Multiply
            "act_fn": str(self.act_fn),  # 激活函数类型
            "optimizer": {
                "cache": self.optimizer.cache,  # 优化器缓存
                "hyperparameters": self.optimizer.hyperparameters,  # 优化器超参数
            },
        }

    # 计算单个小批量的层输出
    def forward(self, X, retain_derived=True):
        """
        Compute the layer output on a single minibatch.

        Parameters
        ----------
        X : list of length `n_inputs`
            A list of tensors, all of the same shape.
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through wrt. this input. Default
            is True.

        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, *)`
            The product over the `n_ex` examples.
        """  # noqa: E501
        # 复制第一个输入张量作为初始输出
        out = X[0].copy()
        # 对于剩余的输入张量，依次与输出张量相乘
        for i in range(1, len(X)):
            out *= X[i]
        # 如果需要保留派生变量，则将输入张量和乘积结果添加到相应列表中
        if retain_derived:
            self.X.append(X)
            self.derived_variables["product"].append(out)
        # 返回激活函数应用于乘积结果的输出
        return self.act_fn(out)
    # 从神经网络层的输出向输入反向传播，计算损失对输入的梯度
    def backward(self, dLdY, retain_grads=True):
        r"""
        Backprop from layer outputs to inputs.

        Parameters
        ----------
        dLdY : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, *)`
            The gradient of the loss wrt. the layer output `Y`.
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.

        Returns
        -------
        dX : list of length `n_inputs`
            The gradient of the loss wrt. each input in `X`.
        """
        # 如果梯度不是列表，则转换为列表形式
        if not isinstance(dLdY, list):
            dLdY = [dLdY]

        # 获取输入数据 X 和派生变量 product
        X = self.X
        _prod = self.derived_variables["product"]
        # 计算每个输入的梯度
        grads = [self._bwd(dy, x, pr) for dy, x, pr in zip(dLdY, X, _prod)]
        # 如果输入只有一个，则返回第一个梯度，否则返回所有梯度
        return grads[0] if len(X) == 1 else grads

    # 实际计算损失对每个输入的梯度
    def _bwd(self, dLdY, X, prod):
        """Actual computation of gradient of loss wrt. each input"""
        # 计算每个输入的梯度
        grads = [dLdY * self.act_fn.grad(prod)] * len(X)
        # 遍历每个输入，计算梯度
        for i, x in enumerate(X):
            grads = [g * x if j != i else g for j, g in enumerate(grads)]
        return grads
# 定义一个Flatten类，继承自LayerBase类
class Flatten(LayerBase):
    def __init__(self, keep_dim="first", optimizer=None):
        """
        Flatten a multidimensional input into a 2D matrix.

        Parameters
        ----------
        keep_dim : {'first', 'last', -1}
            The dimension of the original input to retain. Typically used for
            retaining the minibatch dimension.. If -1, flatten all dimensions.
            Default is 'first'.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the :class:`SGD
            <numpy_ml.neural_nets.optimizers.SGD>` optimizer with
            default parameters. Default is None.

        Attributes
        ----------
        X : list
            Unused
        gradients : dict
            Unused
        parameters : dict
            Unused
        hyperparameters : dict
            Dictionary of layer hyperparameters
        derived_variables : dict
            Dictionary of any intermediate values computed during
            forward/backward propagation.
        """  # noqa: E501
        # 调用父类的构造函数
        super().__init__(optimizer)

        # 初始化keep_dim属性
        self.keep_dim = keep_dim
        # 调用_init_params方法
        self._init_params()

    # 初始化参数方法
    def _init_params(self):
        # 初始化梯度字典
        self.gradients = {}
        # 初始化参数字典
        self.parameters = {}
        # 初始化派生变量字典
        self.derived_variables = {"in_dims": []}

    # 定义hyperparameters属性，返回包含层超参数的字典
    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        return {
            "layer": "Flatten",
            "keep_dim": self.keep_dim,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }
    # 定义一个方法，用于计算单个小批量的层输出
    def forward(self, X, retain_derived=True):
        # 如果需要保留在前向传播期间计算的变量，则将输入数据的维度添加到派生变量中
        if retain_derived:
            self.derived_variables["in_dims"].append(X.shape)
        # 如果 keep_dim 为 -1，则将输入数据展平并重新形状为 (1, -1)
        if self.keep_dim == -1:
            return X.flatten().reshape(1, -1)
        # 根据 keep_dim 的值确定输出数据的形状
        rs = (X.shape[0], -1) if self.keep_dim == "first" else (-1, X.shape[-1])
        # 根据确定的形状重新调整输入数据的形状并返回
        return X.reshape(*rs)
    # 定义反向传播函数，从层输出向输入传播梯度
    def backward(self, dLdy, retain_grads=True):
        r"""
        Backprop from layer outputs to inputs.

        Parameters
        ----------
        dLdY : :py:class:`ndarray <numpy.ndarray>` of shape `(*out_dims)`
            The gradient of the loss wrt. the layer output `Y`.
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.

        Returns
        -------
        dX : :py:class:`ndarray <numpy.ndarray>` of shape `(*in_dims)` or list of arrays
            The gradient of the loss wrt. the layer input(s) `X`.
        """  # noqa: E501
        # 如果梯度不是列表，则转换为列表
        if not isinstance(dLdy, list):
            dLdy = [dLdy]
        # 获取输入维度
        in_dims = self.derived_variables["in_dims"]
        # 将梯度重塑为输入维度形状
        out = [dy.reshape(*dims) for dy, dims in zip(dLdy, in_dims)]
        # 如果只有一个梯度，则返回第一个梯度，否则返回所有梯度
        return out[0] if len(dLdy) == 1 else out
# 正则化层
class BatchNorm2D(LayerBase):
    # 初始化参数
    def _init_params(self):
        # 随机生成缩放器
        scaler = np.random.rand(self.in_ch)
        # 初始化截距为零
        intercept = np.zeros(self.in_ch)

        # 初始化运行均值和标准差为0和1
        running_mean = np.zeros(self.in_ch)
        running_var = np.ones(self.in_ch)

        # 参数字典包含缩放器、截距、运行标准差和均值
        self.parameters = {
            "scaler": scaler,
            "intercept": intercept,
            "running_var": running_var,
            "running_mean": running_mean,
        }

        # 梯度字典包含缩放器和截距
        self.gradients = {
            "scaler": np.zeros_like(scaler),
            "intercept": np.zeros_like(intercept),
        }

        self.is_initialized = True

    @property
    def hyperparameters(self):
        """返回包含层超参数的字典"""
        return {
            "layer": "BatchNorm2D",
            "act_fn": None,
            "in_ch": self.in_ch,
            "out_ch": self.out_ch,
            "epsilon": self.epsilon,
            "momentum": self.momentum,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }

    def reset_running_stats(self):
        """将运行均值和方差估计重置为0和1"""
        assert self.trainable, "Layer is frozen"
        self.parameters["running_mean"] = np.zeros(self.in_ch)
        self.parameters["running_var"] = np.ones(self.in_ch)
    def backward(self, dLdy, retain_grads=True):
        """
        Backprop from layer outputs to inputs.

        Parameters
        ----------
        dLdY : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
            The gradient of the loss wrt. the layer output `Y`.
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.

        Returns
        -------
        dX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
            The gradient of the loss wrt. the layer input `X`.
        """  # noqa: E501
        # 检查当前层是否可训练，如果不可训练则抛出异常
        assert self.trainable, "Layer is frozen"
        # 如果输入的梯度不是列表形式，则转换为列表
        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        # 初始化存储输入梯度的列表
        dX = []
        # 获取当前层的输入数据
        X = self.X
        # 遍历每个输入梯度和对应的输入数据
        for dy, x in zip(dLdy, X):
            # 调用内部方法计算当前层的反向传播，得到输入梯度、参数的梯度
            dx, dScaler, dIntercept = self._bwd(dy, x)
            # 将计算得到的输入梯度添加到列表中
            dX.append(dx)

            # 如果需要保留参数的梯度
            if retain_grads:
                # 累加参数的梯度
                self.gradients["scaler"] += dScaler
                self.gradients["intercept"] += dIntercept

        # 如果输入数据只有一个样本，则返回第一个输入梯度；否则返回整个输入梯度列表
        return dX[0] if len(X) == 1 else dX
    # 定义一个私有方法，用于计算损失对输入 X、缩放器、和截距的梯度
    def _bwd(self, dLdy, X):
        """Computation of gradient of loss wrt. X, scaler, and intercept"""
        # 获取缩放器参数
        scaler = self.parameters["scaler"]
        # 获取超参数中的 epsilon
        ep = self.hyperparameters["epsilon"]

        # 将输入 X 重塑为2D数组，保留通道维度
        X_shape = X.shape
        X = np.reshape(X, (-1, X.shape[3]))
        dLdy = np.reshape(dLdy, (-1, dLdy.shape[3]))

        # 在重塑后的数组上应用1D批量归一化的反向传播
        n_ex, in_ch = X.shape
        # 计算 X 的均值和方差
        X_mean, X_var = X.mean(axis=0), X.var(axis=0)  # , ddof=1)

        # 根据批量归一化的公式计算 N
        N = (X - X_mean) / np.sqrt(X_var + ep)
        # 计算截距的梯度
        dIntercept = dLdy.sum(axis=0)
        # 计算缩放器的梯度
        dScaler = np.sum(dLdy * N, axis=0)

        # 计算 dN
        dN = dLdy * scaler
        # 计算 dX
        dX = (n_ex * dN - dN.sum(axis=0) - N * (dN * N).sum(axis=0)) / (
            n_ex * np.sqrt(X_var + ep)
        )

        # 将 dX 重塑回原始形状，并返回 dX、dScaler、dIntercept
        return np.reshape(dX, X_shape), dScaler, dIntercept
class BatchNorm1D(LayerBase):
    # 初始化参数
    def _init_params(self):
        # 随机生成缩放器
        scaler = np.random.rand(self.n_in)
        # 初始化偏移为零
        intercept = np.zeros(self.n_in)

        # 初始化运行均值和标准差为0和1
        running_mean = np.zeros(self.n_in)
        running_var = np.ones(self.n_in)

        # 设置参数字典
        self.parameters = {
            "scaler": scaler,
            "intercept": intercept,
            "running_mean": running_mean,
            "running_var": running_var,
        }

        # 设置梯度字典
        self.gradients = {
            "scaler": np.zeros_like(scaler),
            "intercept": np.zeros_like(intercept),
        }
        self.is_initialized = True

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        # 返回包含层超参数的字典
        return {
            "layer": "BatchNorm1D",
            "act_fn": None,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "epsilon": self.epsilon,
            "momentum": self.momentum,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }

    def reset_running_stats(self):
        """Reset the running mean and variance estimates to 0 and 1."""
        # 重置运行均值和方差的估计为0和1
        assert self.trainable, "Layer is frozen"
        self.parameters["running_mean"] = np.zeros(self.n_in)
        self.parameters["running_var"] = np.ones(self.n_in)
    def forward(self, X, retain_derived=True):
        """
        计算单个小批量的层输出。

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            层的输入，表示`n_ex`个示例的`n_in`维特征的小批量。
        retain_derived : bool
            是否使用当前输入来调整运行均值和运行方差的计算。将其设置为True相当于冻结当前输入的层。默认为True。

        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            每个`n_ex`示例的层输出
        """
        if not self.is_initialized:
            self.n_in = self.n_out = X.shape[1]
            self._init_params()

        ep = self.hyperparameters["epsilon"]
        mm = self.hyperparameters["momentum"]
        rm = self.parameters["running_mean"]
        rv = self.parameters["running_var"]

        scaler = self.parameters["scaler"]
        intercept = self.parameters["intercept"]

        # 如果层被冻结，使用我们的运行均值/标准差值，而不是新批次的均值/标准差值
        X_mean = self.parameters["running_mean"]
        X_var = self.parameters["running_var"]

        if self.trainable and retain_derived:
            X_mean, X_var = X.mean(axis=0), X.var(axis=0)  # , ddof=1)
            self.parameters["running_mean"] = mm * rm + (1.0 - mm) * X_mean
            self.parameters["running_var"] = mm * rv + (1.0 - mm) * X_var

        if retain_derived:
            self.X.append(X)

        N = (X - X_mean) / np.sqrt(X_var + ep)
        y = scaler * N + intercept
        return y
    def backward(self, dLdy, retain_grads=True):
        """
        Backprop from layer outputs to inputs.

        Parameters
        ----------
        dLdY : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            The gradient of the loss wrt. the layer output `Y`.
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.

        Returns
        -------
        dX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            The gradient of the loss wrt. the layer input `X`.
        """
        assert self.trainable, "Layer is frozen"
        # 检查当前层是否可训练，如果不可训练则抛出异常
        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        dX = []
        X = self.X
        # 遍历每个输入数据和对应的梯度
        for dy, x in zip(dLdy, X):
            # 调用_bwd方法计算梯度
            dx, dScaler, dIntercept = self._bwd(dy, x)
            dX.append(dx)

            if retain_grads:
                # 如果需要保留梯度，则将计算得到的梯度加到参数梯度中
                self.gradients["scaler"] += dScaler
                self.gradients["intercept"] += dIntercept

        return dX[0] if len(X) == 1 else dX

    def _bwd(self, dLdy, X):
        """Computation of gradient of loss wrt X, scaler, and intercept"""
        # 获取参数scaler和epsilon
        scaler = self.parameters["scaler"]
        ep = self.hyperparameters["epsilon"]

        n_ex, n_in = X.shape
        # 计算输入数据X的均值和方差
        X_mean, X_var = X.mean(axis=0), X.var(axis=0)  # , ddof=1)

        # 标准化输入数据X
        N = (X - X_mean) / np.sqrt(X_var + ep)
        # 计算intercept的梯度
        dIntercept = dLdy.sum(axis=0)
        # 计算scaler的梯度
        dScaler = np.sum(dLdy * N, axis=0)

        # 计算dN
        dN = dLdy * scaler
        # 计算dX
        dX = (n_ex * dN - dN.sum(axis=0) - N * (dN * N).sum(axis=0)) / (
            n_ex * np.sqrt(X_var + ep)
        )

        return dX, dScaler, dIntercept
class LayerNorm2D(LayerBase):
    # 定义一个继承自LayerBase的LayerNorm2D类

    def _init_params(self, X_shape):
        # 初始化参数方法，接受输入数据的形状X_shape

        n_ex, in_rows, in_cols, in_ch = X_shape
        # 将输入数据形状X_shape解包为样本数、行数、列数和通道数

        scaler = np.random.rand(in_rows, in_cols, in_ch)
        # 随机生成一个与输入数据形状相同的缩放参数

        intercept = np.zeros((in_rows, in_cols, in_ch))
        # 创建一个与输入数据形状相同的偏置参数，初始值为0

        self.parameters = {"scaler": scaler, "intercept": intercept}
        # 将缩放参数和偏置参数存储在parameters字典中

        self.gradients = {
            "scaler": np.zeros_like(scaler),
            "intercept": np.zeros_like(intercept),
        }
        # 初始化梯度字典，将缩放参数和偏置参数的梯度初始化为与参数相同形状的零数组

        self.is_initialized = True
        # 将初始化标志设置为True，表示参数已经初始化完成

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        # 定义一个属性方法，返回包含层超参数的字典

        return {
            "layer": "LayerNorm2D",
            "act_fn": None,
            "in_ch": self.in_ch,
            "out_ch": self.out_ch,
            "epsilon": self.epsilon,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }
        # 返回包含层类型、激活函数、输入通道数、输出通道数、epsilon值和优化器信息的字典
    # 计算单个 minibatch 上的层输出
    def forward(self, X, retain_derived=True):
        """
        Compute the layer output on a single minibatch.

        Notes
        -----
        Equations [train & test]::

            Y = scaler * norm(X) + intercept
            norm(X) = (X - mean(X)) / sqrt(var(X) + epsilon)

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
            Input volume containing the `in_rows` by `in_cols`-dimensional
            features for a minibatch of `n_ex` examples.
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through wrt. this input. Default
            is True.

        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
            Layer output for each of the `n_ex` examples.
        """  # noqa: E501
        # 如果尚未初始化，则根据输入的形状初始化参数
        if not self.is_initialized:
            self.in_ch = self.out_ch = X.shape[3]
            self._init_params(X.shape)

        # 获取参数
        scaler = self.parameters["scaler"]
        ep = self.hyperparameters["epsilon"]
        intercept = self.parameters["intercept"]

        # 如果需要保留派生变量，则将输入添加到 X 列表中
        if retain_derived:
            self.X.append(X)

        # 计算输入的方差和均值
        X_var = X.var(axis=(1, 2, 3), keepdims=True)
        X_mean = X.mean(axis=(1, 2, 3), keepdims=True)
        # 计算层归一化
        lnorm = (X - X_mean) / np.sqrt(X_var + ep)
        # 计算层输出
        y = scaler * lnorm + intercept
        return y
    def backward(self, dLdy, retain_grads=True):
        """
        Backprop from layer outputs to inputs.

        Parameters
        ----------
        dLdY : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
            The gradient of the loss wrt. the layer output `Y`.
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.

        Returns
        -------
        dX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
            The gradient of the loss wrt. the layer input `X`.
        """  # noqa: E501
        # 检查当前层是否可训练，如果不可训练则抛出异常
        assert self.trainable, "Layer is frozen"
        # 如果输入的梯度不是列表形式，则转换为列表
        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        # 初始化存储输入梯度的列表
        dX = []
        # 获取当前层的输入数据
        X = self.X
        # 遍历每个输入梯度和对应的输入数据
        for dy, x in zip(dLdy, X):
            # 调用内部方法计算当前层的反向传播，得到输入梯度、参数的梯度
            dx, dScaler, dIntercept = self._bwd(dy, x)
            # 将计算得到的输入梯度添加到列表中
            dX.append(dx)

            # 如果需要保留参数的梯度
            if retain_grads:
                # 累加参数的梯度
                self.gradients["scaler"] += dScaler
                self.gradients["intercept"] += dIntercept

        # 如果输入数据只有一个样本，则返回第一个输入梯度；否则返回整个输入梯度列表
        return dX[0] if len(X) == 1 else dX
    # 计算损失对 X、scaler、intercept 的梯度
    def _bwd(self, dy, X):
        # 获取 scaler 参数
        scaler = self.parameters["scaler"]
        # 获取 epsilon 超参数
        ep = self.hyperparameters["epsilon"]

        # 计算 X 的均值
        X_mean = X.mean(axis=(1, 2, 3), keepdims=True)
        # 计算 X 的方差
        X_var = X.var(axis=(1, 2, 3), keepdims=True)
        # 计算归一化后的 X
        lnorm = (X - X_mean) / np.sqrt(X_var + ep)

        # 计算 dLnorm
        dLnorm = dy * scaler
        # 计算 dIntercept
        dIntercept = dy.sum(axis=0)
        # 计算 dScaler
        dScaler = np.sum(dy * lnorm, axis=0)

        # 计算输入 X 的维度
        n_in = np.prod(X.shape[1:])
        lnorm = lnorm.reshape(-1, n_in)
        dLnorm = dLnorm.reshape(lnorm.shape)
        X_var = X_var.reshape(X_var.shape[:2])

        # 计算 dX
        dX = (
            n_in * dLnorm
            - dLnorm.sum(axis=1, keepdims=True)
            - lnorm * (dLnorm * lnorm).sum(axis=1, keepdims=True)
        ) / (n_in * np.sqrt(X_var + ep))

        # 将 X 梯度重新调整为正确的维度
        return np.reshape(dX, X.shape), dScaler, dIntercept
class LayerNorm1D(LayerBase):
    # 初始化层参数
    def _init_params(self):
        # 随机生成缩放器
        scaler = np.random.rand(self.n_in)
        # 初始化偏置为零
        intercept = np.zeros(self.n_in)

        # 设置参数字典，包括缩放器和偏置
        self.parameters = {"scaler": scaler, "intercept": intercept}

        # 设置梯度字典，初始化为与参数相同形状的零数组
        self.gradients = {
            "scaler": np.zeros_like(scaler),
            "intercept": np.zeros_like(intercept),
        }
        # 标记已初始化
        self.is_initialized = True

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        # 返回包含层超参数的字典
        return {
            "layer": "LayerNorm1D",
            "act_fn": None,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "epsilon": self.epsilon,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }
    def forward(self, X, retain_derived=True):
        """
        Compute the layer output on a single minibatch.

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
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            Layer output for each of the `n_ex` examples.
        """
        # 如果该层尚未初始化，则根据输入的特征维度设置输入和输出维度，并初始化参数
        if not self.is_initialized:
            self.n_in = self.n_out = X.shape[1]
            self._init_params()

        # 获取缩放因子、epsilon值和截距参数
        scaler = self.parameters["scaler"]
        ep = self.hyperparameters["epsilon"]
        intercept = self.parameters["intercept"]

        # 如果需要保留派生变量，则将输入数据添加到X列表中
        if retain_derived:
            self.X.append(X)

        # 计算输入数据的均值和方差
        X_mean, X_var = X.mean(axis=1, keepdims=True), X.var(axis=1, keepdims=True)
        # 对输入数据进行局部响应归一化
        lnorm = (X - X_mean) / np.sqrt(X_var + ep)
        # 计算最终输出结果
        y = scaler * lnorm + intercept
        return y
    def backward(self, dLdy, retain_grads=True):
        """
        Backprop from layer outputs to inputs.

        Parameters
        ----------
        dLdY : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            The gradient of the loss wrt. the layer output `Y`.
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.

        Returns
        -------
        dX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            The gradient of the loss wrt. the layer input `X`.
        """
        assert self.trainable, "Layer is frozen"
        # 检查当前层是否可训练，如果不可训练则抛出异常
        if not isinstance(dLdy, list):
            # 如果梯度不是列表形式，则转换为列表
            dLdy = [dLdy]

        dX = []
        X = self.X
        # 遍历梯度和输入数据
        for dy, x in zip(dLdy, X):
            # 调用_bwd方法计算梯度
            dx, dScaler, dIntercept = self._bwd(dy, x)
            dX.append(dx)

            if retain_grads:
                # 如果需要保留梯度，则更新参数梯度
                self.gradients["scaler"] += dScaler
                self.gradients["intercept"] += dIntercept

        return dX[0] if len(X) == 1 else dX

    def _bwd(self, dLdy, X):
        """Computation of gradient of the loss wrt X, scaler, intercept"""
        # 计算损失相对于X、scaler、intercept的梯度
        scaler = self.parameters["scaler"]
        ep = self.hyperparameters["epsilon"]

        n_ex, n_in = X.shape
        X_mean, X_var = X.mean(axis=1, keepdims=True), X.var(axis=1, keepdims=True)

        lnorm = (X - X_mean) / np.sqrt(X_var + ep)
        dIntercept = dLdy.sum(axis=0)
        dScaler = np.sum(dLdy * lnorm, axis=0)

        dLnorm = dLdy * scaler
        dX = (
            n_in * dLnorm
            - dLnorm.sum(axis=1, keepdims=True)
            - lnorm * (dLnorm * lnorm).sum(axis=1, keepdims=True)
        ) / (n_in * np.sqrt(X_var + ep))

        return dX, dScaler, dIntercept
# 定义一个名为 Embedding 的类，继承自 LayerBase 类
class Embedding(LayerBase):
    # 初始化函数，接受输出维度 n_out、词汇表大小 vocab_size、池化方式 pool、初始化方法 init 和优化器 optimizer 作为参数
    def __init__(
        self, n_out, vocab_size, pool=None, init="glorot_uniform", optimizer=None,
    # 初始化参数函数，初始化权重矩阵 W，并将其存储在参数字典中
    def _init_params(self):
        # 使用指定的初始化方法初始化权重矩阵 W
        init_weights = WeightInitializer("Affine(slope=1, intercept=0)", mode=self.init)
        W = init_weights((self.vocab_size, self.n_out))

        # 存储参数、派生变量和梯度信息
        self.parameters = {"W": W}
        self.derived_variables = {}
        self.gradients = {"W": np.zeros_like(W)}
        self.is_initialized = True

    # 返回包含层超参数的字典
    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        return {
            "layer": "Embedding",
            "init": self.init,
            "pool": self.pool,
            "n_out": self.n_out,
            "vocab_size": self.vocab_size,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }

    # 查找给定 IDs 对应的嵌入向量
    def lookup(self, ids):
        """
        Return the embeddings associated with the IDs in `ids`.

        Parameters
        ----------
        word_ids : :py:class:`ndarray <numpy.ndarray>` of shape (`M`,)
            An array of `M` IDs to retrieve embeddings for.

        Returns
        -------
        embeddings : :py:class:`ndarray <numpy.ndarray>` of shape (`M`, `n_out`)
            The embedding vectors for each of the `M` IDs.
        """
        return self.parameters["W"][ids]
    def forward(self, X, retain_derived=True):
        """
        Compute the layer output on a single minibatch.

        Notes
        -----
        Equations:
            Y = W[x]

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)` or list of length `n_ex`
            Layer input, representing a minibatch of `n_ex` examples. If
            ``self.pool`` is None, each example must consist of exactly `n_in`
            integer token IDs. Otherwise, `X` can be a ragged array, with each
            example consisting of a variable number of token IDs.
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through with regard to this input.
            Default is True.

        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in, n_out)`
            Embeddings for each coordinate of each of the `n_ex` examples
        """  # noqa: E501
        # 如果 X 是一个不规则数组
        if isinstance(X, list) and not issubclass(X[0].dtype.type, np.integer):
            fstr = "Input to Embedding layer must be an array of integers, got '{}'"
            raise TypeError(fstr.format(X[0].dtype.type))

        # 否则
        if isinstance(X, np.ndarray) and not issubclass(X.dtype.type, np.integer):
            fstr = "Input to Embedding layer must be an array of integers, got '{}'"
            raise TypeError(fstr.format(X.dtype.type))

        # 调用内部方法 _fwd 计算输出 Y
        Y = self._fwd(X)
        # 如果需要保留计算过程中的变量，则将输入 X 添加到 self.X 中
        if retain_derived:
            self.X.append(X)
        # 返回计算结果 Y
        return Y
    def _fwd(self, X):
        """实际进行前向传播计算"""
        # 获取参数中的权重矩阵
        W = self.parameters["W"]
        # 如果没有池化操作
        if self.pool is None:
            # 直接根据输入索引X获取对应的权重向量
            emb = W[X]
        # 如果池化方式为"sum"
        elif self.pool == "sum":
            # 对每个输入索引X，计算对应的权重向量的和
            emb = np.array([W[x].sum(axis=0) for x in X])[:, None, :]
        # 如果池化方式为"mean"
        elif self.pool == "mean":
            # 对每个输入索引X，计算对应的权重向量的均值
            emb = np.array([W[x].mean(axis=0) for x in X])[:, None, :]
        return emb

    def backward(self, dLdy, retain_grads=True):
        """
        从层输出反向传播到嵌入权重。

        注意
        -----
        因为`X`中的项被解释为索引，所以无法计算层输出相对于`X`的梯度。

        参数
        ----------
        dLdy : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in, n_out)` or list of arrays
            损失相对于层输出的梯度
        retain_grads : bool
            是否在反向传播过程中保留中间参数梯度。默认为True。
        """  # noqa: E501
        assert self.trainable, "Layer is frozen"
        # 如果dLdy不是列表，则转换为列表
        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        # 对于每个dLdy和对应的输入X，计算梯度dw
        for dy, x in zip(dLdy, self.X):
            dw = self._bwd(dy, x)

            # 如果需要保留梯度
            if retain_grads:
                self.gradients["W"] += dw

    def _bwd(self, dLdy, X):
        """实际计算损失相对于W的梯度"""
        # 初始化梯度矩阵dW
        dW = np.zeros_like(self.parameters["W"])
        # 将dLdy重塑为二维数组
        dLdy = dLdy.reshape(-1, self.n_out)

        # 如果没有池化操作
        if self.pool is None:
            # 对于每个X中的索引，更新对应的权重梯度
            for ix, v_id in enumerate(X.flatten()):
                dW[v_id] += dLdy[ix]
        # 如果池化方式为"sum"
        elif self.pool == "sum":
            # 对于每个X中的索引列表，更新对应的权重梯度
            for ix, v_ids in enumerate(X):
                dW[v_ids] += dLdy[ix]
        # 如果池化方式为"mean"
        elif self.pool == "mean":
            # 对于每个X中的索引列表，更新对应的权重梯度（除以索引列表长度）
            for ix, v_ids in enumerate(X):
                dW[v_ids] += dLdy[ix] / len(v_ids)
        return dW
class FullyConnected(LayerBase):
    # FullyConnected 类继承自 LayerBase 类
    def _init_params(self):
        # 初始化参数方法
        init_weights = WeightInitializer(str(self.act_fn), mode=self.init)
        # 使用激活函数和初始化方式初始化权重

        b = np.zeros((1, self.n_out))
        # 初始化偏置为零向量
        W = init_weights((self.n_in, self.n_out))
        # 初始化权重矩阵

        self.parameters = {"W": W, "b": b}
        # 将参数保存在字典中
        self.derived_variables = {"Z": []}
        # 初始化派生变量字典，包含 Z 键
        self.gradients = {"W": np.zeros_like(W), "b": np.zeros_like(b)}
        # 初始化梯度字典，包含 W 和 b 键
        self.is_initialized = True
        # 标记已初始化

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        # 返回包含层超参数的字典
        return {
            "layer": "FullyConnected",
            "init": self.init,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }

    def forward(self, X, retain_derived=True):
        """
        Compute the layer output on a single minibatch.

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
        if not self.is_initialized:
            # 如果未初始化
            self.n_in = X.shape[1]
            # 设置输入维度为 X 的第二维度
            self._init_params()
            # 初始化参数

        Y, Z = self._fwd(X)
        # 计算前向传播得到输出 Y 和派生变量 Z

        if retain_derived:
            # 如果需要保留派生变量
            self.X.append(X)
            # 将输入 X 添加到 X 列表中
            self.derived_variables["Z"].append(Z)
            # 将派生变量 Z 添加到 Z 列表中

        return Y
        # 返回输出 Y
    def _fwd(self, X):
        """实际进行前向传播计算"""
        # 获取参数 W
        W = self.parameters["W"]
        # 获取参数 b
        b = self.parameters["b"]

        # 计算线性变换
        Z = X @ W + b
        # 应用激活函数
        Y = self.act_fn(Z)
        return Y, Z

    def backward(self, dLdy, retain_grads=True):
        """
        从层输出反向传播到输入。

        Parameters
        ----------
        dLdy : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)` or list of arrays
            损失相对于层输出的梯度。
        retain_grads : bool
            是否在反向传播过程中保留计算的中间参数梯度以用于最终参数更新。默认为 True。

        Returns
        -------
        dLdX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)` or list of arrays
            损失相对于层输入 `X` 的梯度。
        """  # noqa: E501
        assert self.trainable, "Layer is frozen"
        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        dX = []
        X = self.X
        for dy, x in zip(dLdy, X):
            dx, dw, db = self._bwd(dy, x)
            dX.append(dx)

            if retain_grads:
                self.gradients["W"] += dw
                self.gradients["b"] += db

        return dX[0] if len(X) == 1 else dX

    def _bwd(self, dLdy, X):
        """实际计算损失相对于 X、W 和 b 的梯度"""
        # 获取参数 W
        W = self.parameters["W"]
        # 获取参数 b
        b = self.parameters["b"]

        # 计算线性变换
        Z = X @ W + b
        # 计算激活函数的梯度
        dZ = dLdy * self.act_fn.grad(Z)

        # 计算损失相对于输入 X 的梯度
        dX = dZ @ W.T
        # 计算损失相对于参数 W 的梯度
        dW = X.T @ dZ
        # 计算损失相对于参数 b 的梯度
        dB = dZ.sum(axis=0, keepdims=True)
        return dX, dW, dB
    # 计算损失对输入 X 的二阶导数/导数，以及对权重 W 和偏置 b 的导数
    def _bwd2(self, dLdy, X, dLdy_bwd):
        # 获取权重 W 和偏置 b
        W = self.parameters["W"]
        b = self.parameters["b"]

        # 计算激活函数在 XW + b 处的一阶导数
        dZ = self.act_fn.grad(X @ W + b)
        # 计算激活函数在 XW + b 处的二阶导数
        ddZ = self.act_fn.grad2(X @ W + b)

        # 计算损失对输入 X 的二阶导数
        ddX = dLdy @ W * dZ
        # 计算损失对权重 W 的导数
        ddW = dLdy.T @ (dLdy_bwd * dZ)
        # 计算损失对偏置 b 的导数
        ddB = np.sum(dLdy @ W * dLdy_bwd * ddZ, axis=0, keepdims=True)
        # 返回结果：损失对输入 X 的二阶导数，损失对权重 W 的导数，损失对偏置 b 的导数
        return ddX, ddW, ddB
class Softmax(LayerBase):
    # Softmax 类继承自 LayerBase 类
    def _init_params(self):
        # 初始化梯度、参数、派生变量和初始化标志
        self.gradients = {}
        self.parameters = {}
        self.derived_variables = {}
        self.is_initialized = True

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        # 返回包含层超参数的字典
        return {
            "layer": "SoftmaxLayer",
            "n_in": self.n_in,
            "n_out": self.n_in,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }

    def forward(self, X, retain_derived=True):
        """
        Compute the layer output on a single minibatch.

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
        if not self.is_initialized:
            # 如果未初始化，则根据输入的形状设置 n_in，并初始化参数
            self.n_in = X.shape[1]
            self._init_params()

        # 调用 _fwd 方法计算前向传播结果
        Y = self._fwd(X)

        if retain_derived:
            # 如果需要保留派生变量，则将输入数据添加到 X 列表中
            self.X.append(X)

        return Y

    def _fwd(self, X):
        """Actual computation of softmax forward pass"""
        # 将数据居中以避免溢出
        e_X = np.exp(X - np.max(X, axis=self.dim, keepdims=True))
        return e_X / e_X.sum(axis=self.dim, keepdims=True)
    def backward(self, dLdy, retain_grads=True):
        """
        Backprop from layer outputs to inputs.

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
        dLdX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            The gradient of the loss wrt. the layer input `X`.
        """  # noqa: E501
        assert self.trainable, "Layer is frozen"
        # 检查当前层是否可训练，如果不可训练则抛出异常
        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        dX = []
        X = self.X
        # 遍历梯度和输入数据，计算梯度
        for dy, x in zip(dLdy, X):
            dx = self._bwd(dy, x)
            dX.append(dx)

        return dX[0] if len(X) == 1 else dX

    def _bwd(self, dLdy, X):
        """
        Actual computation of the gradient of the loss wrt. the input X.

        The Jacobian, J, of the softmax for input x = [x1, ..., xn] is:
            J[i, j] =
                softmax(x_i)  * (1 - softmax(x_j))  if i = j
                -softmax(x_i) * softmax(x_j)        if i != j
            where
                x_n is input example n (ie., the n'th row in X)
        """
        dX = []
        # 遍历梯度和输入数据，计算梯度
        for dy, x in zip(dLdy, X):
            dxi = []
            # 遍历每个样本的梯度和输入数据，计算梯度
            for dyi, xi in zip(*np.atleast_2d(dy, x)):
                yi = self._fwd(xi.reshape(1, -1)).reshape(-1, 1)
                dyidxi = np.diagflat(yi) - yi @ yi.T  # jacobian wrt. input sample xi
                dxi.append(dyi @ dyidxi)
            dX.append(dxi)
        return np.array(dX).reshape(*X.shape)
class SparseEvolution(LayerBase):
    # 定义 SparseEvolution 类，继承自 LayerBase 类

    def __init__(
        self,
        n_out,
        zeta=0.3,
        epsilon=20,
        act_fn=None,
        init="glorot_uniform",
        optimizer=None,
    ):
        # 初始化 SparseEvolution 类的实例
        # n_out: 输出神经元数量
        # zeta: 稀疏度参数，默认为 0.3
        # epsilon: 稀疏性参数，默认为 20
        # act_fn: 激活函数，默认为 None
        # init: 初始化方法，默认为 "glorot_uniform"
        # optimizer: 优化器，默认为 None

    def _init_params(self):
        # 初始化参数方法
        init_weights = WeightInitializer(str(self.act_fn), mode=self.init)
        # 使用激活函数和初始化方法初始化权重

        b = np.zeros((1, self.n_out))
        # 初始化偏置 b 为全零数组

        W = init_weights((self.n_in, self.n_out))
        # 初始化权重 W

        # convert a fully connected base layer into a sparse layer
        # 将全连接基础层转换为稀疏层
        n_in, n_out = W.shape
        # 获取权重 W 的形状
        p = (self.epsilon * (n_in + n_out)) / (n_in * n_out)
        # 计算稀疏性参数 p
        mask = np.random.binomial(1, p, shape=W.shape)
        # 生成稀疏性掩码 mask

        self.derived_variables = {"Z": []}
        # 初始化派生变量字典，包含键 "Z"

        self.parameters = {"W": W, "b": b, "W_mask": mask}
        # 初始化参数字典，包含权重 W、偏置 b 和权重掩码 W_mask

        self.gradients = {"W": np.zeros_like(W), "b": np.zeros_like(b)}
        # 初始化梯度字典，包含权重 W 和偏置 b 的梯度

        self.is_initialized = True
        # 设置初始化标志为 True

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        # 返回包含层超参数的字典
        return {
            "layer": "SparseEvolutionary",
            "init": self.init,
            "zeta": self.zeta,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "epsilon": self.epsilon,
            "act_fn": str(self.act_fn),
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }
        # 返回包含层超参数的字典
    # 计算单个小批量的层输出
    def forward(self, X, retain_derived=True):
        """
        Compute the layer output on a single minibatch.

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
        # 如果尚未初始化，则根据输入的形状设置输入维度并初始化参数
        if not self.is_initialized:
            self.n_in = X.shape[1]
            self._init_params()

        # 调用_fwd方法计算前向传播得到输出Y和中间变量Z
        Y, Z = self._fwd(X)

        # 如果需要保留计算过程中的变量，则将输入X和中间变量Z添加到相应列表中
        if retain_derived:
            self.X.append(X)
            self.derived_variables["Z"].append(Z)

        # 返回计算得到的输出Y
        return Y

    # 实际的前向传播计算
    def _fwd(self, X):
        """Actual computation of forward pass"""
        # 获取参数W、b和W_mask
        W = self.parameters["W"]
        b = self.parameters["b"]
        W_mask = self.parameters["W_mask"]

        # 计算线性变换结果Z和激活函数作用后的输出Y
        Z = X @ (W * W_mask) + b
        Y = self.act_fn(Z)
        return Y, Z
    def backward(self, dLdy, retain_grads=True):
        """
        Backprop from layer outputs to inputs

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
        dLdX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            The gradient of the loss wrt. the layer input `X`.
        """  # noqa: E501
        assert self.trainable, "Layer is frozen"
        # 如果 dLdy 不是列表，则将其转换为列表
        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        # 初始化一个空列表用于存储梯度
        dX = []
        # 获取输入数据 X
        X = self.X
        # 遍历 dLdy 和 X，计算梯度
        for dy, x in zip(dLdy, X):
            # 调用 _bwd 方法计算梯度
            dx, dw, db = self._bwd(dy, x)
            # 将计算得到的输入梯度添加到列表中
            dX.append(dx)

            # 如果需要保留梯度，则更新参数梯度
            if retain_grads:
                self.gradients["W"] += dw
                self.gradients["b"] += db

        # 如果输入数据 X 只有一个，则返回第一个梯度，否则返回所有梯度
        return dX[0] if len(X) == 1 else dX

    def _bwd(self, dLdy, X):
        """Actual computation of gradient of the loss wrt. X, W, and b"""
        # 获取参数 W 和 b
        W = self.parameters["W"]
        b = self.parameters["b"]
        # 根据参数掩码获取稀疏矩阵 W_sparse
        W_sparse = W * self.parameters["W_mask"]

        # 计算 Z
        Z = X @ W_sparse + b
        # 计算 dZ
        dZ = dLdy * self.act_fn.grad(Z)

        # 计算 dX, dW, dB
        dX = dZ @ W_sparse.T
        dW = X.T @ dZ
        dB = dZ.sum(axis=0, keepdims=True)
        return dX, dW, dB
    def _bwd2(self, dLdy, X, dLdy_bwd):
        """计算损失对dX、dW和db的二阶导数/导数"""
        W = self.parameters["W"]
        b = self.parameters["b"]
        W_sparse = W * self.parameters["W_mask"]

        dZ = self.act_fn.grad(X @ W_sparse + b)
        ddZ = self.act_fn.grad2(X @ W_sparse + b)

        ddX = dLdy @ W * dZ
        ddW = dLdy.T @ (dLdy_bwd * dZ)
        ddB = np.sum(dLdy @ W_sparse * dLdy_bwd * ddZ, axis=0, keepdims=True)
        return ddX, ddW, ddB

    def update(self):
        """
        使用当前梯度更新参数，并通过SET演化网络连接。
        """
        assert self.trainable, "Layer is frozen"
        for k, v in self.gradients.items():
            if k in self.parameters:
                self.parameters[k] = self.optimizer(self.parameters[k], v, k)
        self.flush_gradients()
        self._evolve_connections()

    def _evolve_connections(self):
        assert self.trainable, "Layer is frozen"
        W = self.parameters["W"]
        W_mask = self.parameters["W_mask"]
        W_flat = (W * W_mask).reshape(-1)

        k = int(np.prod(W.shape) * self.zeta)

        (p_ix,) = np.where(W_flat > 0)
        (n_ix,) = np.where(W_flat < 0)

        # 移除k个最大的负权重和k个最小的正权重
        k_smallest_p = p_ix[np.argsort(W_flat[p_ix])][:k]
        k_largest_n = n_ix[np.argsort(W_flat[n_ix])][-k:]
        n_rewired = len(k_smallest_p) + len(k_largest_n)

        self.mask = np.ones_like(W_flat)
        self.mask[k_largest_n] = 0
        self.mask[k_smallest_p] = 0

        zero_ixs = np.where(self.mask == 0)

        # 重新采样新的连接并更新掩码
        np.shuffle(zero_ixs)
        self.mask[zero_ixs[:n_rewired]] = 1
        self.mask = self.mask.reshape(*W.shape)
# 定义一个名为 Conv1D 的类，继承自 LayerBase 类
class Conv1D(LayerBase):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        out_ch,
        kernel_width,
        pad=0,
        stride=1,
        dilation=0,
        act_fn=None,
        init="glorot_uniform",
        optimizer=None,
    ):
    
    # 初始化参数函数
    def _init_params(self):
        # 根据激活函数和初始化方式初始化权重
        init_weights = WeightInitializer(str(self.act_fn), mode=self.init)
        
        # 初始化权重矩阵 W 和偏置 b
        W = init_weights((self.kernel_width, self.in_ch, self.out_ch))
        b = np.zeros((1, 1, self.out_ch))
        
        # 存储参数、梯度、派生变量，并标记已初始化
        self.parameters = {"W": W, "b": b}
        self.gradients = {"W": np.zeros_like(W), "b": np.zeros_like(b)}
        self.derived_variables = {"Z": [], "out_rows": [], "out_cols": []}
        self.is_initialized = True

    # 返回包含层超参数的字典
    @property
    def hyperparameters(self):
        return {
            "layer": "Conv1D",
            "pad": self.pad,
            "init": self.init,
            "in_ch": self.in_ch,
            "out_ch": self.out_ch,
            "stride": self.stride,
            "dilation": self.dilation,
            "act_fn": str(self.act_fn),
            "kernel_width": self.kernel_width,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }
    def forward(self, X, retain_derived=True):
        """
        Compute the layer output given input volume `X`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, l_in, in_ch)`
            The input volume consisting of `n_ex` examples, each of length
            `l_in` and with `in_ch` input channels
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through wrt. this input. Default
            is True.

        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, l_out, out_ch)`
            The layer output.
        """
        # 如果该层尚未初始化，则根据输入的通道数初始化参数
        if not self.is_initialized:
            self.in_ch = X.shape[2]
            self._init_params()

        # 获取该层的权重和偏置参数
        W = self.parameters["W"]
        b = self.parameters["b"]

        # 获取输入数据的形状信息
        n_ex, l_in, in_ch = X.shape
        s, p, d = self.stride, self.pad, self.dilation

        # 对输入进行填充并执行前向卷积操作
        Z = conv1D(X, W, s, p, d) + b
        Y = self.act_fn(Z)

        # 如果需要保留派生变量，则将计算过程中的变量保存起来
        if retain_derived:
            self.X.append(X)
            self.derived_variables["Z"].append(Z)
            self.derived_variables["out_rows"].append(Z.shape[1])
            self.derived_variables["out_cols"].append(Z.shape[2])

        return Y
    def backward(self, dLdy, retain_grads=True):
        """
        Compute the gradient of the loss with respect to the layer parameters.

        Notes
        -----
        Relies on :meth:`~numpy_ml.neural_nets.utils.im2col` and
        :meth:`~numpy_ml.neural_nets.utils.col2im` to vectorize the
        gradient calculation.  See the private method :meth:`_backward_naive`
        for a more straightforward implementation.

        Parameters
        ----------
        dLdy : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, l_out, out_ch)` or list of arrays
            The gradient(s) of the loss with respect to the layer output(s).
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.

        Returns
        -------
        dX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, l_in, in_ch)`
            The gradient of the loss with respect to the layer input volume.
        """  # noqa: E501

        # 检查当前层是否可训练，如果不可训练则抛出异常
        assert self.trainable, "Layer is frozen"
        
        # 如果输入的梯度不是列表，则转换为列表
        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        # 获取当前层的输入数据和派生变量 Z
        X = self.X
        Z = self.derived_variables["Z"]

        # 初始化存储输入梯度的列表
        dX = []
        
        # 遍历每个输入梯度、输入数据和派生变量，计算并存储输入梯度
        for dy, x, z in zip(dLdy, X, Z):
            dx, dw, db = self._bwd(dy, x, z)
            dX.append(dx)

            # 如果需要保留中间参数梯度，则更新参数梯度
            if retain_grads:
                self.gradients["W"] += dw
                self.gradients["b"] += db

        # 如果输入数据只有一个，则返回第一个输入梯度；否则返回所有输入梯度
        return dX[0] if len(X) == 1 else dX
    # 计算损失对 X、W 和 b 的梯度
    def _bwd(self, dLdy, X, Z):
        # 获取参数 W
        W = self.parameters["W"]

        # 为 X、W 和 dZ 添加一个行维度，以便使用 im2col/col2im
        X2D = np.expand_dims(X, axis=1)
        W2D = np.expand_dims(W, axis=0)
        dLdZ = np.expand_dims(dLdy * self.act_fn.grad(Z), axis=1)

        # 获取参数的相关维度信息
        d = self.dilation
        fr, fc, in_ch, out_ch = W2D.shape
        n_ex, l_out, out_ch = dLdy.shape
        fr, fc, s = 1, self.kernel_width, self.stride

        # 使用 pad1D 处理 self.pad = 'causal'，因为 pad2D 不支持这种情况
        _, p = pad1D(X, self.pad, self.kernel_width, s, d)
        p2D = (0, 0, p[0], p[1])

        # 对 W、X 和 dLdy 进行列化处理
        dLdZ_col = dLdZ.transpose(3, 1, 2, 0).reshape(out_ch, -1)
        W_col = W2D.transpose(3, 2, 0, 1).reshape(out_ch, -1).T
        X_col, _ = im2col(X2D, W2D.shape, p2D, s, d)

        # 通过矩阵乘法计算梯度并重塑形状
        dB = dLdZ_col.sum(axis=1).reshape(1, 1, -1)
        dW = (dLdZ_col @ X_col.T).reshape(out_ch, in_ch, fr, fc).transpose(2, 3, 1, 0)

        # 将列化的 dX 重塑回与输入体积相同的格式
        dX_col = W_col @ dLdZ_col
        dX = col2im(dX_col, X2D.shape, W2D.shape, p2D, s, d).transpose(0, 2, 3, 1)

        # 去除多余的维度
        return np.squeeze(dX, axis=1), np.squeeze(dW, axis=0), dB
class Conv2D(LayerBase):
    # 定义 Conv2D 类，继承自 LayerBase 类
    def __init__(
        self,
        out_ch,
        kernel_shape,
        pad=0,
        stride=1,
        dilation=0,
        act_fn=None,
        optimizer=None,
        init="glorot_uniform",
    # 初始化函数，接受输出通道数、卷积核形状、填充、步幅、膨胀率、激活函数、优化器和初始化方式等参数
    def _init_params(self):
        # 初始化参数函数
        init_weights = WeightInitializer(str(self.act_fn), mode=self.init)
        # 使用激活函数和初始化方式初始化权重

        fr, fc = self.kernel_shape
        # 获取卷积核的行数和列数
        W = init_weights((fr, fc, self.in_ch, self.out_ch))
        # 初始化权重矩阵
        b = np.zeros((1, 1, 1, self.out_ch))
        # 初始化偏置矩阵

        self.parameters = {"W": W, "b": b}
        # 存储参数字典
        self.gradients = {"W": np.zeros_like(W), "b": np.zeros_like(b)}
        # 存储梯度字典
        self.derived_variables = {"Z": [], "out_rows": [], "out_cols": []}
        # 存储派生变量字典
        self.is_initialized = True
        # 初始化标志为 True

    @property
    def hyperparameters(self):
        """A dictionary containing the layer hyperparameters."""
        # 返回包含层超参数的字典
        return {
            "layer": "Conv2D",
            "pad": self.pad,
            "init": self.init,
            "in_ch": self.in_ch,
            "out_ch": self.out_ch,
            "stride": self.stride,
            "dilation": self.dilation,
            "act_fn": str(self.act_fn),
            "kernel_shape": self.kernel_shape,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }
        # 返回包含层超参数的字典
    def forward(self, X, retain_derived=True):
        """
        Compute the layer output given input volume `X`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
            The input volume consisting of `n_ex` examples, each with dimension
            (`in_rows`, `in_cols`, `in_ch`).
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through wrt. this input. Default
            is True.

        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows, out_cols, out_ch)`
            The layer output.
        """  # noqa: E501
        # 如果该层还未初始化，则根据输入的通道数初始化参数
        if not self.is_initialized:
            self.in_ch = X.shape[3]
            self._init_params()

        # 获取参数 W 和 b
        W = self.parameters["W"]
        b = self.parameters["b"]

        # 获取输入 X 的形状信息
        n_ex, in_rows, in_cols, in_ch = X.shape
        s, p, d = self.stride, self.pad, self.dilation

        # 对输入进行填充并执行前向卷积
        Z = conv2D(X, W, s, p, d) + b
        # 对 Z 应用激活函数得到输出 Y
        Y = self.act_fn(Z)

        # 如果需要保留派生变量，则将计算过程中的变量保存起来
        if retain_derived:
            self.X.append(X)
            self.derived_variables["Z"].append(Z)
            self.derived_variables["out_rows"].append(Z.shape[1])
            self.derived_variables["out_cols"].append(Z.shape[2])

        # 返回输出 Y
        return Y
    def backward(self, dLdy, retain_grads=True):
        """
        Compute the gradient of the loss with respect to the layer parameters.

        Notes
        -----
        Relies on :meth:`~numpy_ml.neural_nets.utils.im2col` and
        :meth:`~numpy_ml.neural_nets.utils.col2im` to vectorize the
        gradient calculation.

        See the private method :meth:`_backward_naive` for a more straightforward
        implementation.

        Parameters
        ----------
        dLdy : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows, out_cols, out_ch)` or list of arrays
            The gradient(s) of the loss with respect to the layer output(s).
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.

        Returns
        -------
        dX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
            The gradient of the loss with respect to the layer input volume.
        """  # noqa: E501
        assert self.trainable, "Layer is frozen"
        # 检查当前层是否可训练，如果不可训练则抛出异常
        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        dX = []
        X = self.X
        Z = self.derived_variables["Z"]

        for dy, x, z in zip(dLdy, X, Z):
            # 调用私有方法 _bwd 计算损失相对于参数的梯度
            dx, dw, db = self._bwd(dy, x, z)
            dX.append(dx)

            if retain_grads:
                # 如果需要保留梯度，则将计算得到的参数梯度累加到梯度字典中
                self.gradients["W"] += dw
                self.gradients["b"] += db

        # 返回损失相对于输入的梯度
        return dX[0] if len(X) == 1 else dX
    # 计算损失函数对输入 X、权重 W 和偏置 b 的梯度
    def _bwd(self, dLdy, X, Z):
        """Actual computation of gradient of the loss wrt. X, W, and b"""
        # 获取权重 W
        W = self.parameters["W"]

        # 获取卷积核的尺寸和输入输出通道数
        d = self.dilation
        fr, fc, in_ch, out_ch = W.shape
        n_ex, out_rows, out_cols, out_ch = dLdy.shape
        (fr, fc), s, p = self.kernel_shape, self.stride, self.pad

        # 将 dLdy 乘以激活函数关于 Z 的导数，得到 dLdZ
        dLdZ = dLdy * self.act_fn.grad(Z)
        # 将 dLdZ 转置并展平成二维数组
        dLdZ_col = dLdZ.transpose(3, 1, 2, 0).reshape(out_ch, -1)
        # 将 W 转置并展平成二维数组
        W_col = W.transpose(3, 2, 0, 1).reshape(out_ch, -1).T
        # 将输入 X 转换成列向量
        X_col, p = im2col(X, W.shape, p, s, d)

        # 计算偏置 b 的梯度
        dB = dLdZ_col.sum(axis=1).reshape(1, 1, 1, -1)
        # 计算权重 W 的梯度
        dW = (dLdZ_col @ X_col.T).reshape(out_ch, in_ch, fr, fc).transpose(2, 3, 1, 0)

        # 将列向量 dX_col 重塑回与输入体积相同的格式
        dX_col = W_col @ dLdZ_col
        dX = col2im(dX_col, X.shape, W.shape, p, s, d).transpose(0, 2, 3, 1)

        return dX, dW, dB
class Pool2D(LayerBase):
    def __init__(self, kernel_shape, stride=1, pad=0, mode="max", optimizer=None):
        """
        A single two-dimensional pooling layer.

        Parameters
        ----------
        kernel_shape : 2-tuple
            The dimension of a single 2D filter/kernel in the current layer
        stride : int
            The stride/hop of the convolution kernels as they move over the
            input volume. Default is 1.
        pad : int, tuple, or 'same'
            The number of rows/columns of 0's to pad the input. Default is 0.
        mode : {"max", "average"}
            The pooling function to apply.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the :class:`SGD
            <numpy_ml.neural_nets.optimizers.SGD>` optimizer with
            default parameters. Default is None.
        """  # noqa: E501
        # 调用父类的构造函数，初始化优化器
        super().__init__(optimizer)

        # 初始化池化层的参数
        self.pad = pad
        self.mode = mode
        self.in_ch = None
        self.out_ch = None
        self.stride = stride
        self.kernel_shape = kernel_shape
        self.is_initialized = False

    def _init_params(self):
        # 初始化派生变量，用于存储输出行数和列数
        self.derived_variables = {"out_rows": [], "out_cols": []}
        self.is_initialized = True

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        # 返回包含层超参数的字典
        return {
            "layer": "Pool2D",
            "act_fn": None,
            "pad": self.pad,
            "mode": self.mode,
            "in_ch": self.in_ch,
            "out_ch": self.out_ch,
            "stride": self.stride,
            "kernel_shape": self.kernel_shape,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }
class Deconv2D(LayerBase):
    # 定义 Deconv2D 类，继承自 LayerBase 类
    def __init__(
        self,
        out_ch,
        kernel_shape,
        pad=0,
        stride=1,
        act_fn=None,
        optimizer=None,
        init="glorot_uniform",
    # 初始化函数，接受输出通道数、卷积核形状、填充、步长、激活函数、优化器和初始化方式等参数
    def _init_params(self):
        # 初始化权重
        init_weights = WeightInitializer(str(self.act_fn), mode=self.init)
        # 获取卷积核的行数和列数
        fr, fc = self.kernel_shape
        # 初始化权重矩阵 W 和偏置 b
        W = init_weights((fr, fc, self.in_ch, self.out_ch))
        b = np.zeros((1, 1, 1, self.out_ch))
        # 初始化参数字典和梯度字典
        self.parameters = {"W": W, "b": b}
        self.gradients = {"W": np.zeros_like(W), "b": np.zeros_like(b)}
        # 初始化派生变量字典
        self.derived_variables = {"Z": [], "out_rows": [], "out_cols": []}
        self.is_initialized = True

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        # 返回包含层超参数的字典
        return {
            "layer": "Deconv2D",
            "pad": self.pad,
            "init": self.init,
            "in_ch": self.in_ch,
            "out_ch": self.out_ch,
            "stride": self.stride,
            "act_fn": str(self.act_fn),
            "kernel_shape": self.kernel_shape,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }
    def forward(self, X, retain_derived=True):
        """
        Compute the layer output given input volume `X`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
            The input volume consisting of `n_ex` examples, each with dimension
            (`in_rows`, `in_cols`, `in_ch`).
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through wrt. this input. Default
            is True.

        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows, out_cols, out_ch)`
            The layer output.
        """  # noqa: E501
        # 如果该层尚未初始化，则根据输入的通道数初始化参数
        if not self.is_initialized:
            self.in_ch = X.shape[3]
            self._init_params()

        # 获取该层的权重和偏置参数
        W = self.parameters["W"]
        b = self.parameters["b"]

        # 获取步长和填充值
        s, p = self.stride, self.pad
        n_ex, in_rows, in_cols, in_ch = X.shape

        # 对输入进行填充并执行前向反卷积
        Z = deconv2D_naive(X, W, s, p, 0) + b
        # 对 Z 应用激活函数得到输出 Y
        Y = self.act_fn(Z)

        # 如果需要保留派生变量，则将计算过程中的变量保存起来
        if retain_derived:
            self.X.append(X)
            self.derived_variables["Z"].append(Z)
            self.derived_variables["out_rows"].append(Z.shape[1])
            self.derived_variables["out_cols"].append(Z.shape[2])

        # 返回输出 Y
        return Y
    def backward(self, dLdY, retain_grads=True):
        """
        Compute the gradient of the loss with respect to the layer parameters.

        Notes
        -----
        Relies on :meth:`~numpy_ml.neural_nets.utils.im2col` and
        :meth:`~numpy_ml.neural_nets.utils.col2im` to vectorize the
        gradient calculations.

        Parameters
        ----------
        dLdY : :py:class:`ndarray <numpy.ndarray>` of shape (`n_ex, out_rows, out_cols, out_ch`)
            The gradient of the loss with respect to the layer output.
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.

        Returns
        -------
        dX : :py:class:`ndarray <numpy.ndarray>` of shape (`n_ex, in_rows, in_cols, in_ch`)
            The gradient of the loss with respect to the layer input volume.
        """  # noqa: E501
        assert self.trainable, "Layer is frozen"
        # 如果层被冻结，则抛出异常
        if not isinstance(dLdY, list):
            dLdY = [dLdY]

        # 初始化梯度列表
        dX = []
        # 获取输入数据 X 和派生变量 Z
        X, Z = self.X, self.derived_variables["Z"]

        # 遍历每个梯度和对应的输入数据 X 和派生变量 Z
        for dy, x, z in zip(dLdY, X, Z):
            # 计算当前梯度对应的输入数据的梯度，参数梯度和偏置梯度
            dx, dw, db = self._bwd(dy, x, z)
            # 将输入数据的梯度添加到梯度列表中
            dX.append(dx)

            # 如果需要保留梯度
            if retain_grads:
                # 累加参数梯度和偏置梯度
                self.gradients["W"] += dw
                self.gradients["b"] += db

        # 如果输入数据 X 只有一个，则返回第一个梯度，否则返回整个梯度列表
        return dX[0] if len(X) == 1 else dX
    # 计算损失对 X、W 和 b 的梯度
    def _bwd(self, dLdY, X, Z):
        # 旋转权重矩阵 W 180 度
        W = np.rot90(self.parameters["W"], 2)

        # 获取步长
        s = self.stride
        # 如果步长大于1，则对输入 X 进行扩张
        if self.stride > 1:
            X = dilate(X, s - 1)
            s = 1

        # 获取权重矩阵的形状信息
        fr, fc, in_ch, out_ch = W.shape
        # 获取卷积核的形状和填充信息
        (fr, fc), p = self.kernel_shape, self.pad
        n_ex, out_rows, out_cols, out_ch = dLdY.shape

        # 对输入 X 进行填充
        X_pad, p = pad2D(X, p, W.shape[:2], s)
        n_ex, in_rows, in_cols, in_ch = X_pad.shape
        pr1, pr2, pc1, pc2 = p

        # 计算额外的填充以产生反卷积
        out_rows = s * (in_rows - 1) - pr1 - pr2 + fr
        out_cols = s * (in_cols - 1) - pc1 - pc2 + fc
        out_dim = (out_rows, out_cols)

        # 添加额外的“反卷积”填充
        _p = calc_pad_dims_2D(X_pad.shape, out_dim, W.shape[:2], s, 0)
        X_pad, _ = pad2D(X_pad, _p, W.shape[:2], s)

        # 对 dLdY 进行列化
        dLdZ = dLdY * self.act_fn.grad(Z)
        dLdZ, _ = pad2D(dLdZ, p, W.shape[:2], s)

        # 对 dLdZ 进行列化
        dLdZ_col = dLdZ.transpose(3, 1, 2, 0).reshape(out_ch, -1)
        W_col = W.transpose(3, 2, 0, 1).reshape(out_ch, -1)
        X_col, _ = im2col(X_pad, W.shape, 0, s, 0)

        # 通过矩阵乘法计算梯度并重塑
        dB = dLdZ_col.sum(axis=1).reshape(1, 1, 1, -1)
        dW = (dLdZ_col @ X_col.T).reshape(out_ch, in_ch, fr, fc).transpose(2, 3, 1, 0)
        dW = np.rot90(dW, 2)

        # 将列化的 dX 重塑回与输入体积相同的格式
        dX_col = W_col.T @ dLdZ_col

        total_pad = tuple(i + j for i, j in zip(p, _p))
        dX = col2im(dX_col, X.shape, W.shape, total_pad, s, 0).transpose(0, 2, 3, 1)
        dX = dX[:, :: self.stride, :: self.stride, :]

        return dX, dW, dB
# 定义 RNNCell 类，继承自 LayerBase 类
class RNNCell(LayerBase):
    # 初始化参数
    def _init_params(self):
        # 初始化输入数据列表
        self.X = []
        # 使用指定激活函数和初始化方式初始化权重
        init_weights = WeightInitializer(str(self.act_fn), mode=self.init)

        # 初始化权重矩阵 Wax
        Wax = init_weights((self.n_in, self.n_out))
        # 初始化权重矩阵 Waa
        Waa = init_weights((self.n_out, self.n_out))
        # 初始化偏置向量 ba
        ba = np.zeros((self.n_out, 1))
        # 初始化偏置向量 bx
        bx = np.zeros((self.n_out, 1))

        # 存储参数字典，包括权重和偏置
        self.parameters = {"Waa": Waa, "Wax": Wax, "ba": ba, "bx": bx}

        # 存储梯度字典，初始化为零矩阵
        self.gradients = {
            "Waa": np.zeros_like(Waa),
            "Wax": np.zeros_like(Wax),
            "ba": np.zeros_like(ba),
            "bx": np.zeros_like(bx),
        }

        # 存储派生变量字典，包括中间变量和累加器
        self.derived_variables = {
            "A": [],
            "Z": [],
            "n_timesteps": 0,
            "current_step": 0,
            "dLdA_accumulator": None,
        }

        # 标记初始化完成
        self.is_initialized = True

    # 定义 hyperparameters 属性，返回包含层超参数的字典
    @property
    def hyperparameters(self):
        return {
            "layer": "RNNCell",
            "init": self.init,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }
    def forward(self, Xt):
        """
        Compute the network output for a single timestep.

        Parameters
        ----------
        Xt : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            Input at timestep `t` consisting of `n_ex` examples each of
            dimensionality `n_in`.

        Returns
        -------
        At: :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)`
            The value of the hidden state at timestep `t` for each of the
            `n_ex` examples.
        """
        # 如果网络未初始化，则设置输入维度并初始化参数
        if not self.is_initialized:
            self.n_in = Xt.shape[1]
            self._init_params()

        # 增加时间步数
        self.derived_variables["n_timesteps"] += 1
        self.derived_variables["current_step"] += 1

        # 获取参数
        ba = self.parameters["ba"]
        bx = self.parameters["bx"]
        Wax = self.parameters["Wax"]
        Waa = self.parameters["Waa"]

        # 初始化隐藏状态为零
        As = self.derived_variables["A"]
        if len(As) == 0:
            n_ex, n_in = Xt.shape
            A0 = np.zeros((n_ex, self.n_out))
            As.append(A0)

        # 计算下一个隐藏状态
        Zt = As[-1] @ Waa + ba.T + Xt @ Wax + bx.T
        At = self.act_fn(Zt)

        self.derived_variables["Z"].append(Zt)
        self.derived_variables["A"].append(At)

        # 存储中间变量
        self.X.append(Xt)
        return At
    def backward(self, dLdAt):
        """
        Backprop for a single timestep.

        Parameters
        ----------
        dLdAt : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out)`
            The gradient of the loss wrt. the layer outputs (ie., hidden
            states) at timestep `t`.

        Returns
        -------
        dLdXt : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in)`
            The gradient of the loss wrt. the layer inputs at timestep `t`.
        """
        assert self.trainable, "Layer is frozen"

        # decrement current step
        self.derived_variables["current_step"] -= 1

        # extract context variables
        Zs = self.derived_variables["Z"]
        As = self.derived_variables["A"]
        t = self.derived_variables["current_step"]
        dA_acc = self.derived_variables["dLdA_accumulator"]

        # initialize accumulator if it is None
        if dA_acc is None:
            dA_acc = np.zeros_like(As[0])

        # get network weights for gradient calculations
        Wax = self.parameters["Wax"]
        Waa = self.parameters["Waa"]

        # compute gradient components at timestep t
        dA = dLdAt + dA_acc
        dZ = self.act_fn.grad(Zs[t]) * dA
        dXt = dZ @ Wax.T

        # update parameter gradients with signal from current step
        self.gradients["Waa"] += As[t].T @ dZ
        self.gradients["Wax"] += self.X[t].T @ dZ
        self.gradients["ba"] += dZ.sum(axis=0, keepdims=True).T
        self.gradients["bx"] += dZ.sum(axis=0, keepdims=True).T

        # update accumulator variable for hidden state
        self.derived_variables["dLdA_accumulator"] = dZ @ Waa.T
        return dXt
    # 清空所有层的派生变量和梯度
    def flush_gradients(self):
        # 检查层是否可训练
        assert self.trainable, "Layer is frozen"

        # 重置输入数据列表
        self.X = []
        # 遍历派生变量字典，将值清空
        for k, v in self.derived_variables.items():
            self.derived_variables[k] = []

        # 重置时间步数和当前步数
        self.derived_variables["n_timesteps"] = 0
        self.derived_variables["current_step"] = 0

        # 将参数梯度重置为0
        # 遍历参数字典，将梯度设置为与参数相同形状的零数组
        for k, v in self.parameters.items():
            self.gradients[k] = np.zeros_like(v)
# 定义一个 LSTM 单元的类，继承自 LayerBase 类
class LSTMCell(LayerBase):
    # 初始化方法，接受输出维度、激活函数、门函数、初始化方法和优化器等参数
    def __init__(
        self,
        n_out,
        act_fn="Tanh",
        gate_fn="Sigmoid",
        init="glorot_uniform",
        optimizer=None,
    # 初始化参数方法
    def _init_params(self):
        # 初始化输入数据列表
        self.X = []
        # 初始化门权重的初始化器
        init_weights_gate = WeightInitializer(str(self.gate_fn), mode=self.init)
        # 初始化激活函数权重的初始化器
        init_weights_act = WeightInitializer(str(self.act_fn), mode=self.init)

        # 初始化遗忘门权重
        Wf = init_weights_gate((self.n_in + self.n_out, self.n_out))
        # 初始化更新门权重
        Wu = init_weights_gate((self.n_in + self.n_out, self.n_out))
        # 初始化细胞状态权重
        Wc = init_weights_act((self.n_in + self.n_out, self.n_out))
        # 初始化输出门权重
        Wo = init_weights_gate((self.n_in + self.n_out, self.n_out))

        # 初始化遗忘门偏置
        bf = np.zeros((1, self.n_out))
        # 初始化更新门偏置
        bu = np.zeros((1, self.n_out))
        # 初始化细胞状态偏置
        bc = np.zeros((1, self.n_out))
        # 初始化输出门偏置
        bo = np.zeros((1, self.n_out))

        # 初始化参数字典
        self.parameters = {
            "Wf": Wf,
            "Wu": Wu,
            "Wc": Wc,
            "Wo": Wo,
            "bf": bf,
            "bu": bu,
            "bc": bc,
            "bo": bo,
        }

        # 初始化梯度字典
        self.gradients = {
            "Wf": np.zeros_like(Wf),
            "Wu": np.zeros_like(Wu),
            "Wc": np.zeros_like(Wc),
            "Wo": np.zeros_like(Wo),
            "bf": np.zeros_like(bf),
            "bu": np.zeros_like(bu),
            "bc": np.zeros_like(bc),
            "bo": np.zeros_like(bo),
        }

        # 初始化派生变量字典
        self.derived_variables = {
            "C": [],
            "A": [],
            "Gf": [],
            "Gu": [],
            "Go": [],
            "Gc": [],
            "Cc": [],
            "n_timesteps": 0,
            "current_step": 0,
            "dLdA_accumulator": None,
            "dLdC_accumulator": None,
        }

        # 设置初始化标志为 True
        self.is_initialized = True
    # 获取神经网络层的参数
    def _get_params(self):
        Wf = self.parameters["Wf"]  # 获取遗忘门的权重参数
        Wu = self.parameters["Wu"]  # 获取更新门的权重参数
        Wc = self.parameters["Wc"]  # 获取细胞状态的权重参数
        Wo = self.parameters["Wo"]  # 获取输出门的权重参数
        bf = self.parameters["bf"]  # 获取遗忘门的偏置参数
        bu = self.parameters["bu"]  # 获取更新门的偏置参数
        bc = self.parameters["bc"]  # 获取细胞状态的偏置参数
        bo = self.parameters["bo"]  # 获取输出门的偏置参数
        return Wf, Wu, Wc, Wo, bf, bu, bc, bo

    @property
    def hyperparameters(self):
        """返回包含层超参数的字典"""
        return {
            "layer": "LSTMCell",
            "init": self.init,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
            "gate_fn": str(self.gate_fn),
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }

    def flush_gradients(self):
        """擦除所有层的派生变量和梯度"""
        assert self.trainable, "Layer is frozen"  # 断言层是否可训练

        self.X = []  # 重置输入数据
        for k, v in self.derived_variables.items():
            self.derived_variables[k] = []  # 清空派生变量

        self.derived_variables["n_timesteps"] = 0  # 重置时间步数
        self.derived_variables["current_step"] = 0  # 重置当前步数

        # 将参数梯度重置为0
        for k, v in self.parameters.items():
            self.gradients[k] = np.zeros_like(v)
class RNN(LayerBase):
    def __init__(self, n_out, act_fn="Tanh", init="glorot_uniform", optimizer=None):
        """
        A single vanilla (Elman)-RNN layer.

        Parameters
        ----------
        n_out : int
            The dimension of a single hidden state / output on a given
            timestep.
        act_fn : str, :doc:`Activation <numpy_ml.neural_nets.activations>` object, or None
            The activation function for computing ``A[t]``. Default is
            `'Tanh'`.
        init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
            The weight initialization strategy. Default is `'glorot_uniform'`.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the :class:`SGD
            <numpy_ml.neural_nets.optimizers.SGD>` optimizer with default
            parameters. Default is None.
        """  # noqa: E501
        # 调用父类的构造函数，传入优化器参数
        super().__init__(optimizer)

        # 初始化权重初始化策略、输入维度、输出维度、时间步数、激活函数和初始化状态
        self.init = init
        self.n_in = None
        self.n_out = n_out
        self.n_timesteps = None
        self.act_fn = ActivationInitializer(act_fn)()
        self.is_initialized = False

    def _init_params(self):
        # 初始化 RNNCell 层，传入输入维度、输出维度、激活函数、权重初始化策略和优化器
        self.cell = RNNCell(
            n_in=self.n_in,
            n_out=self.n_out,
            act_fn=self.act_fn,
            init=self.init,
            optimizer=self.optimizer,
        )
        # 设置初始化状态为 True
        self.is_initialized = True

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        # 返回包含层超参数的字典
        return {
            "layer": "RNN",
            "init": self.init,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
            "optimizer": self.cell.hyperparameters["optimizer"],
        }
    def forward(self, X):
        """
        Run a forward pass across all timesteps in the input.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in, n_t)`
            Input consisting of `n_ex` examples each of dimensionality `n_in`
            and extending for `n_t` timesteps.

        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out, n_t)`
            The value of the hidden state for each of the `n_ex` examples
            across each of the `n_t` timesteps.
        """
        # 如果网络未初始化，则根据输入的形状初始化网络参数
        if not self.is_initialized:
            self.n_in = X.shape[1]
            self._init_params()

        Y = []
        n_ex, n_in, n_t = X.shape
        # 遍历每个时间步
        for t in range(n_t):
            # 在每个时间步上运行 cell 的前向传播
            yt = self.cell.forward(X[:, :, t])
            Y.append(yt)
        return np.dstack(Y)

    def backward(self, dLdA):
        """
        Run a backward pass across all timesteps in the input.

        Parameters
        ----------
        dLdA : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out, n_t)`
            The gradient of the loss with respect to the layer output for each
            of the `n_ex` examples across all `n_t` timesteps.

        Returns
        -------
        dLdX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in, n_t)`
            The value of the hidden state for each of the `n_ex` examples
            across each of the `n_t` timesteps.
        """
        # 检查网络是否可训练
        assert self.cell.trainable, "Layer is frozen"
        dLdX = []
        n_ex, n_out, n_t = dLdA.shape
        # 逆序遍历每个时间步
        for t in reversed(range(n_t)):
            # 在每个时间步上运行 cell 的反向传播
            dLdXt = self.cell.backward(dLdA[:, :, t])
            dLdX.insert(0, dLdXt)
        dLdX = np.dstack(dLdX)
        return dLdX

    @property
    def derived_variables(self):
        """
        Return a dictionary containing any intermediate variables computed
        during the forward / backward passes.
        """
        # 返回包含在前向/后向传递过程中计算的任何中间变量的字典
        return self.cell.derived_variables

    @property
    def gradients(self):
        """
        Return a dictionary of the gradients computed during the backward
        pass
        """
        # 返回在反向传递过程中计算的梯度的字典
        return self.cell.gradients

    @property
    def parameters(self):
        """Return a dictionary of the current layer parameters"""
        # 返回当前层参数的字典
        return self.cell.parameters

    def set_params(self, summary_dict):
        """
        Set the layer parameters from a dictionary of values.

        Parameters
        ----------
        summary_dict : dict
            A dictionary of layer parameters and hyperparameters. If a required
            parameter or hyperparameter is not included within `summary_dict`,
            this method will use the value in the current layer's
            :meth:`summary` method.

        Returns
        -------
        layer : :doc:`Layer <numpy_ml.neural_nets.layers>` object
            The newly-initialized layer.
        """
        # 从值字典中设置层参数
        self = super().set_params(summary_dict)
        return self.cell.set_parameters(summary_dict)

    def freeze(self):
        """
        Freeze the layer parameters at their current values so they can no
        longer be updated.
        """
        # 冻结层参数，使其无法再更新
        self.cell.freeze()

    def unfreeze(self):
        """Unfreeze the layer parameters so they can be updated."""
        # 解冻层参数，使其可以更新
        self.cell.unfreeze()

    def flush_gradients(self):
        """Erase all the layer's derived variables and gradients."""
        # 擦除所有层的派生变量和梯度
        self.cell.flush_gradients()

    def update(self):
        """
        Update the layer parameters using the accrued gradients and layer
        optimizer. Flush all gradients once the update is complete.
        """
        # 使用累积的梯度和层优化器更新层参数。更新完成后刷新所有梯度
        self.cell.update()
        self.flush_gradients()
class LSTM(LayerBase):
    def __init__(
        self,
        n_out,
        act_fn="Tanh",
        gate_fn="Sigmoid",
        init="glorot_uniform",
        optimizer=None,
    ):
        """
        A single long short-term memory (LSTM) RNN layer.

        Parameters
        ----------
        n_out : int
            The dimension of a single hidden state / output on a given timestep.
        act_fn : str, :doc:`Activation <numpy_ml.neural_nets.activations>` object, or None
            The activation function for computing ``A[t]``. Default is `'Tanh'`.
        gate_fn : str, :doc:`Activation <numpy_ml.neural_nets.activations>` object, or None
            The gate function for computing the update, forget, and output
            gates. Default is `'Sigmoid'`.
        init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
            The weight initialization strategy. Default is `'glorot_uniform'`.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the :class:`SGD
            <numpy_ml.neural_nets.optimizers.SGD>` optimizer with
            default parameters. Default is None.
        """  # noqa: E501
        # 调用父类的构造函数，传入优化器参数
        super().__init__(optimizer)

        # 初始化参数
        self.init = init
        self.n_in = None
        self.n_out = n_out
        self.n_timesteps = None
        # 初始化激活函数和门函数
        self.act_fn = ActivationInitializer(act_fn)()
        self.gate_fn = ActivationInitializer(gate_fn)()
        self.is_initialized = False

    # 初始化参数
    def _init_params(self):
        self.cell = LSTMCell(
            n_in=self.n_in,
            n_out=self.n_out,
            act_fn=self.act_fn,
            gate_fn=self.gate_fn,
            init=self.init,
        )
        self.is_initialized = True

    @property
    # 返回包含层超参数的字典
    def hyperparameters(self):
        return {
            "layer": "LSTM",
            "init": self.init,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
            "gate_fn": str(self.gate_fn),
            "optimizer": self.cell.hyperparameters["optimizer"],
        }

    # 在输入的所有时间步上运行前向传播
    def forward(self, X):
        """
        Run a forward pass across all timesteps in the input.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_in, n_t)`
            Input consisting of `n_ex` examples each of dimensionality `n_in`
            and extending for `n_t` timesteps.

        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out, n_t)`
            The value of the hidden state for each of the `n_ex` examples
            across each of the `n_t` timesteps.
        """
        # 如果未初始化，则设置输入维度并初始化参数
        if not self.is_initialized:
            self.n_in = X.shape[1]
            self._init_params()

        Y = []
        n_ex, n_in, n_t = X.shape
        # 遍历所有时间步
        for t in range(n_t):
            # 在当前时间步上运行前向传播
            yt, _ = self.cell.forward(X[:, :, t])
            Y.append(yt)
        # 沿着第三个维度堆叠结果
        return np.dstack(Y)
    def backward(self, dLdA):
        """
        Run a backward pass across all timesteps in the input.

        Parameters
        ----------
        dLdA : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_out, n_t)`
            The gradient of the loss with respect to the layer output for each
            of the `n_ex` examples across all `n_t` timesteps.

        Returns
        -------
        dLdX : :py:class:`ndarray <numpy.ndarray>` of shape (`n_ex`, `n_in`, `n_t`)
            The value of the hidden state for each of the `n_ex` examples
            across each of the `n_t` timesteps.
        """  # noqa: E501
        assert self.cell.trainable, "Layer is frozen"
        # 初始化一个空列表用于存储每个时间步的隐藏状态梯度
        dLdX = []
        n_ex, n_out, n_t = dLdA.shape
        # 逆序遍历所有时间步
        for t in reversed(range(n_t)):
            # 对每个时间步的梯度进行反向传播计算
            dLdXt, _ = self.cell.backward(dLdA[:, :, t])
            # 将计算得到的隐藏状态梯度插入到列表的开头
            dLdX.insert(0, dLdXt)
        # 沿着第三个维度将隐藏状态梯度堆叠起来
        dLdX = np.dstack(dLdX)
        return dLdX

    @property
    def derived_variables(self):
        """
        Return a dictionary containing any intermediate variables computed
        during the forward / backward passes.
        """
        return self.cell.derived_variables

    @property
    def gradients(self):
        """
        Return a dictionary of the gradients computed during the backward
        pass
        """
        return self.cell.gradients

    @property
    def parameters(self):
        """Return a dictionary of the current layer parameters"""
        return self.cell.parameters

    def freeze(self):
        """
        Freeze the layer parameters at their current values so they can no
        longer be updated.
        """
        # 冻结层的参数，使其不能再被更新
        self.cell.freeze()

    def unfreeze(self):
        """Unfreeze the layer parameters so they can be updated."""
        # 解冻层的参数，使其可以被更新
        self.cell.unfreeze()
    def set_params(self, summary_dict):
        """
        从一个包含值的字典中设置层的参数。

        Parameters
        ----------
        summary_dict : dict
            一个包含层参数和超参数的字典。如果在`summary_dict`中没有包含所需的参数或超参数，
            这个方法将使用当前层的:meth:`summary`方法中的值。

        Returns
        -------
        layer : :doc:`Layer <numpy_ml.neural_nets.layers>` object
            新初始化的层。
        """
        # 调用父类的set_params方法设置参数
        self = super().set_params(summary_dict)
        # 调用cell对象的set_parameters方法设置参数
        return self.cell.set_parameters(summary_dict)

    def flush_gradients(self):
        """擦除所有层的派生变量和梯度。"""
        # 调用cell对象的flush_gradients方法
        self.cell.flush_gradients()

    def update(self):
        """
        使用累积的梯度和层优化器更新层参数。更新完成后清除所有梯度。
        """
        # 调用cell对象的update方法
        self.cell.update()
        # 调用flush_gradients方法清除所有梯度
        self.flush_gradients()
```