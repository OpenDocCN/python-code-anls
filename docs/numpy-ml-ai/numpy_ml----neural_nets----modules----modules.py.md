# `numpy-ml\numpy_ml\neural_nets\modules\modules.py`

```py
# 从 abc 模块导入 ABC 和 abstractmethod 装饰器
from abc import ABC, abstractmethod

# 导入 re 模块用于正则表达式操作
import re
# 导入 numpy 模块并重命名为 np
import numpy as np

# 从当前包的 wrappers 模块中导入 Dropout 类
from ..wrappers import Dropout
# 从当前包的 utils 模块中导入 calc_pad_dims_2D 函数
from ..utils import calc_pad_dims_2D
# 从当前包的 activations 模块中导入 Tanh, Sigmoid, ReLU, LeakyReLU, Affine 类
from ..activations import Tanh, Sigmoid, ReLU, LeakyReLU, Affine
# 从当前包的 layers 模块中导入各种层类
from ..layers import (
    DotProductAttention,
    FullyConnected,
    BatchNorm2D,
    Conv1D,
    Conv2D,
    Multiply,
    LSTMCell,
    Add,
)

# 定义一个抽象基类 ModuleBase
class ModuleBase(ABC):
    # 初始化方法
    def __init__(self):
        # 初始化 X 属性为 None
        self.X = None
        # 初始化 trainable 属性为 True
        self.trainable = True

        # 调用父类的初始化方法
        super().__init__()

    # 抽象方法，用于初始化参数
    @abstractmethod
    def _init_params(self, **kwargs):
        raise NotImplementedError

    # 抽象方法，前向传播
    @abstractmethod
    def forward(self, z, **kwargs):
        raise NotImplementedError

    # 抽象方法，反向传播
    @abstractmethod
    def backward(self, out, **kwargs):
        raise NotImplementedError

    # components 属性，返回组件列表
    @property
    def components(self):
        comps = []
        # 遍历组件列表中的组件 ID
        for c in self.hyperparameters["component_ids"]:
            # 如果当前对象有该组件，则添加到 comps 列表中
            if hasattr(self, c):
                comps.append(getattr(self, c))
        return comps

    # 冻结方法，将当前层及其组件设为不可训练
    def freeze(self):
        self.trainable = False
        for c in self.components:
            c.freeze()

    # 解冻方法，将当前层及其组件设为可训练
    def unfreeze(self):
        self.trainable = True
        for c in self.components:
            c.unfreeze()

    # 更新方法，更新参数
    def update(self, cur_loss=None):
        assert self.trainable, "Layer is frozen"
        for c in self.components:
            c.update(cur_loss)
        self.flush_gradients()

    # 清空梯度方法
    def flush_gradients(self):
        assert self.trainable, "Layer is frozen"

        # 清空梯度相关变量
        self.X = []
        self._dv = {}
        for c in self.components:
            for k, v in c.derived_variables.items():
                c.derived_variables[k] = None

            for k, v in c.gradients.items():
                c.gradients[k] = np.zeros_like(v)
    # 设置模型的参数，根据传入的字典 summary_dict
    def set_params(self, summary_dict):
        # 获取组件的 ID 列表
        cids = self.hyperparameters["component_ids"]
        # 遍历 summary_dict 中的参数
        for k, v in summary_dict["parameters"].items():
            # 如果参数是 "components"，则进一步处理
            if k == "components":
                # 遍历组件参数字典
                for c, cd in summary_dict["parameters"][k].items():
                    # 如果组件在组件 ID 列表中，则设置组件的参数
                    if c in cids:
                        getattr(self, c).set_params(cd)

            # 如果参数在模型的参数列表中，则更新模型的参数值
            elif k in self.parameters:
                self.parameters[k] = v

        # 遍历 summary_dict 中的超参数
        for k, v in summary_dict["hyperparameters"].items():
            # 如果参数是 "components"，则进一步处理
            if k == "components":
                # 遍历组件超参数字典
                for c, cd in summary_dict["hyperparameters"][k].items():
                    # 如果组件在组件 ID 列表中，则设置组件的超参数
                    if c in cids:
                        getattr(self, c).set_params(cd)

            # 如果超参数在模型的超参数列表中，则更新模型的超参数值
            if k in self.hyperparameters:
                # 根据不同的超参数值类型进行处理
                if k == "act_fn" and v == "ReLU":
                    self.hyperparameters[k] = ReLU()
                elif v == "act_fn" and v == "Sigmoid":
                    self.hyperparameters[k] = Sigmoid()
                elif v == "act_fn" and v == "Tanh":
                    self.hyperparameters[k] = Tanh()
                elif v == "act_fn" and "Affine" in v:
                    # 解析 Affine 函数的参数值
                    r = r"Affine\(slope=(.*), intercept=(.*)\)"
                    slope, intercept = re.match(r, v).groups()
                    self.hyperparameters[k] = Affine(float(slope), float(intercept))
                elif v == "act_fn" and "Leaky ReLU" in v:
                    # 解析 Leaky ReLU 函数的参数值
                    r = r"Leaky ReLU\(alpha=(.*)\)"
                    alpha = re.match(r, v).groups()[0]
                    self.hyperparameters[k] = LeakyReLU(float(alpha))
                else:
                    # 其他情况直接更新超参数值
                    self.hyperparameters[k] = v

    # 返回模型的摘要信息，包括参数、层信息和超参数
    def summary(self):
        return {
            "parameters": self.parameters,
            "layer": self.hyperparameters["layer"],
            "hyperparameters": self.hyperparameters,
        }
class WavenetResidualModule(ModuleBase):
    # 定义一个 Wavenet 残差模块类，继承自 ModuleBase 类
    def __init__(
        self,
        ch_residual,
        ch_dilation,
        dilation,
        kernel_width,
        optimizer=None,
        init="glorot_uniform",
    # 初始化函数，接受残差通道数、扩张通道数、扩张率、卷积核宽度、优化器和初始化方式等参数
    def _init_params(self):
        # 初始化参数字典
        self._dv = {}

        # 创建扩张卷积层对象
        self.conv_dilation = Conv1D(
            stride=1,
            pad="causal",
            init=self.init,
            kernel_width=2,
            dilation=self.dilation,
            out_ch=self.ch_dilation,
            optimizer=self.optimizer,
            act_fn=Affine(slope=1, intercept=0),
        )

        # 创建 Tanh 激活函数对象
        self.tanh = Tanh()
        # 创建 Sigmoid 激活函数对象
        self.sigm = Sigmoid()
        # 创建 Multiply 门对象
        self.multiply_gate = Multiply(act_fn=Affine(slope=1, intercept=0))

        # 创建 1x1 卷积层对象
        self.conv_1x1 = Conv1D(
            stride=1,
            pad="same",
            dilation=0,
            init=self.init,
            kernel_width=1,
            out_ch=self.ch_residual,
            optimizer=self.optimizer,
            act_fn=Affine(slope=1, intercept=0),
        )

        # 创建残差相加层对象
        self.add_residual = Add(act_fn=Affine(slope=1, intercept=0))
        # 创建跳跃连接相加层对象
        self.add_skip = Add(act_fn=Affine(slope=1, intercept=0))

    @property
    def parameters(self):
        """A dictionary of the module parameters."""
        # 返回模块参数的字典
        return {
            "components": {
                "conv_1x1": self.conv_1x1.parameters,
                "add_skip": self.add_skip.parameters,
                "add_residual": self.add_residual.parameters,
                "conv_dilation": self.conv_dilation.parameters,
                "multiply_gate": self.multiply_gate.parameters,
            }
        }

    @property
    # 返回模块的参数
    # 返回模块的超参数字典
    def hyperparameters(self):
        """A dictionary of the module hyperparameters"""
        return {
            "layer": "WavenetResidualModule",
            "init": self.init,
            "dilation": self.dilation,
            "optimizer": self.optimizer,
            "ch_residual": self.ch_residual,
            "ch_dilation": self.ch_dilation,
            "kernel_width": self.kernel_width,
            "component_ids": [
                "conv_1x1",
                "add_skip",
                "add_residual",
                "conv_dilation",
                "multiply_gate",
            ],
            "components": {
                "conv_1x1": self.conv_1x1.hyperparameters,
                "add_skip": self.add_skip.hyperparameters,
                "add_residual": self.add_residual.hyperparameters,
                "conv_dilation": self.conv_dilation.hyperparameters,
                "multiply_gate": self.multiply_gate.hyperparameters,
            },
        }

    # 返回计算过程中前向/后向传播期间计算的中间值的字典
    @property
    def derived_variables(self):
        """A dictionary of intermediate values computed during the
        forward/backward passes."""
        dv = {
            "conv_1x1_out": None,
            "conv_dilation_out": None,
            "multiply_gate_out": None,
            "components": {
                "conv_1x1": self.conv_1x1.derived_variables,
                "add_skip": self.add_skip.derived_variables,
                "add_residual": self.add_residual.derived_variables,
                "conv_dilation": self.conv_dilation.derived_variables,
                "multiply_gate": self.multiply_gate.derived_variables,
            },
        }
        # 更新中间值字典
        dv.update(self._dv)
        return dv

    @property
    # 返回模块参数梯度的字典
    def gradients(self):
        # 返回包含各组件参数梯度的字典
        return {
            "components": {
                # 获取 conv_1x1 组件的参数梯度
                "conv_1x1": self.conv_1x1.gradients,
                # 获取 add_skip 组件的参数梯度
                "add_skip": self.add_skip.gradients,
                # 获取 add_residual 组件的参数梯度
                "add_residual": self.add_residual.gradients,
                # 获取 conv_dilation 组件的参数梯度
                "conv_dilation": self.conv_dilation.gradients,
                # 获取 multiply_gate 组件的参数梯度
                "multiply_gate": self.multiply_gate.gradients,
            }
        }
    def forward(self, X_main, X_skip=None):
        """
        Compute the module output on a single minibatch.

        Parameters
        ----------
        X_main : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
            The input volume consisting of `n_ex` examples, each with dimension
            (`in_rows`, `in_cols`, `in_ch`).
        X_skip : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`, or None
            The output of the preceding skip-connection if this is not the
            first module in the network.

        Returns
        -------
        Y_main : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows, out_cols, out_ch)`
            The output of the main pathway.
        Y_skip : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows, out_cols, out_ch)`
            The output of the skip-connection pathway.
        """
        # 设置输入数据和跳跃连接数据
        self.X_main, self.X_skip = X_main, X_skip
        # 计算卷积扩张层的输出
        conv_dilation_out = self.conv_dilation.forward(X_main)

        # 计算tanh门的输出
        tanh_gate = self.tanh.fn(conv_dilation_out)
        # 计算sigmoid门的输出
        sigm_gate = self.sigm.fn(conv_dilation_out)

        # 计算门控乘积的输出
        multiply_gate_out = self.multiply_gate.forward([tanh_gate, sigm_gate])
        # 计算1x1卷积层的输出
        conv_1x1_out = self.conv_1x1.forward(multiply_gate_out)

        # 如果这是第一个Wavenet块，则将“前一个”跳跃连接和卷积1x1输出的和初始化为0
        self.X_skip = np.zeros_like(conv_1x1_out) if X_skip is None else X_skip

        # 计算跳跃连接路径的输出
        Y_skip = self.add_skip.forward([X_skip, conv_1x1_out])
        # 计算主路径的输出
        Y_main = self.add_residual.forward([X_main, conv_1x1_out])

        # 保存各个中间结果，以便后续调试
        self._dv["tanh_out"] = tanh_gate
        self._dv["sigm_out"] = sigm_gate
        self._dv["conv_dilation_out"] = conv_dilation_out
        self._dv["multiply_gate_out"] = multiply_gate_out
        self._dv["conv_1x1_out"] = conv_1x1_out
        # 返回主路径和跳跃连接路径的输出
        return Y_main, Y_skip
    # 反向传播函数，计算梯度并返回
    def backward(self, dY_skip, dY_main=None):
        # 调用 add_skip 模块的反向传播函数，计算 skip path 的梯度和输出
        dX_skip, dConv_1x1_out = self.add_skip.backward(dY_skip)

        # 如果这是最后一个 wavenet block，dY_main 将为 None。如果不是，
        # 计算来自 dY_main 的误差贡献并添加到 skip path 的贡献中
        dX_main = np.zeros_like(self.X_main)
        if dY_main is not None:
            # 调用 add_residual 模块的反向传播函数，计算 main path 的梯度和输出
            dX_main, dConv_1x1_main = self.add_residual.backward(dY_main)
            dConv_1x1_out += dConv_1x1_main

        # 调用 conv_1x1 模块的反向传播函数，计算梯度并返回
        dMultiply_out = self.conv_1x1.backward(dConv_1x1_out)
        # 调用 multiply_gate 模块的反向传播函数，计算梯度并返回
        dTanh_out, dSigm_out = self.multiply_gate.backward(dMultiply_out)

        # 获取派生变量中的 conv_dilation_out
        conv_dilation_out = self.derived_variables["conv_dilation_out"]
        # 计算 dTanh_in 和 dSigm_in，并乘以对应激活函数的梯度
        dTanh_in = dTanh_out * self.tanh.grad(conv_dilation_out)
        dSigm_in = dSigm_out * self.sigm.grad(conv_dilation_out)
        # 计算 dDilation_out
        dDilation_out = dTanh_in + dSigm_in

        # 调用 conv_dilation 模块的反向传播函数，计算梯度并返回
        conv_back = self.conv_dilation.backward(dDilation_out)
        dX_main += conv_back

        # 存储各个梯度到派生变量中
        self._dv["dLdTanh"] = dTanh_out
        self._dv["dLdSigmoid"] = dSigm_out
        self._dv["dLdConv_1x1"] = dConv_1x1_out
        self._dv["dLdMultiply"] = dMultiply_out
        self._dv["dLdConv_dilation"] = dDilation_out
        # 返回 main path 和 skip path 的梯度
        return dX_main, dX_skip
class SkipConnectionIdentityModule(ModuleBase):
    # 定义一个继承自 ModuleBase 的 SkipConnectionIdentityModule 类
    def __init__(
        self,
        out_ch,
        kernel_shape1,
        kernel_shape2,
        stride1=1,
        stride2=1,
        act_fn=None,
        epsilon=1e-5,
        momentum=0.9,
        optimizer=None,
        init="glorot_uniform",
    # 初始化函数，接受多个参数，包括输出通道数、卷积核形状等
    def _init_params(self):
        # 初始化参数字典
        self._dv = {}

        # 创建第一个卷积层对象
        self.conv1 = Conv2D(
            pad="same",
            init=self.init,
            out_ch=self.out_ch,
            act_fn=self.act_fn,
            stride=self.stride1,
            optimizer=self.optimizer,
            kernel_shape=self.kernel_shape1,
        )
        # 无法初始化 `conv2`，需要 X 的维度；参见 `forward` 获取更多细节
        self.batchnorm1 = BatchNorm2D(epsilon=self.epsilon, momentum=self.momentum)
        self.batchnorm2 = BatchNorm2D(epsilon=self.epsilon, momentum=self.momentum)
        self.add3 = Add(self.act_fn)

    def _init_conv2(self):
        # 创建第二个卷积层对象
        self.conv2 = Conv2D(
            pad="same",
            init=self.init,
            out_ch=self.in_ch,
            stride=self.stride2,
            optimizer=self.optimizer,
            kernel_shape=self.kernel_shape2,
            act_fn=Affine(slope=1, intercept=0),
        )

    @property
    def parameters(self):
        """A dictionary of the module parameters."""
        # 返回模块参数的字典
        return {
            "components": {
                "add3": self.add3.parameters,
                "conv1": self.conv1.parameters,
                "conv2": self.conv2.parameters,
                "batchnorm1": self.batchnorm1.parameters,
                "batchnorm2": self.batchnorm2.parameters,
            }
        }

    @property
    # 返回模块的参数
    # 返回模块的超参数字典
    def hyperparameters(self):
        """A dictionary of the module hyperparameters."""
        return {
            "layer": "SkipConnectionIdentityModule",
            "init": self.init,
            "in_ch": self.in_ch,
            "out_ch": self.out_ch,
            "epsilon": self.epsilon,
            "stride1": self.stride1,
            "stride2": self.stride2,
            "momentum": self.momentum,
            "optimizer": self.optimizer,
            "act_fn": str(self.act_fn),
            "kernel_shape1": self.kernel_shape1,
            "kernel_shape2": self.kernel_shape2,
            "component_ids": ["conv1", "batchnorm1", "conv2", "batchnorm2", "add3"],
            "components": {
                "add3": self.add3.hyperparameters,
                "conv1": self.conv1.hyperparameters,
                "conv2": self.conv2.hyperparameters,
                "batchnorm1": self.batchnorm1.hyperparameters,
                "batchnorm2": self.batchnorm2.hyperparameters,
            },
        }

    # 返回模块计算过程中的中间值字典
    @property
    def derived_variables(self):
        """A dictionary of intermediate values computed during the
        forward/backward passes."""
        dv = {
            "conv1_out": None,
            "conv2_out": None,
            "batchnorm1_out": None,
            "batchnorm2_out": None,
            "components": {
                "add3": self.add3.derived_variables,
                "conv1": self.conv1.derived_variables,
                "conv2": self.conv2.derived_variables,
                "batchnorm1": self.batchnorm1.derived_variables,
                "batchnorm2": self.batchnorm2.derived_variables,
            },
        }
        # 更新中间值字典
        dv.update(self._dv)
        return dv

    @property
    # 返回累积模块参数梯度的字典
    def gradients(self):
        return {
            "components": {
                "add3": self.add3.gradients,
                "conv1": self.conv1.gradients,
                "conv2": self.conv2.gradients,
                "batchnorm1": self.batchnorm1.gradients,
                "batchnorm2": self.batchnorm2.gradients,
            }
        }

    # 计算给定输入体积 X 的模块输出
    def forward(self, X, retain_derived=True):
        """
        Compute the module output given input volume `X`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape (n_ex, in_rows, in_cols, in_ch)
            The input volume consisting of `n_ex` examples, each with dimension
            (`in_rows`, `in_cols`, `in_ch`).
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through wrt. this input. Default
            is True.

        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape (n_ex, out_rows, out_cols, out_ch)
            The module output volume.
        """
        # 如果 self 没有属性 "conv2"，则初始化 conv2
        if not hasattr(self, "conv2"):
            self.in_ch = X.shape[3]
            self._init_conv2()

        # 计算 conv1 的输出
        conv1_out = self.conv1.forward(X, retain_derived)
        # 计算 batchnorm1 的输出
        bn1_out = self.batchnorm1.forward(conv1_out, retain_derived)
        # 计算 conv2 的输出
        conv2_out = self.conv2.forward(bn1_out, retain_derived)
        # 计算 batchnorm2 的输出
        bn2_out = self.batchnorm2.forward(conv2_out, retain_derived)
        # 计算 add3 的输出
        Y = self.add3.forward([X, bn2_out], retain_derived)

        # 如果 retain_derived 为 True，则保存中间变量以备后用
        if retain_derived:
            self._dv["conv1_out"] = conv1_out
            self._dv["conv2_out"] = conv2_out
            self._dv["batchnorm1_out"] = bn1_out
            self._dv["batchnorm2_out"] = bn2_out
        # 返回模块输出
        return Y
    def backward(self, dLdY, retain_grads=True):
        """
        Compute the gradient of the loss with respect to the layer parameters.

        Parameters
        ----------
        dLdy : :py:class:`ndarray <numpy.ndarray>` of shape (`n_ex, out_rows, out_cols, out_ch`) or list of arrays
            The gradient(s) of the loss with respect to the module output(s).
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.

        Returns
        -------
        dX : :py:class:`ndarray <numpy.ndarray>` of shape (n_ex, in_rows, in_cols, in_ch)
            The gradient of the loss with respect to the module input volume.
        """
        # Compute the backward pass for the add3 layer and get the gradient of the loss with respect to the input and output of add3
        dX, dBn2_out = self.add3.backward(dLdY, retain_grads)
        # Compute the backward pass for the batchnorm2 layer using the gradient from add3 layer
        dConv2_out = self.batchnorm2.backward(dBn2_out, retain_grads)
        # Compute the backward pass for the conv2 layer using the gradient from batchnorm2 layer
        dBn1_out = self.conv2.backward(dConv2_out, retain_grads)
        # Compute the backward pass for the batchnorm1 layer using the gradient from conv2 layer
        dConv1_out = self.batchnorm1.backward(dBn1_out, retain_grads)
        # Compute the backward pass for the conv1 layer using the gradient from batchnorm1 layer and add it to the existing gradient
        dX += self.conv1.backward(dConv1_out, retain_grads)

        # Store the gradients for each layer in the internal dictionary for reference
        self._dv["dLdAdd3_X"] = dX
        self._dv["dLdBn2"] = dBn2_out
        self._dv["dLdBn1"] = dBn1_out
        self._dv["dLdConv2"] = dConv2_out
        self._dv["dLdConv1"] = dConv1_out
        # Return the final gradient of the loss with respect to the module input volume
        return dX
class SkipConnectionConvModule(ModuleBase):
    # 定义一个继承自 ModuleBase 的 SkipConnectionConvModule 类
    def __init__(
        self,
        out_ch1,
        out_ch2,
        kernel_shape1,
        kernel_shape2,
        kernel_shape_skip,
        pad1=0,
        pad2=0,
        stride1=1,
        stride2=1,
        act_fn=None,
        epsilon=1e-5,
        momentum=0.9,
        stride_skip=1,
        optimizer=None,
        init="glorot_uniform",
    ):
        # 初始化函数，接受多个参数，包括卷积层的参数、激活函数、优化器等
    def _init_params(self, X=None):
        # 初始化参数函数，接受输入 X，但在此处未使用
        self._dv = {}
        # 初始化一个空字典 _dv
        self.conv1 = Conv2D(
            pad=self.pad1,
            init=self.init,
            act_fn=self.act_fn,
            out_ch=self.out_ch1,
            stride=self.stride1,
            optimizer=self.optimizer,
            kernel_shape=self.kernel_shape1,
        )
        # 初始化第一个卷积层，设置卷积参数、激活函数、输出通道数等
        self.conv2 = Conv2D(
            pad=self.pad2,
            init=self.init,
            out_ch=self.out_ch2,
            stride=self.stride2,
            optimizer=self.optimizer,
            kernel_shape=self.kernel_shape2,
            act_fn=Affine(slope=1, intercept=0),
        )
        # 初始化第二个卷积层，设置卷积参数、输出通道数、激活函数等
        # 无法初始化 `conv_skip`，需要 X 的维度；参见 `forward` 获取更多细节
        self.batchnorm1 = BatchNorm2D(epsilon=self.epsilon, momentum=self.momentum)
        # 初始化第一个批归一化层，设置 epsilon 和 momentum 参数
        self.batchnorm2 = BatchNorm2D(epsilon=self.epsilon, momentum=self.momentum)
        # 初始化第二个批归一化层，设置 epsilon 和 momentum 参数
        self.batchnorm_skip = BatchNorm2D(epsilon=self.epsilon, momentum=self.momentum)
        # 初始化跳跃连接的批归一化层，设置 epsilon 和 momentum 参数
        self.add3 = Add(self.act_fn)
        # 初始化一个加法层，使用给定的激活函数
    # 计算卷积层之间的填充大小
    def _calc_skip_padding(self, X):
        # 初始化填充列表
        pads = []
        # 遍历每个填充参数
        for p in [self.pad1, self.pad2]:
            # 如果填充参数是整数，则转换为四元组
            if isinstance(p, int):
                pads.append((p, p, p, p))
            # 如果填充参数是二元组且长度为2，则转换为四元组
            elif isinstance(p, tuple) and len(p) == 2:
                pads.append((p[0], p[0], p[1], p[1])
        # 更新填充参数
        self.pad1, self.pad2 = pads

        # 计算卷积1输出的维度
        s1 = self.stride1
        fr1, fc1 = self.kernel_shape1
        _, in_rows, in_cols, _ = X.shape
        pr11, pr12, pc11, pc12 = self.pad1

        out_rows1 = np.floor(1 + (in_rows + pr11 + pr12 - fr1) / s1).astype(int)
        out_cols1 = np.floor(1 + (in_cols + pc11 + pc12 - fc1) / s1).astype(int)

        # 计算卷积2输出的维度
        s2 = self.stride2
        fr2, fc2 = self.kernel_shape2
        pr21, pr22, pc21, pc22 = self.pad2

        out_rows2 = np.floor(1 + (out_rows1 + pr21 + pr22 - fr2) / s2).astype(int)
        out_cols2 = np.floor(1 + (out_cols1 + pc21 + pc22 - fc2) / s2).astype(int)

        # 最后，计算跳跃卷积的适当填充维度
        desired_dims = (out_rows2, out_cols2)
        self.pad_skip = calc_pad_dims_2D(
            X.shape,
            desired_dims,
            stride=self.stride_skip,
            kernel_shape=self.kernel_shape_skip,
        )

    # 初始化跳跃卷积层
    def _init_conv_skip(self, X):
        # 计算跳跃卷积的填充大小
        self._calc_skip_padding(X)
        # 创建跳跃卷积层对象
        self.conv_skip = Conv2D(
            init=self.init,
            pad=self.pad_skip,
            out_ch=self.out_ch2,
            stride=self.stride_skip,
            kernel_shape=self.kernel_shape_skip,
            act_fn=Affine(slope=1, intercept=0),
            optimizer=self.optimizer,
        )

    # 属性方法
    @property
    # 返回模块参数的字典
    def parameters(self):
        """A dictionary of the module parameters."""
        return {
            # 返回包含各组件参数的字典
            "components": {
                # 添加3的参数
                "add3": self.add3.parameters,
                # 卷积层1的参数
                "conv1": self.conv1.parameters,
                # 卷积层2的参数
                "conv2": self.conv2.parameters,
                # 如果存在跳跃连接的卷积层，返回其参数；否则返回None
                "conv_skip": self.conv_skip.parameters
                if hasattr(self, "conv_skip")
                else None,
                # 批归一化层1的参数
                "batchnorm1": self.batchnorm1.parameters,
                # 批归一化层2的参数
                "batchnorm2": self.batchnorm2.parameters,
                # 如果存在跳跃连接的批归一化层，返回其参数
                "batchnorm_skip": self.batchnorm_skip.parameters,
            }
        }

    @property
    # 返回模块超参数的字典
    def hyperparameters(self):
        """A dictionary of the module hyperparameters."""
        return {
            "layer": "SkipConnectionConvModule",
            "init": self.init,
            "pad1": self.pad1,
            "pad2": self.pad2,
            "in_ch": self.in_ch,
            "out_ch1": self.out_ch1,
            "out_ch2": self.out_ch2,
            "epsilon": self.epsilon,
            "stride1": self.stride1,
            "stride2": self.stride2,
            "momentum": self.momentum,
            "act_fn": str(self.act_fn),
            "stride_skip": self.stride_skip,
            "kernel_shape1": self.kernel_shape1,
            "kernel_shape2": self.kernel_shape2,
            "kernel_shape_skip": self.kernel_shape_skip,
            "pad_skip": self.pad_skip if hasattr(self, "pad_skip") else None,
            "component_ids": [
                "add3",
                "conv1",
                "conv2",
                "conv_skip",
                "batchnorm1",
                "batchnorm2",
                "batchnorm_skip",
            ],
            "components": {
                "add3": self.add3.hyperparameters,
                "conv1": self.conv1.hyperparameters,
                "conv2": self.conv2.hyperparameters,
                "conv_skip": self.conv_skip.hyperparameters
                if hasattr(self, "conv_skip")
                else None,
                "batchnorm1": self.batchnorm1.hyperparameters,
                "batchnorm2": self.batchnorm2.hyperparameters,
                "batchnorm_skip": self.batchnorm_skip.hyperparameters,
            },
        }

    @property
    # 计算前向/后向传播过程中计算的中间值的字典
    def derived_variables(self):
        dv = {
            "conv1_out": None,
            "conv2_out": None,
            "conv_skip_out": None,
            "batchnorm1_out": None,
            "batchnorm2_out": None,
            "batchnorm_skip_out": None,
            "components": {
                "add3": self.add3.derived_variables,  # 计算 add3 模块的派生变量
                "conv1": self.conv1.derived_variables,  # 计算 conv1 模块的派生变量
                "conv2": self.conv2.derived_variables,  # 计算 conv2 模块的派生变量
                "conv_skip": self.conv_skip.derived_variables  # 如果存在 conv_skip 模块，则计算其派生变量
                if hasattr(self, "conv_skip")  # 检查是否存在 conv_skip 模块
                else None,
                "batchnorm1": self.batchnorm1.derived_variables,  # 计算 batchnorm1 模块的派生变量
                "batchnorm2": self.batchnorm2.derived_variables,  # 计算 batchnorm2 模块的派生变量
                "batchnorm_skip": self.batchnorm_skip.derived_variables,  # 计算 batchnorm_skip 模块的派生变量
            },
        }
        # 更新派生变量字典
        dv.update(self._dv)
        return dv

    @property
    # 累积模块参数梯度的字典
    def gradients(self):
        return {
            "components": {
                "add3": self.add3.gradients,  # 获取 add3 模块的梯度
                "conv1": self.conv1.gradients,  # 获取 conv1 模块的梯度
                "conv2": self.conv2.gradients,  # 获取 conv2 模块的梯度
                "conv_skip": self.conv_skip.gradients  # 获取 conv_skip 模块的梯度
                if hasattr(self, "conv_skip")  # 检查是否存在 conv_skip 模块
                else None,
                "batchnorm1": self.batchnorm1.gradients,  # 获取 batchnorm1 模块的梯度
                "batchnorm2": self.batchnorm2.gradients,  # 获取 batchnorm2 模块的梯度
                "batchnorm_skip": self.batchnorm_skip.gradients,  # 获取 batchnorm_skip 模块的梯度
            }
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
            The module output volume.
        """
        # 现在我们有了输入 X 的维度，可以初始化 `conv_skip` 层的正确填充
        if not hasattr(self, "conv_skip"):
            self._init_conv_skip(X)
            self.in_ch = X.shape[3]

        # 计算第一个卷积层的输出
        conv1_out = self.conv1.forward(X, retain_derived)
        # 计算第一个批归一化层的输出
        bn1_out = self.batchnorm1.forward(conv1_out, retain_derived)
        # 计算第二个卷积层的输出
        conv2_out = self.conv2.forward(bn1_out, retain_derived)
        # 计算第二个批归一化层的输出
        bn2_out = self.batchnorm2.forward(conv2_out, retain_derived)
        # 计算跳跃连接卷积层的输出
        conv_skip_out = self.conv_skip.forward(X, retain_derived)
        # 计算跳跃连接批归一化层的输出
        bn_skip_out = self.batchnorm_skip.forward(conv_skip_out, retain_derived)
        # 计算三个层的输出相加的结果
        Y = self.add3.forward([bn_skip_out, bn2_out], retain_derived)

        # 如果需要保留派生变量，则将它们保存在 _dv 字典中
        if retain_derived:
            self._dv["conv1_out"] = conv1_out
            self._dv["conv2_out"] = conv2_out
            self._dv["batchnorm1_out"] = bn1_out
            self._dv["batchnorm2_out"] = bn2_out
            self._dv["conv_skip_out"] = conv_skip_out
            self._dv["batchnorm_skip_out"] = bn_skip_out
        # 返回模块的输出
        return Y
    def backward(self, dLdY, retain_grads=True):
        """
        Compute the gradient of the loss with respect to the module parameters.

        Parameters
        ----------
        dLdy : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, out_rows, out_cols, out_ch)`
        or list of arrays
            The gradient(s) of the loss with respect to the module output(s).
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.

        Returns
        -------
        dX : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, in_rows, in_cols, in_ch)`
            The gradient of the loss with respect to the module input volume.
        """
        # 计算 Add3 模块的反向传播，得到对应的梯度
        dBnskip_out, dBn2_out = self.add3.backward(dLdY)
        # 计算 BatchNorm 模块的反向传播，得到对应的梯度
        dConvskip_out = self.batchnorm_skip.backward(dBnskip_out)
        # 计算 Convolution 模块的反向传播，得到对应的梯度
        dX = self.conv_skip.backward(dConvskip_out)

        # 计算 BatchNorm2 模块的反向传播，得到对应的梯度
        dConv2_out = self.batchnorm2.backward(dBn2_out)
        # 计算 Convolution2 模块的反向传播，得到对应的梯度
        dBn1_out = self.conv2.backward(dConv2_out)
        # 计算 BatchNorm1 模块的反向传播，得到对应的梯度
        dConv1_out = self.batchnorm1.backward(dBn1_out)
        # 将 Convolution1 模块的反向传播梯度加到之前的梯度上
        dX += self.conv1.backward(dConv1_out)

        # 如果需要保留中间参数梯度，则将它们保存在对应的变量中
        if retain_grads:
            self._dv["dLdAdd3_X"] = dX
            self._dv["dLdBn1"] = dBn1_out
            self._dv["dLdBn2"] = dBn2_out
            self._dv["dLdConv1"] = dConv1_out
            self._dv["dLdConv2"] = dConv2_out
            self._dv["dLdBnSkip"] = dBnskip_out
            self._dv["dLdConvSkip"] = dConvskip_out
        # 返回输入体积的梯度
        return dX
class BidirectionalLSTM(ModuleBase):
    # 定义一个双向长短期记忆（LSTM）层
    def __init__(
        self,
        n_out,
        act_fn=None,
        gate_fn=None,
        merge_mode="concat",
        init="glorot_uniform",
        optimizer=None,
    ):
        """
        A single bidirectional long short-term memory (LSTM) layer.

        Parameters
        ----------
        n_out : int
            The dimension of a single hidden state / output on a given timestep
        act_fn : :doc:`Activation <numpy_ml.neural_nets.activations>` object or None
            The activation function for computing ``A[t]``. If not specified,
            use :class:`~numpy_ml.neural_nets.activations.Tanh` by default.
        gate_fn : :doc:`Activation <numpy_ml.neural_nets.activations>` object or None
            The gate function for computing the update, forget, and output
            gates. If not specified, use
            :class:`~numpy_ml.neural_nets.activations.Sigmoid` by default.
        merge_mode : {"sum", "multiply", "concat", "average"}
            Mode by which outputs of the forward and backward LSTMs will be
            combined. Default is 'concat'.
        optimizer : str or :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object or None
            The optimization strategy to use when performing gradient updates
            within the `update` method.  If None, use the
            :class:`~numpy_ml.neural_nets.optimizers.SGD` optimizer with
            default parameters. Default is None.
        init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
            The weight initialization strategy. Default is 'glorot_uniform'.
        """
        # 调用父类的构造函数
        super().__init__()

        # 初始化参数
        self.init = init
        self.n_in = None
        self.n_out = n_out
        self.optimizer = optimizer
        self.merge_mode = merge_mode
        # 如果未指定激活函数，则使用Tanh作为默认激活函数
        self.act_fn = Tanh() if act_fn is None else act_fn
        # 如果未指定门函数，则使用Sigmoid作为默认门函数
        self.gate_fn = Sigmoid() if gate_fn is None else gate_fn
        # 初始化参数
        self._init_params()
    # 初始化参数，创建前向和后向的LSTM单元
    def _init_params(self):
        self.cell_fwd = LSTMCell(
            init=self.init,
            n_out=self.n_out,
            act_fn=self.act_fn,
            gate_fn=self.gate_fn,
            optimizer=self.optimizer,
        )
        self.cell_bwd = LSTMCell(
            init=self.init,
            n_out=self.n_out,
            act_fn=self.act_fn,
            gate_fn=self.gate_fn,
            optimizer=self.optimizer,
        )

    # 前向传播函数，对输入的所有时间步进行前向传播
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
        Y_fwd, Y_bwd, Y = [], [], []  # 初始化前向、后向和合并后的隐藏状态列表
        n_ex, self.n_in, n_t = X.shape  # 获取输入数据的形状信息

        # 前向LSTM
        for t in range(n_t):
            yt, ct = self.cell_fwd.forward(X[:, :, t])  # 对每个时间步进行前向传播
            Y_fwd.append(yt)  # 将隐藏状态添加到前向隐藏状态列表中

        # 后向LSTM
        for t in reversed(range(n_t)):
            yt, ct = self.cell_bwd.forward(X[:, :, t])  # 对每个时间步进行后向传播
            Y_bwd.insert(0, yt)  # 将隐藏状态添加到后向隐藏状态列表中

        # 合并前向和后向状态
        for t in range(n_t):
            if self.merge_mode == "concat":
                Y.append(np.concatenate([Y_fwd[t], Y_bwd[t]], axis=1))  # 按照指定方式合并隐藏状态
            elif self.merge_mode == "sum":
                Y.append(Y_fwd[t] + Y_bwd[t])  # 按照指定方式合并隐藏状态
            elif self.merge_mode == "average":
                Y.append((Y_fwd[t] + Y_bwd[t]) / 2)  # 按照指定方式合并隐藏状态
            elif self.merge_mode == "multiply":
                Y.append(Y_fwd[t] * Y_bwd[t])  # 按照指定方式合并隐藏状态

        self.Y_fwd, self.Y_bwd = Y_fwd, Y_bwd  # 保存前向和后向隐藏状态列表
        return np.dstack(Y)  # 返回合并后的隐藏状态
    # 在输入的所有时间步上运行反向传播

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
        # 检查层是否可训练
        assert self.trainable, "Layer is frozen"

        # 获取输入的形状信息
        n_ex, n_out, n_t = dLdA.shape
        dLdX_f, dLdX_b, dLdX = [], [], []

        # 前向 LSTM
        for t in reversed(range(n_t)):
            # 根据合并模式选择不同的反向传播方式
            if self.merge_mode == "concat":
                dLdXt_f = self.cell_fwd.backward(dLdA[:, : self.n_out, t])
            elif self.merge_mode == "sum":
                dLdXt_f = self.cell_fwd.backward(dLdA[:, :, t])
            elif self.merge_mode == "multiplty":
                dLdXt_f = self.cell_fwd.backward(dLdA[:, :, t] * self.Y_bwd[t])
            elif self.merge_mode == "average":
                dLdXt_f = self.cell_fwd.backward(dLdA[:, :, t] * 0.5)
            dLdX_f.insert(0, dLdXt_f)

        # 后向 LSTM
        for t in range(n_t):
            # 根据合并模式选择不同的反向传播方式
            if self.merge_mode == "concat":
                dLdXt_b = self.cell_bwd.backward(dLdA[:, self.n_out :, t])
            elif self.merge_mode == "sum":
                dLdXt_b = self.cell_bwd.backward(dLdA[:, :, t])
            elif self.merge_mode == "multiplty":
                dLdXt_b = self.cell_bwd.backward(dLdA[:, :, t] * self.Y_fwd[t])
            elif self.merge_mode == "average":
                dLdXt_b = self.cell_bwd.backward(dLdA[:, :, t] * 0.5)
            dLdX_b.append(dLdXt_b)

        # 将前向和后向 LSTM 的结果相加
        for t in range(n_t):
            dLdX.append(dLdX_f[t] + dLdX_b[t])

        # 沿着第三个维度堆叠结果
        return np.dstack(dLdX)
    @property
    def derived_variables(self):
        """返回在前向/后向传递过程中计算的中间值的字典。"""
        return {
            "components": {
                "cell_fwd": self.cell_fwd.derived_variables,
                "cell_bwd": self.cell_bwd.derived_variables,
            }
        }

    @property
    def gradients(self):
        """返回累积的模块参数梯度的字典。"""
        return {
            "components": {
                "cell_fwd": self.cell_fwd.gradients,
                "cell_bwd": self.cell_bwd.gradients,
            }
        }

    @property
    def parameters(self):
        """返回模块参数的字典。"""
        return {
            "components": {
                "cell_fwd": self.cell_fwd.parameters,
                "cell_bwd": self.cell_bwd.parameters,
            }
        }

    @property
    def hyperparameters(self):
        """返回模块超参数的字典。"""
        return {
            "layer": "BidirectionalLSTM",
            "init": self.init,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
            "optimizer": self.optimizer,
            "merge_mode": self.merge_mode,
            "component_ids": ["cell_fwd", "cell_bwd"],
            "components": {
                "cell_fwd": self.cell_fwd.hyperparameters,
                "cell_bwd": self.cell_bwd.hyperparameters,
            },
        }
class MultiHeadedAttentionModule(ModuleBase):
    # 多头注意力模块类，继承自 ModuleBase 类
    def _init_params(self):
        # 初始化参数字典
        self._dv = {}

        # 假设 keys、query、values 的维度相同
        assert self.kqv_dim % self.n_heads == 0
        # 计算每个头的潜在维度
        self.latent_dim = self.kqv_dim // self.n_heads

        # 创建点积注意力对象，设置缩放参数和丢弃率
        self.attention = DotProductAttention(scale=True, dropout_p=self.dropout_p)
        # 创建投影矩阵字典，包括 Q、K、V、O 四个投影矩阵
        self.projections = {
            k: Dropout(
                FullyConnected(
                    init=self.init,
                    n_out=self.kqv_dim,
                    optimizer=self.optimizer,
                    act_fn="Affine(slope=1, intercept=0)",
                ),
                self.dropout_p,
            )
            for k in ["Q", "K", "V", "O"]
        }

        # 标记初始化完成
        self.is_initialized = True
    # 实现多头注意力机制的前向传播过程
    def forward(self, Q, K, V):
        # 如果模型未初始化，则初始化参数
        if not self.is_initialized:
            # 获取查询向量的维度
            self.kqv_dim = Q.shape[-1]
            # 初始化参数
            self._init_params()

        # 将查询、键和值投影到 `latent_dim` 维度的子空间
        n_ex = Q.shape[0]
        for k, x in zip(["Q", "K", "V"], [Q, K, V]):
            # 对输入数据进行投影
            proj = self.projections[k].forward(x)
            # 重塑投影后的数据形状
            proj = proj.reshape(n_ex, -1, self.n_heads, self.latent_dim).swapaxes(1, 2)
            self._dv["{}_proj".format(k)] = proj

        # 获取派生变量
        dv = self.derived_variables
        Q_proj, K_proj, V_proj = dv["Q_proj"], dv["K_proj"], dv["V_proj"]

        # 对投影后的向量应用缩放点积注意力机制
        attn = self.attention
        attn_out = attn.forward(Q_proj, K_proj, V_proj)
        self._dv["attention_weights"] = attn.derived_variables["attention_weights"]

        # 使用 `reshape` 连接不同头的注意力输出，创建一个 `kqv_dim` 维向量
        attn_out = attn_out.swapaxes(1, 2).reshape(n_ex, self.kqv_dim)
        self._dv["attention_out"] = attn_out.reshape(n_ex, -1, self.kqv_dim)

        # 应用最终的输出投影
        Y = self.projections["O"].forward(attn_out)
        Y = Y.reshape(n_ex, -1, self.kqv_dim)
        # 返回最终输出结果
        return Y
    # 反向传播函数，计算损失对查询、键、值的梯度
    def backward(self, dLdy):
        # 获取样本数量
        n_ex = dLdy.shape[0]
        # 重塑梯度形状
        dLdy = dLdy.reshape(n_ex, self.kqv_dim)
        # 调用投影层的反向传播函数
        dLdX = self.projections["O"].backward(dLdy)
        # 重塑梯度形状
        dLdX = dLdX.reshape(n_ex, self.n_heads, -1, self.latent_dim)

        # 调用注意力机制的反向传播函数
        dLdQ_proj, dLdK_proj, dLdV_proj = self.attention.backward(dLdX)

        # 更新导数字典
        self._dv["dQ_proj"] = dLdQ_proj
        self._dv["dK_proj"] = dLdK_proj
        self._dv["dV_proj"] = dLdV_proj

        # 重塑梯度形状
        dLdQ_proj = dLdQ_proj.reshape(n_ex, self.kqv_dim)
        dLdK_proj = dLdK_proj.reshape(n_ex, self.kqv_dim)
        dLdV_proj = dLdV_proj.reshape(n_ex, self.kqv_dim)

        # 调用投影层的反向传播函数
        dLdQ = self.projections["Q"].backward(dLdQ_proj)
        dLdK = self.projections["K"].backward(dLdK_proj)
        dLdV = self.projections["V"].backward(dLdV_proj)
        # 返回查询、键、值的梯度
        return dLdQ, dLdK, dLdV

    # 派生变量属性，存储前向/反向传播过程中计算的中间值
    @property
    def derived_variables(self):
        """A dictionary of intermediate values computed during the
        forward/backward passes."""
        dv = {
            "Q_proj": None,
            "K_proj": None,
            "V_proj": None,
            "components": {
                "Q": self.projections["Q"].derived_variables,
                "K": self.projections["K"].derived_variables,
                "V": self.projections["V"].derived_variables,
                "O": self.projections["O"].derived_variables,
                "attention": self.attention.derived_variables,
            },
        }
        # 更新派生变量字典
        dv.update(self._dv)
        return dv

    # 梯度属性，存储累积的模块参数梯度
    @property
    def gradients(self):
        """A dictionary of the accumulated module parameter gradients."""
        return {
            "components": {
                "Q": self.projections["Q"].gradients,
                "K": self.projections["K"].gradients,
                "V": self.projections["V"].gradients,
                "O": self.projections["O"].gradients,
                "attention": self.attention.gradients,
            }
        }

    @property
    # 返回模块参数的字典
    def parameters(self):
        """A dictionary of the module parameters."""
        return {
            "components": {
                "Q": self.projections["Q"].parameters,
                "K": self.projections["K"].parameters,
                "V": self.projections["V"].parameters,
                "O": self.projections["O"].parameters,
                "attention": self.attention.parameters,
            }
        }

    # 返回模块超参数的字典
    @property
    def hyperparameters(self):
        """A dictionary of the module hyperparameters."""
        return {
            "layer": "MultiHeadedAttentionModule",
            "init": self.init,
            "kqv_dim": self.kqv_dim,
            "latent_dim": self.latent_dim,
            "n_heads": self.n_heads,
            "dropout_p": self.dropout_p,
            "component_ids": ["attention", "Q", "K", "V", "O"],
            "components": {
                "Q": self.projections["Q"].hyperparameters,
                "K": self.projections["K"].hyperparameters,
                "V": self.projections["V"].hyperparameters,
                "O": self.projections["O"].hyperparameters,
                "attention": self.attention.hyperparameters,
            },
        }
```