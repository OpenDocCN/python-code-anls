# `numpy-ml\numpy_ml\neural_nets\models\wgan_gp.py`

```
# 导入所需的模块和库
from time import time
from collections import OrderedDict
import numpy as np
# 导入自定义的模块
from ..utils import minibatch
from ..layers import FullyConnected
from ..losses import WGAN_GPLoss

# 定义 WGAN_GP 类，实现 Wasserstein 生成对抗网络（WGAN）结构和梯度惩罚（GP）
class WGAN_GP(object):
    """
    A Wasserstein generative adversarial network (WGAN) architecture with
    gradient penalty (GP).

    Notes
    -----
    In contrast to a regular WGAN, WGAN-GP uses gradient penalty on the
    generator rather than weight clipping to encourage the 1-Lipschitz
    constraint:

    .. math::

        | \\text{Generator}(\mathbf{x}_1) - \\text{Generator}(\mathbf{x}_2) |
            \leq |\mathbf{x}_1 - \mathbf{x}_2 | \ \ \ \ \\forall \mathbf{x}_1, \mathbf{x}_2

    In other words, the generator must have input gradients with a norm of at
    most 1 under the :math:`\mathbf{X}_{real}` and :math:`\mathbf{X}_{fake}`
    data distributions.

    To enforce this constraint, WGAN-GP penalizes the model if the generator
    gradient norm moves away from a target norm of 1. See
    :class:`~numpy_ml.neural_nets.losses.WGAN_GPLoss` for more details.

    In contrast to a standard WGAN, WGAN-GP avoids using BatchNorm in the
    critic, as correlation between samples in a batch can impact the stability
    of the gradient penalty.

    WGAP-GP architecture:

    .. code-block:: text

        X_real ------------------------|
                                        >---> [Critic] --> Y_out
        Z --> [Generator] --> X_fake --|

    where ``[Generator]`` is

    .. code-block:: text

        FC1 -> ReLU -> FC2 -> ReLU -> FC3 -> ReLU -> FC4

    and ``[Critic]`` is

    .. code-block:: text

        FC1 -> ReLU -> FC2 -> ReLU -> FC3 -> ReLU -> FC4

    and

    .. math::

        Z \sim \mathcal{N}(0, 1)
    """

    # 初始化函数，设置网络参数和优化器
    def __init__(
        self,
        g_hidden=512,
        init="he_uniform",
        optimizer="RMSProp(lr=0.0001)",
        debug=False,
    ):
        """
        Wasserstein generative adversarial network with gradient penalty.

        Parameters
        ----------
        g_hidden : int
            The number of units in the critic and generator hidden layers.
            Default is 512.
        init : str
            The weight initialization strategy. Valid entries are
            {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform',
            'std_normal', 'trunc_normal'}. Default is "he_uniform".
        optimizer : str or :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object or None
            The optimization strategy to use when performing gradient updates.
            If None, use the :class:`~numpy_ml.neural_nets.optimizers.SGD`
            optimizer with default parameters. Default is "RMSProp(lr=0.0001)".
        debug : bool
            Whether to store additional intermediate output within
            ``self.derived_variables``. Default is False.
        """
        # 初始化函数，设置初始参数
        self.init = init
        self.debug = debug
        self.g_hidden = g_hidden
        self.optimizer = optimizer

        self.lambda_ = None
        self.n_steps = None
        self.batchsize = None

        self.is_initialized = False

    # 初始化参数函数
    def _init_params(self):
        # 初始化存储派生变量的字典
        self._dv = {}
        # 初始化存储梯度的字典
        self._gr = {}
        # 构建评论者网络
        self._build_critic()
        # 构建生成器网络
        self._build_generator()
        # 设置初始化标志为True
        self.is_initialized = True
    def _build_generator(self):
        """
        构建生成器网络结构：FC1 -> ReLU -> FC2 -> ReLU -> FC3 -> ReLU -> FC4
        """
        # 初始化生成器网络结构为有序字典
        self.generator = OrderedDict()
        # 添加全连接层 FC1 到生成器网络结构中
        self.generator["FC1"] = FullyConnected(
            self.g_hidden, act_fn="ReLU", optimizer=self.optimizer, init=self.init
        )
        # 添加全连接层 FC2 到生成器网络结构中
        self.generator["FC2"] = FullyConnected(
            self.g_hidden, act_fn="ReLU", optimizer=self.optimizer, init=self.init
        )
        # 添加全连接层 FC3 到生成器网络结构中
        self.generator["FC3"] = FullyConnected(
            self.g_hidden, act_fn="ReLU", optimizer=self.optimizer, init=self.init
        )
        # 添加全连接层 FC4 到生成器网络结构中
        self.generator["FC4"] = FullyConnected(
            self.n_feats,
            act_fn="Affine(slope=1, intercept=0)",
            optimizer=self.optimizer,
            init=self.init,
        )

    def _build_critic(self):
        """
        构建评论者网络结构：FC1 -> ReLU -> FC2 -> ReLU -> FC3 -> ReLU -> FC4
        """
        # 初始化评论者网络结构为有序字典
        self.critic = OrderedDict()
        # 添加全连接层 FC1 到评论者网络结构中
        self.critic["FC1"] = FullyConnected(
            self.g_hidden, act_fn="ReLU", optimizer=self.optimizer, init=self.init
        )
        # 添加全连接层 FC2 到评论者网络结构中
        self.critic["FC2"] = FullyConnected(
            self.g_hidden, act_fn="ReLU", optimizer=self.optimizer, init=self.init
        )
        # 添加全连接层 FC3 到评论者网络结构中
        self.critic["FC3"] = FullyConnected(
            self.g_hidden, act_fn="ReLU", optimizer=self.optimizer, init=self.init
        )
        # 添加全连接层 FC4 到评论者网络结构中
        self.critic["FC4"] = FullyConnected(
            1,
            act_fn="Affine(slope=1, intercept=0)",
            optimizer=self.optimizer,
            init=self.init,
        )

    @property
    # 返回超参数字典，包括初始化方法、lambda参数、生成器隐藏层大小、训练步数、优化器、批大小、每轮更新critic的次数等
    def hyperparameters(self):
        return {
            "init": self.init,
            "lambda_": self.lambda_,
            "g_hidden": self.g_hidden,
            "n_steps": self.n_steps,
            "optimizer": self.optimizer,
            "batchsize": self.batchsize,
            "c_updates_per_epoch": self.c_updates_per_epoch,
            "components": {
                # 获取critic组件的超参数字典
                "critic": {k: v.hyperparameters for k, v in self.critic.items()},
                # 获取generator组件的超参数字典
                "generator": {k: v.hyperparameters for k, v in self.generator.items()},
            },
        }

    # 返回参数字典，包括critic和generator组件的参数
    @property
    def parameters(self):
        return {
            "components": {
                # 获取critic组件的参数字典
                "critic": {k: v.parameters for k, v in self.critic.items()},
                # 获取generator组件的参数字典
                "generator": {k: v.parameters for k, v in self.generator.items()},
            }
        }

    # 返回派生变量字典，包括critic和generator组件的派生变量
    @property
    def derived_variables(self):
        C = self.critic.items()
        G = self.generator.items()
        dv = {
            "components": {
                # 获取critic组件的派生变量字典
                "critic": {k: v.derived_variables for k, v in C},
                # 获取generator组件的派生变量字典
                "generator": {k: v.derived_variables for k, v in G},
            }
        }
        # 更新派生变量字典
        dv.update(self._dv)
        return dv

    # 返回梯度字典，包括critic和generator组件的梯度
    @property
    def gradients(self):
        grads = {
            "dC_Y_fake": None,
            "dC_Y_real": None,
            "dG_Y_fake": None,
            "dC_gradInterp": None,
            "components": {
                # 获取critic组件的梯度字典
                "critic": {k: v.gradients for k, v in self.critic.items()},
                # 获取generator组件的梯度字典
                "generator": {k: v.gradients for k, v in self.generator.items()},
            },
        }
        # 更新梯度字典
        grads.update(self._gr)
        return grads
    # 执行生成器或评论者的前向传播

    def forward(self, X, module, retain_derived=True):
        """
        Perform the forward pass for either the generator or the critic.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(batchsize, \*)`
            Input data
        module : {'C' or 'G'}
            Whether to perform the forward pass for the critic ('C') or for the
            generator ('G').
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through wrt. this input. Default
            is True.

        Returns
        -------
        out : :py:class:`ndarray <numpy.ndarray>` of shape `(batchsize, \*)`
            The output of the final layer of the module.
        Xs : dict
            A dictionary with layer ids as keys and values corresponding to the
            input to each intermediate layer during the forward pass. Useful
            during debugging.
        """
        # 根据模块类型选择要执行的模块
        if module == "G":
            mod = self.generator
        elif module == "C":
            mod = self.critic
        else:
            raise ValueError("Unrecognized module name: {}".format(module))

        # 初始化存储中间结果的字典
        Xs = {}
        # 初始化输出和是否保留派生变量的标志
        out, rd = X, retain_derived
        # 遍历模块中的每一层，执行前向传播
        for k, v in mod.items():
            # 将当前层的输入存储到字典中
            Xs[k] = out
            # 执行当前层的前向传播
            out = v.forward(out, retain_derived=rd)
        # 返回最终层的输出和中间结果字典
        return out, Xs
    def backward(self, grad, module, retain_grads=True):
        """
        Perform the backward pass for either the generator or the critic.

        Parameters
        ----------
        grad : :py:class:`ndarray <numpy.ndarray>` of shape `(batchsize, \*)` or list of arrays
            Gradient of the loss with respect to module output(s).
        module : {'C' or 'G'}
            Whether to perform the backward pass for the critic ('C') or for the
            generator ('G').
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is True.

        Returns
        -------
        out : :py:class:`ndarray <numpy.ndarray>` of shape `(batchsize, \*)`
            The gradient of the loss with respect to the module input.
        dXs : dict
            A dictionary with layer ids as keys and values corresponding to the
            input to each intermediate layer during the backward pass. Useful
            during debugging.
        """
        # 根据传入的 module 参数选择要执行反向传播的模块
        if module == "G":
            mod = self.generator
        elif module == "C":
            mod = self.critic
        else:
            raise ValueError("Unrecognized module name: {}".format(module))

        # 初始化存储每个中间层输入的字典
        dXs = {}
        # 初始化输出和是否保留梯度的标志
        out, rg = grad, retain_grads
        # 遍历模块中的层，执行反向传播
        for k, v in reversed(list(mod.items())):
            # 将当前输出保存到中间层输入字典中
            dXs[k] = out
            # 执行当前层的反向传播，更新输出
            out = v.backward(out, retain_grads=rg)
        # 返回最终输出和中间层输入字典
        return out, dXs
    def _dGradInterp(self, dLdGradInterp, dYi_outs):
        """
        Compute the gradient penalty's contribution to the critic loss and
        update the parameter gradients accordingly.

        Parameters
        ----------
        dLdGradInterp : :py:class:`ndarray <numpy.ndarray>` of shape `(batchsize, critic_in_dim)`
            Gradient of `Y_interp` with respect to `X_interp`.
        dYi_outs : dict
            The intermediate outputs generated during the backward pass when
            computing `dLdGradInterp`.
        """
        # 将梯度值初始化为dLdGradInterp
        dy = dLdGradInterp
        # 遍历critic字典中的每个键值对
        for k, v in self.critic.items():
            # 获取当前层的输入X
            X = v.X[-1]  # layer input during forward pass
            # 计算当前层的梯度、权重和偏置
            dy, dW, dB = v._bwd2(dy, X, dYi_outs[k][2])
            # 更新当前层的权重梯度
            self.critic[k].gradients["W"] += dW
            # 更新当前层的偏置梯度
            self.critic[k].gradients["b"] += dB

    def update_generator(self, X_shape):
        """
        Compute parameter gradients for the generator on a single minibatch.

        Parameters
        ----------
        X_shape : tuple of `(batchsize, n_feats)`
            Shape for the input batch.

        Returns
        -------
        G_loss : float
            The generator loss on the fake data (generated during the critic
            update)
        """
        # 重置生成器的梯度为0
        self.flush_gradients("G")
        # 获取生成器生成的假数据Y_fake
        Y_fake = self.derived_variables["Y_fake"]

        # 获取Y_fake的行数和列数
        n_ex, _ = Y_fake.shape
        # 计算生成器损失
        G_loss = -Y_fake.mean()
        # 计算生成器损失的梯度
        dG_loss = -np.ones_like(Y_fake) / n_ex
        # 反向传播计算生成器的梯度
        self.backward(dG_loss, "G")

        # 如果开启了调试模式，则保存生成器损失和梯度
        if self.debug:
            self._dv["G_loss"] = G_loss
            self._dv["dG_Y_fake"] = dG_loss

        return G_loss

    def flush_gradients(self, module):
        """Reset parameter gradients to 0 after an update."""
        # 根据模块名称选择要重置梯度的模块
        if module == "G":
            mod = self.generator
        elif module == "C":
            mod = self.critic
        else:
            raise ValueError("Unrecognized module name: {}".format(module))

        # 将选定模块的所有参数梯度重置为0
        for k, v in mod.items():
            v.flush_gradients()
    # 更新模型参数，根据传入的模块名称选择对应的模型
    def update(self, module, module_loss=None):
        # 如果模块名称为 "G"，则选择生成器模型
        if module == "G":
            mod = self.generator
        # 如果模块名称为 "C"，则选择评论者模型
        elif module == "C":
            mod = self.critic
        # 如果模块名称不是 "G" 或 "C"，则抛出数值错误异常
        else:
            raise ValueError("Unrecognized module name: {}".format(module))

        # 遍历模型中的参数，逆序遍历参数列表
        for k, v in reversed(list(mod.items())):
            # 更新模型参数
            v.update(module_loss)
        # 清空梯度
        self.flush_gradients(module)

    # 拟合模型
    def fit(
        self,
        X_real,
        lambda_,
        n_steps=1000,
        batchsize=128,
        c_updates_per_epoch=5,
        verbose=True,
```