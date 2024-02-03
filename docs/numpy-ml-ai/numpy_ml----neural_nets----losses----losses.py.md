# `numpy-ml\numpy_ml\neural_nets\losses\losses.py`

```py
# 从 abc 模块导入 ABC 和 abstractmethod 装饰器
from abc import ABC, abstractmethod

# 导入 numpy 库并重命名为 np
import numpy as np

# 从相对路径中导入 is_binary 和 is_stochastic 函数
from ...utils.testing import is_binary, is_stochastic
# 从相对路径中导入 WeightInitializer、ActivationInitializer 和 OptimizerInitializer 类
from ..initializers import (
    WeightInitializer,
    ActivationInitializer,
    OptimizerInitializer,
)


# 定义一个抽象基类 ObjectiveBase
class ObjectiveBase(ABC):
    # 初始化方法
    def __init__(self):
        super().__init__()

    # 抽象方法，计算损失
    @abstractmethod
    def loss(self, y_true, y_pred):
        pass

    # 抽象方法，计算梯度
    @abstractmethod
    def grad(self, y_true, y_pred, **kwargs):
        pass


# 定义一个 SquaredError 类，继承自 ObjectiveBase
class SquaredError(ObjectiveBase):
    # 初始化方法
    def __init__(self):
        """
        A squared-error / `L2` loss.

        Notes
        -----
        For real-valued target **y** and predictions :math:`\hat{\mathbf{y}}`, the
        squared error is

        .. math::
                \mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})
                    = 0.5 ||\hat{\mathbf{y}} - \mathbf{y}||_2^2
        """
        super().__init__()

    # 调用方法，返回损失值
    def __call__(self, y, y_pred):
        return self.loss(y, y_pred)

    # 返回类的字符串表示
    def __str__(self):
        return "SquaredError"

    # 静态方法，计算损失
    @staticmethod
    def loss(y, y_pred):
        """
        Compute the squared error between `y` and `y_pred`.

        Parameters
        ----------
        y : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            Ground truth values for each of `n` examples
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            Predictions for the `n` examples in the batch.

        Returns
        -------
        loss : float
            The sum of the squared error across dimensions and examples.
        """
        return 0.5 * np.linalg.norm(y_pred - y) ** 2

    # 静态方法
    @staticmethod
    # 计算均方误差损失相对于非线性输入 `z` 的梯度
    def grad(y, y_pred, z, act_fn):
        """
        Gradient of the squared error loss with respect to the pre-nonlinearity
        input, `z`.

        Notes
        -----
        The current method computes the gradient :math:`\\frac{\partial
        \mathcal{L}}{\partial \mathbf{z}}`, where

        .. math::

            \mathcal{L}(\mathbf{z})
                &=  \\text{squared_error}(\mathbf{y}, g(\mathbf{z})) \\\\
            g(\mathbf{z})
                &=  \\text{act_fn}(\mathbf{z})

        The gradient with respect to :math:`\mathbf{z}` is then

        .. math::

            \\frac{\partial \mathcal{L}}{\partial \mathbf{z}}
                = (g(\mathbf{z}) - \mathbf{y}) \left(
                    \\frac{\partial g}{\partial \mathbf{z}} \\right)

        Parameters
        ----------
        y : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            Ground truth values for each of `n` examples.
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            Predictions for the `n` examples in the batch.
        act_fn : :doc:`Activation <numpy_ml.neural_nets.activations>` object
            The activation function for the output layer of the network.

        Returns
        -------
        grad : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The gradient of the squared error loss with respect to `z`.
        """
        # 返回损失函数相对于 `z` 的梯度
        return (y_pred - y) * act_fn.grad(z)
class CrossEntropy(ObjectiveBase):
    # 定义交叉熵损失函数类，继承自ObjectiveBase类
    def __init__(self):
        """
        A cross-entropy loss.

        Notes
        -----
        For a one-hot target **y** and predicted class probabilities
        :math:`\hat{\mathbf{y}}`, the cross entropy is

        .. math::
                \mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})
                    = \sum_i y_i \log \hat{y}_i
        """
        # 初始化函数，包含交叉熵损失的数学定义和说明
        super().__init__()

    def __call__(self, y, y_pred):
        return self.loss(y, y_pred)

    def __str__(self):
        return "CrossEntropy"

    @staticmethod
    def loss(y, y_pred):
        """
        Compute the cross-entropy (log) loss.

        Notes
        -----
        This method returns the sum (not the average!) of the losses for each
        sample.

        Parameters
        ----------
        y : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            Class labels (one-hot with `m` possible classes) for each of `n`
            examples.
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            Probabilities of each of `m` classes for the `n` examples in the
            batch.

        Returns
        -------
        loss : float
            The sum of the cross-entropy across classes and examples.
        """
        # 计算交叉熵损失的静态方法，包含参数说明和返回值说明
        is_binary(y)
        is_stochastic(y_pred)

        # prevent taking the log of 0
        eps = np.finfo(float).eps

        # each example is associated with a single class; sum the negative log
        # probability of the correct label over all samples in the batch.
        # observe that we are taking advantage of the fact that y is one-hot
        # encoded
        cross_entropy = -np.sum(y * np.log(y_pred + eps))
        return cross_entropy

    @staticmethod
    def grad(y, y_pred):
        """
        Compute the gradient of the cross entropy loss with regard to the
        softmax input, `z`.

        Notes
        -----
        The gradient for this method goes through both the cross-entropy loss
        AND the softmax non-linearity to return :math:`\\frac{\partial
        \mathcal{L}}{\partial \mathbf{z}}` (rather than :math:`\\frac{\partial
        \mathcal{L}}{\partial \\text{softmax}(\mathbf{z})}`).

        In particular, let:

        .. math::

            \mathcal{L}(\mathbf{z})
                = \\text{cross_entropy}(\\text{softmax}(\mathbf{z})).

        The current method computes:

        .. math::

            \\frac{\partial \mathcal{L}}{\partial \mathbf{z}}
                &= \\text{softmax}(\mathbf{z}) - \mathbf{y} \\\\
                &=  \hat{\mathbf{y}} - \mathbf{y}

        Parameters
        ----------
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(n, m)`
            A one-hot encoding of the true class labels. Each row constitues a
            training example, and each column is a different class.
        y_pred: :py:class:`ndarray <numpy.ndarray>` of shape `(n, m)`
            The network predictions for the probability of each of `m` class
            labels on each of `n` examples in a batch.

        Returns
        -------
        grad : :py:class:`ndarray <numpy.ndarray>` of shape (n, m)
            The gradient of the cross-entropy loss with respect to the *input*
            to the softmax function.
        """
        # 检查 y 是否为二进制编码
        is_binary(y)
        # 检查 y_pred 是否为随机值
        is_stochastic(y_pred)

        # 交叉熵损失相对于 z 的梯度是 y_pred - y，因此可以从正确类别标签的概率中减去 1
        grad = y_pred - y

        # [可选] 将梯度按批次中的示例数量进行缩放
        # n, m = y.shape
        # grad /= n
        return grad
class VAELoss(ObjectiveBase):
    def __init__(self):
        """
        The variational lower bound for a variational autoencoder with Bernoulli
        units.

        Notes
        -----
        The VLB to the sum of the binary cross entropy between the true input and
        the predicted output (the "reconstruction loss") and the KL divergence
        between the learned variational distribution :math:`q` and the prior,
        :math:`p`, assumed to be a unit Gaussian.

        .. math::

            \\text{VAELoss} =
                \\text{cross_entropy}(\mathbf{y}, \hat{\mathbf{y}})
                    + \\mathbb{KL}[q \ || \ p]

        where :math:`\mathbb{KL}[q \ || \ p]` is the Kullback-Leibler
        divergence between the distributions :math:`q` and :math:`p`.

        References
        ----------
        .. [1] Kingma, D. P. & Welling, M. (2014). "Auto-encoding variational Bayes".
           *arXiv preprint arXiv:1312.6114.* https://arxiv.org/pdf/1312.6114.pdf
        """
        # 调用父类的构造函数
        super().__init__()

    def __call__(self, y, y_pred, t_mean, t_log_var):
        # 调用 loss 方法计算损失
        return self.loss(y, y_pred, t_mean, t_log_var)

    def __str__(self):
        # 返回字符串 "VAELoss"
        return "VAELoss"

    @staticmethod
    # 计算 Bernoulli VAE 的变分下界

    def loss(y, y_pred, t_mean, t_log_var):
        """
        Variational lower bound for a Bernoulli VAE.

        Parameters
        ----------
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, N)`
            The original images.
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, N)`
            The VAE reconstruction of the images.
        t_mean: :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, T)`
            Mean of the variational distribution :math:`q(t \mid x)`.
        t_log_var: :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, T)`
            Log of the variance vector of the variational distribution
            :math:`q(t \mid x)`.

        Returns
        -------
        loss : float
            The VLB, averaged across the batch.
        """
        # 防止在 log(0) 时出现 nan
        eps = np.finfo(float).eps
        y_pred = np.clip(y_pred, eps, 1 - eps)

        # 重构损失：二元交叉熵
        rec_loss = -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred), axis=1)

        # 变分分布 q 和先验分布 p 之间的 KL 散度，一个单位高斯分布
        kl_loss = -0.5 * np.sum(1 + t_log_var - t_mean ** 2 - np.exp(t_log_var), axis=1)
        loss = np.mean(kl_loss + rec_loss)
        return loss

    @staticmethod
    # 定义计算 VLB 相对于网络参数的梯度的函数
    def grad(y, y_pred, t_mean, t_log_var):
        """
        Compute the gradient of the VLB with regard to the network parameters.

        Parameters
        ----------
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, N)`
            The original images.
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, N)`
            The VAE reconstruction of the images.
        t_mean: :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, T)`
            Mean of the variational distribution :math:`q(t | x)`.
        t_log_var: :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, T)`
            Log of the variance vector of the variational distribution
            :math:`q(t | x)`.

        Returns
        -------
        dY_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, N)`
            The gradient of the VLB with regard to `y_pred`.
        dLogVar : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, T)`
            The gradient of the VLB with regard to `t_log_var`.
        dMean : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, T)`
            The gradient of the VLB with regard to `t_mean`.
        """
        # 获取原始图像的数量
        N = y.shape[0]
        # 定义一个极小值，避免除零错误
        eps = np.finfo(float).eps
        # 将预测值限制在 eps 和 1-eps 之间
        y_pred = np.clip(y_pred, eps, 1 - eps)

        # 计算相对于 y_pred 的 VLB 梯度
        dY_pred = -y / (N * y_pred) - (y - 1) / (N - N * y_pred)
        # 计算相对于 t_log_var 的 VLB 梯度
        dLogVar = (np.exp(t_log_var) - 1) / (2 * N)
        # 计算相对于 t_mean 的 VLB 梯度
        dMean = t_mean / N
        # 返回计算得到的梯度
        return dY_pred, dLogVar, dMean
class WGAN_GPLoss(ObjectiveBase):
    # 定义一个继承自 ObjectiveBase 的 WGAN_GPLoss 类
    def __init__(self, lambda_=10):
        """
        The loss function for a Wasserstein GAN [*]_ [*]_ with gradient penalty.

        Notes
        -----
        Assuming an optimal critic, minimizing this quantity wrt. the generator
        parameters corresponds to minimizing the Wasserstein-1 (earth-mover)
        distance between the fake and real data distributions.

        The formula for the WGAN-GP critic loss is

        .. math::

            \\text{WGANLoss}
                &=  \sum_{x \in X_{real}} p(x) D(x)
                    - \sum_{x' \in X_{fake}} p(x') D(x') \\\\
            \\text{WGANLossGP}
                &=  \\text{WGANLoss} + \lambda
                    (||\\nabla_{X_{interp}} D(X_{interp})||_2 - 1)^2

        where

        .. math::

            X_{fake}  &=   \\text{Generator}(\mathbf{z}) \\\\
            X_{interp}   &=   \\alpha X_{real} + (1 - \\alpha) X_{fake} \\\\

        and

        .. math::

            \mathbf{z}  &\sim  \mathcal{N}(0, \mathbb{1}) \\\\
            \\alpha  &\sim  \\text{Uniform}(0, 1)

        References
        ----------
        .. [*] Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., &
           Courville, A. (2017) "Improved training of Wasserstein GANs"
           *Advances in Neural Information Processing Systems, 31*: 5769-5779.
        .. [*] Goodfellow, I. J, Abadie, P. A., Mirza, M., Xu, B., Farley, D.
           W., Ozair, S., Courville, A., & Bengio, Y. (2014) "Generative
           adversarial nets" *Advances in Neural Information Processing
           Systems, 27*: 2672-2680.

        Parameters
        ----------
        lambda_ : float
            The gradient penalty coefficient. Default is 10.
        """
        # 初始化函数，设置 lambda_ 参数，默认值为 10
        self.lambda_ = lambda_
        # 调用父类的初始化函数
        super().__init__()
    # 定义一个方法，用于计算生成器和评论者损失，使用WGAN-GP值函数
    def __call__(self, Y_fake, module, Y_real=None, gradInterp=None):
        """
        Computes the generator and critic loss using the WGAN-GP value
        function.

        Parameters
        ----------
        Y_fake : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex,)`
            The output of the critic for `X_fake`.
        module : {'C', 'G'}
            Whether to calculate the loss for the critic ('C') or the generator
            ('G'). If calculating loss for the critic, `Y_real` and
            `gradInterp` must not be None.
        Y_real : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex,)`, or None
            The output of the critic for `X_real`. Default is None.
        gradInterp : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_feats)`, or None
            The gradient of the critic output for `X_interp` wrt. `X_interp`.
            Default is None.

        Returns
        -------
        loss : float
            Depending on the setting for `module`, either the critic or
            generator loss, averaged over examples in the minibatch.
        """
        # 调用loss方法计算损失
        return self.loss(Y_fake, module, Y_real=Y_real, gradInterp=gradInterp)

    # 定义一个方法，返回描述WGANLossGP对象的字符串
    def __str__(self):
        return "WGANLossGP(lambda_={})".format(self.lambda_)
    # 定义一个方法，计算生成器和评论家的损失，使用 WGAN-GP 值函数
    def loss(self, Y_fake, module, Y_real=None, gradInterp=None):
        """
        Computes the generator and critic loss using the WGAN-GP value
        function.

        Parameters
        ----------
        Y_fake : :py:class:`ndarray <numpy.ndarray>` of shape (n_ex,)
            The output of the critic for `X_fake`.
        module : {'C', 'G'}
            Whether to calculate the loss for the critic ('C') or the generator
            ('G'). If calculating loss for the critic, `Y_real` and
            `gradInterp` must not be None.
        Y_real : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex,)` or None
            The output of the critic for `X_real`. Default is None.
        gradInterp : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, n_feats)` or None
            The gradient of the critic output for `X_interp` wrt. `X_interp`.
            Default is None.

        Returns
        -------
        loss : float
            Depending on the setting for `module`, either the critic or
            generator loss, averaged over examples in the minibatch.
        """
        # 计算评论家损失，包括梯度惩罚
        if module == "C":
            # 计算 X_interp_norm，即 gradInterp 的 L2 范数
            X_interp_norm = np.linalg.norm(gradInterp, axis=1, keepdims=True)
            # 计算梯度惩罚
            gradient_penalty = (X_interp_norm - 1) ** 2
            # 计算损失，包括评论家输出均值、真实数据输出均值和 lambda_ 乘以梯度惩罚均值
            loss = (
                Y_fake.mean() - Y_real.mean() + self.lambda_ * gradient_penalty.mean()
            )

        # 计算生成器损失
        elif module == "G":
            # 计算损失，取 Y_fake 的均值的负值
            loss = -Y_fake.mean()

        else:
            # 如果 module 不是 'C' 或 'G'，抛出数值错误
            raise ValueError("Unrecognized module: {}".format(module))

        # 返回计算得到的损失
        return loss
# 定义 NCELoss 类，继承自 ObjectiveBase 类
class NCELoss(ObjectiveBase):
    """
    """

    # 初始化方法，接受参数 n_classes, noise_sampler, num_negative_samples, optimizer, init, subtract_log_label_prob
    def __init__(
        self,
        n_classes,
        noise_sampler,
        num_negative_samples,
        optimizer=None,
        init="glorot_uniform",
        subtract_log_label_prob=True,
    def _init_params(self):
        # 使用 WeightInitializer 初始化权重
        init_weights = WeightInitializer(str(self.act_fn), mode=self.init)

        # 初始化参数 X 为空列表
        self.X = []
        # 初始化偏置 b 为全零数组
        b = np.zeros((1, self.n_classes))
        # 初始化权重 W 使用 init_weights 初始化
        W = init_weights((self.n_classes, self.n_in))

        # 参数字典包含权重 W 和偏置 b
        self.parameters = {"W": W, "b": b}

        # 梯度字典包含权重 W 和偏置 b 的梯度
        self.gradients = {"W": np.zeros_like(W), "b": np.zeros_like(b)}

        # 衍生变量字典包含多个空列表
        self.derived_variables = {
            "y_pred": [],
            "target": [],
            "true_w": [],
            "true_b": [],
            "sampled_b": [],
            "sampled_w": [],
            "out_labels": [],
            "target_logits": [],
            "noise_samples": [],
            "noise_logits": [],
        }

        # 初始化标志为 True
        self.is_initialized = True

    # hyperparameters 属性返回包含超参数的字典
    @property
    def hyperparameters(self):
        return {
            "id": "NCELoss",
            "n_in": self.n_in,
            "init": self.init,
            "n_classes": self.n_classes,
            "noise_sampler": self.noise_sampler,
            "num_negative_samples": self.num_negative_samples,
            "subtract_log_label_prob": self.subtract_log_label_prob,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }

    # 调用方法，调用 loss 方法
    def __call__(self, X, target, neg_samples=None, retain_derived=True):
        return self.loss(X, target, neg_samples, retain_derived)

    # 字符串表示方法，返回包含超参数的字符串
    def __str__(self):
        keys = [
            "{}={}".format(k, v)
            for k, v in self.hyperparameters.items()
            if k not in ["id", "optimizer"]
        ] + ["optimizer={}".format(self.optimizer)]
        return "NCELoss({})".format(", ".join(keys))
    def freeze(self):
        """
        Freeze the loss parameters at their current values so they can no
        longer be updated.
        """
        # 冻结损失参数，使其不能再被更新
        self.trainable = False

    def unfreeze(self):
        """Unfreeze the layer parameters so they can be updated."""
        # 解冻层参数，使其可以被更新
        self.trainable = True

    def flush_gradients(self):
        """Erase all the layer's derived variables and gradients."""
        # 清空层的所有派生变量和梯度
        assert self.trainable, "NCELoss is frozen"
        self.X = []
        for k, v in self.derived_variables.items():
            self.derived_variables[k] = []

        for k, v in self.gradients.items():
            self.gradients[k] = np.zeros_like(v)

    def update(self, cur_loss=None):
        """
        Update the loss parameters using the accrued gradients and optimizer.
        Flush all gradients once the update is complete.
        """
        # 使用累积的梯度和优化器更新损失参数
        assert self.trainable, "NCELoss is frozen"
        self.optimizer.step()
        for k, v in self.gradients.items():
            if k in self.parameters:
                self.parameters[k] = self.optimizer(self.parameters[k], v, k, cur_loss)
        self.flush_gradients()
    # 计算 NCE 损失相对于输入、权重和偏置的梯度
    def grad(self, retain_grads=True, update_params=True):
        """
        Compute the gradient of the NCE loss with regard to the inputs,
        weights, and biases.

        Parameters
        ----------
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.
        update_params : bool
            Whether to perform a single step of gradient descent on the layer
            weights and bias using the calculated gradients. If `retain_grads`
            is False, this option is ignored and the parameter gradients are
            not updated. Default is True.

        Returns
        -------
        dLdX : :py:class:`ndarray <numpy.ndarray>` of shape (`n_ex`, `n_in`) or list of arrays
            The gradient of the loss with regard to the layer input(s) `X`.
        """
        # 检查 NCE 损失是否可训练
        assert self.trainable, "NCE loss is frozen"

        # 初始化存储输入梯度的列表
        dX = []
        # 遍历输入数据的索引和数据
        for input_idx, x in enumerate(self.X):
            # 计算输入数据 x 的梯度、权重梯度和偏置梯度
            dx, dw, db = self._grad(x, input_idx)
            # 将输入数据 x 的梯度添加到列表中
            dX.append(dx)

            # 如果需要保留梯度
            if retain_grads:
                # 累加权重梯度和偏置梯度到梯度字典中
                self.gradients["W"] += dw
                self.gradients["b"] += db

        # 如果输入数据只有一个，则取第一个梯度；否则保持列表形式
        dX = dX[0] if len(self.X) == 1 else dX

        # 如果需要保留梯度并且更新参数
        if retain_grads and update_params:
            # 更新参数
            self.update()

        # 返回输入数据的梯度
        return dX
```