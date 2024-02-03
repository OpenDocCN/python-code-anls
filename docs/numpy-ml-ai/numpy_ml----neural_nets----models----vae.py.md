# `numpy-ml\numpy_ml\neural_nets\models\vae.py`

```
# 从 time 模块中导入 time 函数
# 从 collections 模块中导入 OrderedDict 类
# 从 numpy 模块中导入 np 别名
# 从相对路径中导入 VAELoss 类
# 从相对路径中导入 minibatch 函数
# 从相对路径中导入 ReLU、Affine、Sigmoid 类
# 从相对路径中导入 Conv2D、Pool2D、Flatten、FullyConnected 类
from time import time
from collections import OrderedDict
import numpy as np
from ..losses import VAELoss
from ..utils import minibatch
from ..activations import ReLU, Affine, Sigmoid
from ..layers import Conv2D, Pool2D, Flatten, FullyConnected

# 定义 BernoulliVAE 类
class BernoulliVAE(object):
    # 初始化函数
    def __init__(
        self,
        T=5,
        latent_dim=256,
        enc_conv1_pad=0,
        enc_conv2_pad=0,
        enc_conv1_out_ch=32,
        enc_conv2_out_ch=64,
        enc_conv1_stride=1,
        enc_pool1_stride=2,
        enc_conv2_stride=1,
        enc_pool2_stride=1,
        enc_conv1_kernel_shape=(5, 5),
        enc_pool1_kernel_shape=(2, 2),
        enc_conv2_kernel_shape=(5, 5),
        enc_pool2_kernel_shape=(2, 2),
        optimizer="RMSProp(lr=0.0001)",
        init="glorot_uniform",
    # 初始化参数函数
    def _init_params(self):
        # 初始化参数字典
        self._dv = {}
        # 构建编码器
        self._build_encoder()
        # 构建解码器
        self._build_decoder()
    def _build_encoder(self):
        """
        构建 CNN 编码器

        Conv1 -> ReLU -> MaxPool1 -> Conv2 -> ReLU -> MaxPool2 ->
            Flatten -> FC1 -> ReLU -> FC2
        """
        # 初始化编码器为有序字典
        self.encoder = OrderedDict()
        # 添加第一层卷积层 Conv1
        self.encoder["Conv1"] = Conv2D(
            act_fn=ReLU(),
            init=self.init,
            pad=self.enc_conv1_pad,
            optimizer=self.optimizer,
            out_ch=self.enc_conv1_out_ch,
            stride=self.enc_conv1_stride,
            kernel_shape=self.enc_conv1_kernel_shape,
        )
        # 添加第一层池化层 Pool1
        self.encoder["Pool1"] = Pool2D(
            mode="max",
            optimizer=self.optimizer,
            stride=self.enc_pool1_stride,
            kernel_shape=self.enc_pool1_kernel_shape,
        )
        # 添加第二层卷积层 Conv2
        self.encoder["Conv2"] = Conv2D(
            act_fn=ReLU(),
            init=self.init,
            pad=self.enc_conv2_pad,
            optimizer=self.optimizer,
            out_ch=self.enc_conv2_out_ch,
            stride=self.enc_conv2_stride,
            kernel_shape=self.enc_conv2_kernel_shape,
        )
        # 添加第二层池化层 Pool2
        self.encoder["Pool2"] = Pool2D(
            mode="max",
            optimizer=self.optimizer,
            stride=self.enc_pool2_stride,
            kernel_shape=self.enc_pool2_kernel_shape,
        )
        # 添加展平层 Flatten
        self.encoder["Flatten3"] = Flatten(optimizer=self.optimizer)
        # 添加第一层全连接层 FC4
        self.encoder["FC4"] = FullyConnected(
            n_out=self.latent_dim, act_fn=ReLU(), optimizer=self.optimizer
        )
        # 添加第二层全连接层 FC5
        self.encoder["FC5"] = FullyConnected(
            n_out=self.T * 2,
            optimizer=self.optimizer,
            act_fn=Affine(slope=1, intercept=0),
            init=self.init,
        )
    # 构建 MLP 解码器
    def _build_decoder(self):
        """
        MLP decoder

        FC1 -> ReLU -> FC2 -> Sigmoid
        """
        # 初始化解码器为有序字典
        self.decoder = OrderedDict()
        # 添加全连接层 FC1 到解码器，使用 ReLU 激活函数
        self.decoder["FC1"] = FullyConnected(
            act_fn=ReLU(),
            init=self.init,
            n_out=self.latent_dim,
            optimizer=self.optimizer,
        )
        # 注意：`n_out` 取决于 X 的维度。我们现在使用占位符，并在 `forward` 方法中更新它
        # 添加全连接层 FC2 到解码器，使用 Sigmoid 激活函数
        self.decoder["FC2"] = FullyConnected(
            n_out=None, act_fn=Sigmoid(), optimizer=self.optimizer, init=self.init
        )

    # 返回模型的参数
    @property
    def parameters(self):
        return {
            "components": {
                # 返回编码器的参数
                "encoder": {k: v.parameters for k, v in self.encoder.items()},
                # 返回解码器的参数
                "decoder": {k: v.parameters for k, v in self.decoder.items()},
            }
        }

    @property
    # 返回模型的超参数字典
    def hyperparameters(self):
        return {
            "layer": "BernoulliVAE",  # 模型层类型
            "T": self.T,  # T 参数
            "init": self.init,  # 初始化方法
            "loss": str(self.loss),  # 损失函数
            "optimizer": self.optimizer,  # 优化器
            "latent_dim": self.latent_dim,  # 潜在空间维度
            "enc_conv1_pad": self.enc_conv1_pad,  # 编码器第一层卷积填充
            "enc_conv2_pad": self.enc_conv2_pad,  # 编码器第二层卷积填充
            "enc_conv1_in_ch": self.enc_conv1_in_ch,  # 编码器第一层卷积输入通道数
            "enc_conv1_stride": self.enc_conv1_stride,  # 编码器第一层卷积步长
            "enc_conv1_out_ch": self.enc_conv1_out_ch,  # 编码器第一层卷积输出通道数
            "enc_pool1_stride": self.enc_pool1_stride,  # 编码器第一层池化步长
            "enc_conv2_out_ch": self.enc_conv2_out_ch,  # 编码器第二层卷积输出通道数
            "enc_conv2_stride": self.enc_conv2_stride,  # 编码器第二层卷积步长
            "enc_pool2_stride": self.enc_pool2_stride,  # 编码器第二层池化步长
            "enc_conv2_kernel_shape": self.enc_conv2_kernel_shape,  # 编码器第二层卷积核形状
            "enc_pool2_kernel_shape": self.enc_pool2_kernel_shape,  # 编码器第二层池化核形状
            "enc_conv1_kernel_shape": self.enc_conv1_kernel_shape,  # 编码器第一层卷积核形状
            "enc_pool1_kernel_shape": self.enc_pool1_kernel_shape,  # 编码器第一层池化核形状
            "encoder_ids": list(self.encoder.keys()),  # 编码器 ID 列表
            "decoder_ids": list(self.decoder.keys()),  # 解码器 ID 列表
            "components": {
                "encoder": {k: v.hyperparameters for k, v in self.encoder.items()},  # 编码器超参数字典
                "decoder": {k: v.hyperparameters for k, v in self.decoder.items()},  # 解码器超参数字典
            },
        }

    @property
    # 计算派生变量，包括噪声、均值、对数方差等
    def derived_variables(self):
        # 初始化派生变量字典
        dv = {
            "noise": None,
            "t_mean": None,
            "t_log_var": None,
            "dDecoder_FC1_in": None,
            "dDecoder_t_mean": None,
            "dEncoder_FC5_out": None,
            "dDecoder_FC1_out": None,
            "dEncoder_FC4_out": None,
            "dEncoder_Pool2_out": None,
            "dEncoder_Conv2_out": None,
            "dEncoder_Pool1_out": None,
            "dEncoder_Conv1_out": None,
            "dDecoder_t_log_var": None,
            "dEncoder_Flatten3_out": None,
            # 初始化组件字典，包括编码器和解码器
            "components": {
                "encoder": {k: v.derived_variables for k, v in self.encoder.items()},
                "decoder": {k: v.derived_variables for k, v in self.decoder.items()},
            },
        }
        # 更新派生变量字典
        dv.update(self._dv)
        # 返回派生变量字典
        return dv

    # 获取梯度信息
    @property
    def gradients(self):
        # 返回梯度信息字典，包括编码器和解码器
        return {
            "components": {
                "encoder": {k: v.gradients for k, v in self.encoder.items()},
                "decoder": {k: v.gradients for k, v in self.decoder.items()},
            }
        }

    # 从分布中抽样
    def _sample(self, t_mean, t_log_var):
        """
        Returns a sample from the distribution

            q(t | x) = N(t_mean, diag(exp(t_log_var)))

        using the reparameterization trick.

        Parameters
        ----------
        t_mean : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, latent_dim)`
            Mean of the desired distribution.
        t_log_var : :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, latent_dim)`
            Log variance vector of the desired distribution.

        Returns
        -------
        samples: :py:class:`ndarray <numpy.ndarray>` of shape `(n_ex, latent_dim)`
        """
        # 生成服从标准正态分布的噪声
        noise = np.random.normal(loc=0.0, scale=1.0, size=t_mean.shape)
        # 使用重参数化技巧从分布中抽样
        samples = noise * np.exp(t_log_var) + t_mean
        # 保存抽样的噪声用于反向传播
        self._dv["noise"] = noise
        # 返回抽样结果
        return samples
    # VAE 前向传播
    def forward(self, X_train):
        """VAE forward pass"""
        # 如果解码器的输出大小未知，则设置为 N
        if self.decoder["FC2"].n_out is None:
            fc2 = self.decoder["FC2"]
            self.decoder["FC2"] = fc2.set_params({"n_out": self.N})

        # 假设每个图像被表示为一个扁平化的行向量
        n_ex, in_rows, N, in_ch = X_train.shape

        # 对训练批次进行编码，以估计变分分布的均值和方差
        out = X_train
        for k, v in self.encoder.items():
            out = v.forward(out)

        # 从编码器输出中提取变分分布的均值和对数方差
        t_mean = out[:, : self.T]
        t_log_var = out[:, self.T :]

        # 使用重参数化技巧从 q(t | x) 中采样 t
        t = self._sample(t_mean, t_log_var)

        # 将采样的潜在值 t 通过解码器传递，生成平均重构
        X_recon = t
        for k, v in self.decoder.items():
            X_recon = v.forward(X_recon)

        self._dv["t_mean"] = t_mean
        self._dv["t_log_var"] = t_log_var
        return X_recon

    # 执行梯度更新
    def update(self, cur_loss=None):
        """Perform gradient updates"""
        # 对解码器进行反向梯度更新
        for k, v in reversed(list(self.decoder.items())):
            v.update(cur_loss)
        # 对编码器进行反向梯度更新
        for k, v in reversed(list(self.encoder.items())):
            v.update(cur_loss)
        # 清空梯度
        self.flush_gradients()

    # 在更新后重置参数梯度
    def flush_gradients(self):
        """Reset parameter gradients after update"""
        # 重置解码器参数梯度
        for k, v in self.decoder.items():
            v.flush_gradients()
        # 重置编码器参数梯度
        for k, v in self.encoder.items():
            v.flush_gradients()
```