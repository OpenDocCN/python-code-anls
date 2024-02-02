# `so-vits-svc\modules\F0Predictor\fcpe\model.py`

```py
import numpy as np  # 导入 NumPy 库，用于科学计算
import torch  # 导入 PyTorch 库，用于深度学习
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 中的函数模块
from torch.nn.utils import weight_norm  # 导入 PyTorch 中的权重归一化函数
from torchaudio.transforms import Resample  # 导入 PyTorch 音频处理模块中的重采样函数

from .nvSTFT import STFT  # 从当前目录下的 nvSTFT 模块中导入 STFT 类
from .pcmer import PCmer  # 从当前目录下的 pcmer 模块中导入 PCmer 类

# 定义一个函数，用于计算 L2 正则化损失
def l2_regularization(model, l2_alpha):
    l2_loss = []  # 初始化一个空列表，用于存储每个模块的 L2 损失
    for module in model.modules():  # 遍历模型中的每个模块
        if type(module) is nn.Conv2d:  # 如果模块是二维卷积层
            l2_loss.append((module.weight ** 2).sum() / 2.0)  # 计算 L2 损失并添加到列表中
    return l2_alpha * sum(l2_loss)  # 返回 L2 损失的加权和

# 定义一个类，用于实现 Fully Convolutional Pitch Extraction (FCPE) 模型
class FCPE(nn.Module):
    def __init__(
            self,
            input_channel=128,  # 输入通道数，默认为 128
            out_dims=360,  # 输出维度，默认为 360
            n_layers=12,  # 网络层数，默认为 12
            n_chans=512,  # 通道数，默认为 512
            use_siren=False,  # 是否使用 Siren 激活函数，默认为 False
            use_full=False,  # 是否使用全连接层，默认为 False
            loss_mse_scale=10,  # MSE 损失的缩放因子，默认为 10
            loss_l2_regularization=False,  # 是否使用 L2 正则化损失，默认为 False
            loss_l2_regularization_scale=1,  # L2 正则化损失的缩放因子，默认为 1
            loss_grad1_mse=False,  # 是否使用一阶梯度 MSE 损失，默认为 False
            loss_grad1_mse_scale=1,  # 一阶梯度 MSE 损失的缩放因子，默认为 1
            f0_max=1975.5,  # 最大基频，默认为 1975.5
            f0_min=32.70,  # 最小基频，默认为 32.70
            confidence=False,  # 是否输出置信度，默认为 False
            threshold=0.05,  # 置信度阈值，默认为 0.05
            use_input_conv=True  # 是否使用输入卷积层，默认为 True
    # 定义一个方法，用于将输入的mel数据进行前向推理，返回预测结果
    def forward(self, mel, infer=True, gt_f0=None, return_hz_f0=False, cdecoder = "local_argmax"):
        """
        input:
            B x n_frames x n_unit
        return:
            dict of B x n_frames x feat
        """
        # 如果解码器选择为"argmax"，则使用cents_decoder作为解码器
        if cdecoder == "argmax":
            self.cdecoder = self.cents_decoder
        # 如果解码器选择为"local_argmax"，则使用cents_local_decoder作为解码器
        elif cdecoder == "local_argmax":
            self.cdecoder = self.cents_local_decoder
        # 如果使用输入卷积，则对mel数据进行转置和堆叠操作
        if self.use_input_conv:
            x = self.stack(mel.transpose(1, 2)).transpose(1, 2)
        # 否则直接使用mel数据
        else:
            x = mel
        # 对输入数据进行解码
        x = self.decoder(x)
        # 对解码后的数据进行归一化
        x = self.norm(x)
        # 对归一化后的数据进行密集输出
        x = self.dense_out(x)  # [B,N,D]
        # 对输出数据进行sigmoid激活
        x = torch.sigmoid(x)
        # 如果不是推理模式，则计算损失
        if not infer:
            # 将真实的f0频率转换为cent表示
            gt_cent_f0 = self.f0_to_cent(gt_f0)  # mel f0  #[B,N,1]
            # 对cent表示的f0频率进行高斯模糊处理
            gt_cent_f0 = self.gaussian_blurred_cent(gt_cent_f0)  # #[B,N,out_dim]
            # 计算二进制交叉熵损失
            loss_all = self.loss_mse_scale * F.binary_cross_entropy(x, gt_cent_f0)  # bce loss
            # 如果开启L2正则化，则添加L2正则化损失
            if self.loss_l2_regularization:
                loss_all = loss_all + l2_regularization(model=self, l2_alpha=self.loss_l2_regularization_scale)
            # 将损失值赋给x
            x = loss_all
        # 如果是推理模式，则进行解码和频率转换
        if infer:
            x = self.cdecoder(x)
            x = self.cent_to_f0(x)
            # 如果不需要返回Hz频率，则对频率进行log变换
            if not return_hz_f0:
                x = (1 + x / 700).log()
        # 返回结果
        return x

    # 定义一个方法，用于将cent表示的频率进行解码
    def cents_decoder(self, y, mask=True):
        B, N, _ = y.size()
        # 将cent_table进行扩展，用于计算解码后的频率
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        # 计算解码后的频率
        rtn = torch.sum(ci * y, dim=-1, keepdim=True) / torch.sum(y, dim=-1, keepdim=True)  # cents: [B,N,1]
        # 如果需要进行掩码处理
        if mask:
            # 计算置信度
            confident = torch.max(y, dim=-1, keepdim=True)[0]
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= self.threshold] = float("-INF")
            rtn = rtn * confident_mask
        # 如果开启置信度返回，则返回置信度
        if self.confidence:
            return rtn, confident
        # 否则只返回解码后的频率
        else:
            return rtn
    # 定义一个本地解码器函数，用于将输入的y进行本地解码
    def cents_local_decoder(self, y, mask=True):
        # 获取输入y的维度信息
        B, N, _ = y.size()
        # 从self.cent_table中获取ci的值，并扩展到与y相同的维度
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        # 获取y中每一行的最大值和对应的索引
        confident, max_index = torch.max(y, dim=-1, keepdim=True)
        # 计算本地最大值的索引
        local_argmax_index = torch.arange(0,9).to(max_index.device) + (max_index - 4)
        local_argmax_index[local_argmax_index<0] = 0
        local_argmax_index[local_argmax_index>=self.n_out] = self.n_out - 1
        # 从ci和y中根据本地最大值的索引获取对应的值
        ci_l = torch.gather(ci,-1,local_argmax_index)
        y_l = torch.gather(y,-1,local_argmax_index)
        # 计算本地加权平均值
        rtn = torch.sum(ci_l * y_l, dim=-1, keepdim=True) / torch.sum(y_l, dim=-1, keepdim=True)  # cents: [B,N,1]
        # 如果mask为True，则进行置信度掩码处理
        if mask:
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= self.threshold] = float("-INF")
            rtn = rtn * confident_mask
        # 如果self.confidence为True，则返回解码结果和置信度
        if self.confidence:
            return rtn, confident
        # 否则只返回解码结果
        else:
            return rtn

    # 将cent转换为f0
    def cent_to_f0(self, cent):
        return 10. * 2 ** (cent / 1200.)

    # 将f0转换为cent
    def f0_to_cent(self, f0):
        return 1200. * torch.log2(f0 / 10.)

    # 对cents进行高斯模糊处理
    def gaussian_blurred_cent(self, cents):  # cents: [B,N,1]
        # 创建一个mask，用于过滤不符合条件的值
        mask = (cents > 0.1) & (cents < (1200. * np.log2(self.f0_max / 10.)))
        # 获取cents的维度信息
        B, N, _ = cents.size()
        # 从self.cent_table中获取ci的值，并扩展到与cents相同的维度
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        # 计算高斯模糊后的值
        return torch.exp(-torch.square(ci - cents) / 1250) * mask.float()
# 定义一个类 FCPEInfer
class FCPEInfer:
    # 初始化方法，接受模型路径、设备和数据类型作为参数
    def __init__(self, model_path, device=None, dtype=torch.float32):
        # 如果设备未指定，则根据 CUDA 是否可用来选择设备
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 将设备保存到实例变量中
        self.device = device
        # 加载模型参数
        ckpt = torch.load(model_path, map_location=torch.device(self.device))
        # 从模型参数中获取配置信息，并保存到实例变量中
        self.args = DotDict(ckpt["config"])
        # 保存数据类型到实例变量中
        self.dtype = dtype
        # 创建 FCPE 模型实例
        model = FCPE(
            input_channel=self.args.model.input_channel,
            out_dims=self.args.model.out_dims,
            n_layers=self.args.model.n_layers,
            n_chans=self.args.model.n_chans,
            use_siren=self.args.model.use_siren,
            use_full=self.args.model.use_full,
            loss_mse_scale=self.args.loss.loss_mse_scale,
            loss_l2_regularization=self.args.loss.loss_l2_regularization,
            loss_l2_regularization_scale=self.args.loss.loss_l2_regularization_scale,
            loss_grad1_mse=self.args.loss.loss_grad1_mse,
            loss_grad1_mse_scale=self.args.loss.loss_grad1_mse_scale,
            f0_max=self.args.model.f0_max,
            f0_min=self.args.model.f0_min,
            confidence=self.args.model.confidence,
        )
        # 将模型移动到指定设备和数据类型
        model.to(self.device).to(self.dtype)
        # 加载模型参数
        model.load_state_dict(ckpt['model'])
        # 设置模型为评估模式
        model.eval()
        # 保存模型实例到实例变量中
        self.model = model
        # 创建 Wav2Mel 实例
        self.wav2mel = Wav2Mel(self.args, dtype=self.dtype, device=self.device)

    # 定义一个不需要梯度的方法
    @torch.no_grad()
    # 定义一个调用方法，接受音频、采样率和阈值作为参数
    def __call__(self, audio, sr, threshold=0.05):
        # 设置模型阈值
        self.model.threshold = threshold
        # 将音频转换为二维数组
        audio = audio[None,:]
        # 将音频转换为梅尔频谱，并转换为指定数据类型
        mel = self.wav2mel(audio=audio, sample_rate=sr).to(self.dtype)
        # 使用模型进行推理，返回基频
        f0 = self.model(mel=mel, infer=True, return_hz_f0=True)
        # 返回基频
        return f0


# 定义一个类 Wav2Mel
class Wav2Mel:
    # 初始化函数，接受参数 args、device 和 dtype，默认为 torch.float32
    def __init__(self, args, device=None, dtype=torch.float32):
        # 设置采样率为 args.mel.sampling_rate
        self.sampling_rate = args.mel.sampling_rate
        # 设置帧移大小为 args.mel.hop_size
        self.hop_size = args.mel.hop_size
        # 如果未指定设备，则根据 CUDA 是否可用来选择设备
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 设置设备
        self.device = device
        # 设置数据类型
        self.dtype = dtype
        # 创建 STFT 对象
        self.stft = STFT(
            args.mel.sampling_rate,
            args.mel.num_mels,
            args.mel.n_fft,
            args.mel.win_size,
            args.mel.hop_size,
            args.mel.fmin,
            args.mel.fmax
        )
        # 初始化重采样核字典
        self.resample_kernel = {}

    # 提取 NVSTFT 特征
    def extract_nvstft(self, audio, keyshift=0, train=False):
        # 获取梅尔频谱并转置
        mel = self.stft.get_mel(audio, keyshift=keyshift, train=train).transpose(1, 2)  # B, n_frames, bins
        return mel

    # 提取梅尔频谱
    def extract_mel(self, audio, sample_rate, keyshift=0, train=False):
        # 将音频转换为指定数据类型和设备
        audio = audio.to(self.dtype).to(self.device)
        # 如果采样率与设定的采样率相同，则不进行重采样
        if sample_rate == self.sampling_rate:
            audio_res = audio
        else:
            # 否则进行重采样
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.sampling_rate, lowpass_filter_width=128)
            self.resample_kernel[key_str] = self.resample_kernel[key_str].to(self.dtype).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)
        # 提取梅尔频谱
        mel = self.extract_nvstft(audio_res, keyshift=keyshift, train=train)  # B, n_frames, bins
        # 根据音频帧数和梅尔频谱帧数进行处理
        n_frames = int(audio.shape[1] // self.hop_size) + 1
        if n_frames > int(mel.shape[1]):
            mel = torch.cat((mel, mel[:, -1:, :]), 1)
        if n_frames < int(mel.shape[1]):
            mel = mel[:, :n_frames, :]
        return mel

    # 调用函数，返回提取的梅尔频谱
    def __call__(self, audio, sample_rate, keyshift=0, train=False):
        return self.extract_mel(audio, sample_rate, keyshift=keyshift, train=train)
# 定义一个继承自 dict 的 DotDict 类
class DotDict(dict):
    # 定义一个特殊方法，用于获取属性
    def __getattr__(*args):
        # 获取字典中的值
        val = dict.get(*args)
        # 如果值是字典类型，则将其转换为 DotDict 类型
        return DotDict(val) if type(val) is dict else val

    # 定义一个特殊方法，用于设置属性
    __setattr__ = dict.__setitem__
    # 定义一个特殊方法，用于删除属性
    __delattr__ = dict.__delitem__
```