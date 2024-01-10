# `so-vits-svc\modules\F0Predictor\rmvpe\inference.py`

```
# 导入 torch 库
import torch
# 导入 torch.nn.functional 模块，并重命名为 F
import torch.nn.functional as F
# 从 torchaudio.transforms 模块中导入 Resample 类
from torchaudio.transforms import Resample
# 从当前目录下的 constants 模块中导入所有内容
from .constants import *  # noqa: F403
# 从当前目录下的 model 模块中导入 E2E0 类
from .model import E2E0
# 从当前目录下的 spec 模块中导入 MelSpectrogram 类
from .spec import MelSpectrogram
# 从当前目录下的 utils 模块中导入 to_local_average_cents 和 to_viterbi_cents 函数
from .utils import to_local_average_cents, to_viterbi_cents

# 定义 RMVPE 类
class RMVPE:
    # 初始化方法
    def __init__(self, model_path, device=None, dtype = torch.float32, hop_length=160):
        # 初始化 resample_kernel 字典
        self.resample_kernel = {}
        # 如果 device 为 None，则根据是否有 CUDA 加速选择设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        # 创建 E2E0 模型对象
        model = E2E0(4, 1, (2, 2))
        # 加载模型参数
        ckpt = torch.load(model_path, map_location=torch.device(self.device))
        model.load_state_dict(ckpt['model'])
        # 将模型转移到指定设备和数据类型
        model = model.to(dtype).to(self.device)
        # 设置模型为评估模式
        model.eval()
        # 保存模型对象
        self.model = model
        # 保存数据类型
        self.dtype = dtype
        # 创建 MelSpectrogram 对象
        self.mel_extractor = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX)  # noqa: F405
        # 初始化 resample_kernel 字典
        self.resample_kernel = {}

    # 将梅尔频谱转换为隐藏表示的方法
    def mel2hidden(self, mel):
        # 禁止梯度计算
        with torch.no_grad():
            # 获取梅尔频谱的帧数
            n_frames = mel.shape[-1]
            # 对梅尔频谱进行填充
            mel = F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode='constant')
            # 通过模型获取隐藏表示
            hidden = self.model(mel)
            # 返回隐藏表示的前 n_frames 列
            return hidden[:, :n_frames]

    # 解码隐藏表示得到音高的方法
    def decode(self, hidden, thred=0.03, use_viterbi=False):
        # 如果 use_viterbi 为 True，则使用维特比算法得到音高
        if use_viterbi:
            cents_pred = to_viterbi_cents(hidden, thred=thred)
        # 否则使用局部平均算法得到音高
        else:
            cents_pred = to_local_average_cents(hidden, thred=thred)
        # 根据音高得到对应的频率
        f0 = torch.Tensor([10 * (2 ** (cent_pred / 1200)) if cent_pred else 0 for cent_pred in cents_pred]).to(self.device)
        # 返回频率
        return f0
    # 从音频中推断基频（f0），默认采样率为16000，阈值为0.05，是否使用维特比算法为False
    def infer_from_audio(self, audio, sample_rate=16000, thred=0.05, use_viterbi=False):
        # 将音频扩展一个维度，并转换为指定数据类型和设备
        audio = audio.unsqueeze(0).to(self.dtype).to(self.device)
        # 如果采样率为16000，则不进行重采样
        if sample_rate == 16000:
            audio_res = audio
        else:
            # 将采样率转换为字符串
            key_str = str(sample_rate)
            # 如果重采样核心不在预先存储的字典中，则创建一个Resample对象并存储
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, 16000, lowpass_filter_width=128)
            self.resample_kernel[key_str] = self.resample_kernel[key_str].to(self.dtype).to(self.device)
            # 使用对应采样率的重采样核心对音频进行重采样
            audio_res = self.resample_kernel[key_str](audio)
        # 将mel_extractor对象转移到指定设备
        mel_extractor = self.mel_extractor.to(self.device)
        # 提取音频的mel频谱，并转换为指定数据类型
        mel = mel_extractor(audio_res, center=True).to(self.dtype)
        # 使用mel频谱生成隐藏状态
        hidden = self.mel2hidden(mel)
        # 使用隐藏状态解码得到基频（f0）
        f0 = self.decode(hidden.squeeze(0), thred=thred, use_viterbi=use_viterbi)
        # 返回基频（f0）
        return f0
```