# `so-vits-svc\vencoder\WhisperPPGLarge.py`

```
# 导入 torch 库
import torch

# 从 vencoder.encoder 模块中导入 SpeechEncoder 类
from vencoder.encoder import SpeechEncoder
# 从 vencoder.whisper.audio 模块中导入 log_mel_spectrogram 和 pad_or_trim 函数
from vencoder.whisper.audio import log_mel_spectrogram, pad_or_trim
# 从 vencoder.whisper.model 模块中导入 ModelDimensions 和 Whisper 类
from vencoder.whisper.model import ModelDimensions, Whisper

# 定义 WhisperPPGLarge 类，继承自 SpeechEncoder 类
class WhisperPPGLarge(SpeechEncoder):
    # 初始化方法
    def __init__(self, vec_path="pretrain/large-v2.pt", device=None):
        super().__init__()
        # 如果设备为空，则根据是否有 CUDA 设备选择设备
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        # 加载预训练模型参数
        checkpoint = torch.load(vec_path, map_location=device)
        # 根据模型参数创建 ModelDimensions 对象
        dims = ModelDimensions(**checkpoint["dims"])
        # 创建 Whisper 模型对象
        model = Whisper(dims)
        # 加载模型参数
        model.load_state_dict(checkpoint["model_state_dict"])
        # 将隐藏维度设置为模型维度
        self.hidden_dim = dims
        # 将模型移动到指定设备
        self.model = model.to(self.dev)

    # 编码器方法
    def encoder(self, wav):
        # 将音频数据赋值给变量 audio
        audio = wav
        # 获取音频数据的长度
        audln = audio.shape[0]
        # 计算音频数据对应的 PPG 长度
        ppgln = audln // 320
        # 对音频数据进行填充或修剪
        audio = pad_or_trim(audio)
        # 计算音频数据的对数梅尔频谱图，并移动到指定设备
        mel = log_mel_spectrogram(audio).to(self.dev)
        # 禁止梯度计算，计算 PPG
        with torch.no_grad():
            ppg = self.model.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
            # 将 PPG 转换为张量，并移动到指定设备
            ppg = torch.FloatTensor(ppg[:ppgln, ]).to(self.dev)
            # 返回 PPG 的转置
            return ppg[None, :, :].transpose(1, 2)
```