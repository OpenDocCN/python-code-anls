# `so-vits-svc\vencoder\DPHubert.py`

```py
# 导入 torch 库
import torch

# 从 vencoder.dphubert.model 中导入 wav2vec2_model
from vencoder.dphubert.model import wav2vec2_model

# 从 vencoder.encoder 中导入 SpeechEncoder
from vencoder.encoder import SpeechEncoder

# 定义 DPHubert 类，继承自 SpeechEncoder
class DPHubert(SpeechEncoder):
    # 初始化方法
    def __init__(self, vec_path="pretrain/DPHuBERT-sp0.75.pth", device=None):
        # 调用父类的初始化方法
        super().__init__()
        # 打印加载模型的信息
        print("load model(s) from {}".format(vec_path))
        # 如果设备为空
        if device is None:
            # 如果 CUDA 可用，则使用 CUDA，否则使用 CPU
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            # 使用指定的设备
            self.dev = torch.device(device)
        # 加载模型参数
        ckpt = torch.load(vec_path)
        # 设置隐藏维度为 768
        self.hidden_dim = 768
        # 创建 wav2vec2 模型，并将其移动到指定设备上
        self.model = wav2vec2_model(**ckpt["config"]).to(self.dev)
        # 加载模型参数
        self.model.load_state_dict(ckpt["state_dict"], strict=False)

    # 编码器方法
    def encoder(self, wav):
        # 将音频特征赋值给 feats
        feats = wav
        # 如果特征维度为 2，表示双声道
        if feats.dim() == 2:  # double channels
            # 取平均值，将双声道转为单声道
            feats = feats.mean(-1)
        # 断言特征维度为 1
        assert feats.dim() == 1, feats.dim()
        # 在第一个维度上增加一个维度
        feats = feats[None, :]
        # 禁止梯度计算
        with torch.no_grad():
            # 进入推理模式
            with torch.inference_mode():
                # 使用模型对特征进行编码
                units = self.model(feats)[0]
                # 转置结果
                return units.transpose(1,2)
```