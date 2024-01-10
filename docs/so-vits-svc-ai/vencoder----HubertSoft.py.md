# `so-vits-svc\vencoder\HubertSoft.py`

```
# 导入 torch 库
import torch

# 从 vencoder.encoder 模块中导入 SpeechEncoder 类
from vencoder.encoder import SpeechEncoder
# 从 vencoder.hubert 模块中导入 hubert_model 函数
from vencoder.hubert import hubert_model

# 定义 HubertSoft 类，继承自 SpeechEncoder 类
class HubertSoft(SpeechEncoder):
    # 初始化方法，接受 vec_path 和 device 两个参数
    def __init__(self, vec_path="pretrain/hubert-soft-0d54a1f4.pt", device=None):
        # 调用父类的初始化方法
        super().__init__()
        # 打印加载模型信息
        print("load model(s) from {}".format(vec_path))
        # 调用 hubert_model.hubert_soft 函数加载模型
        hubert_soft = hubert_model.hubert_soft(vec_path)
        # 如果 device 参数为 None，则根据是否有 CUDA 设备选择设备
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 否则，根据 device 参数选择设备
        else:
            self.dev = torch.device(device)
        # 设置隐藏维度为 256
        self.hidden_dim = 256
        # 将加载的模型移动到选择的设备上
        self.model = hubert_soft.to(self.dev)

    # 编码器方法，接受音频数据作为输入
    def encoder(self, wav):
        # 将音频数据赋值给 feats
        feats = wav
        # 如果音频数据的维度为 2，表示有双声道，取平均值
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        # 断言音频数据的维度为 1
        assert feats.dim() == 1, feats.dim()
        # 在 feats 上添加两个维度
        feats = feats[None,None,:]  
        # 使用 torch.no_grad() 上下文管理器，关闭梯度计算
        with torch.no_grad():
            # 使用 torch.inference_mode() 上下文管理器，设置模型为推理模式
            with torch.inference_mode():
                # 使用模型对特征进行编码
                units = self.model.units(feats)
                # 转置编码结果的维度
                return units.transpose(1,2)
```