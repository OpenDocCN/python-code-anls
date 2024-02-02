# `so-vits-svc\vencoder\WavLMBasePlus.py`

```py
# 导入 torch 库
import torch

# 从 vencoder.encoder 模块中导入 SpeechEncoder 类
from vencoder.encoder import SpeechEncoder
# 从 vencoder.wavlm.WavLM 模块中导入 WavLM 类和 WavLMConfig 类
from vencoder.wavlm.WavLM import WavLM, WavLMConfig

# 定义 WavLMBasePlus 类，继承自 SpeechEncoder 类
class WavLMBasePlus(SpeechEncoder):
    # 初始化方法，接受 vec_path 和 device 两个参数
    def __init__(self, vec_path="pretrain/WavLM-Base+.pt", device=None):
        # 调用父类的初始化方法
        super().__init__()
        # 打印加载模型的信息
        print("load model(s) from {}".format(vec_path))
        # 加载模型的检查点
        checkpoint = torch.load(vec_path)
        # 根据检查点中的配置信息创建 WavLMConfig 对象
        self.cfg = WavLMConfig(checkpoint['cfg'])
        # 如果 device 参数为 None，则根据是否有 GPU 来选择设备
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 否则，根据传入的 device 参数选择设备
        else:
            self.dev = torch.device(device)
        # 设置隐藏层维度为配置中的编码器嵌入维度
        self.hidden_dim = self.cfg.encoder_embed_dim
        # 创建 WavLM 模型对象
        self.model = WavLM(self.cfg)
        # 加载模型的状态字典
        self.model.load_state_dict(checkpoint['model'])
        # 将模型移动到指定设备并设置为评估模式
        self.model.to(self.dev).eval()

    # 编码器方法，接受音频数据作为输入
    def encoder(self, wav):
        # 将音频特征赋值给 feats
        feats = wav
        # 如果特征的维度为 2，表示双声道音频
        if feats.dim() == 2:  # double channels
            # 对双声道音频进行平均处理，转为单声道
            feats = feats.mean(-1)
        # 断言特征的维度为 1
        assert feats.dim() == 1, feats.dim()
        # 如果配置中指定了归一化操作
        if self.cfg.normalize:
            # 对特征进行层归一化
            feats = torch.nn.functional.layer_norm(feats, feats.shape)
        # 使用 torch.no_grad() 上下文管理器，关闭梯度计算
        with torch.no_grad():
            # 使用 torch.inference_mode() 上下文管理器，设置为推断模式
            with torch.inference_mode():
                # 提取特征并转置
                units = self.model.extract_features(feats[None, :])[0]
                return units.transpose(1, 2)
```