# `so-vits-svc\vencoder\CNHubertLarge.py`

```
# 导入 torch 库
import torch
# 从 fairseq 库中导入 checkpoint_utils 模块
from fairseq import checkpoint_utils
# 从 vencoder.encoder 模块中导入 SpeechEncoder 类
from vencoder.encoder import SpeechEncoder

# 定义 CNHubertLarge 类，继承自 SpeechEncoder 类
class CNHubertLarge(SpeechEncoder):
    # 初始化方法，接受 vec_path 和 device 两个参数
    def __init__(self, vec_path="pretrain/chinese-hubert-large-fairseq-ckpt.pt", device=None):
        # 调用父类的初始化方法
        super().__init__()
        # 打印加载模型的信息
        print("load model(s) from {}".format(vec_path))
        # 加载模型集合、保存的配置和任务
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
          [vec_path],
          suffix="",
        )
        # 如果 device 为 None，则根据是否有 CUDA 设备选择设备
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 否则，根据给定的 device 参数选择设备
        else:
            self.dev = torch.device(device)
        # 将模型移动到选择的设备上
        self.model = models[0].to(self.dev)
        # 设置模型为评估模式
        self.model.eval()

    # 定义编码器方法，接受音频数据作为输入
    def encoder(self, wav):
        # 将音频特征赋值给 feats
        feats = wav
        # 如果特征的维度为 2，表示有双声道，取平均值
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        # 断言特征的维度为 1
        assert feats.dim() == 1, feats.dim()
        # 将特征重塑为 1 行，-1 列的形状
        feats = feats.view(1, -1)
        # 创建与 feats 相同形状的填充掩码，填充值为 False
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        # 构建输入字典，包含源数据和填充掩码
        inputs = {
          "source": feats.to(wav.device),
          "padding_mask": padding_mask.to(wav.device)
        }
        # 使用 torch.no_grad() 上下文，不计算梯度
        with torch.no_grad():
            # 提取特征
            logits = self.model.extract_features(**inputs)
        # 返回特征的转置
        return logits[0].transpose(1, 2)
```