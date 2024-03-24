# `.\lucidrains\audiolm-pytorch\audiolm_pytorch\hubert_kmeans.py`

```
# 导入必要的库
from pathlib import Path
import torch
from torch import nn, einsum
from torchaudio.functional import resample
from einops import rearrange, repeat, pack, unpack
from audiolm_pytorch.utils import curtail_to_multiple

# 定义一个空函数用于忽略警告
def noop(*args, **kwargs):
    pass

import warnings
import logging

# 设置日志级别为 ERROR
logging.root.setLevel(logging.ERROR)

# 忽略警告
warnings.warn = noop

# 导入 fairseq 和 joblib 用于 hubert 模型
import joblib
import fairseq

# 定义辅助函数
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# 定义一个带有 kmeans 的 Hubert 模型类
class HubertWithKmeans(nn.Module):
    """
    checkpoint and kmeans can be downloaded at https://github.com/facebookresearch/fairseq/tree/main/examples/hubert
    or you can train your own
    """

    def __init__(
        self,
        checkpoint_path,
        kmeans_path,
        target_sample_hz = 16000,
        seq_len_multiple_of = None,
        output_layer = 9
    ):
        super().__init__()

        # 初始化模型参数
        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = seq_len_multiple_of
        self.output_layer = output_layer

        # 加载模型和 kmeans
        model_path = Path(checkpoint_path)
        kmeans_path = Path(kmeans_path)

        assert model_path.exists(), f'path {checkpoint_path} does not exist'
        assert kmeans_path.exists(), f'path {kmeans_path} does not exist'

        checkpoint = torch.load(checkpoint_path)
        load_model_input = {checkpoint_path: checkpoint}
        model, *_ = fairseq.checkpoint_utils.load_model_ensemble_and_task(load_model_input)

        self.model = model[0]
        self.model.eval()

        kmeans = joblib.load(kmeans_path)

        self.kmeans = kmeans

        # 注册缓冲区
        self.register_buffer(
            'cluster_centers',
            torch.from_numpy(kmeans.cluster_centers_)
        )

    @property
    def groups(self):
        return 1

    @property
    def codebook_size(self):
        return self.kmeans.n_clusters

    @property
    def downsample_factor(self):
        # todo: double check
        return 320

    @torch.inference_mode()
    def forward(
        self,
        wav_input,
        flatten = True,
        input_sample_hz = None
    ):
        # 获取输入数据的批次和设备
        batch, device = wav_input.shape[0], wav_input.device

        # 如果输入采样率存在，则对输入进行重采样
        if exists(input_sample_hz):
            wav_input = resample(wav_input, input_sample_hz, self.target_sample_hz)

        # 如果设置了 seq_len_multiple_of，则对输入进行截断
        if exists(self.seq_len_multiple_of):
            wav_input = curtail_to_multiple(wav_input, self.seq_len_multiple_of)

        # 提取特征
        embed = self.model(
            wav_input,
            features_only = True,
            mask = False,
            output_layer = self.output_layer
        )['x']

        # 重复聚类中心以匹配嵌入的形状
        batched_cluster_centers = repeat(self.cluster_centers, 'c d -> b c d', b = embed.shape[0])
        # 计算嵌入和聚类中心之间的欧氏距离
        dists = -torch.cdist(embed, batched_cluster_centers, p = 2)
        # 获取最大距离对应的聚类
        clusters = dists.argmax(dim = -1)

        # 如果 flatten 为 True，则返回平坦的聚类结果
        if flatten:
            return clusters

        # 否则返回重排后的聚类结果
        return rearrange(clusters, 'b ... -> b (...)')
```