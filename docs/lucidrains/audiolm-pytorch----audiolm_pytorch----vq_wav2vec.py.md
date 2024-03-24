# `.\lucidrains\audiolm-pytorch\audiolm_pytorch\vq_wav2vec.py`

```py
# 导入所需的模块
from pathlib import Path
# 导入 torch 模块
import torch
# 导入 torch 中的 nn 模块
from torch import nn
# 导入 einops 中的 rearrange 函数
from einops import rearrange
# 导入 fairseq 模块
import fairseq
# 导入 torchaudio 中的 resample 函数
from torchaudio.functional import resample
# 导入自定义的 curtail_to_multiple 函数
from audiolm_pytorch.utils import curtail_to_multiple
# 导入 logging 模块
import logging
# 设置日志级别为 ERROR
logging.root.setLevel(logging.ERROR)

# 定义一个函数，用于判断值是否存在
def exists(val):
    return val is not None

# 定义 FairseqVQWav2Vec 类
class FairseqVQWav2Vec(nn.Module):
    """
    checkpoint path can be found at https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md#vq-wav2vec
    specifically download the kmeans model for now

    $ wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt
    """

    # 初始化函数
    def __init__(
        self,
        checkpoint_path,
        target_sample_hz = 24000,
        seq_len_multiple_of = None
    ):
        super().__init__()
        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = seq_len_multiple_of

        # 获取路径对象
        path = Path(checkpoint_path)
        # 断言路径存在
        assert path.exists(), f'path {checkpoint_path} does not exist'

        # 加载模型
        checkpoint = torch.load(checkpoint_path)
        load_model_input = {checkpoint_path: checkpoint}
        model, *_ = fairseq.checkpoint_utils.load_model_ensemble_and_task(load_model_input)

        self.model = model[0]
        self.model.eval()

        # 断言模型有效
        assert hasattr(self.model, 'vector_quantizer') and hasattr(self.model.vector_quantizer, 'embedding'), 'the vq wav2vec model does not seem to be valid'

    # 获取 groups 属性
    @property
    def groups(self):
        return self.model.vector_quantizer.groups

    # 获取 downsample_factor 属性
    @property
    def downsample_factor(self):
        # todo: double check architecture
        return 80

    # 获取 codebook_size 属性
    @property
    def codebook_size(self):
        return self.model.vector_quantizer.embedding.shape[0]

    # 前向传播函数
    @torch.inference_mode()
    def forward(
        self,
        wav_input,
        flatten = True,
        input_sample_hz = None
    ):
        # 如果输入采样率存在，则对输入进行重采样
        if exists(input_sample_hz):
            wav_input = resample(wav_input, input_sample_hz, self.target_sample_hz)

        # 如果 seq_len_multiple_of 存在，则对输入进行截断
        if exists(self.seq_len_multiple_of):
            wav_input = curtail_to_multiple(wav_input, self.seq_len_multiple_of)

        # 提取特征
        embed = self.model.feature_extractor(wav_input)
        # 获取 codebook 索引
        _, codebook_indices = self.model.vector_quantizer.forward_idx(embed)

        # 如果不需要展平，则返回 codebook 索引
        if not flatten:
            return codebook_indices

        # 对 codebook 索引进行重新排列
        return rearrange(codebook_indices, 'b ... -> b (...)')
```