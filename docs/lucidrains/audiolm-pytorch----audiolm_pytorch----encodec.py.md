# `.\lucidrains\audiolm-pytorch\audiolm_pytorch\encodec.py`

```py
# 导入所需的库和模块
from functools import reduce
from einops import rearrange, pack, unpack
import torch
from torch import nn
from torchaudio.functional import resample
from vector_quantize_pytorch import ResidualVQ
from encodec import EncodecModel
from encodec.utils import _linear_overlap_add

# 定义一个辅助函数，用于检查变量是否存在
def exists(val):
    return val is not None

# 获取模型中的量化器数量
def get_num_quantizers(model: EncodecModel, audio_length = 512):
    out = model.encode(torch.randn(1, 1, audio_length))
    return out[0][0].shape[1]

# 定义一个包装器类，用于支持预训练的 24kHz Encodec 模型
class EncodecWrapper(nn.Module):
    def __init__(
        self,
        target_sample_hz = 24000,
        strides = (2, 4, 5, 8),
        num_quantizers = 8,
        bandwidth = 6.0
    ):
        super().__init__()
        # 实例化一个预训练的 Encodec 模型
        self.model = EncodecModel.encodec_model_24khz()
        self.model.normalize = False

        # 设置目标带宽，影响量化器数量
        self.model.set_target_bandwidth(bandwidth)
        num_quantizers = get_num_quantizers(self.model)

        # 设置一些字段
        self.target_sample_hz = target_sample_hz
        assert self.target_sample_hz == 24000, "haven't done anything with non-24kHz yet"
        self.codebook_dim = 128
        self.rq_groups = 1
        self.num_quantizers = num_quantizers
        self.strides = strides

        # 初始化 ResidualVQ 模块
        self.rq = ResidualVQ(
            dim = 128,
            codebook_size = 1024,
            num_quantizers = num_quantizers
        )

        # 复制编码器的码书到 ResidualVQ 模块
        for encodec_rq_layer, rq_layer in zip(self.model.quantizer.vq.layers, self.rq.layers):
            encodec_codebook = dict(encodec_rq_layer._codebook.named_buffers()).get('embed')
            vq_codebook = dict(rq_layer._codebook.named_buffers()).get('embed')
            encodec_codebook = rearrange(encodec_codebook, '... -> 1 ...')
            vq_codebook.copy_(encodec_codebook)

    @property
    def seq_len_multiple_of(self):
        return reduce(lambda x, y: x * y, self.strides)

    @property
    def downsample_factor(self):
        return self.seq_len_multiple_of

    def forward(
        self,
        x,
        input_sample_hz = None,
        return_encoded = False,
        **kwargs
    ):
        x, ps = pack([x], '* n')

        if exists(input_sample_hz):
            x = resample(x, input_sample_hz, self.target_sample_hz)

        assert not self.model.training, "Encodec is pretrained and should never be called outside eval mode."

        wav = rearrange(x, f'b t -> b {self.model.channels} t')

        with torch.inference_mode():
            encoded_frames = self.model.encode(wav)

        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        codes = rearrange(codes, 'b q n -> b n q')

        emb = None

        if return_encoded:
            emb = self.get_emb_from_indices(codes)
            emb, = unpack(emb, ps, '* n c')

        codes, = unpack(codes, ps, '* n q')

        return emb, codes, None

    def decode_from_codebook_indices(self, quantized_indices):
        frames = self._decode_frame(quantized_indices)
        result = _linear_overlap_add(frames, self.model.segment_stride or 1)
        return rearrange(result, 'b n -> b 1 n')

    def get_emb_from_indices(self, indices):
        codes = rearrange(indices, 'b t q -> q b t')
        emb = self.model.quantizer.decode(codes)
        return rearrange(emb, 'b c n -> b n c')

    def decode(self, emb):
        emb = rearrange(emb, 'b n c -> b c n')
        return self.model.decoder(emb)
    # 解码帧数据，输入为量化后的索引
    def _decode_frame(self, quantized_indices):
        # 以下代码是从 self.model._decode_frame() (Encodec 版本 0.1.1) 中插入的，假设我们已经解包了 EncodedFrame
        # 输入: batch x num tokens x num quantizers
        # 输出: batch x new_num_samples，其中 new_num_samples 是 num_frames * stride 的乘积（可能略大于原始 num samples，因为最后一帧可能不是完全填满的）
        # num_frames == 你拥有的声学标记数量，每个标记对应一帧
        # 重新排列量化后的索引，形状为 'b t q -> q b t'
        codes = rearrange(quantized_indices, 'b t q -> q b t')
        # 使用量化器解码得到的嵌入
        emb = self.model.quantizer.decode(codes)
        # emb 形状: batch x self.model.quantizer.dimension x T。注意 self.model.quantizer.dimension 是嵌入维度
        return self.model.decoder(emb)
```