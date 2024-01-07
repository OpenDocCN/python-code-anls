# `Bert-VITS2\train_ms.py`

```

# 导入必要的库
import platform  # 用于获取操作系统信息
import os  # 用于与操作系统交互
import torch  # PyTorch深度学习库
from torch.nn import functional as F  # PyTorch中的神经网络模块
from torch.utils.data import DataLoader  # PyTorch中用于加载数据的工具
from torch.utils.tensorboard import SummaryWriter  # 用于创建TensorBoard可视化的工具
import torch.distributed as dist  # PyTorch中的分布式训练工具
from torch.nn.parallel import DistributedDataParallel as DDP  # PyTorch中的分布式数据并行工具
from torch.cuda.amp import autocast, GradScaler  # PyTorch中的混合精度训练工具
from tqdm import tqdm  # 用于在循环中显示进度条
import logging  # 用于记录日志信息
from config import config  # 导入配置文件
import argparse  # 用于解析命令行参数
import datetime  # 用于处理日期和时间

# 设置日志级别
logging.getLogger("numba").setLevel(logging.WARNING)
# 导入自定义的工具和函数
import commons
import utils
from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler,
)
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
    DurationDiscriminator,
    WavLMDiscriminator,
)
from losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss,
    WavLMLoss,
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch  # 处理音频信号的工具
from text.symbols import symbols  # 导入文本符号

# 设置一些PyTorch的参数
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = (
    True  # 如果遇到训练问题，请尝试禁用TF32。
)
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.sdp_kernel("flash")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(
    True
)  # 如果torch版本低于2.0，则不可用
global_step = 0

# 定义训练和评估函数
def train_and_evaluate(
    rank,
    local_rank,
    epoch,
    hps,
    nets,
    optims,
    schedulers,
    scaler,
    loaders,
    logger,
    writers,
def evaluate(hps, generator, eval_loader, writer_eval):
    # 将生成器设置为评估模式
    generator.eval()
    image_dict = {}  # 用于存储图像数据的字典
    audio_dict = {}  # 用于存储音频数据的字典
    print("Evaluating ...")  # 打印评估信息
    with torch.no_grad():  # 禁用梯度计算
        for batch_idx, (
            x,
            x_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            speakers,
            tone,
            language,
            bert,
            ja_bert,
            en_bert,
        ) in enumerate(eval_loader):
            # 将数据移动到GPU上
            x, x_lengths = x.cuda(), x_lengths.cuda()
            spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
            y, y_lengths = y.cuda(), y_lengths.cuda()
            speakers = speakers.cuda()
            bert = bert.cuda()
            ja_bert = ja_bert.cuda()
            en_bert = en_bert.cuda()
            tone = tone.cuda()
            language = language.cuda()
            for use_sdp in [True, False]:  # 遍历两种情况
                # 生成音频数据
                y_hat, attn, mask, *_ = generator.module.infer(
                    x,
                    x_lengths,
                    speakers,
                    tone,
                    language,
                    bert,
                    ja_bert,
                    en_bert,
                    y=spec,
                    max_len=1000,
                    sdp_ratio=0.0 if not use_sdp else 1.0,
                )
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

                # 将频谱转换为梅尔频谱
                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                # 更新图像数据字典
                image_dict.update(
                    {
                        f"gen/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0].cpu().numpy()
                        )
                    }
                )
                # 更新音频数据字典
                audio_dict.update(
                    {
                        f"gen/audio_{batch_idx}_{use_sdp}": y_hat[
                            0, :, : y_hat_lengths[0]
                        ]
                    }
                )
                # 更新图像数据字典
                image_dict.update(
                    {
                        f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                            mel[0].cpu().numpy()
                        )
                    }
                )
                # 更新音频数据字典
                audio_dict.update({f"gt/audio_{batch_idx}": y[0, :, : y_lengths[0]})

    # 将评估结果写入TensorBoard
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    # 将生成器设置为训练模式
    generator.train()


if __name__ == "__main__":
    run()  # 运行主函数

```