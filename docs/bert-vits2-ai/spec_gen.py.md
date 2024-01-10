# `Bert-VITS2\spec_gen.py`

```
# 导入 torch 库
import torch
# 从 tqdm 库中导入 tqdm 函数
from tqdm import tqdm
# 从 multiprocessing 库中导入 Pool 类
from multiprocessing import Pool
# 从 mel_processing 模块中导入 spectrogram_torch 和 mel_spectrogram_torch 函数
from mel_processing import spectrogram_torch, mel_spectrogram_torch
# 从 utils 模块中导入 load_wav_to_torch 函数
from utils import load_wav_to_torch

# 定义 AudioProcessor 类
class AudioProcessor:
    # 初始化函数，接受多个参数
    def __init__(
        self,
        max_wav_value,
        use_mel_spec_posterior,
        filter_length,
        n_mel_channels,
        sampling_rate,
        hop_length,
        win_length,
        mel_fmin,
        mel_fmax,
    ):
        # 设置实例变量 max_wav_value
        self.max_wav_value = max_wav_value
        # 设置实例变量 use_mel_spec_posterior
        self.use_mel_spec_posterior = use_mel_spec_posterior
        # 设置实例变量 filter_length
        self.filter_length = filter_length
        # 设置实例变量 n_mel_channels
        self.n_mel_channels = n_mel_channels
        # 设置实例变量 sampling_rate
        self.sampling_rate = sampling_rate
        # 设置实例变量 hop_length
        self.hop_length = hop_length
        # 设置实例变量 win_length
        self.win_length = win_length
        # 设置实例变量 mel_fmin
        self.mel_fmin = mel_fmin
        # 设置实例变量 mel_fmax
        self.mel_fmax = mel_fmax
    # 处理音频文件，返回音频的频谱和归一化后的音频数据
    def process_audio(self, filename):
        # 载入音频文件并获取采样率
        audio, sampling_rate = load_wav_to_torch(filename)
        # 对音频数据进行归一化处理
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        # 生成频谱文件名
        spec_filename = filename.replace(".wav", ".spec.pt")
        # 如果使用梅尔频谱后处理，则替换文件名后缀
        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
        try:
            # 尝试加载预先计算的频谱数据
            spec = torch.load(spec_filename)
        except:
            # 如果未找到预先计算的频谱数据，则根据是否使用梅尔频谱后处理来计算频谱
            if self.use_mel_spec_posterior:
                spec = mel_spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.n_mel_channels,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    self.mel_fmin,
                    self.mel_fmax,
                    center=False,
                )
            else:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
            # 去除频谱数据的批次维度，并保存计算得到的频谱数据
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        # 返回频谱数据和归一化后的音频数据
        return spec, audio_norm
# 创建一个音频处理器对象，设置各项参数
processor = AudioProcessor(
    max_wav_value=32768.0,  # 最大音频值
    use_mel_spec_posterior=False,  # 是否使用梅尔频谱后验
    filter_length=2048,  # 滤波器长度
    n_mel_channels=128,  # 梅尔频道数
    sampling_rate=44100,  # 采样率
    hop_length=512,  # 跳跃长度
    win_length=2048,  # 窗口长度
    mel_fmin=0.0,  # 梅尔频率最小值
    mel_fmax="null",  # 梅尔频率最大值
)

# 打开文件 "filelists/train.list" 以只读模式
with open("filelists/train.list", "r") as f:
    # 从每一行中取出以 "|" 分割的第一部分作为文件路径，存入列表 filepaths
    filepaths = [line.split("|")[0] for line in f]

# 使用多进程处理
# 创建一个拥有32个进程的进程池
with Pool(processes=32) as pool:
    # 使用 tqdm 创建一个进度条，总长度为 filepaths 的长度
    with tqdm(total=len(filepaths)) as pbar:
        # 使用 pool.imap_unordered 方法并行处理 filepaths 中的文件路径
        for i, _ in enumerate(pool.imap_unordered(processor.process_audio, filepaths)):
            # 更新进度条
            pbar.update()
```