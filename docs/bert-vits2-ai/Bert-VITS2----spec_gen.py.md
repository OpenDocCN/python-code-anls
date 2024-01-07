# `Bert-VITS2\spec_gen.py`

```

# 导入需要的库
import torch
from tqdm import tqdm
from multiprocessing import Pool
from mel_processing import spectrogram_torch, mel_spectrogram_torch
from utils import load_wav_to_torch

# 定义音频处理类
class AudioProcessor:
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
        # 初始化音频处理器的参数
        self.max_wav_value = max_wav_value
        self.use_mel_spec_posterior = use_mel_spec_posterior
        self.filter_length = filter_length
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax

    # 处理音频文件
    def process_audio(self, filename):
        # 加载音频文件并获取采样率
        audio, sampling_rate = load_wav_to_torch(filename)
        # 归一化音频数据
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        # 生成保存频谱数据的文件名
        spec_filename = filename.replace(".wav", ".spec.pt")
        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
        try:
            # 尝试加载已经存在的频谱数据文件
            spec = torch.load(spec_filename)
        except:
            # 如果文件不存在，则根据需要生成频谱数据
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
            spec = torch.squeeze(spec, 0)
            # 保存生成的频谱数据
            torch.save(spec, spec_filename)
        return spec, audio_norm

# 创建音频处理器实例
processor = AudioProcessor(
    max_wav_value=32768.0,
    use_mel_spec_posterior=False,
    filter_length=2048,
    n_mel_channels=128,
    sampling_rate=44100,
    hop_length=512,
    win_length=2048,
    mel_fmin=0.0,
    mel_fmax="null",
)

# 读取文件列表
with open("filelists/train.list", "r") as f:
    filepaths = [line.split("|")[0] for line in f]  # 取每一行的第一部分作为audiopath

# 使用多进程处理音频文件
with Pool(processes=32) as pool:  # 使用32个进程
    with tqdm(total=len(filepaths)) as pbar:
        for i, _ in enumerate(pool.imap_unordered(processor.process_audio, filepaths)):
            pbar.update()

```