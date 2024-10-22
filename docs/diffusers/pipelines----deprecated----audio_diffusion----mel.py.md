# `.\diffusers\pipelines\deprecated\audio_diffusion\mel.py`

```py
# 版权声明，标识该代码的版权信息
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 许可证信息，指出该文件的使用条款
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 说明可以在此处获得许可证
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 指明软件分发的条件，强调其无担保性质
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入 NumPy 库，供后续使用
import numpy as np  # noqa: E402

# 从配置工具导入混合类和注册装饰器
from ....configuration_utils import ConfigMixin, register_to_config
# 从调度工具导入调度混合类
from ....schedulers.scheduling_utils import SchedulerMixin

# 尝试导入 librosa 库
try:
    import librosa  # noqa: E402

    # 如果导入成功，设置标志为真
    _librosa_can_be_imported = True
    # 初始化导入错误信息为空
    _import_error = ""
# 捕获导入错误并设置相关标志和错误信息
except Exception as e:
    _librosa_can_be_imported = False
    # 设置错误信息，指示如何解决导入问题
    _import_error = (
        f"Cannot import librosa because {e}. Make sure to correctly install librosa to be able to install it."
    )

# 导入 PIL 库中的 Image 模块
from PIL import Image  # noqa: E402

# 定义 Mel 类，继承配置和调度混合类
class Mel(ConfigMixin, SchedulerMixin):
    """
    参数说明：
        x_res (`int`):
            频谱图的 x 方向分辨率（时间）。
        y_res (`int`):
            频谱图的 y 方向分辨率（频率区间）。
        sample_rate (`int`):
            音频的采样率。
        n_fft (`int`):
            快速傅里叶变换的数量。
        hop_length (`int`):
            每次移动的长度（当 `y_res` < 256 时，推荐更大的值）。
        top_db (`int`):
            最大分贝值。
        n_iter (`int`):
            Griffin-Lim Mel 反转的迭代次数。
    """

    # 指定配置文件名
    config_name = "mel_config.json"

    # 初始化方法，注册到配置中
    @register_to_config
    def __init__(
        self,
        x_res: int = 256,  # x 方向分辨率，默认值为 256
        y_res: int = 256,  # y 方向分辨率，默认值为 256
        sample_rate: int = 22050,  # 默认采样率为 22050
        n_fft: int = 2048,  # 默认 FFT 数量为 2048
        hop_length: int = 512,  # 默认移动长度为 512
        top_db: int = 80,  # 默认最大分贝值为 80
        n_iter: int = 32,  # 默认迭代次数为 32
    ):
        # 设置 hop_length 属性
        self.hop_length = hop_length
        # 设置采样率属性
        self.sr = sample_rate
        # 设置 FFT 数量属性
        self.n_fft = n_fft
        # 设置最大分贝值属性
        self.top_db = top_db
        # 设置迭代次数属性
        self.n_iter = n_iter
        # 调用方法设置频谱图分辨率
        self.set_resolution(x_res, y_res)
        # 初始化音频属性为 None
        self.audio = None

        # 检查 librosa 是否成功导入，若未导入则引发错误
        if not _librosa_can_be_imported:
            raise ValueError(_import_error)

    # 设置频谱图分辨率的方法
    def set_resolution(self, x_res: int, y_res: int):
        """设置分辨率。

        参数：
            x_res (`int`):
                频谱图的 x 方向分辨率（时间）。
            y_res (`int`):
                频谱图的 y 方向分辨率（频率区间）。
        """
        # 设置 x 方向分辨率
        self.x_res = x_res
        # 设置 y 方向分辨率
        self.y_res = y_res
        # 设置梅尔频率数量
        self.n_mels = self.y_res
        # 计算切片大小
        self.slice_size = self.x_res * self.hop_length - 1
    # 加载音频文件或原始音频数据
        def load_audio(self, audio_file: str = None, raw_audio: np.ndarray = None):
            """Load audio.
    
            Args:
                audio_file (`str`):
                    An audio file that must be on disk due to [Librosa](https://librosa.org/) limitation.
                raw_audio (`np.ndarray`):
                    The raw audio file as a NumPy array.
            """
            # 如果提供了音频文件名，则使用 Librosa 加载音频
            if audio_file is not None:
                self.audio, _ = librosa.load(audio_file, mono=True, sr=self.sr)
            # 否则，使用提供的原始音频数据
            else:
                self.audio = raw_audio
    
            # 如果音频长度不足，使用静音进行填充
            if len(self.audio) < self.x_res * self.hop_length:
                self.audio = np.concatenate([self.audio, np.zeros((self.x_res * self.hop_length - len(self.audio),))])
    
        # 获取音频切片的数量
        def get_number_of_slices(self) -> int:
            """Get number of slices in audio.
    
            Returns:
                `int`:
                    Number of spectograms audio can be sliced into.
            """
            # 返回音频长度除以每个切片的大小，得到切片数量
            return len(self.audio) // self.slice_size
    
        # 获取指定音频切片
        def get_audio_slice(self, slice: int = 0) -> np.ndarray:
            """Get slice of audio.
    
            Args:
                slice (`int`):
                    Slice number of audio (out of `get_number_of_slices()`).
    
            Returns:
                `np.ndarray`:
                    The audio slice as a NumPy array.
            """
            # 返回指定切片的音频数据
            return self.audio[self.slice_size * slice : self.slice_size * (slice + 1)]
    
        # 获取音频的采样率
        def get_sample_rate(self) -> int:
            """Get sample rate.
    
            Returns:
                `int`:
                    Sample rate of audio.
            """
            # 返回音频的采样率
            return self.sr
    
        # 将音频切片转换为声谱图
        def audio_slice_to_image(self, slice: int) -> Image.Image:
            """Convert slice of audio to spectrogram.
    
            Args:
                slice (`int`):
                    Slice number of audio to convert (out of `get_number_of_slices()`).
    
            Returns:
                `PIL Image`:
                    A grayscale image of `x_res x y_res`.
            """
            # 计算音频切片的梅尔声谱图
            S = librosa.feature.melspectrogram(
                y=self.get_audio_slice(slice), sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
            )
            # 将声谱图转换为对数刻度
            log_S = librosa.power_to_db(S, ref=np.max, top_db=self.top_db)
            # 将对数声谱图归一化并转换为8位无符号整数
            bytedata = (((log_S + self.top_db) * 255 / self.top_db).clip(0, 255) + 0.5).astype(np.uint8)
            # 从数组创建图像
            image = Image.fromarray(bytedata)
            # 返回生成的图像
            return image
    # 定义一个将光谱图转换为音频的函数
    def image_to_audio(self, image: Image.Image) -> np.ndarray:
        """将光谱图转换为音频。
    
        参数:
            image (`PIL Image`):
                一个灰度图像，尺寸为 `x_res x y_res`。
    
        返回:
            audio (`np.ndarray`):
                以 NumPy 数组形式返回的音频。
        """
        # 将图像数据转换为字节并从字节缓冲区创建一个 NumPy 数组，数据类型为无符号 8 位整型
        bytedata = np.frombuffer(image.tobytes(), dtype="uint8").reshape((image.height, image.width))
        # 将字节数据转换为浮点数，并进行归一化处理，计算对数幅度谱
        log_S = bytedata.astype("float") * self.top_db / 255 - self.top_db
        # 将对数幅度谱转换为功率谱
        S = librosa.db_to_power(log_S)
        # 使用逆梅尔频谱将功率谱转换为音频信号
        audio = librosa.feature.inverse.mel_to_audio(
            S, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_iter=self.n_iter
        )
        # 返回生成的音频数组
        return audio
```