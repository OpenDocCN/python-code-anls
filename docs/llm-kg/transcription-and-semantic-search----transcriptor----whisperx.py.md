# `.\transcription-and-semantic-search\transcriptor\whisperx.py`

```
import whisper  # 导入 whisper 模块
import whisperx  # 导入 whisperx 模块

from base.config import Config  # 从 base.config 模块中导入 Config 类


class WhisperX(Config):
    """Transcription Model"""

    def __init__(self, model_name: str):
        """
        Init method
        Args:
            model_name (str): 模型名称，可以是 'whisper' 或 'whisperx'
        """
        super().__init__()  # 调用父类的初始化方法
        self.model_name = model_name  # 设置实例变量 model_name
        if self.model_name == "whisper":
            # 如果模型名称是 'whisper'，则加载 whisper 模型
            self.model = whisper.load_model(
                self.config["transcriptor"][model_name]["model"],  # 加载指定的 whisper 模型
                self.config["transcriptor"]["device"],  # 指定设备
            )
        else:
            # 如果模型名称是 'whisperx'，则加载 whisperx 模型
            self.model = whisperx.load_model(
                self.config["transcriptor"][model_name]["model"],  # 加载指定的 whisperx 模型
                self.config["transcriptor"]["device"],  # 指定设备
                compute_type=self.config["transcriptor"][model_name]["compute_type"],  # 指定计算类型
            )

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribes the audio
        Args:
            audio_path (str): .wav 文件的路径
        Returns:
            result_aligned (dict): 包含分段和元数据的字典
        """
        if self.model_name == "whisperx":
            # 如果模型名称是 'whisperx'，则加载音频文件
            audio_path = whisperx.load_audio(audio_path)

        result = self.model.transcribe(audio_path)  # 对音频进行转录
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],  # 使用转录结果中的语言代码
            device=self.config["transcriptor"]["device"],  # 指定设备
        )
        result_aligned = whisperx.align(
            result["segments"],  # 转录结果中的音频分段
            model_a,  # 对齐模型
            metadata,  # 元数据
            audio_path,  # 音频文件路径
            self.config["transcriptor"]["device"],  # 指定设备
        )

        return result_aligned  # 返回对齐后的结果字典
```