# `.\Chat-Haruhi-Suzumiya\yuki_builder\audio_feature_ext\utils\record.py`

```py
import wave

import pyaudio


class RecordAudio:
    def __init__(self):
        # 录音参数
        self.chunk = 1024  # 设置每次读取的音频数据块大小为1024字节
        self.format = pyaudio.paInt16  # 设置音频格式为16位PCM格式
        self.channels = 1  # 设置音频通道数为1（单声道）
        self.rate = 16000  # 设置音频采样率为16000 Hz

        # 打开录音流
        self.p = pyaudio.PyAudio()  # 初始化PyAudio对象
        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk)  # 打开音频输入流

    def record(self, output_path="audio/temp.wav", record_seconds=3):
        """
        录音
        :param output_path: 录音保存的路径，后缀名为wav
        :param record_seconds: 录音时间，默认3秒
        :return: 录音的文件路径
        """
        i = input("按下回车键开机录音，录音3秒中：")  # 提示用户按回车键开始录音
        print("开始录音......")
        frames = []
        for i in range(0, int(self.rate / self.chunk * record_seconds)):
            data = self.stream.read(self.chunk)  # 从音频流中读取数据
            frames.append(data)  # 将读取的数据块添加到frames列表中

        print("录音已结束!")
        wf = wave.open(output_path, 'wb')  # 创建一个wave文件对象
        wf.setnchannels(self.channels)  # 设置wave文件的通道数
        wf.setsampwidth(self.p.get_sample_size(self.format))  # 设置wave文件的样本宽度
        wf.setframerate(self.rate)  # 设置wave文件的采样率
        wf.writeframes(b''.join(frames))  # 将所有音频数据写入wave文件
        wf.close()  # 关闭wave文件对象
        return output_path  # 返回录音文件的路径
```