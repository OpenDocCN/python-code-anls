# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\uvr5\MDXNet.py`

```py
import soundfile as sf  # 导入处理音频文件的库
import torch, pdb, time, argparse, os, warnings, sys, librosa  # 导入多个常用库
import numpy as np  # 导入处理数组的库
import onnxruntime as ort  # 导入用于运行 ONNX 模型的库
from scipy.io.wavfile import write  # 导入用于写入 WAV 文件的函数
from tqdm import tqdm  # 导入用于显示进度条的库
import torch  # 导入 PyTorch
import torch.nn as nn  # 导入 PyTorch 神经网络模块

dim_c = 4  # 设置全局变量 dim_c 为 4


class Conv_TDF_net_trim:
    def __init__(
        self, device, model_name, target_name, L, dim_f, dim_t, n_fft, hop=1024
    ):
        super(Conv_TDF_net_trim, self).__init__()  # 初始化父类构造函数

        self.dim_f = dim_f  # 设置频率维度
        self.dim_t = 2**dim_t  # 设置时间维度为 2 的 dim_t 次方
        self.n_fft = n_fft  # 设置 FFT 点数
        self.hop = hop  # 设置帧移大小
        self.n_bins = self.n_fft // 2 + 1  # 计算频谱帧数
        self.chunk_size = hop * (self.dim_t - 1)  # 计算块大小
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(
            device
        )  # 创建汉宁窗口
        self.target_name = target_name  # 设置目标名称
        self.blender = "blender" in model_name  # 检查模型名称中是否包含 "blender"

        out_c = dim_c * 4 if target_name == "*" else dim_c  # 根据目标名称调整输出通道数
        self.freq_pad = torch.zeros(
            [1, out_c, self.n_bins - self.dim_f, self.dim_t]
        ).to(device)  # 创建用于填充频率的张量，维度为 [1, out_c, 频谱点数-频率维度, 时间维度]

        self.n = L // 2  # 设置 n 的值为 L 的一半

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])  # 重塑输入张量 x
        x = torch.stft(  # 执行短时傅里叶变换
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)  # 转换为实部和虚部
        x = x.permute([0, 3, 1, 2])  # 调整维度顺序
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, dim_c, self.n_bins, self.dim_t]
        )  # 重塑张量维度
        return x[:, :, : self.dim_f]  # 返回频率维度范围内的数据

    def istft(self, x, freq_pad=None):
        freq_pad = (
            self.freq_pad.repeat([x.shape[0], 1, 1, 1])
            if freq_pad is None
            else freq_pad
        )  # 复制填充频率的张量，如果未提供则使用 self.freq_pad
        x = torch.cat([x, freq_pad], -2)  # 在最后一个维度上拼接张量 x 和 freq_pad
        c = 4 * 2 if self.target_name == "*" else 2  # 根据目标名称调整 c 的值
        x = x.reshape([-1, c, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 2, self.n_bins, self.dim_t]
        )  # 重塑张量维度
        x = x.permute([0, 2, 3, 1])  # 调整维度顺序
        x = x.contiguous()  # 确保张量是连续的
        x = torch.view_as_complex(x)  # 转换为复数形式
        x = torch.istft(
            x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True
        )  # 执行反短时傅里叶变换
        return x.reshape([-1, c, self.chunk_size])  # 重塑输出张量维度


def get_models(device, dim_f, dim_t, n_fft):
    return Conv_TDF_net_trim(  # 返回 Conv_TDF_net_trim 类的实例
        device=device,
        model_name="Conv-TDF",
        target_name="vocals",
        L=11,
        dim_f=dim_f,
        dim_t=dim_t,
        n_fft=n_fft,
    )


warnings.filterwarnings("ignore")  # 忽略警告信息
cpu = torch.device("cpu")  # 设置 CPU 设备
if torch.cuda.is_available():  # 如果 CUDA 可用
    device = torch.device("cuda:0")  # 设置为第一个 CUDA 设备
elif torch.backends.mps.is_available():  # 如果存在 MPS 后端
    device = torch.device("mps")  # 设置为 MPS 设备
else:  # 否则
    device = torch.device("cpu")  # 设置为 CPU 设备


class Predictor:
    # 初始化方法，接受参数 args
    def __init__(self, args):
        # 将参数 args 存储为对象的属性
        self.args = args
        # 调用函数 get_models 获取模型，返回的是一个模型对象
        self.model_ = get_models(
            device=cpu, dim_f=args.dim_f, dim_t=args.dim_t, n_fft=args.n_fft
        )
        # 使用 ONNX 文件路径创建 ONNX 推理会话对象，指定使用 CUDA 和 CPU 执行提供者
        self.model = ort.InferenceSession(
            os.path.join(args.onnx, self.model_.target_name + ".onnx"),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        # 打印信息，表示 ONNX 模型加载完成
        print("onnx load done")

    # 分离方法，接受混合音频数据 mix 作为参数
    def demix(self, mix):
        # 获取混合音频的样本数
        samples = mix.shape[-1]
        # 获取参数中定义的边缘大小
        margin = self.args.margin
        # 定义每个分段的大小为参数中指定的块数乘以采样率
        chunk_size = self.args.chunks * 44100
        # 断言边缘大小不能为零
        assert not margin == 0, "margin cannot be zero!"
        # 如果定义的边缘大小大于分段大小，则将边缘大小设置为分段大小
        if margin > chunk_size:
            margin = chunk_size

        # 创建空字典来存储分段后的混合音频数据
        segmented_mix = {}

        # 如果块数为 0 或者音频样本数小于分段大小，则将分段大小设置为音频样本数
        if self.args.chunks == 0 or samples < chunk_size:
            chunk_size = samples

        # 初始化计数器为 -1，用于迭代处理分段
        counter = -1
        # 循环处理每个分段，从 0 开始，步长为 chunk_size
        for skip in range(0, samples, chunk_size):
            counter += 1

            # 如果是第一次迭代，s_margin 设为 0，否则设为预定义的边缘大小
            s_margin = 0 if counter == 0 else margin
            # 计算当前分段的结束位置，确保不超过样本数
            end = min(skip + chunk_size + margin, samples)

            # 计算当前分段的起始位置，考虑上一个分段的边缘
            start = skip - s_margin

            # 将当前分段的混合音频数据复制到 segmented_mix 字典中
            segmented_mix[skip] = mix[:, start:end].copy()
            # 如果已经处理到音频末尾，则结束循环
            if end == samples:
                break

        # 调用 demix_base 方法，传入分段后的混合音频数据和边缘大小，获取分离后的音频源
        """
        mix:(2,big_sample)
        segmented_mix:offset->(2,small_sample)
        sources:(1,2,big_sample)
        """
        sources = self.demix_base(segmented_mix, margin_size=margin)
        
        # 返回分离后的音频源数据
        return sources
    # 定义一个方法 demix_base，用于对混合信号进行分离处理
    def demix_base(self, mixes, margin_size):
        # 初始化一个空列表，用于存储分离后的信号块
        chunked_sources = []
        # 初始化一个进度条，总数为 mixes 的长度，显示 "Processing" 作为描述信息
        progress_bar = tqdm(total=len(mixes))
        progress_bar.set_description("Processing")
        # 遍历 mixes 中的每一个混合信号
        for mix in mixes:
            # 获取当前混合信号的内容
            cmix = mixes[mix]
            # 初始化一个空列表，用于存储分离后的信号
            sources = []
            # 获取当前混合信号的样本数
            n_sample = cmix.shape[1]
            # 获取模型对象
            model = self.model_
            # 计算修剪值，为模型 n_fft 的一半
            trim = model.n_fft // 2
            # 计算生成大小，为模型 chunk_size 减去两倍的 trim
            gen_size = model.chunk_size - 2 * trim
            # 计算填充大小，为 gen_size 减去 n_sample 取模的结果
            pad = gen_size - n_sample % gen_size
            # 对混合信号进行填充处理，拼接得到 mix_p
            mix_p = np.concatenate(
                (np.zeros((2, trim)), cmix, np.zeros((2, pad)), np.zeros((2, trim))), 1
            )
            # 初始化一个空列表，用于存储切分后的混合信号块
            mix_waves = []
            # 初始化索引 i
            i = 0
            # 循环切分混合信号，直到处理完所有样本
            while i < n_sample + pad:
                # 提取 mix_p 中的片段，形成 waves 数组，并添加到 mix_waves 列表中
                waves = np.array(mix_p[:, i : i + model.chunk_size])
                mix_waves.append(waves)
                i += gen_size
            # 将 mix_waves 转换为 torch.tensor 类型，并移到 CPU 上
            mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(cpu)
            # 关闭梯度计算
            with torch.no_grad():
                # 获取模型对象
                _ort = self.model
                # 对混合信号的时频数据进行 STFT 变换
                spek = model.stft(mix_waves)
                # 如果开启了去噪选项
                if self.args.denoise:
                    # 使用 ONNX 运行去噪模型，得到预测的频谱
                    spec_pred = (
                        -_ort.run(None, {"input": -spek.cpu().numpy()})[0] * 0.5
                        + _ort.run(None, {"input": spek.cpu().numpy()})[0] * 0.5
                    )
                    # 将预测的频谱反变换为波形数据
                    tar_waves = model.istft(torch.tensor(spec_pred))
                else:
                    # 否则直接对频谱数据进行反变换为波形数据
                    tar_waves = model.istft(
                        torch.tensor(_ort.run(None, {"input": spek.cpu().numpy()})[0])
                    )
                # 对反变换后的波形数据进行处理，截取有效信号部分
                tar_signal = (
                    tar_waves[:, :, trim:-trim]
                    .transpose(0, 1)
                    .reshape(2, -1)
                    .numpy()[:, :-pad]
                )

                # 计算每个源信号的起始和结束位置
                start = 0 if mix == 0 else margin_size
                end = None if mix == list(mixes.keys())[::-1][0] else -margin_size
                if margin_size == 0:
                    end = None
                # 截取对应范围内的源信号，并添加到 sources 列表中
                sources.append(tar_signal[:, start:end])

                # 更新进度条
                progress_bar.update(1)

            # 将当前混合信号处理后的源信号列表 sources 添加到 chunked_sources 中
            chunked_sources.append(sources)
        
        # 将所有混合信号处理后的源信号按最后一个轴进行连接
        _sources = np.concatenate(chunked_sources, axis=-1)
        # 删除模型对象
        # del self.model
        # 关闭进度条
        progress_bar.close()
        # 返回处理后的源信号数组
        return _sources
    # 定义一个方法用于预测处理音频混合文件的声音分离
    def prediction(self, m, vocal_root, others_root, format):
        # 确保目录存在，如果不存在则创建
        os.makedirs(vocal_root, exist_ok=True)
        os.makedirs(others_root, exist_ok=True)
        # 获取文件的基本名称部分
        basename = os.path.basename(m)
        # 使用 librosa 库加载音频文件，返回混合音频数据和采样率
        mix, rate = librosa.load(m, mono=False, sr=44100)
        # 如果混合音频数据的维度为1，将其转换为二维数组
        if mix.ndim == 1:
            mix = np.asfortranarray([mix, mix])
        # 转置混合音频数据，确保与 librosa 的预期格式一致
        mix = mix.T
        # 对混合音频进行声音分离处理，得到各个分离后的声音源
        sources = self.demix(mix.T)
        # 选取其中一个声音源作为优化后的声音数据
        opt = sources[0].T
        vocal_path = None
        others_path = None

        # 根据指定的格式写入声音分离后的文件
        if format in ["wav", "flac"]:
            # 构建保存主唱音轨文件路径
            vocal_path = "%s/%s_main_vocal.%s" % (vocal_root, basename, format)
            # 构建保存其他音轨文件路径
            others_path = "%s/%s_others.%s" % (others_root, basename, format)
            # 将混合音频减去优化后的声音数据，写入主唱音轨文件
            sf.write(vocal_path, mix - opt, rate)
            # 将优化后的声音数据写入其他音轨文件
            sf.write(others_path, opt, rate)
        else:
            # 如果格式不是 wav 或 flac，使用默认的 wav 格式保存
            vocal_path = "%s/%s_main_vocal.wav" % (vocal_root, basename)
            others_path = "%s/%s_others.wav" % (others_root, basename)
            # 将混合音频减去优化后的声音数据，写入主唱音轨文件
            sf.write(vocal_path, mix - opt, rate)
            # 将优化后的声音数据写入其他音轨文件
            sf.write(others_path, opt, rate)
            # 如果已经存在主唱音轨文件，使用 ffmpeg 转换为指定格式
            if os.path.exists(vocal_path):
                os.system(
                    "ffmpeg -i %s -vn %s -q:a 2 -y"
                    % (vocal_path, vocal_path[:-4] + ".%s" % format)
                )
            # 如果已经存在其他音轨文件，使用 ffmpeg 转换为指定格式
            if os.path.exists(others_path):
                os.system(
                    "ffmpeg -i %s -vn %s -q:a 2 -y"
                    % (others_path, others_path[:-4] + ".%s" % format)
                )

        # 返回保存的主唱音轨和其他音轨文件路径
        return vocal_path, others_path
class MDXNetDereverb:
    def __init__(self, onnx, chunks):
        # 初始化函数，接收两个参数：onnx 和 chunks
        self.onnx = onnx
        # 设定属性 shifts 为固定值 10，用于预测时的随机等变稳定
        self.shifts = 10  #'Predict with randomised equivariant stabilisation'
        # 设定属性 mixing 为字符串 "min_mag"，表示混合方式选择
        self.mixing = "min_mag"  # ['default','min_mag','max_mag']
        # 初始化函数的第二个参数 chunks 赋值给属性 chunks
        self.chunks = chunks
        # 设定属性 margin 为固定值 44100，可能是音频处理中的边缘值
        self.margin = 44100
        # 设定属性 dim_t 为固定值 9，可能是时间维度上的大小
        self.dim_t = 9
        # 设定属性 dim_f 为固定值 3072，可能是频率维度上的大小
        self.dim_f = 3072
        # 设定属性 n_fft 为固定值 6144，可能是FFT（快速傅里叶变换）的窗口大小
        self.n_fft = 6144
        # 设定属性 denoise 为 True，表示是否进行降噪处理
        self.denoise = True
        # 创建 Predictor 类的实例，传入当前对象 self 作为参数
        self.pred = Predictor(self)

    def _path_audio_(self, input, vocal_root, others_root, format):
        """
        处理音频
        :param input: 输入音频文件名
        :param vocal_root: vocal 音频文件的根目录
        :param others_root: others 音频文件的根目录
        :param format: 音频文件格式
        :return: vocal_path, others_path，返回处理后的 vocal 和 others 音频文件路径
        """
        # 使用 self.pred 的 prediction 方法处理输入的音频，返回处理后的 vocal 和 others 的路径
        vocal_path, others_path = self.pred.prediction(input, vocal_root, others_root, format)
        # 返回处理后的 vocal 和 others 的路径
        return vocal_path, others_path


if __name__ == "__main__":
    # 创建 MDXNetDereverb 类的实例 dereverb，传入参数 15
    dereverb = MDXNetDereverb(15)
    # 导入 time 模块的 time 方法，并重命名为 ttime
    from time import time as ttime

    # 记录开始时间
    t0 = ttime()
    # 调用 dereverb 实例的 _path_audio_ 方法，处理指定的音频文件
    dereverb._path_audio_(
        "雪雪伴奏对消HP5.wav",
        "vocal",
        "others",
    )
    # 记录结束时间
    t1 = ttime()
    # 输出处理时间的差值
    print(t1 - t0)
```