# `so-vits-svc\utils.py`

```
# 导入必要的库
import argparse  # 用于解析命令行参数
import glob  # 用于查找文件路径
import json  # 用于处理 JSON 数据
import logging  # 用于记录日志
import os  # 用于操作文件和目录
import re  # 用于处理正则表达式
import subprocess  # 用于执行外部命令
import sys  # 用于访问与 Python 解释器交互的变量和函数
import traceback  # 用于跟踪异常
from multiprocessing import cpu_count  # 用于获取 CPU 核心数量

import faiss  # 用于高效相似性搜索和聚类
import librosa  # 用于音频处理
import numpy as np  # 用于数值计算
import torch  # 用于构建神经网络
from scipy.io.wavfile import read  # 用于读取音频文件
from sklearn.cluster import MiniBatchKMeans  # 用于 K 均值聚类
from torch.nn import functional as F  # 用于神经网络的函数

# 设置全局变量 MATPLOTLIB_FLAG 为 False
MATPLOTLIB_FLAG = False

# 配置日志记录器，将日志输出到标准输出，日志级别为警告
logging.basicConfig(stream=sys.stdout, level=logging.WARN)
# 创建日志记录器对象
logger = logging

# 设置音频特征参数
f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)

# 定义函数，用于对音频的基频进行归一化处理
def normalize_f0(f0, x_mask, uv, random_scale=True):
    # 根据 x_mask 计算 uv 的和
    uv_sum = torch.sum(uv, dim=1, keepdim=True)
    # 将 uv_sum 中值为 0 的元素替换为 9999
    uv_sum[uv_sum == 0] = 9999
    # 根据 uv 计算 f0 的均值
    means = torch.sum(f0[:, 0, :] * uv, dim=1, keepdim=True) / uv_sum

    if random_scale:
        # 如果 random_scale 为 True，则生成随机缩放因子
        factor = torch.Tensor(f0.shape[0], 1).uniform_(0.8, 1.2).to(f0.device)
    else:
        # 否则，使用全为 1 的缩放因子
        factor = torch.ones(f0.shape[0], 1).to(f0.device)
    # 根据均值和缩放因子对 f0 进行归一化处理
    f0_norm = (f0 - means.unsqueeze(-1)) * factor.unsqueeze(-1)
    # 如果归一化后的 f0 中存在 NaN 值，则退出程序
    if torch.isnan(f0_norm).any():
        exit(0)
    return f0_norm * x_mask

# 定义函数，将绘制的数据转换为 NumPy 数组
def plot_data_to_numpy(x, y):
    global MATPLOTLIB_FLAG
    # 如果 MATPLOTLIB_FLAG 为 False，则导入 matplotlib 库并设置为使用非交互式后端
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    # 创建绘图对象
    fig, ax = plt.subplots(figsize=(10, 2))
    # 绘制数据
    plt.plot(x)
    plt.plot(y)
    plt.tight_layout()

    # 绘图并将结果转换为 NumPy 数组
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data
# 将基频转换为粗糙的频率
def f0_to_coarse(f0):
  # 将基频转换为梅尔频率
  f0_mel = 1127 * (1 + f0 / 700).log()
  # 计算缩放系数 a
  a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
  # 计算偏移量 b
  b = f0_mel_min * a - 1.
  # 对梅尔频率进行缩放和偏移
  f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
  # 对梅尔频率进行裁剪
  # torch.clip_(f0_mel, min=1., max=float(f0_bin - 1))
  # 对梅尔频率进行四舍五入并转换为整数
  f0_coarse = torch.round(f0_mel).long()
  # 将小于等于 0 的值设为 0
  f0_coarse = f0_coarse * (f0_coarse > 0)
  # 将小于 1 的值加 1
  f0_coarse = f0_coarse + ((f0_coarse < 1) * 1)
  # 将大于等于 f0_bin 的值设为 f0_bin - 1
  f0_coarse = f0_coarse * (f0_coarse < f0_bin)
  f0_coarse = f0_coarse + ((f0_coarse >= f0_bin) * (f0_bin - 1))
  return f0_coarse

# 获取音频的内容特征
def get_content(cmodel, y):
    # 禁用梯度计算
    with torch.no_grad():
        # 提取音频的特征
        c = cmodel.extract_features(y.squeeze(1))[0]
    # 转置特征张量的维度
    c = c.transpose(1, 2)
    return c

# 获取基频预测器
def get_f0_predictor(f0_predictor,hop_length,sampling_rate,**kargs):
    # 根据不同的基频预测器类型选择对应的类
    if f0_predictor == "pm":
        from modules.F0Predictor.PMF0Predictor import PMF0Predictor
        f0_predictor_object = PMF0Predictor(hop_length=hop_length,sampling_rate=sampling_rate)
    elif f0_predictor == "crepe":
        from modules.F0Predictor.CrepeF0Predictor import CrepeF0Predictor
        f0_predictor_object = CrepeF0Predictor(hop_length=hop_length,sampling_rate=sampling_rate,device=kargs["device"],threshold=kargs["threshold"])
    elif f0_predictor == "harvest":
        from modules.F0Predictor.HarvestF0Predictor import HarvestF0Predictor
        f0_predictor_object = HarvestF0Predictor(hop_length=hop_length,sampling_rate=sampling_rate)
    elif f0_predictor == "dio":
        from modules.F0Predictor.DioF0Predictor import DioF0Predictor
        f0_predictor_object = DioF0Predictor(hop_length=hop_length,sampling_rate=sampling_rate) 
    elif f0_predictor == "rmvpe":
        from modules.F0Predictor.RMVPEF0Predictor import RMVPEF0Predictor
        f0_predictor_object = RMVPEF0Predictor(hop_length=hop_length,sampling_rate=sampling_rate,dtype=torch.float32 ,device=kargs["device"],threshold=kargs["threshold"])
    # 如果 f0_predictor 等于 "fcpe"，则导入 FCPEF0Predictor 模块
    elif f0_predictor == "fcpe":
        from modules.F0Predictor.FCPEF0Predictor import FCPEF0Predictor
        # 创建 FCPEF0Predictor 对象，设置参数并赋值给 f0_predictor_object
        f0_predictor_object = FCPEF0Predictor(hop_length=hop_length,sampling_rate=sampling_rate,dtype=torch.float32 ,device=kargs["device"],threshold=kargs["threshold"])
    # 如果 f0_predictor 不等于 "fcpe"，则抛出异常
    else:
        raise Exception("Unknown f0 predictor")
    # 返回 f0_predictor_object 对象
    return f0_predictor_object
# 定义一个函数，用于获取语音编码器对象
def get_speech_encoder(speech_encoder, device=None, **kargs):
    # 如果选择的语音编码器是"vec768l12"，则导入对应的模块并创建对象
    if speech_encoder == "vec768l12":
        from vencoder.ContentVec768L12 import ContentVec768L12
        speech_encoder_object = ContentVec768L12(device=device)
    # 如果选择的语音编码器是"vec256l9"，则导入对应的模块并创建对象
    elif speech_encoder == "vec256l9":
        from vencoder.ContentVec256L9 import ContentVec256L9
        speech_encoder_object = ContentVec256L9(device=device)
    # 如果选择的语音编码器是"vec256l9-onnx"，则导入对应的模块并创建对象
    elif speech_encoder == "vec256l9-onnx":
        from vencoder.ContentVec256L9_Onnx import ContentVec256L9_Onnx
        speech_encoder_object = ContentVec256L9_Onnx(device=device)
    # 如果选择的语音编码器是"vec256l12-onnx"，则导入对应的模块并创建对象
    elif speech_encoder == "vec256l12-onnx":
        from vencoder.ContentVec256L12_Onnx import ContentVec256L12_Onnx
        speech_encoder_object = ContentVec256L12_Onnx(device=device)
    # 如果选择的语音编码器是"vec768l9-onnx"，则导入对应的模块并创建对象
    elif speech_encoder == "vec768l9-onnx":
        from vencoder.ContentVec768L9_Onnx import ContentVec768L9_Onnx
        speech_encoder_object = ContentVec768L9_Onnx(device=device)
    # 如果选择的语音编码器是"vec768l12-onnx"，则导入对应的模块并创建对象
    elif speech_encoder == "vec768l12-onnx":
        from vencoder.ContentVec768L12_Onnx import ContentVec768L12_Onnx
        speech_encoder_object = ContentVec768L12_Onnx(device=device)
    # 如果选择的语音编码器是"hubertsoft-onnx"，则导入对应的模块并创建对象
    elif speech_encoder == "hubertsoft-onnx":
        from vencoder.HubertSoft_Onnx import HubertSoft_Onnx
        speech_encoder_object = HubertSoft_Onnx(device=device)
    # 如果选择的语音编码器是"hubertsoft"，则导入对应的模块并创建对象
    elif speech_encoder == "hubertsoft":
        from vencoder.HubertSoft import HubertSoft
        speech_encoder_object = HubertSoft(device=device)
    # 如果选择的语音编码器是"whisper-ppg"，则导入对应的模块并创建对象
    elif speech_encoder == "whisper-ppg":
        from vencoder.WhisperPPG import WhisperPPG
        speech_encoder_object = WhisperPPG(device=device)
    # 如果选择的语音编码器是"cnhubertlarge"，则导入对应的模块并创建对象
    elif speech_encoder == "cnhubertlarge":
        from vencoder.CNHubertLarge import CNHubertLarge
        speech_encoder_object = CNHubertLarge(device=device)
    # 如果选择的语音编码器是"dphubert"，则导入对应的模块并创建对象
    elif speech_encoder == "dphubert":
        from vencoder.DPHubert import DPHubert
        speech_encoder_object = DPHubert(device=device)
    # 如果语音编码器是"whisper-ppg-large"，则导入WhisperPPGLarge类并创建对象
    elif speech_encoder == "whisper-ppg-large":
        from vencoder.WhisperPPGLarge import WhisperPPGLarge
        speech_encoder_object = WhisperPPGLarge(device = device)
    # 如果语音编码器是"wavlmbase+"，则导入WavLMBasePlus类并创建对象
    elif speech_encoder == "wavlmbase+":
        from vencoder.WavLMBasePlus import WavLMBasePlus
        speech_encoder_object = WavLMBasePlus(device = device)
    # 如果语音编码器不是以上两种情况，则抛出异常
    else:
        raise Exception("Unknown speech encoder")
    # 返回语音编码器对象
    return speech_encoder_object 
# 加载检查点文件，更新模型和优化器状态
def load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False):
    # 确保检查点文件存在
    assert os.path.isfile(checkpoint_path)
    # 加载检查点文件内容到字典中
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    # 获取迭代次数和学习率
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    # 如果优化器存在且不跳过优化器状态，并且检查点文件中包含优化器状态，则加载优化器状态
    if optimizer is not None and not skip_optimizer and checkpoint_dict['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    # 获取模型的保存状态
    saved_state_dict = checkpoint_dict['model']
    # 将模型转移到与保存状态相同的数据类型
    model = model.to(list(saved_state_dict.values())[0].dtype)
    # 如果模型有 'module' 属性，则获取模型状态字典
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    # 遍历模型状态字典，更新模型状态
    for k, v in state_dict.items():
        try:
            # 如果保存状态中的键存在于模型状态中，则更新模型状态
            new_state_dict[k] = saved_state_dict[k]
            # 确保保存状态中的形状与模型状态中的形状相同
            assert saved_state_dict[k].shape == v.shape, (saved_state_dict[k].shape, v.shape)
        except Exception:
            # 如果出现异常，打印警告信息，并记录日志
            if "enc_q" not in k or "emb_g" not in k:
                print("%s is not in the checkpoint,please check your checkpoint.If you're using pretrain model,just ignore this warning." % k)
                logger.info("%s is not in the checkpoint" % k)
                # 使用模型状态中的值
                new_state_dict[k] = v
    # 如果模型有 'module' 属性，则加载新的模型状态字典
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    # 打印加载信息，并记录日志
    print("load ")
    logger.info("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    # 返回更新后的模型、优化器、学习率和迭代次数
    return model, optimizer, learning_rate, iteration

# 保存模型和优化器状态到检查点文件
def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info("Saving model and optimizer state at iteration {} to {}".format(
        iteration, checkpoint_path))
    # 如果模型有 'module' 属性，则获取模型状态字典
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
    # 获取模型的状态字典
    state_dict = model.state_dict()
    # 保存模型的状态字典、迭代次数、优化器的状态字典和学习率到指定的检查点路径
    torch.save({'model': state_dict,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path)
# 清理检查点文件以释放空间
def clean_checkpoints(path_to_models='logs/44k/', n_ckpts_to_keep=2, sort_by_time=True):
  """Freeing up space by deleting saved ckpts

  Arguments:
  path_to_models    --  Path to the model directory
  n_ckpts_to_keep   --  Number of ckpts to keep, excluding G_0.pth and D_0.pth
  sort_by_time      --  True -> chronologically delete ckpts
                        False -> lexicographically delete ckpts
  """
  # 获取目录下的所有文件
  ckpts_files = [f for f in os.listdir(path_to_models) if os.path.isfile(os.path.join(path_to_models, f))]
  # 定义按文件名排序的函数
  def name_key(_f):
      return int(re.compile("._(\\d+)\\.pth").match(_f).group(1))
  # 定义按修改时间排序的函数
  def time_key(_f):
      return os.path.getmtime(os.path.join(path_to_models, _f))
  # 根据排序方式选择排序函数
  sort_key = time_key if sort_by_time else name_key
  # 定义按指定前缀排序并排除特定文件的函数
  def x_sorted(_x):
      return sorted([f for f in ckpts_files if f.startswith(_x) and not f.endswith("_0.pth")], key=sort_key)
  # 获取待删除的文件列表
  to_del = [os.path.join(path_to_models, fn) for fn in
            (x_sorted('G')[:-n_ckpts_to_keep] + x_sorted('D')[:-n_ckpts_to_keep])]
  # 定义删除文件的信息记录函数
  def del_info(fn):
      return logger.info(f".. Free up space by deleting ckpt {fn}")
  # 定义删除文件的例行程序函数
  def del_routine(x):
      return [os.remove(x), del_info(x)]
  # 执行删除文件的例行程序
  [del_routine(fn) for fn in to_del]

# 将标量、直方图、图像和音频数据汇总到 TensorBoard
def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
  for k, v in scalars.items():
    writer.add_scalar(k, v, global_step)
  for k, v in histograms.items():
    writer.add_histogram(k, v, global_step)
  for k, v in images.items():
    writer.add_image(k, v, global_step, dataformats='HWC')
  for k, v in audios.items():
    writer.add_audio(k, v, global_step, audio_sampling_rate)

# 获取最新的检查点文件路径
def latest_checkpoint_path(dir_path, regex="G_*.pth"):
  f_list = glob.glob(os.path.join(dir_path, regex))
  f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  x = f_list[-1]
  print(x)
  return x

# 将频谱图绘制为 numpy 数组
def plot_spectrogram_to_numpy(spectrogram):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    # 设置 MATPLOTLIB_FLAG 为 True
    MATPLOTLIB_FLAG = True
    # 获取 matplotlib 的日志记录器，并设置日志级别为 WARNING
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
    # 导入 matplotlib.pylab 和 numpy 模块
    import matplotlib.pylab as plt
    import numpy as np
    
    # 创建一个新的图形和一个子图，设置图形大小为 (10,2)
    fig, ax = plt.subplots(figsize=(10,2))
    # 在子图上显示频谱图像，设置纵横比为 "auto"，原点为 lower，插值方式为 none
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')
    # 在子图上添加颜色条
    plt.colorbar(im, ax=ax)
    # 设置 x 轴标签为 "Frames"，y 轴标签为 "Channels"
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    # 调整子图布局
    plt.tight_layout()
    
    # 绘制图形
    fig.canvas.draw()
    # 从图形画布中获取 RGB 数据，转换为一维数组
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # 将一维数组转换为图像数据的形状
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # 关闭图形
    plt.close()
    # 返回图像数据
    return data
# 将对齐矩阵转换为 numpy 数组
def plot_alignment_to_numpy(alignment, info=None):
  # 声明全局变量 MATPLOTLIB_FLAG
  global MATPLOTLIB_FLAG
  # 如果 MATPLOTLIB_FLAG 为假
  if not MATPLOTLIB_FLAG:
    # 导入 matplotlib 库并设置使用 "Agg" 后端
    import matplotlib
    matplotlib.use("Agg")
    # 将 MATPLOTLIB_FLAG 设置为真
    MATPLOTLIB_FLAG = True
    # 获取 matplotlib 的日志记录器并设置日志级别为 WARNING
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  # 导入 matplotlib.pylab 和 numpy 库
  import matplotlib.pylab as plt
  import numpy as np

  # 创建一个新的图形和子图
  fig, ax = plt.subplots(figsize=(6, 4))
  # 在子图上显示对齐矩阵的图像
  im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower',
                  interpolation='none')
  # 在图上添加颜色条
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  # 如果 info 不为空，则在 xlabel 上添加额外信息
  if info is not None:
      xlabel += '\n\n' + info
  # 设置 x 轴标签和 y 轴标签
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  # 调整子图布局
  plt.tight_layout()

  # 绘制图形
  fig.canvas.draw()
  # 从绘图对象中获取 RGB 数据并转换为 numpy 数组
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  # 关闭图形
  plt.close()
  # 返回转换后的 numpy 数组
  return data


# 将音频文件加载为 torch 张量
def load_wav_to_torch(full_path):
  # 读取音频文件的采样率和数据
  sampling_rate, data = read(full_path)
  # 将数据转换为 float32 类型的 torch 张量并返回
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate


# 加载文件路径和文本信息
def load_filepaths_and_text(filename, split="|"):
  # 打开文件并按指定分隔符分割每行的内容，返回文件路径和文本信息的列表
  with open(filename, encoding='utf-8') as f:
    filepaths_and_text = [line.strip().split(split) for line in f]
  return filepaths_and_text


# 获取超参数
def get_hparams(init=True):
  # 创建参数解析器
  parser = argparse.ArgumentParser()
  # 添加命令行参数选项
  parser.add_argument('-c', '--config', type=str, default="./configs/config.json",
                      help='JSON file for configuration')
  parser.add_argument('-m', '--model', type=str, required=True,
                      help='Model name')

  # 解析命令行参数
  args = parser.parse_args()
  # 模型目录为 "./logs" 下的模型名称目录
  model_dir = os.path.join("./logs", args.model)

  # 如果模型目录不存在，则创建
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  # 配置文件路径为命令行参数指定的路径
  config_path = args.config
  # 配置保存路径为模型目录下的 "config.json" 文件
  config_save_path = os.path.join(model_dir, "config.json")
  # 如果 init 为真
  if init:
    # 读取配置文件内容并将其写入配置保存路径
    with open(config_path, "r") as f:
      data = f.read()
    with open(config_save_path, "w") as f:
      f.write(data)
  else:
    # 使用只读模式打开配置文件
    with open(config_save_path, "r") as f:
        # 读取文件内容
        data = f.read()
    # 将读取的文件内容解析为 JSON 格式
    config = json.loads(data)
    # 使用解析后的配置创建 HParams 对象
    hparams = HParams(**config)
    # 设置 HParams 对象的模型目录
    hparams.model_dir = model_dir
    # 返回创建的 HParams 对象
    return hparams
# 从给定的模型目录中获取超参数
def get_hparams_from_dir(model_dir):
  # 构建配置文件保存路径
  config_save_path = os.path.join(model_dir, "config.json")
  # 打开配置文件，读取数据
  with open(config_save_path, "r") as f:
    data = f.read()
  # 将读取的数据转换为 JSON 格式
  config = json.loads(data)

  # 使用配置文件中的参数创建超参数对象
  hparams = HParams(**config)
  # 设置超参数对象的模型目录
  hparams.model_dir = model_dir
  # 返回超参数对象
  return hparams


# 从给定的配置文件中获取超参数
def get_hparams_from_file(config_path, infer_mode = False):
  # 打开配置文件，读取数据
  with open(config_path, "r") as f:
    data = f.read()
  # 将读取的数据转换为 JSON 格式
  config = json.loads(data)
  # 根据推断模式选择不同的超参数对象
  hparams = HParams(**config) if not infer_mode else InferHParams(**config)
  # 返回超参数对象
  return hparams


# 检查模型目录中的 Git 哈希值
def check_git_hash(model_dir):
  # 获取当前文件的目录
  source_dir = os.path.dirname(os.path.realpath(__file__))
  # 如果当前文件不是一个 Git 仓库，则忽略哈希值比较
  if not os.path.exists(os.path.join(source_dir, ".git")):
    logger.warn("{} is not a git repository, therefore hash value comparison will be ignored.".format(
      source_dir
    ))
    return

  # 获取当前 Git 仓库的哈希值
  cur_hash = subprocess.getoutput("git rev-parse HEAD")

  # 构建保存哈希值的文件路径
  path = os.path.join(model_dir, "githash")
  # 如果保存哈希值的文件存在
  if os.path.exists(path):
    # 读取保存的哈希值
    saved_hash = open(path).read()
    # 如果保存的哈希值与当前哈希值不同，则发出警告
    if saved_hash != cur_hash:
      logger.warn("git hash values are different. {}(saved) != {}(current)".format(
        saved_hash[:8], cur_hash[:8]))
  else:
    # 否则，将当前哈希值写入文件
    open(path, "w").write(cur_hash)


# 获取日志记录器
def get_logger(model_dir, filename="train.log"):
  global logger
  # 创建日志记录器
  logger = logging.getLogger(os.path.basename(model_dir))
  logger.setLevel(logging.DEBUG)

  # 设置日志格式
  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  # 如果模型目录不存在，则创建
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  # 创建日志文件处理器
  h = logging.FileHandler(os.path.join(model_dir, filename))
  h.setLevel(logging.DEBUG)
  h.setFormatter(formatter)
  # 将文件处理器添加到日志记录器中
  logger.addHandler(h)
  # 返回日志记录器
  return logger


# 根据模式对内容进行二维重复扩展
def repeat_expand_2d(content, target_len, mode = 'left'):
    # content : [h, t]
    # 根据模式选择不同的二维重复扩展方法
    return repeat_expand_2d_left(content, target_len) if mode == 'left' else repeat_expand_2d_other(content, target_len, mode)


# 以左侧为基准对内容进行二维重复扩展
def repeat_expand_2d_left(content, target_len):
    # content : [h, t]
    # 获取内容的长度
    src_len = content.shape[-1]
    # 创建一个全零张量，形状为[content.shape[0], target_len]，数据类型为torch.float，并将其放置在与content相同的设备上
    target = torch.zeros([content.shape[0], target_len], dtype=torch.float).to(content.device)
    # 创建一个长度为src_len+1的张量，每个元素为0到target_len的等差数列，用于确定content在target中的位置
    temp = torch.arange(src_len+1) * target_len / src_len
    # 初始化当前位置为0
    current_pos = 0
    # 遍历target的每个位置
    for i in range(target_len):
        # 如果当前位置小于temp中下一个位置的值
        if i < temp[current_pos+1]:
            # 将content[:, current_pos]赋值给target[:, i]
            target[:, i] = content[:, current_pos]
        else:
            # 否则，当前位置加1，将content[:, current_pos]赋值给target[:, i]
            current_pos += 1
            target[:, i] = content[:, current_pos]
    # 返回target张量
    return target
# mode : 'nearest'| 'linear'| 'bilinear'| 'bicubic'| 'trilinear'| 'area'
# 定义函数 repeat_expand_2d_other，用于将二维内容进行重复扩展
def repeat_expand_2d_other(content, target_len, mode = 'nearest'):
    # content : [h, t]
    # 将输入的二维内容扩展为三维，增加一个维度
    content = content[None,:,:]
    # 使用双线性插值将内容扩展到目标长度，返回结果
    target = F.interpolate(content,size=target_len,mode=mode)[0]
    return target

# 定义函数 mix_model，用于混合多个模型
def mix_model(model_paths,mix_rate,mode):
  # 将混合比例转换为浮点数
  mix_rate = torch.FloatTensor(mix_rate)/100
  # 加载第一个模型
  model_tem = torch.load(model_paths[0])
  # 加载所有模型
  models = [torch.load(path)["model"] for path in model_paths]
  # 如果模式为0，对混合比例进行 softmax 处理
  if mode == 0:
     mix_rate = F.softmax(mix_rate,dim=0)
  # 遍历模型的键
  for k in model_tem["model"].keys():
     # 将模型的对应键初始化为与模型相同形状的零张量
     model_tem["model"][k] = torch.zeros_like(model_tem["model"][k])
     # 遍历所有模型
     for i,model in enumerate(models):
        # 按照混合比例将模型参数进行加权求和
        model_tem["model"][k] += model[k]*mix_rate[i]
  # 保存混合后的模型
  torch.save(model_tem,os.path.join(os.path.curdir,"output.pth"))
  # 返回混合后的模型路径
  return os.path.join(os.path.curdir,"output.pth")

# 定义函数 change_rms，用于改变音频的 RMS 值
def change_rms(data1, sr1, data2, sr2, rate):  # 1是输入音频，2是输出音频,rate是2的占比 from RVC
    # 计算输入音频的 RMS 值
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 每半秒一个点
    # 计算输出音频的 RMS 值
    rms2 = librosa.feature.rms(y=data2.detach().cpu().numpy(), frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    # 将 numpy 数组转换为张量，并进行线性插值，使其与输出音频长度相同
    rms1 = torch.from_numpy(rms1).to(data2.device)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    # 将输出音频的 RMS 值转换为张量，并进行线性插值，使其与输出音频长度相同
    rms2 = torch.from_numpy(rms2).to(data2.device)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    # 将输出音频的 RMS 值限制为大于等于 1e-6
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    # 根据 RMS 值的比例改变输出音频的幅度
    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    )
    return data2

# 定义函数 train_index，用于训练索引
def train_index(spk_name,root_dir = "dataset/44k/"):  #from: RVC https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
    # 获取 CPU 核心数
    n_cpu = cpu_count()
    # 打印信息
    print("The feature index is constructing.")
    # 构建实验目录
    exp_dir = os.path.join(root_dir,spk_name)
    # 存储列表
    listdir_res = []
    # 遍历指定目录下的文件
    for file in os.listdir(exp_dir):
        # 如果文件名包含 ".wav.soft.pt"，则将其路径添加到列表中
        if ".wav.soft.pt" in file:
            listdir_res.append(os.path.join(exp_dir,file))
    # 如果列表为空，抛出异常
    if len(listdir_res) == 0:
        raise Exception("You need to run preprocess_hubert_f0.py!")
    # 创建空列表用于存储数据
    npys = []
    # 遍历排序后的文件路径列表
    for name in sorted(listdir_res):
        # 从文件中加载数据，转置并转换为 numpy 数组，然后添加到列表中
        phone = torch.load(name)[0].transpose(-1,-2).numpy()
        npys.append(phone)
    # 将列表中的数组沿着指定轴拼接成一个大数组
    big_npy = np.concatenate(npys, 0)
    # 创建一个与 big_npy 大小相同的数组，用于存储乱序后的索引
    big_npy_idx = np.arange(big_npy.shape[0])
    # 对索引数组进行随机乱序
    np.random.shuffle(big_npy_idx)
    # 根据乱序后的索引对大数组进行重新排序
    big_npy = big_npy[big_npy_idx]
    # 如果大数组的行数大于 2e5
    if big_npy.shape[0] > 2e5:
        # 打印提示信息
        info = "Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0]
        print(info)
        # 尝试进行 kmeans 聚类
        try:
            # 使用 MiniBatchKMeans 进行聚类，得到聚类中心
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        # 如果出现异常，打印异常信息
        except Exception:
            info = traceback.format_exc()
            print(info)
    # 计算 IVF 索引的数量
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    # 创建指定维度和 IVF 参数的索引
    index = faiss.index_factory(big_npy.shape[1] , "IVF%s,Flat" % n_ivf)
    # 提取 IVF 索引
    index_ivf = faiss.extract_index_ivf(index)  #
    # 设置 IVF 索引的探测次数
    index_ivf.nprobe = 1
    # 训练索引
    index.train(big_npy)
    # 设置批量添加数据的大小
    batch_size_add = 8192
    # 循环批量添加数据到索引中
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    # 打印成功构建索引的信息
    print("Successfully build index")
    # 返回索引对象
    return index
# 定义一个参数类，用于存储和管理模型的超参数
class HParams():
  # 初始化方法，接受一个字典作为参数
  def __init__(self, **kwargs):
    # 遍历传入的参数字典
    for k, v in kwargs.items():
      # 如果值的类型是字典，则递归创建 HParams 对象
      if type(v) == dict:
        v = HParams(**v)
      # 将参数存储到实例对象的属性中
      self[k] = v

  # 返回参数字典的键
  def keys(self):
    return self.__dict__.keys()

  # 返回参数字典的键值对
  def items(self):
    return self.__dict__.items()

  # 返回参数字典的值
  def values(self):
    return self.__dict__.values()

  # 返回参数字典的长度
  def __len__(self):
    return len(self.__dict__)

  # 获取参数字典中指定键的值
  def __getitem__(self, key):
    return getattr(self, key)

  # 设置参数字典中指定键的值
  def __setitem__(self, key, value):
    return setattr(self, key, value)

  # 判断参数字典中是否包含指定键
  def __contains__(self, key):
    return key in self.__dict__

  # 返回参数字典的字符串表示形式
  def __repr__(self):
    return self.__dict__.__repr__()

  # 获取参数字典中指定键的值
  def get(self,index):
    return self.__dict__.get(index)

# 定义一个推断参数类，继承自 HParams 类
class InferHParams(HParams):
  # 初始化方法，接受一个字典作为参数
  def __init__(self, **kwargs):
    # 遍历传入的参数字典
    for k, v in kwargs.items():
      # 如果值的类型是字典，则递归创建 InferHParams 对象
      if type(v) == dict:
        v = InferHParams(**v)
      # 将参数存储到实例对象的属性中
      self[k] = v

  # 获取参数字典中指定键的值
  def __getattr__(self,index):
    return self.get(index)

# 定义一个音量提取器类
class Volume_Extractor:
    # 初始化方法，接受一个跳跃大小参数
    def __init__(self, hop_size = 512):
        # 将跳跃大小参数存储到实例对象的属性中
        self.hop_size = hop_size
        
    # 提取音频的音量
    def extract(self, audio): # audio: 2d tensor array
        # 如果音频不是 torch.Tensor 类型，则转换为 torch.Tensor 类型
        if not isinstance(audio,torch.Tensor):
           audio = torch.Tensor(audio)
        # 计算音频帧数
        n_frames = int(audio.size(-1) // self.hop_size)
        # 计算音频的平方
        audio2 = audio ** 2
        # 对音频的平方进行填充
        audio2 = torch.nn.functional.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)), mode = 'reflect')
        # 对填充后的音频进行展开和计算平均值，得到音量
        volume = torch.nn.functional.unfold(audio2[:,None,None,:],(1,self.hop_size),stride=self.hop_size)[:,:,:n_frames].mean(dim=1)[0]
        # 对音量进行平方根处理
        volume = torch.sqrt(volume)
        # 返回音量
        return volume
```