# `Bert-VITS2\utils.py`

```py
# 导入所需的模块
import os
import glob
import argparse
import logging
import json
import shutil
import subprocess
import numpy as np
from huggingface_hub import hf_hub_download
from scipy.io.wavfile import read
import torch
import re

# 设置全局变量 MATPLOTLIB_FLAG
MATPLOTLIB_FLAG = False

# 获取日志记录器
logger = logging.getLogger(__name__)

# 下载情感模型
def download_emo_models(mirror, repo_id, model_name):
    # 如果镜像是 "openi"，则使用 openi 模块下载模型
    if mirror == "openi":
        import openi

        openi.model.download_model(
            "Stardust_minus/Bert-VITS2",
            repo_id.split("/")[-1],
            "./emotional",
        )
    # 否则使用 hf_hub_download 下载模型
    else:
        hf_hub_download(
            repo_id,
            "pytorch_model.bin",
            local_dir=model_name,
            local_dir_use_symlinks=False,
        )

# 下载检查点
def download_checkpoint(
    dir_path, repo_config, token=None, regex="G_*.pth", mirror="openi"
):
    repo_id = repo_config["repo_id"]
    # 获取指定目录下符合正则表达式的文件列表
    f_list = glob.glob(os.path.join(dir_path, regex))
    # 如果文件列表不为空，则使用已存在的模型，跳过下载
    if f_list:
        print("Use existed model, skip downloading.")
        return
    # 如果镜像是 "openi"，则使用 openi 模块下载模型
    if mirror.lower() == "openi":
        import openi

        kwargs = {"token": token} if token else {}
        openi.login(**kwargs)

        model_image = repo_config["model_image"]
        openi.model.download_model(repo_id, model_image, dir_path)

        fs = glob.glob(os.path.join(dir_path, model_image, "*.pth"))
        for file in fs:
            shutil.move(file, dir_path)
        shutil.rmtree(os.path.join(dir_path, model_image))
    # 否则使用 hf_hub_download 下载模型
    else:
        for file in ["DUR_0.pth", "D_0.pth", "G_0.pth"]:
            hf_hub_download(
                repo_id, file, local_dir=dir_path, local_dir_use_symlinks=False
            )

# 加载检查点
def load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False):
    # 确保检查点文件存在
    assert os.path.isfile(checkpoint_path)
    # 使用 torch.load 加载检查点文件
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    # 如果优化器不为空且不跳过优化器，并且检查点字典中存在优化器状态，则加载优化器状态
    if (
        optimizer is not None
        and not skip_optimizer
        and checkpoint_dict["optimizer"] is not None
    ):
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    # 如果优化器为空且不跳过优化器
    elif optimizer is None and not skip_optimizer:
        # 创建新的优化器状态字典
        new_opt_dict = optimizer.state_dict()
        # 获取新的优化器参数
        new_opt_dict_params = new_opt_dict["param_groups"][0]["params"]
        # 更新新的优化器参数组
        new_opt_dict["param_groups"] = checkpoint_dict["optimizer"]["param_groups"]
        new_opt_dict["param_groups"][0]["params"] = new_opt_dict_params
        # 加载新的优化器状态字典
        optimizer.load_state_dict(new_opt_dict)

    # 保存模型状态字典
    saved_state_dict = checkpoint_dict["model"]
    # 如果模型有 "module" 属性
    if hasattr(model, "module"):
        # 获取模型的状态字典
        state_dict = model.module.state_dict()
    else:
        # 获取模型的状态字典
        state_dict = model.state_dict()

    # 创建新的模型状态字典
    new_state_dict = {}
    # 遍历模型状态字典的键值对
    for k, v in state_dict.items():
        try:
            # 如果检查点中的键存在于模型状态字典中
            new_state_dict[k] = saved_state_dict[k]
            # 断言检查点中的值的形状与模型状态字典中的值的形状相同
            assert saved_state_dict[k].shape == v.shape, (
                saved_state_dict[k].shape,
                v.shape,
            )
        except:
            # 对于从旧版本升级的情况
            if "ja_bert_proj" in k:
                # 将值设置为与模型状态字典中相同形状的零张量
                v = torch.zeros_like(v)
                logger.warn(
                    f"Seems you are using the old version of the model, the {k} is automatically set to zero for backward compatibility"
                )
            else:
                # 记录错误日志
                logger.error(f"{k} is not in the checkpoint")

            # 更新新的模型状态字典
            new_state_dict[k] = v

    # 如果模型有 "module" 属性
    if hasattr(model, "module"):
        # 加载新的模型状态字典，允许不严格匹配
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        # 加载新的模型状态字典，允许不严格匹配
        model.load_state_dict(new_state_dict, strict=False)

    # 记录日志，显示加载的检查点和迭代次数
    logger.info(
        "Loaded checkpoint '{}' (iteration {})".format(checkpoint_path, iteration)
    )

    # 返回模型、优化器、学习率和迭代次数
    return model, optimizer, learning_rate, iteration
# 保存模型和优化器状态的检查点
def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    # 记录保存模型和优化器状态的信息
    logger.info(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, checkpoint_path
        )
    )
    # 如果模型有 module 属性，则获取其状态字典，否则获取模型自身的状态字典
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    # 保存模型、迭代次数、优化器状态和学习率到指定路径
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


# 将标量、直方图、图像和音频数据汇总到 TensorBoard
def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    # 将标量数据添加到 TensorBoard
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    # 将直方图数据添加到 TensorBoard
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    # 将图像数据添加到 TensorBoard
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    # 将音频数据添加到 TensorBoard
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


# 获取指定目录下最新的检查点文件路径
def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    # 获取目录下符合指定正则表达式的文件列表，并按文件名中的数字排序
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    # 返回排序后的文件列表中的最后一个文件路径
    x = f_list[-1]
    return x


# 将频谱图绘制为 numpy 数组
def plot_spectrogram_to_numpy(spectrogram):
    # 设置全局变量 MATPLOTLIB_FLAG，确保只导入 matplotlib 一次
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        # 设置 matplotlib 使用后端为 "Agg"
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        # 设置 matplotlib 日志级别为 WARNING
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np
    # 创建绘图对象和子图
    fig, ax = plt.subplots(figsize=(10, 2))
    # 绘制频谱图
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    # 添加颜色条
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    # 绘制画布并将其转换为 RGB 数组
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    # 重新调整数据的形状，使其符合画布的宽高和通道数
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # 关闭图形
    plt.close()
    # 返回调整后的数据
    return data
# 将对齐矩阵转换为 numpy 数组
def plot_alignment_to_numpy(alignment, info=None):
    global MATPLOTLIB_FLAG
    # 如果 MATPLOTLIB_FLAG 为假，则导入 matplotlib 库并设置使用 Agg 后端
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        # 设置 matplotlib 日志级别为警告
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    # 创建一个新的图形和子图
    fig, ax = plt.subplots(figsize=(6, 4))
    # 在子图上显示对齐矩阵的图像
    im = ax.imshow(
        alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
    )
    # 在图上添加颜色条
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    # 如果有额外信息，则在 x 轴标签上添加额外信息
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    # 绘制图形
    fig.canvas.draw()
    # 从绘制的图形中获取 RGB 数据并转换为 numpy 数组
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # 关闭图形
    plt.close()
    return data


# 将音频文件加载为 torch 张量
def load_wav_to_torch(full_path):
    # 读取音频文件的采样率和数据
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


# 加载文件路径和文本信息
def load_filepaths_and_text(filename, split="|"):
    # 打开文件并按指定分隔符分割每行的内容
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


# 获取超参数
def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    # 添加命令行参数：配置文件路径和模型名称
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/base.json",
        help="JSON file for configuration",
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")

    # 解析命令行参数
    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)

    # 如果模型目录不存在，则创建
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.json")
    # 如果 init 为真，则执行以下代码块
    if init:
        # 以只读方式打开配置文件，并读取文件内容
        with open(config_path, "r", encoding="utf-8") as f:
            data = f.read()
        # 以写入方式打开配置保存路径，并将读取的文件内容写入
        with open(config_save_path, "w", encoding="utf-8") as f:
            f.write(data)
    # 如果 init 为假，则执行以下代码块
    else:
        # 以只读方式打开配置保存路径，并读取文件内容
        with open(config_save_path, "r", vencoding="utf-8") as f:
            data = f.read()
    # 将读取的文件内容解析为 JSON 格式
    config = json.loads(data)
    # 使用解析后的配置创建 HParams 对象
    hparams = HParams(**config)
    # 设置 HParams 对象的模型目录为 model_dir
    hparams.model_dir = model_dir
    # 返回 HParams 对象
    return hparams
# 清理检查点文件，释放空间
def clean_checkpoints(path_to_models="logs/44k/", n_ckpts_to_keep=2, sort_by_time=True):
    """Freeing up space by deleting saved ckpts

    Arguments:
    path_to_models    --  模型目录的路径
    n_ckpts_to_keep   --  要保留的检查点数量，不包括 G_0.pth 和 D_0.pth
    sort_by_time      --  True -> 按时间顺序删除检查点
                          False -> 按字典顺序删除检查点
    """
    import re

    # 获取模型目录下的所有文件
    ckpts_files = [
        f
        for f in os.listdir(path_to_models)
        if os.path.isfile(os.path.join(path_to_models, f))
    ]

    # 根据文件名中的数字排序
    def name_key(_f):
        return int(re.compile("._(\\d+)\\.pth").match(_f).group(1))

    # 根据文件修改时间排序
    def time_key(_f):
        return os.path.getmtime(os.path.join(path_to_models, _f))

    sort_key = time_key if sort_by_time else name_key

    # 按指定规则排序并获取要删除的文件列表
    def x_sorted(_x):
        return sorted(
            [f for f in ckpts_files if f.startswith(_x) and not f.endswith("_0.pth")],
            key=sort_key,
        )

    to_del = [
        os.path.join(path_to_models, fn)
        for fn in (
            x_sorted("G")[:-n_ckpts_to_keep]
            + x_sorted("D")[:-n_ckpts_to_keep]
            + x_sorted("WD")[:-n_ckpts_to_keep]
        )
    ]

    # 删除文件的信息
    def del_info(fn):
        return logger.info(f".. Free up space by deleting ckpt {fn}")

    # 删除文件的例行程序
    def del_routine(x):
        return [os.remove(x), del_info(x)]

    # 对要删除的文件执行删除例行程序
    [del_routine(fn) for fn in to_del]


# 从模型目录中获取超参数
def get_hparams_from_dir(model_dir):
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


# 从配置文件中获取超参数
def get_hparams_from_file(config_path):
    # print("config_path: ", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


# 检查 Git 哈希值
def check_git_hash(model_dir):
    # 获取当前文件所在目录的绝对路径
    source_dir = os.path.dirname(os.path.realpath(__file__))
    # 如果当前目录下不存在 .git 文件夹，则输出警告信息并返回
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warn(
            "{} is not a git repository, therefore hash value comparison will be ignored.".format(
                source_dir
            )
        )
        return
    
    # 获取当前代码的 git 版本号
    cur_hash = subprocess.getoutput("git rev-parse HEAD")
    
    # 设置保存 git 版本号的文件路径
    path = os.path.join(model_dir, "githash")
    # 如果保存 git 版本号的文件存在，则读取其中的版本号
    if os.path.exists(path):
        saved_hash = open(path).read()
        # 如果保存的版本号与当前版本号不同，则输出警告信息
        if saved_hash != cur_hash:
            logger.warn(
                "git hash values are different. {}(saved) != {}(current)".format(
                    saved_hash[:8], cur_hash[:8]
                )
            )
    # 如果保存 git 版本号的文件不存在，则创建该文件并写入当前版本号
    else:
        open(path, "w").write(cur_hash)
# 根据模型目录和文件名创建日志记录器，设置记录级别为 DEBUG
def get_logger(model_dir, filename="train.log"):
    global logger
    # 使用模型目录的基本名称创建日志记录器
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    # 设置日志格式
    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    # 如果模型目录不存在，则创建该目录
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # 创建文件处理器，将日志记录到指定文件中
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    # 将文件处理器添加到日志记录器中
    logger.addHandler(h)
    # 返回日志记录器
    return logger


# 定义 HParams 类，用于存储模型超参数
class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


# 加载模型
def load_model(model_path, config_path):
    # 从配置文件中获取超参数
    hps = get_hparams_from_file(config_path)
    # 创建合成器模型对象
    net = SynthesizerTrn(
        108,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to("cpu")
    _ = net.eval()
    # 加载模型的参数
    _ = load_checkpoint(model_path, net, None, skip_optimizer=True)
    # 返回加载的模型
    return net


# 混合模型
def mix_model(
    network1, network2, output_path, voice_ratio=(0.5, 0.5), tone_ratio=(0.5, 0.5)
):
    # 检查网络是否有 module 属性，获取对应的状态字典
    if hasattr(network1, "module"):
        state_dict1 = network1.module.state_dict()
        state_dict2 = network2.module.state_dict()
    else:
        state_dict1 = network1.state_dict()
        state_dict2 = network2.state_dict()
    # 遍历 state_dict1 的键
    for k in state_dict1.keys():
        # 如果 state_dict2 中不包含当前键，则跳过当前循环
        if k not in state_dict2.keys():
            continue
        # 如果当前键包含 "enc_p"，则按照 tone_ratio 来混合 state_dict1 和 state_dict2 中对应键的值
        if "enc_p" in k:
            state_dict1[k] = (
                state_dict1[k].clone() * tone_ratio[0]
                + state_dict2[k].clone() * tone_ratio[1]
            )
        # 如果当前键不包含 "enc_p"，则按照 voice_ratio 来混合 state_dict1 和 state_dict2 中对应键的值
        else:
            state_dict1[k] = (
                state_dict1[k].clone() * voice_ratio[0]
                + state_dict2[k].clone() * voice_ratio[1]
            )
    # 遍历 state_dict2 的键
    for k in state_dict2.keys():
        # 如果 state_dict1 中不包含当前键，则将 state_dict2 中对应键的值复制给 state_dict1
        if k not in state_dict1.keys():
            state_dict1[k] = state_dict2[k].clone()
    # 保存 state_dict1 到指定路径，包括模型、迭代次数、优化器、学习率等信息
    torch.save(
        {"model": state_dict1, "iteration": 0, "optimizer": None, "learning_rate": 0},
        output_path,
    )
# 从模型路径中获取步骤数
def get_steps(model_path):
    # 使用正则表达式查找模型路径中的数字
    matches = re.findall(r"\d+", model_path)
    # 如果找到数字，则返回最后一个数字，否则返回 None
    return matches[-1] if matches else None
```