# `so-vits-svc\preprocess_flist_config.py`

```
# 导入必要的模块
import argparse
import json
import os
import re
import wave
from random import shuffle
# 导入日志记录模块
from loguru import logger
# 导入进度条模块
from tqdm import tqdm
# 导入自定义的日志工具模块
import diffusion.logger.utils as du

# 定义正则表达式模式，用于匹配文件路径
pattern = re.compile(r'^[\.a-zA-Z0-9_\/]+$')

# 定义函数，用于获取音频文件的时长
def get_wav_duration(file_path):
    try:
        # 打开音频文件
        with wave.open(file_path, 'rb') as wav_file:
            # 获取音频帧数
            n_frames = wav_file.getnframes()
            # 获取采样率
            framerate = wav_file.getframerate()
            # 计算时长（秒）
            return n_frames / float(framerate)
    except Exception as e:
        # 记录错误日志
        logger.error(f"Reading {file_path}")
        # 抛出异常
        raise e

# 主程序入口
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument("--train_list", type=str, default="./filelists/train.txt", help="path to train list")
    parser.add_argument("--val_list", type=str, default="./filelists/val.txt", help="path to val list")
    parser.add_argument("--source_dir", type=str, default="./dataset/44k", help="path to source dir")
    parser.add_argument("--speech_encoder", type=str, default="vec768l12", help="choice a speech encoder|'vec768l12','vec256l9','hubertsoft','whisper-ppg','cnhubertlarge','dphubert','whisper-ppg-large','wavlmbase+'")
    parser.add_argument("--vol_aug", action="store_true", help="Whether to use volume embedding and volume augmentation")
    parser.add_argument("--tiny", action="store_true", help="Whether to train sovits tiny")
    # 解析命令行参数
    args = parser.parse_args()
    
    # 根据是否使用 tiny 模型选择配置模板
    config_template =  json.load(open("configs_template/config_tiny_template.json")) if args.tiny else json.load(open("configs_template/config_template.json"))
    # 初始化训练集和验证集
    train = []
    val = []
    # 初始化索引
    idx = 0
    # 初始化说话人字典
    spk_dict = {}
    # 初始化说话人 ID
    spk_id = 0
    # 遍历源目录下的所有文件夹，将每个文件夹的名称作为键，spk_id作为值，存入spk_dict字典中
    for speaker in tqdm(os.listdir(args.source_dir)):
        spk_dict[speaker] = spk_id
        spk_id += 1
        wavs = []

        # 遍历每个文件夹下的文件，如果文件不是wav格式则跳过
        for file_name in os.listdir(os.path.join(args.source_dir, speaker)):
            if not file_name.endswith("wav"):
                continue
            if file_name.startswith("."):
                continue

            # 拼接文件路径
            file_path = "/".join([args.source_dir, speaker, file_name])

            # 如果文件名不符合ASCII编码，则记录警告
            if not pattern.match(file_name):
                logger.warning("Detected non-ASCII file name: " + file_path)

            # 如果音频文件时长小于0.3秒，则记录日志并跳过
            if get_wav_duration(file_path) < 0.3:
                logger.info("Skip too short audio: " + file_path)
                continue

            # 将符合条件的音频文件路径添加到wavs列表中
            wavs.append(file_path)

        # 打乱wavs列表中的顺序，将后面的文件路径添加到train列表中，前面的文件路径添加到val列表中
        shuffle(wavs)
        train += wavs[2:]
        val += wavs[:2]

    # 打乱train列表和val列表中的顺序
    shuffle(train)
    shuffle(val)

    # 写入训练集文件列表
    logger.info("Writing " + args.train_list)
    with open(args.train_list, "w") as f:
        for fname in tqdm(train):
            wavpath = fname
            f.write(wavpath + "\n")

    # 写入验证集文件列表
    logger.info("Writing " + args.val_list)
    with open(args.val_list, "w") as f:
        for fname in tqdm(val):
            wavpath = fname
            f.write(wavpath + "\n")

    # 加载diffusion_template.yaml配置文件模板
    d_config_template = du.load_config("configs_template/diffusion_template.yaml")
    # 设置模型中说话人数量为spk_id
    d_config_template["model"]["n_spk"] = spk_id
    # 设置数据编码器为args.speech_encoder
    d_config_template["data"]["encoder"] = args.speech_encoder
    # 设置spk字段为spk_dict
    d_config_template["spk"] = spk_dict
    
    # 设置config_template中的spk字段为spk_dict
    config_template["spk"] = spk_dict
    # 设置config_template中模型的说话人数量为spk_id
    config_template["model"]["n_speakers"] = spk_id
    # 设置config_template中模型的语音编码器为args.speech_encoder
    config_template["model"]["speech_encoder"] = args.speech_encoder
    
    # 如果语音编码器为"vec768l12"、"dphubert"或"wavlmbase+"，则设置模型中的ssl_dim、filter_channels和gin_channels为768
    if args.speech_encoder == "vec768l12" or args.speech_encoder == "dphubert" or args.speech_encoder == "wavlmbase+":
        config_template["model"]["ssl_dim"] = config_template["model"]["filter_channels"] = config_template["model"]["gin_channels"] = 768
        # 设置d_config_template中数据的编码器输出通道数为768
        d_config_template["data"]["encoder_out_channels"] = 768
    # 如果语音编码器为"vec256l9"或"hubertsoft"，则设置模型的ssl_dim和gin_channels为256
    elif args.speech_encoder == "vec256l9" or args.speech_encoder == 'hubertsoft':
        config_template["model"]["ssl_dim"] = config_template["model"]["gin_channels"] = 256
        # 设置数据的encoder_out_channels为256
        d_config_template["data"]["encoder_out_channels"] = 256
    # 如果语音编码器为"whisper-ppg"或"cnhubertlarge"，则设置模型的ssl_dim、filter_channels和gin_channels为1024
    elif args.speech_encoder == "whisper-ppg" or args.speech_encoder == 'cnhubertlarge':
        config_template["model"]["ssl_dim"] = config_template["model"]["filter_channels"] = config_template["model"]["gin_channels"] = 1024
        # 设置数据的encoder_out_channels为1024
        d_config_template["data"]["encoder_out_channels"] = 1024
    # 如果语音编码器为"whisper-ppg-large"，则设置模型的ssl_dim、filter_channels和gin_channels为1280
    elif args.speech_encoder == "whisper-ppg-large":
        config_template["model"]["ssl_dim"] = config_template["model"]["filter_channels"] = config_template["model"]["gin_channels"] = 1280
        # 设置数据的encoder_out_channels为1280
        
    # 如果启用了音量增强，则设置训练和模型的vol_aug为True
    if args.vol_aug:
        config_template["train"]["vol_aug"] = config_template["model"]["vol_embedding"] = True

    # 如果启用了tiny模式，则设置模型的filter_channels为512
    if args.tiny:
        config_template["model"]["filter_channels"] = 512

    # 记录日志，将config_template写入到configs/config.json文件中
    logger.info("Writing to configs/config.json")
    with open("configs/config.json", "w") as f:
        json.dump(config_template, f, indent=2)
    # 记录日志，将d_config_template保存到configs/diffusion.yaml文件中
    logger.info("Writing to configs/diffusion.yaml")
    du.save_config("configs/diffusion.yaml",d_config_template)
```