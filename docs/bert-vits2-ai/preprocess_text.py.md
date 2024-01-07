# `Bert-VITS2\preprocess_text.py`

```

# 导入所需的模块
import json  # 导入json模块，用于处理JSON格式的数据
from collections import defaultdict  # 导入defaultdict类，用于创建默认字典
from random import shuffle  # 导入shuffle函数，用于随机打乱序列
from typing import Optional  # 导入Optional类型，用于指定可选参数的类型
import os  # 导入os模块，用于与操作系统交互

from tqdm import tqdm  # 导入tqdm模块，用于显示进度条
import click  # 导入click模块，用于创建命令行界面
from text.cleaner import clean_text  # 从text模块中导入clean_text函数
from config import config  # 从config模块中导入config对象
from infer import latest_version  # 从infer模块中导入latest_version函数

# 从config对象中获取预处理文本的配置
preprocess_text_config = config.preprocess_text_config

# 定义命令行参数和选项
@click.command()
@click.option(
    "--transcription-path",
    default=preprocess_text_config.transcription_path,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--cleaned-path", default=preprocess_text_config.cleaned_path)
@click.option("--train-path", default=preprocess_text_config.train_path)
@click.option("--val-path", default=preprocess_text_config.val_path)
@click.option(
    "--config-path",
    default=preprocess_text_config.config_path,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--val-per-lang", default=preprocess_text_config.val_per_lang)
@click.option("--max-val-total", default=preprocess_text_config.max_val_total)
@click.option("--clean/--no-clean", default=preprocess_text_config.clean)
@click.option("-y", "--yml_config")
def preprocess(
    transcription_path: str,
    cleaned_path: Optional[str],
    train_path: str,
    val_path: str,
    config_path: str,
    val_per_lang: int,
    max_val_total: int,
    clean: bool,
    yml_config: str,  # 这个不要删
):
    # 如果cleaned_path为空或者为None，则将其设置为transcription_path加上".cleaned"
    if cleaned_path == "" or cleaned_path is None:
        cleaned_path = transcription_path + ".cleaned"

    # 如果clean为True，则进行文本清洗
    if clean:
        # 打开cleaned_path文件，准备写入清洗后的文本
        with open(cleaned_path, "w", encoding="utf-8") as out_file:
            # 打开transcription_path文件，准备读取原始文本
            with open(transcription_path, "r", encoding="utf-8") as trans_file:
                lines = trans_file.readlines()  # 读取所有行
                if len(lines) != 0:
                    for line in tqdm(lines):  # 遍历每一行文本
                        try:
                            utt, spk, language, text = line.strip().split("|")  # 按照"|"分割文本
                            norm_text, phones, tones, word2ph = clean_text(
                                text, language
                            )  # 调用clean_text函数进行文本清洗
                            out_file.write(  # 将清洗后的文本写入文件
                                "{}|{}|{}|{}|{}|{}|{}\n".format(
                                    utt,
                                    spk,
                                    language,
                                    norm_text,
                                    " ".join(phones),
                                    " ".join([str(i) for i in tones]),
                                    " ".join([str(i) for i in word2ph]),
                                )
                            )
                        except Exception as e:
                            print(line)  # 打印出错的行
                            print(f"生成训练集和验证集时发生错误！, 详细信息:\n{e}")  # 打印错误信息

    transcription_path = cleaned_path  # 将transcription_path设置为cleaned_path
    spk_utt_map = defaultdict(list)  # 创建默认字典，用于存储说话人和对应的句子列表
    spk_id_map = {}  # 创建空字典，用于存储说话人和对应的ID
    current_sid = 0  # 初始化当前说话人ID为0

    with open(transcription_path, "r", encoding="utf-8") as f:  # 打开cleaned_path文件，准备读取清洗后的文本
        audioPaths = set()  # 创建空集合，用于存储音频路径
        countSame = 0  # 初始化重复音频计数为0
        countNotFound = 0  # 初始化未找到音频计数为0
        for line in f.readlines():  # 遍历每一行文本
            utt, spk, language, text, phones, tones, word2ph = line.strip().split("|")  # 按照"|"分割文本
            if utt in audioPaths:  # 如果音频路径已经存在于集合中
                print(f"重复音频文本：{line}")  # 打印重复音频文本
                countSame += 1  # 重复音频计数加1
                continue
            if not os.path.isfile(utt):  # 如果音频文件不存在
                print(f"没有找到对应的音频：{utt}")  # 打印未找到音频的信息
                countNotFound += 1  # 未找到音频计数加1
                continue
            audioPaths.add(utt)  # 将音频路径添加到集合中
            spk_utt_map[language].append(line)  # 将文本按照语言添加到spk_utt_map中
            if spk not in spk_id_map.keys():  # 如果说话人不在spk_id_map中
                spk_id_map[spk] = current_sid  # 将说话人添加到spk_id_map中，并分配ID
                current_sid += 1  # 当前说话人ID加1
        print(f"总重复音频数：{countSame}，总未找到的音频数:{countNotFound}")  # 打印重复音频和未找到音频的总数

    train_list = []  # 创建空列表，用于存储训练集
    val_list = []  # 创建空列表，用于存储验证集

    for spk, utts in spk_utt_map.items():  # 遍历每个说话人和对应的句子列表
        shuffle(utts)  # 随机打乱句子列表
        val_list += utts[:val_per_lang]  # 将部分句子添加到验证集
        train_list += utts[val_per_lang:]  # 将剩余句子添加到训练集

    shuffle(val_list)  # 随机打乱验证集
    if len(val_list) > max_val_total:  # 如果验证集长度超过最大值
        train_list += val_list[max_val_total:]  # 将多余的部分添加到训练集
        val_list = val_list[:max_val_total]  # 截取最大长度的部分作为验证集

    with open(train_path, "w", encoding="utf-8") as f:  # 打开训练集文件，准备写入训练集
        for line in train_list:  # 遍历训练集
            f.write(line)  # 写入训练集

    with open(val_path, "w", encoding="utf-8") as f:  # 打开验证集文件，准备写入验证集
        for line in val_list:  # 遍历验证集
            f.write(line)  # 写入验证集

    json_config = json.load(open(config_path, encoding="utf-8"))  # 从配置文件中加载JSON数据
    json_config["data"]["spk2id"] = spk_id_map  # 将说话人ID映射添加到JSON数据中
    json_config["data"]["n_speakers"] = len(spk_id_map)  # 将说话人数量添加到JSON数据中
    json_config["version"] = latest_version  # 将最新版本号添加到JSON数据中
    json_config["data"]["training_files"] = os.path.normpath(train_path).replace(
        "\\", "/"
    )  # 将训练集文件路径添加到JSON数据中
    json_config["data"]["validation_files"] = os.path.normpath(val_path).replace(
        "\\", "/"
    )  # 将验证集文件路径添加到JSON数据中
    with open(config_path, "w", encoding="utf-8") as f:  # 打开配置文件，准备写入JSON数据
        json.dump(json_config, f, indent=2, ensure_ascii=False)  # 将JSON数据写入配置文件
    print("训练集和验证集生成完成！")  # 打印生成完成的信息


if __name__ == "__main__":
    preprocess()  # 调用preprocess函数进行数据预处理

```