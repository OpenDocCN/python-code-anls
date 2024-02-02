# `Bert-VITS2\preprocess_text.py`

```py
# 导入所需的模块
import json
from collections import defaultdict
from random import shuffle
from typing import Optional
import os

# 导入 tqdm 和 click 模块中的所有内容
from tqdm import tqdm
import click
# 导入 clean_text 函数和 config 模块中的 latest_version 函数
from text.cleaner import clean_text
from config import config
from infer import latest_version

# 从 config 模块中获取预处理文本的配置
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
# 定义预处理函数，接受命令行参数和选项作为输入
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
    # 如果 cleaned_path 为空或者为 None，则将其设置为 transcription_path + ".cleaned"
    if cleaned_path == "" or cleaned_path is None:
        cleaned_path = transcription_path + ".cleaned"
    # 如果需要清洗数据
    if clean:
        # 打开清洗后的文件，准备写入
        with open(cleaned_path, "w", encoding="utf-8") as out_file:
            # 打开转录文件，准备读取
            with open(transcription_path, "r", encoding="utf-8") as trans_file:
                # 读取所有行
                lines = trans_file.readlines()
                # 如果文件不为空
                if len(lines) != 0:
                    # 遍历每一行
                    for line in tqdm(lines):
                        try:
                            # 尝试按照特定格式拆分每一行的内容
                            utt, spk, language, text = line.strip().split("|")
                            # 清洗文本并获取相关信息
                            norm_text, phones, tones, word2ph = clean_text(
                                text, language
                            )
                            # 将清洗后的内容写入清洗后的文件
                            out_file.write(
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
                        # 如果出现异常
                        except Exception as e:
                            # 打印出错的行和详细信息
                            print(line)
                            print(f"生成训练集和验证集时发生错误！, 详细信息:\n{e}")

    # 将清洗后的文件路径赋值给转录文件路径
    transcription_path = cleaned_path
    # 创建说话者到句子的映射字典
    spk_utt_map = defaultdict(list)
    # 创建说话者到ID的映射字典
    spk_id_map = {}
    # 初始化当前说话者ID
    current_sid = 0
    # 打开文本文件，使用 utf-8 编码方式读取
    with open(transcription_path, "r", encoding="utf-8") as f:
        # 初始化音频路径集合、相同音频计数、未找到音频计数
        audioPaths = set()
        countSame = 0
        countNotFound = 0
        # 遍历文本文件的每一行
        for line in f.readlines():
            # 拆分每行数据
            utt, spk, language, text, phones, tones, word2ph = line.strip().split("|")
            # 如果音频路径已经存在于集合中，则打印重复音频信息并跳过
            if utt in audioPaths:
                print(f"重复音频文本：{line}")
                countSame += 1
                continue
            # 如果音频文件不存在，则打印未找到音频信息并跳过
            if not os.path.isfile(utt):
                print(f"没有找到对应的音频：{utt}")
                countNotFound += 1
                continue
            # 将音频路径添加到集合中
            audioPaths.add(utt)
            # 将语言对应的行数据添加到 spk_utt_map 中
            spk_utt_map[language].append(line)
            # 如果说话者不在 spk_id_map 中，则将其添加，并更新当前说话者 id
            if spk not in spk_id_map.keys():
                spk_id_map[spk] = current_sid
                current_sid += 1
        # 打印总重复音频数和总未找到的音频数
        print(f"总重复音频数：{countSame}，总未找到的音频数:{countNotFound}")

    # 初始化训练集和验证集列表
    train_list = []
    val_list = []

    # 遍历每个说话者的音频数据
    for spk, utts in spk_utt_map.items():
        # 打乱每个说话者的音频数据顺序
        shuffle(utts)
        # 将部分数据添加到验证集，将剩余数据添加到训练集
        val_list += utts[:val_per_lang]
        train_list += utts[val_per_lang:]

    # 打乱验证集的顺序，并根据最大验证集数量进行截取
    shuffle(val_list)
    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]

    # 将训练集写入到文件中
    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    # 将验证集写入到文件中
    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)

    # 读取配置文件，并更新其中的数据
    json_config = json.load(open(config_path, encoding="utf-8"))
    json_config["data"]["spk2id"] = spk_id_map
    json_config["data"]["n_speakers"] = len(spk_id_map)
    # 更新配置文件中的版本和数据集路径信息
    json_config["version"] = latest_version
    json_config["data"]["training_files"] = os.path.normpath(train_path).replace(
        "\\", "/"
    )
    json_config["data"]["validation_files"] = os.path.normpath(val_path).replace(
        "\\", "/"
    )
    # 使用指定的编码打开配置文件，准备写入内容
    with open(config_path, "w", encoding="utf-8") as f:
        # 将 JSON 配置数据以可读性更好的格式写入到文件中，禁用 ASCII 编码
        json.dump(json_config, f, indent=2, ensure_ascii=False)
    # 打印训练集和验证集生成完成的提示信息
    print("训练集和验证集生成完成！")
# 如果当前脚本被直接执行，则调用 preprocess() 函数进行预处理
if __name__ == "__main__":
    preprocess()
```