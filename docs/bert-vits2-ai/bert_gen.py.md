# `d:/src/tocomm/Bert-VITS2\bert_gen.py`

```
import torch  # 导入 PyTorch 库
from multiprocessing import Pool  # 导入多进程池
import commons  # 导入自定义的 commons 模块
import utils  # 导入自定义的 utils 模块
from tqdm import tqdm  # 导入 tqdm 库，用于显示进度条
from text import check_bert_models, cleaned_text_to_sequence, get_bert  # 从 text 模块中导入指定函数
import argparse  # 导入 argparse 库，用于解析命令行参数
import torch.multiprocessing as mp  # 导入 PyTorch 多进程库
from config import config  # 从 config 模块中导入 config 对象


def process_line(x):
    line, add_blank = x  # 从输入参数 x 中解包得到 line 和 add_blank
    device = config.bert_gen_config.device  # 从 config 对象中获取设备信息
    if config.bert_gen_config.use_multi_device:  # 检查是否使用多设备
        rank = mp.current_process()._identity  # 获取当前进程的标识
        rank = rank[0] if len(rank) > 0 else 0  # 如果标识列表不为空，则取第一个元素，否则为 0
        if torch.cuda.is_available():  # 检查是否有可用的 CUDA 设备
            gpu_id = rank % torch.cuda.device_count()  # 计算当前进程应该使用的 GPU ID
            device = torch.device(f"cuda:{gpu_id}")  # 根据 GPU ID 创建对应的 CUDA 设备对象
        else:
            # 如果没有GPU可用，则将设备设置为CPU
            device = torch.device("cpu")
    # 从输入行中提取出音频路径、语言字符串、文本、音素、音调和单词到音素的映射关系
    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")
    # 将音素字符串分割成列表
    phone = phones.split(" ")
    # 将音调字符串分割成整数列表
    tone = [int(i) for i in tone.split(" ")]
    # 将单词到音素的映射关系字符串分割成整数列表
    word2ph = [int(i) for i in word2ph.split(" ")]
    # 将单词到音素的映射关系列表中的每个元素加倍
    word2ph = [i for i in word2ph]
    # 将音素、音调和语言字符串转换为序列
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    # 如果需要添加空白符
    if add_blank:
        # 在音素、音调和语言序列中插入空白符
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        # 将单词到音素的映射关系列表中的每个元素乘以2
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        # 将单词到音素的映射关系列表的第一个元素加1
        word2ph[0] += 1

    # 根据音频路径生成BERT文件路径
    bert_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".bert.pt")

    # 尝试执行以下代码块
    try:
        bert = torch.load(bert_path)  # 从指定路径加载预训练的 BERT 模型
        assert bert.shape[-1] == len(phone)  # 检查加载的 BERT 模型的最后一个维度是否与电话号码的长度相等
    except Exception:  # 如果出现异常
        bert = get_bert(text, word2ph, language_str, device)  # 调用函数获取 BERT 模型
        assert bert.shape[-1] == len(phone)  # 检查获取的 BERT 模型的最后一个维度是否与电话号码的长度相等
        torch.save(bert, bert_path)  # 将获取的 BERT 模型保存到指定路径


preprocess_text_config = config.preprocess_text_config  # 从配置文件中获取预处理文本的配置信息

if __name__ == "__main__":  # 如果当前脚本作为主程序运行
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument(
        "-c", "--config", type=str, default=config.bert_gen_config.config_path
    )  # 添加一个名为 "-c" 或 "--config" 的命令行参数，用于指定配置文件路径，默认值为预先定义的配置文件路径
    parser.add_argument(
        "--num_processes", type=int, default=config.bert_gen_config.num_processes
    )  # 添加一个名为 "--num_processes" 的命令行参数，用于指定进程数量，默认值为预先定义的进程数量
    args, _ = parser.parse_known_args()  # 解析命令行参数
    config_path = args.config  # 从解析的参数中获取配置文件路径
    hps = utils.get_hparams_from_file(config_path)  # 从配置文件中获取超参数
    check_bert_models()  # 检查 BERT 模型是否存在或可用
    lines = []  # 初始化一个空列表用于存储读取的文本行
    with open(hps.data.training_files, encoding="utf-8") as f:  # 打开训练文件
        lines.extend(f.readlines())  # 读取文件的所有行并添加到列表中

    with open(hps.data.validation_files, encoding="utf-8") as f:  # 打开验证文件
        lines.extend(f.readlines())  # 读取文件的所有行并添加到列表中
    add_blank = [hps.data.add_blank] * len(lines)  # 创建一个与 lines 长度相同的包含 hps.data.add_blank 的列表

    if len(lines) != 0:  # 如果 lines 列表不为空
        num_processes = args.num_processes  # 获取进程数量
        with Pool(processes=num_processes) as pool:  # 创建进程池
            for _ in tqdm(  # 使用 tqdm 显示进度条
                pool.imap_unordered(process_line, zip(lines, add_blank)),  # 并行处理每行文本和对应的 add_blank
                total=len(lines),  # 设置进度条的总数
            ):
                # 这里是缩进的代码块，表示循环体
                pass  # 使用 pass 语句作为占位符
# 打印一条消息，包括变量 lines 的长度信息，表示 bert 生成完毕并显示生成的 bert.pt 文件的数量。
```