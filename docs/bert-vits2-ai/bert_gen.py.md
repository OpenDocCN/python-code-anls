# `Bert-VITS2\bert_gen.py`

```
# 导入 torch 库
import torch
# 从 multiprocessing 库中导入 Pool 类
from multiprocessing import Pool
# 导入自定义的 commons 模块
import commons
# 导入自定义的 utils 模块
import utils
# 从 tqdm 库中导入 tqdm 函数
from tqdm import tqdm
# 从 text 模块中导入 check_bert_models, cleaned_text_to_sequence, get_bert 函数
from text import check_bert_models, cleaned_text_to_sequence, get_bert
# 从 argparse 库中导入 ArgumentParser 类
import argparse
# 从 torch.multiprocessing 库中导入 mp 模块
import torch.multiprocessing as mp
# 从 config 模块中导入 config 对象
from config import config

# 定义处理每行数据的函数
def process_line(x):
    # 解包参数
    line, add_blank = x
    # 获取设备信息
    device = config.bert_gen_config.device
    # 如果使用多设备
    if config.bert_gen_config.use_multi_device:
        # 获取当前进程的标识
        rank = mp.current_process()._identity
        # 如果标识长度大于 0，则取第一个元素，否则为 0
        rank = rank[0] if len(rank) > 0 else 0
        # 如果 CUDA 可用
        if torch.cuda.is_available():
            # 计算 GPU ID
            gpu_id = rank % torch.cuda.device_count()
            # 设置设备为 CUDA 设备
            device = torch.device(f"cuda:{gpu_id}")
        else:
            # 设置设备为 CPU
            device = torch.device("cpu")
    # 解析行数据
    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")
    phone = phones.split(" ")
    tone = [int(i) for i in tone.split(" ")]
    word2ph = [int(i) for i in word2ph.split(" ")]
    word2ph = [i for i in word2ph]
    # 调用 cleaned_text_to_sequence 函数处理文本数据
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    # 如果需要添加空白
    if add_blank:
        # 在 phone、tone、language 列表中插入 0
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        # 修改 word2ph 列表中的值
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    # 根据 wav_path 生成对应的 bert_path
    bert_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".bert.pt")

    try:
        # 尝试加载 bert_path 对应的数据
        bert = torch.load(bert_path)
        # 断言 bert 的形状
        assert bert.shape[0] == 2048
    except Exception:
        # 如果加载失败，则调用 get_bert 函数生成数据
        bert = get_bert(text, word2ph, language_str, device)
        # 断言 bert 的形状
        assert bert.shape[-1] == len(phone)
        # 保存生成的 bert 数据
        torch.save(bert, bert_path)

# 获取预处理文本的配置信息
preprocess_text_config = config.preprocess_text_config

# 如果当前模块为主模块
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加配置参数
    parser.add_argument(
        "-c", "--config", type=str, default=config.bert_gen_config.config_path
    )
    # 添加进程数量参数
    parser.add_argument(
        "--num_processes", type=int, default=config.bert_gen_config.num_processes
    )
    # 解析命令行参数
    args, _ = parser.parse_known_args()
    # 从命令行参数中获取配置文件路径
    config_path = args.config
    # 从配置文件中获取超参数
    hps = utils.get_hparams_from_file(config_path)
    # 检查BERT模型是否存在
    check_bert_models()
    # 初始化一个空列表用于存储数据
    lines = []
    # 以UTF-8编码打开训练文件，并将内容添加到lines列表中
    with open(hps.data.training_files, encoding="utf-8") as f:
        lines.extend(f.readlines())
    # 以UTF-8编码打开验证文件，并将内容添加到lines列表中
    with open(hps.data.validation_files, encoding="utf-8") as f:
        lines.extend(f.readlines())
    # 创建一个与lines列表长度相同的add_blank列表，其元素为hps.data.add_blank
    add_blank = [hps.data.add_blank] * len(lines)
    
    # 如果lines列表不为空
    if len(lines) != 0:
        # 从命令行参数中获取进程数
        num_processes = args.num_processes
        # 使用进程池并行处理数据
        with Pool(processes=num_processes) as pool:
            # 使用进度条展示处理进度，并调用process_line函数处理lines和add_blank的元素
            for _ in tqdm(
                pool.imap_unordered(process_line, zip(lines, add_blank)),
                total=len(lines),
            ):
                # 这里是缩进的代码块，表示循环体
                pass  # 使用pass语句作为占位符
    
    # 打印BERT生成完毕的信息，以及生成的bert.pt文件数量
    print(f"bert生成完毕!, 共有{len(lines)}个bert.pt生成!")
```