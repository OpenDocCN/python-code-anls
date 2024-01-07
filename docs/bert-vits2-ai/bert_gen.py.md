# `Bert-VITS2\bert_gen.py`

```

# 导入torch模块
import torch
# 从multiprocessing模块中导入Pool类
from multiprocessing import Pool
# 导入自定义的commons模块
import commons
# 导入自定义的utils模块
import utils
# 从tqdm模块中导入tqdm类
from tqdm import tqdm
# 从text模块中导入check_bert_models, cleaned_text_to_sequence, get_bert函数
from text import check_bert_models, cleaned_text_to_sequence, get_bert
# 导入argparse模块
import argparse
# 从torch.multiprocessing模块中导入mp对象
import torch.multiprocessing as mp
# 从config模块中导入config对象

# 定义处理每行数据的函数
def process_line(x):
    # 解包x元组
    line, add_blank = x
    # 获取设备信息
    device = config.bert_gen_config.device
    # 如果使用多设备
    if config.bert_gen_config.use_multi_device:
        # 获取当前进程的rank
        rank = mp.current_process()._identity
        rank = rank[0] if len(rank) > 0 else 0
        # 如果CUDA可用
        if torch.cuda.is_available():
            # 计算GPU ID
            gpu_id = rank % torch.cuda.device_count()
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")
    # 解析每行数据
    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")
    phone = phones.split(" ")
    tone = [int(i) for i in tone.split(" ")]
    word2ph = [int(i) for i in word2ph.split(" ")]
    word2ph = [i for i in word2ph]
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    # 如果需要添加空白
    if add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    # 生成bert文件路径
    bert_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".bert.pt")

    try:
        # 尝试加载bert文件
        bert = torch.load(bert_path)
        assert bert.shape[0] == 2048
    except Exception:
        # 如果加载失败，则重新生成bert文件
        bert = get_bert(text, word2ph, language_str, device)
        assert bert.shape[-1] == len(phone)
        torch.save(bert, bert_path)

# 从config中获取预处理文本的配置
preprocess_text_config = config.preprocess_text_config

# 如果当前脚本为主程序
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数-c，指定配置文件路径，默认为config.bert_gen_config.config_path
    parser.add_argument(
        "-c", "--config", type=str, default=config.bert_gen_config.config_path
    )
    # 添加命令行参数--num_processes，指定进程数，默认为config.bert_gen_config.num_processes
    parser.add_argument(
        "--num_processes", type=int, default=config.bert_gen_config.num_processes
    )
    # 解析命令行参数
    args, _ = parser.parse_known_args()
    config_path = args.config
    # 从配置文件中获取超参数
    hps = utils.get_hparams_from_file(config_path)
    # 检查bert模型
    check_bert_models()
    lines = []
    # 读取训练文件中的行
    with open(hps.data.training_files, encoding="utf-8") as f:
        lines.extend(f.readlines())

    # 读取验证文件中的行
    with open(hps.data.validation_files, encoding="utf-8") as f:
        lines.extend(f.readlines())
    # 创建与lines长度相同的add_blank列表
    add_blank = [hps.data.add_blank] * len(lines)

    # 如果lines不为空
    if len(lines) != 0:
        # 获取进程数
        num_processes = args.num_processes
        # 使用进程池并行处理数据
        with Pool(processes=num_processes) as pool:
            for _ in tqdm(
                pool.imap_unordered(process_line, zip(lines, add_blank)),
                total=len(lines),
            ):
                # 这里是缩进的代码块，表示循环体
                pass  # 使用pass语句作为占位符

    # 打印bert生成完毕的信息
    print(f"bert生成完毕!, 共有{len(lines)}个bert.pt生成!")

```