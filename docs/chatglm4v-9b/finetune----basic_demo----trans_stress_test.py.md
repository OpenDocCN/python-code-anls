# `.\chatglm4-finetune\basic_demo\trans_stress_test.py`

```
# 导入所需的库
import argparse  # 用于解析命令行参数
import time  # 用于时间操作
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig  # 导入模型和tokenizer
import torch  # 导入PyTorch库
from threading import Thread  # 导入线程支持库

MODEL_PATH = 'THUDM/glm-4-9b-chat'  # 定义模型路径


def stress_test(token_len, n, num_gpu):  # 定义压力测试函数，接收token长度、数量和GPU数量
    # 确定使用的设备，优先选择GPU
    device = torch.device(f"cuda:{num_gpu - 1}" if torch.cuda.is_available() and num_gpu > 0 else "cpu")
    # 加载tokenizer，并设置相关参数
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="left"
    )
    # 加载预训练的因果语言模型，设置为评估模式并转移到指定设备
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    
    # 使用INT4权重推理的代码块（注释掉的部分）
    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_PATH,
    #     trust_remote_code=True,
    #     quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    #     low_cpu_mem_usage=True,
    # ).eval()

    times = []  # 用于记录每次生成的时间
    decode_times = []  # 用于记录解码时间

    print("Warming up...")  # 输出热身提示
    vocab_size = tokenizer.vocab_size  # 获取词汇表大小
    warmup_token_len = 20  # 设置热身阶段token的长度
    # 随机生成token ID，范围在3到vocab_size - 200之间
    random_token_ids = torch.randint(3, vocab_size - 200, (warmup_token_len - 5,), dtype=torch.long)
    start_tokens = [151331, 151333, 151336, 198]  # 定义起始token ID
    end_tokens = [151337]  # 定义结束token ID
    # 创建输入ID张量，并添加起始和结束token
    input_ids = torch.tensor(start_tokens + random_token_ids.tolist() + end_tokens, dtype=torch.long).unsqueeze(0).to(
        device)
    # 创建注意力掩码
    attention_mask = torch.ones_like(input_ids, dtype=torch.bfloat16).to(device)
    # 创建位置ID张量
    position_ids = torch.arange(len(input_ids[0]), dtype=torch.bfloat16).unsqueeze(0).to(device)
    # 将输入准备为字典格式
    warmup_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids
    }
    # 禁用梯度计算以节省内存
    with torch.no_grad():
        # 生成模型输出，执行热身操作
        _ = model.generate(
            input_ids=warmup_inputs['input_ids'],
            attention_mask=warmup_inputs['attention_mask'],
            max_new_tokens=2048,  # 设置生成的最大token数
            do_sample=False,  # 不进行随机采样
            repetition_penalty=1.0,  # 设置重复惩罚
            eos_token_id=[151329, 151336, 151338]  # 定义结束token ID
        )
    print("Warming up complete. Starting stress test...")  # 输出热身完成提示
    # 循环 n 次以生成多组输入
    for i in range(n):
        # 生成随机的 token ID，范围在 3 到 vocab_size - 200 之间，长度为 token_len - 5
        random_token_ids = torch.randint(3, vocab_size - 200, (token_len - 5,), dtype=torch.long)
        # 将开始 token、随机 token 和结束 token 合并为输入 ID，并转换为张量，增加维度并转移到指定设备
        input_ids = torch.tensor(start_tokens + random_token_ids.tolist() + end_tokens, dtype=torch.long).unsqueeze(
            0).to(device)
        # 创建与 input_ids 相同形状的注意力掩码，初始值为 1
        attention_mask = torch.ones_like(input_ids, dtype=torch.bfloat16).to(device)
        # 生成位置 ID，表示每个 token 的位置，转换为张量并转移到指定设备
        position_ids = torch.arange(len(input_ids[0]), dtype=torch.bfloat16).unsqueeze(0).to(device)
        # 创建测试输入字典，包含 input_ids、attention_mask 和 position_ids
        test_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }

        # 初始化文本流迭代器，设置超时和是否跳过提示及特殊标记
        streamer = TextIteratorStreamer(
            tokenizer=tokenizer,
            timeout=36000,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # 设置生成文本的参数
        generate_kwargs = {
            "input_ids": test_inputs['input_ids'],
            "attention_mask": test_inputs['attention_mask'],
            "max_new_tokens": 512,
            "do_sample": False,
            "repetition_penalty": 1.0,
            "eos_token_id": [151329, 151336, 151338],
            "streamer": streamer
        }

        # 记录开始时间
        start_time = time.time()
        # 创建并启动线程，调用模型的生成方法
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        # 初始化第一个 token 时间和所有 token 时间的列表
        first_token_time = None
        all_token_times = []

        # 遍历生成的每个 token
        for token in streamer:
            current_time = time.time()  # 获取当前时间
            if first_token_time is None:  # 如果是第一个 token，记录时间
                first_token_time = current_time
                times.append(first_token_time - start_time)  # 计算并存储填充时间
            all_token_times.append(current_time)  # 记录所有 token 的时间

        t.join()  # 等待生成线程结束
        end_time = time.time()  # 记录结束时间

        # 计算每个 token 的平均解码时间
        avg_decode_time_per_token = len(all_token_times) / (end_time - first_token_time) if all_token_times else 0
        decode_times.append(avg_decode_time_per_token)  # 存储平均解码时间
        # 打印当前迭代的填充时间和平均解码时间
        print(
            f"Iteration {i + 1}/{n} - Prefilling Time: {times[-1]:.4f} seconds - Average Decode Time: {avg_decode_time_per_token:.4f} tokens/second")

        # 清空 CUDA 缓存
        torch.cuda.empty_cache()

    # 计算 n 次迭代的平均第一个 token 时间
    avg_first_token_time = sum(times) / n
    # 计算 n 次迭代的平均解码时间
    avg_decode_time = sum(decode_times) / n
    # 打印总体平均时间
    print(f"\nAverage First Token Time over {n} iterations: {avg_first_token_time:.4f} seconds")
    print(f"Average Decode Time per Token over {n} iterations: {avg_decode_time:.4f} tokens/second")
    # 返回填充时间、平均第一个 token 时间、解码时间和平均解码时间
    return times, avg_first_token_time, decode_times, avg_decode_time
# 主函数，用于执行模型推理的压力测试
def main():
    # 创建解析器，提供程序描述
    parser = argparse.ArgumentParser(description="Stress test for model inference")
    # 添加参数，指定每次测试的标记数量，默认为1000
    parser.add_argument('--token_len', type=int, default=1000, help='Number of tokens for each test')
    # 添加参数，指定压力测试的迭代次数，默认为3
    parser.add_argument('--n', type=int, default=3, help='Number of iterations for the stress test')
    # 添加参数，指定用于推理的GPU数量，默认为1
    parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPUs to use for inference')
    # 解析命令行参数并将其存储在args中
    args = parser.parse_args()

    # 从args中获取标记长度
    token_len = args.token_len
    # 从args中获取迭代次数
    n = args.n
    # 从args中获取GPU数量
    num_gpu = args.num_gpu

    # 调用压力测试函数，传入标记长度、迭代次数和GPU数量
    stress_test(token_len, n, num_gpu)


# 如果当前模块是主模块，则调用main函数
if __name__ == "__main__":
    main()
```