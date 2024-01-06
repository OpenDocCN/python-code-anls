# `qlora\eval\qa_baseline_gpt.py`

```
# 从 https://github.com/lm-sys/FastChat/blob/b3c8bd71637d6c88206a360be436e7941b4fffb4/fastchat/eval/qa_baseline_gpt35.py 改编而来
"""使用 GPT-3.5 生成答案"""
# 注意：你需要使用 OpenAI Python v0.27.0 才能运行下面的代码
# 导入必要的库
import argparse  # 用于解析命令行参数
import json  # 用于处理 JSON 数据
import os  # 用于操作系统相关功能
import time  # 用于时间相关功能
import concurrent.futures  # 用于并发执行任务

import openai  # OpenAI 的 Python SDK
import tqdm  # 用于在循环中显示进度条
import shortuuid  # 用于生成短的唯一标识符
openai.api_key = os.getenv("OPENAI_API_KEY")  # 设置 OpenAI API 密钥
MODEL="gpt-4"  # 模型名称
MODEL_ID="gpt-4:20230520"  # 模型 ID

# 定义一个函数，用于获取答案
def get_answer(question_id: int, question: str, max_tokens: int):
    ans = {
        "answer_id": shortuuid.uuid(),  # 生成一个唯一的答案 ID
# 创建包含问题ID和模型ID的字典
{
    "question_id": question_id,
    "model_id": MODEL_ID,
}
# 尝试最多3次发送请求
for _ in range(3):
    try:
        # 使用OpenAI的ChatCompletion模块创建对话完成
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": question,
                },
            ],
            max_tokens=max_tokens,
        )
        # 将回复文本存入结果字典
        ans["text"] = response["choices"][0]["message"]["content"]
        # 返回结果字典
        return ans
    # 捕获异常并打印错误信息
    except Exception as e:
        print("[ERROR]", e)
    # 设置答案文本为错误标识
    ans["text"] = "#ERROR#"
    # 等待1秒
    time.sleep(1)
    # 返回答案
    return ans

# 如果作为主程序运行
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="ChatGPT answer generation.")
    # 添加问题参数
    parser.add_argument("-q", "--question")
    # 添加输出参数
    parser.add_argument("-o", "--output")
    # 添加最大标记数参数
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    # 解析参数
    args = parser.parse_args()

    # 创建问题字典
    questions_dict = {}
    # 打开问题文件
    with open(os.path.expanduser(args.question)) as f:
        # 逐行读取文件内容
        for line in f:
# 如果读取的行为空，则跳过继续下一次循环
if not line:
    continue
# 将读取的 JSON 数据转换为 Python 字典
q = json.loads(line)
# 将问题 ID 和问题文本存储到问题字典中
questions_dict[q["question_id"]] = q["text"]

# 创建一个空列表用于存储答案
answers = []

# 使用线程池执行器创建一个最大工作线程数为32的线程池
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    # 创建一个空的 future 列表
    futures = []
    # 遍历问题字典，为每个问题提交一个任务到线程池中
    for qid, question in questions_dict.items():
        future = executor.submit(get_answer, qid, question, args.max_tokens)
        futures.append(future)

    # 使用 tqdm 显示进度条，等待所有任务完成并将结果添加到答案列表中
    for future in tqdm.tqdm(
        concurrent.futures.as_completed(futures), total=len(futures)
    ):
        answers.append(future.result())

# 根据问题 ID 对答案列表进行排序
answers.sort(key=lambda x: x["question_id"])
# 打开指定路径的文件，以写入模式
with open(os.path.expanduser(args.output), "w") as f:
    # 将答案列表中的每个答案转换为 JSON 格式的字符串，并存入列表中
    table = [json.dumps(ans) for ans in answers]
    # 将列表中的 JSON 字符串按行写入文件
    f.write("\n".join(table))
```