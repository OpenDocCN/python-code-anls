# `qlora\eval\qa_baseline_gpt.py`

```py
# 从 https://github.com/lm-sys/FastChat/blob/b3c8bd71637d6c88206a360be436e7941b4fffb4/fastchat/eval/qa_baseline_gpt35.py 改编而来，用于使用 GPT-3.5 生成答案
"""Generate answers with GPT-3.5"""
# 注意：你需要使用 OpenAI Python v0.27.0 才能让下面的代码正常工作
import argparse  # 导入解析命令行参数的模块
import json  # 导入处理 JSON 数据的模块
import os  # 导入操作系统相关功能的模块
import time  # 导入时间相关功能的模块
import concurrent.futures  # 导入并发执行任务的模块

import openai  # 导入 OpenAI 的 Python SDK
import tqdm  # 导入进度条显示的模块
import shortuuid  # 导入生成短 UUID 的模块
openai.api_key = os.getenv("OPENAI_API_KEY")  # 设置 OpenAI 的 API 密钥
MODEL="gpt-4"  # 设置模型名称
MODEL_ID="gpt-4:20230520"  # 设置模型 ID

def get_answer(question_id: int, question: str, max_tokens: int):
    ans = {  # 创建答案字典
        "answer_id": shortuuid.uuid(),  # 生成短 UUID 作为答案 ID
        "question_id": question_id,  # 设置问题 ID
        "model_id": MODEL_ID,  # 设置模型 ID
    }
    for _ in range(3):  # 循环3次，最多尝试3次
        try:
            response = openai.ChatCompletion.create(  # 调用 OpenAI 的对话完成接口
                model=MODEL,  # 设置模型名称
                messages=[  # 设置对话消息
                    {"role": "system", "content": "You are a helpful assistant."},  # 系统角色消息
                    {
                        "role": "user",
                        "content": question,  # 用户角色消息，内容为问题
                    },
                ],
                max_tokens=max_tokens,  # 设置最大生成标记数
            )
            ans["text"] = response["choices"][0]["message"]["content"]  # 获取生成的文本作为答案
            return ans  # 返回答案字典
        except Exception as e:  # 捕获异常
            print("[ERROR]", e)  # 打印错误信息
            ans["text"] = "#ERROR#"  # 设置答案文本为错误标识
            time.sleep(1)  # 等待1秒
    return ans  # 返回答案字典

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT answer generation.")  # 创建命令行参数解析器
    parser.add_argument("-q", "--question")  # 添加问题文件路径参数
    parser.add_argument("-o", "--output")  # 添加输出文件路径参数
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )  # 添加最大标记数参数
    args = parser.parse_args()  # 解析命令行参数

    questions_dict = {}  # 创建问题字典
    with open(os.path.expanduser(args.question)) as f:  # 打开问题文件
        for line in f:  # 遍历文件的每一行
            if not line:  # 如果行为空
                continue  # 继续下一次循环
            q = json.loads(line)  # 解析 JSON 数据
            questions_dict[q["question_id"]] = q["text"]  # 将问题 ID 和文本存入问题字典

    answers = []  # 创建答案列表
    # 使用最多32个线程的线程池执行并发任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        # 创建一个空的future列表
        futures = []
        # 遍历问题字典中的每个问题，提交任务到线程池，并将future对象添加到列表中
        for qid, question in questions_dict.items():
            future = executor.submit(get_answer, qid, question, args.max_tokens)
            futures.append(future)
    
        # 遍历已完成的future对象，并将结果添加到answers列表中
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            answers.append(future.result())
    
    # 根据问题ID对答案列表进行排序
    answers.sort(key=lambda x: x["question_id"])
    
    # 打开输出文件，将答案列表中的每个答案转换为JSON格式的字符串，并写入文件中
    with open(os.path.expanduser(args.output), "w") as f:
        table = [json.dumps(ans) for ans in answers]
        f.write("\n".join(table))
```