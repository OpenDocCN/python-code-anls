# `qlora\eval\eval_gpt_review.py`

```
# 从指定链接中导入所需的库
import argparse  # 导入命令行参数解析模块
import json  # 导入处理 JSON 数据的模块
import os  # 导入操作系统相关功能的模块
import time  # 导入时间相关功能的模块

import openai  # 导入 OpenAI 的 API 客户端
from tqdm import tqdm  # 导入进度条显示模块
import ray  # 导入分布式计算框架

import shortuuid  # 导入生成短 UUID 的模块
import logging  # 导入日志记录模块
import numpy as np  # 导入处理数组和矩阵的模块
import os  # 导入操作系统相关功能的模块
import openai  # 导入 OpenAI 的 API 客户端
openai.api_key = os.getenv("OPENAI_API_KEY")  # 设置 OpenAI 的 API 密钥

logging.basicConfig(level=logging.INFO)  # 配置日志记录器，设置记录级别为 INFO
logger = logging.getLogger(__name__)  # 获取日志记录器对象

MAX_API_RETRY = 1000  # 设置最大 API 重试次数
REQ_TIME_GAP = 2  # 设置 API 请求时间间隔

# 定义一个远程函数，使用 ray.remote 装饰器标记，指定使用的 CPU 核心数为 4
@ray.remote(num_cpus=4)
def get_eval(sys_prompt, user_prompt: str, max_tokens: int, model: str):
    logging.basicConfig(level=logging.INFO)  # 配置日志记录器，设置记录级别为 INFO
    for i in range(MAX_API_RETRY):  # 循环最大 API 重试次数
        try:
            # 调用 OpenAI 的 ChatCompletion API，获取对话的完成结果
            response = openai.ChatCompletion.create(
                model=model,  # 指定使用的模型
                messages=[
                    {"role": "system", "content": sys_prompt},  # 系统提示消息
                    {
                        "role": "user",
                        "content": user_prompt,  # 用户提示消息
                    },
                ],
                temperature=0.2,  # 温度参数，用于控制生成文本的多样性
                max_tokens=max_tokens,  # 最大生成标记数
            )
            content = response["choices"][0]["message"]["content"]  # 获取生成的文本内容
            logger.info(content)  # 记录生成的文本内容
            return content  # 返回生成的文本内容
        except Exception as e:  # 捕获异常
            logger.error(e)  # 记录异常信息
            time.sleep(min(5*(i+1), 100))  # 休眠一段时间后重试
    logger.error(f"Failed after {MAX_API_RETRY} retries.")  # 记录重试次数用尽后仍失败的信息
    return "error"  # 返回错误提示

# 定义一个函数，用于解析三类评分的评价
def parse_three_class_score(review):
    try:
        score = int(review.strip().split("\n")[-1].strip())  # 解析评价中的分数
        return score  # 返回解析得到的分数
    except Exception as e:  # 捕获异常
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )  # 记录异常信息和评价内容
        return -1  # 返回错误提示

# 定义一个函数，用于解析评价的评分
def parse_score(review):
    # 尝试从评论中提取评分对
    try:
        # 从评论中提取第一行评分对
        score_pair = review.split("\n")[0]
        # 将评分对中的逗号替换为空格
        score_pair = score_pair.replace(",", " ")
        # 将评分对按空格分割
        sp = score_pair.split(" ")
        # 如果分割后的列表长度为2，说明是有效的评分对
        if len(sp) == 2:
            # 返回评分对的浮点数形式
            return [float(sp[0]), float(sp[1])]
        else:
            # 如果不是有效的评分对，抛出异常
            raise Exception("Invalid score pair.")
    # 捕获异常
    except Exception as e:
        # 记录错误日志，包括异常信息和评论内容
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        # 返回默认的无效评分对
        return [-1, -1]
# 生成提示信息，根据评审者的 JSON 数据和提示的 JSON 数据，以及类别、问题和答案生成提示
def gen_prompt(reviewer_jsons, prompt_jsons, cat, ques, ans1, ans2):
    # 默认选择通用类别（索引=0）
    reviewer_idx = 0
    # 遍历评审者的 JSON 数据，找到对应类别的评审者索引
    for idx, reviewer in enumerate(reviewer_jsons):
        if reviewer["category"] == cat:
            reviewer_idx = idx
            break
    # 获取评审者对应的提示 ID
    prompt_id = reviewer_jsons[reviewer_idx]["prompt_id"]
    # 根据提示 ID 获取对应的提示 JSON 数据
    prompt_json = prompt_jsons[prompt_id - 1]
    # 断言提示 ID 与获取的提示 JSON 数据中的提示 ID 相等
    assert prompt_json["prompt_id"] == prompt_id

    # 获取系统提示、提示模板、默认值，并根据模板格式化生成最终提示
    sys_prompt = prompt_json["system_prompt"]
    prompt_template = prompt_json["prompt_template"]
    defaults = prompt_json["defaults"]
    prompt = prompt_template.format(
        question=ques, answer_1=ans1, answer_2=ans2, **defaults
    )

    # 返回系统提示、最终提示和评审者索引加1
    return sys_prompt, prompt, reviewer_idx + 1


# 从文件中获取 JSON 数据列表
def get_json_list(file_path):
    # 将文件路径转换为绝对路径
    file_path = os.path.expanduser(file_path)
    # 打开文件，逐行读取并解析为 JSON 数据，存入列表中
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list

# 主程序入口
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    # 添加命令行参数
    parser.add_argument("-q", "--question-file")
    parser.add_argument("-a", "--answer-file-list", nargs="+", default=[])
    parser.add_argument("-p", "--prompt-file")
    parser.add_argument("-r", "--reviewer-file")
    parser.add_argument("-o", "--output-review-file")
    parser.add_argument("-m", "--model", default='gpt-4')
    parser.add_argument("-id", "--id-key", default='question_id')
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 如果输出评审文件的目录不存在，则将输出路径赋值给 dest
    if not os.path.isdir(args.output_review_file):
        dest = args.output_review_file
    # 如果条件不成立，设置后缀为 "_threeclass"，否则为空字符串
    threeclass_suff = "_threeclass" if 'threeclass' in args.prompt_file else ""
    # 拼接目标文件路径
    dest = os.path.join(
        args.output_review_file,
        '_vs_'.join([elt.split('/')[-1].replace('.jsonl', '') for elt in args.answer_file_list]) + f'_{args.model}_reviewer{threeclass_suff}' + '.jsonl'
    )

    # 初始化 Ray
    ray.init()

    # 获取问题文件的 JSON 列表
    question_jsons = get_json_list(args.question_file)
    # 获取第一个答案文件的 JSON 列表
    answer1_jsons = get_json_list(args.answer_file_list[0])
    # 获取第二个答案文件的 JSON 列表
    answer2_jsons = get_json_list(args.answer_file_list[1])
    # 获取评审者文件的 JSON 列表
    reviewer_jsons = get_json_list(args.reviewer_file)
    # 获取提示文件的 JSON 列表
    prompt_jsons = get_json_list(args.prompt_file)

    # 创建问题 ID 的集合
    question_ids = set(question[args.id_key] for question in question_jsons)
    # 对问题 JSON 列表按照 ID 排序
    question_jsons = sorted(question_jsons, key=lambda x: x[args.id_key])
    # 对第一个答案 JSON 列表按照 ID 排序
    answer1_jsons = sorted(
        [answer for answer in answer1_jsons if answer[args.id_key] in question_ids],
        key=lambda x: x[args.id_key]
    )
    # 对第二个答案 JSON 列表按照 ID 排序
    answer2_jsons = sorted(
        [answer for answer in answer2_jsons if answer[args.id_key] in question_ids],
        key=lambda x: x[args.id_key]
    )

    # 检查问题和答案的数量是否相同
    assert len(question_jsons) == len(answer1_jsons) == len(answer2_jsons)

    # 创建空列表 handles
    handles = []
    # 创建空列表 review_jsons
    review_jsons = []
    # 获取问题 JSON 列表的长度
    total_len = len(question_jsons)
    # 创建问题索引列表
    question_idx_list = list(range(total_len))

    # 从 Ray 中获取 handles
    reviews = ray.get(handles)
    # 打开目标文件，准备写入
    with open(dest, "w") as output_review_file:
        # 遍历 reviews 列表
        for idx, review in enumerate(reviews):
            # 如果提示文件中包含 "threeclass"
            if 'threeclass' in args.prompt_file:
                # 解析三类分数
                scores = parse_three_class_score(review)
            else:
                # 解析分数
                scores = parse_score(review)
            # 将 review 内容和分数添加到 review_jsons 中
            review_jsons[idx]["text"] = review
            review_jsons[idx]["score"] = scores
            # 将 review_jsons 中的内容写入目标文件
            output_review_file.write(json.dumps(review_jsons[idx]) + "\n")
            # 刷新文件缓冲区
            output_review_file.flush()
```