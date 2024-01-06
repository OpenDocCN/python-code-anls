# `qlora\eval\eval_gpt_review.py`

```
# 导入所需的库
# 从指定的 GitHub 仓库中导入代码
import argparse  # 用于解析命令行参数
import json  # 用于处理 JSON 数据
import os  # 用于操作系统相关的功能
import time  # 用于时间相关的功能

import openai  # 导入 OpenAI 的 API
from tqdm import tqdm  # 用于在循环中显示进度条
import ray  # 用于分布式计算

import shortuuid  # 用于生成短的 UUID
import logging  # 用于记录日志
import numpy as np  # 用于数值计算
import os  # 用于操作系统相关的功能
import openai  # 导入 OpenAI 的 API
openai.api_key = os.getenv("OPENAI_API_KEY")  # 设置 OpenAI 的 API 密钥

# 配置日志记录的级别
logging.basicConfig(level=logging.INFO)
# 创建一个记录器
logger = logging.getLogger(__name__)
# 设置最大 API 重试次数
MAX_API_RETRY = 1000
# 设置请求时间间隔
REQ_TIME_GAP = 2

# 使用 ray.remote 装饰器将函数标记为远程函数，并指定可用的 CPU 数量
@ray.remote(num_cpus=4)
# 定义一个名为 get_eval 的函数，接受系统提示、用户提示、最大标记数和模型作为参数
def get_eval(sys_prompt, user_prompt: str, max_tokens: int, model: str):
    # 配置日志记录器的日志级别为 INFO
    logging.basicConfig(level=logging.INFO)
    # 循环尝试 API 调用，最多尝试 MAX_API_RETRY 次
    for i in range(MAX_API_RETRY):
        try:
            # 调用 openai.ChatCompletion.create 方法发送聊天请求
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.2,  # 设置温度参数，用于控制生成文本的多样性
                max_tokens=max_tokens,  # 设置最大标记数
# 从 API 响应中获取第一个选择的消息内容，并记录日志
content = response["choices"][0]["message"]["content"]
logger.info(content)
# 返回消息内容
return content
# 捕获异常，记录错误日志，等待一段时间后重试
except Exception as e:
    logger.error(e)
    time.sleep(min(5*(i+1), 100))
# 记录最大重试次数后仍失败的错误日志
logger.error(f"Failed after {MAX_API_RETRY} retries.")
# 返回错误信息
return "error"

# 解析三个类别的评分
def parse_three_class_score(review):
    try:
        # 去除首尾空格后，按换行符分割，取最后一行并转换为整数
        score = int(review.strip().split("\n")[-1].strip())
        # 返回评分
        return score
    # 捕获异常，记录错误日志，返回默认值-1
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return -1
# 解析评分，将评分字符串转换为列表形式的评分
def parse_score(review):
    try:
        # 尝试从评价中提取评分字符串
        score_pair = review.split("\n")[0]
        # 将逗号替换为空格
        score_pair = score_pair.replace(",", " ")
        # 将评分字符串拆分为列表
        sp = score_pair.split(" ")
        # 如果列表长度为2，返回包含两个评分的列表
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        # 如果列表长度不为2，抛出异常
        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        # 捕获异常并记录日志
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        # 返回默认的无效评分
        return [-1, -1]


def gen_prompt(reviewer_jsons, prompt_jsons, cat, ques, ans1, ans2):
    # 默认使用一般类别（索引为0）
    reviewer_idx = 0
    # 遍历评价者的 JSON 列表
    for idx, reviewer in enumerate(reviewer_jsons):
    # 如果评审者的类别与给定的类别相同，则记录下该评审者的索引并跳出循环
    if reviewer["category"] == cat:
        reviewer_idx = idx
        break
    # 获取评审者对应的提示 ID
    prompt_id = reviewer_jsons[reviewer_idx]["prompt_id"]
    # 获取对应提示 ID 的提示 JSON
    prompt_json = prompt_jsons[prompt_id - 1]
    # 断言提示 JSON 中的提示 ID 与获取的提示 ID 相同
    assert prompt_json["prompt_id"] == prompt_id

    # 获取系统提示
    sys_prompt = prompt_json["system_prompt"]
    # 获取提示模板
    prompt_template = prompt_json["prompt_template"]
    # 获取默认值
    defaults = prompt_json["defaults"]
    # 根据提示模板和默认值生成最终的提示
    prompt = prompt_template.format(
        question=ques, answer_1=ans1, answer_2=ans2, **defaults
    )

    # 返回系统提示、最终提示和评审者索引
    return sys_prompt, prompt, reviewer_idx + 1


# 获取 JSON 文件中的内容列表
def get_json_list(file_path):
    # 将文件路径转换为绝对路径
    file_path = os.path.expanduser(file_path)
    # 打开文件并读取内容
    with open(file_path, "r") as f:
# 创建一个空的 JSON 列表
json_list = []
# 遍历文件 f 中的每一行，将其解析为 JSON 对象并添加到 JSON 列表中
for line in f:
    json_list.append(json.loads(line))
# 返回包含所有 JSON 对象的列表
return json_list

# 如果作为独立程序运行
if __name__ == "__main__":
    # 创建一个参数解析器，描述为“ChatGPT-based QA evaluation.”
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    # 添加问题文件参数
    parser.add_argument("-q", "--question-file")
    # 添加答案文件列表参数，可接受多个文件
    parser.add_argument("-a", "--answer-file-list", nargs="+", default=[])
    # 添加提示文件参数
    parser.add_argument("-p", "--prompt-file")
    # 添加评论者文件参数
    parser.add_argument("-r", "--reviewer-file")
    # 添加输出评论文件参数
    parser.add_argument("-o", "--output-review-file")
    # 添加模型参数，默认为 'gpt-4'
    parser.add_argument("-m", "--model", default='gpt-4')
    # 添加 ID 键参数，默认为 'question_id'
    parser.add_argument("-id", "--id-key", default='question_id')
    # 添加最大标记数参数，默认为 1024，用于限制输出中的标记数
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 如果输出审查文件路径不是一个目录，则将其作为目标路径
    if not os.path.isdir(args.output_review_file):
        dest = args.output_review_file
    # 如果输出审查文件路径是一个目录，则根据条件拼接文件名作为目标路径
    else:
        threeclass_suff = "_threeclass" if 'threeclass' in args.prompt_file else ""
        dest = os.path.join(
            args.output_review_file,
            '_vs_'.join([elt.split('/')[-1].replace('.jsonl', '') for elt in args.answer_file_list]) + f'_{args.model}_reviewer{threeclass_suff}' + '.jsonl'
        )

    # 初始化 Ray
    ray.init()

    # 获取问题文件、答案文件、审查者文件和提示文件的 JSON 列表
    question_jsons = get_json_list(args.question_file)
    answer1_jsons = get_json_list(args.answer_file_list[0])
    answer2_jsons = get_json_list(args.answer_file_list[1])
    reviewer_jsons = get_json_list(args.reviewer_file)
    prompt_jsons = get_json_list(args.prompt_file)

    # 获取问题 ID 的集合
    question_ids = set(question[args.id_key] for question in question_jsons)
    # 根据指定的键对问题 JSON 列表进行排序
    question_jsons = sorted(question_jsons, key=lambda x: x[args.id_key])
    # 根据指定的键对答案1 JSON 列表进行排序，并且只保留与问题 ID 匹配的答案
    answer1_jsons = sorted(
        [answer for answer in answer1_jsons if answer[args.id_key] in question_ids],
        key=lambda x: x[args.id_key]
    )
    # 根据指定的键对答案2 JSON 列表进行排序，并且只保留与问题 ID 匹配的答案
    answer2_jsons = sorted(
        [answer for answer in answer2_jsons if answer[args.id_key] in question_ids],
        key=lambda x: x[args.id_key]
    )

    # 检查问题和答案的数量是否相同
    assert len(question_jsons) == len(answer1_jsons) == len(answer2_jsons)

    # 初始化变量
    handles = []
    review_jsons = []
    total_len = len(question_jsons)
    question_idx_list = list(range(total_len))

    # 遍历问题索引列表
    for i in tqdm(question_idx_list):
        # 断言条件是否成立
        assert (
        # 检查三个 JSON 对象中的 id_key 是否相等
        answer1_jsons[i][args.id_key]
        == question_jsons[i][args.id_key]
        == answer2_jsons[i][args.id_key]
    )

    # 获取问题文本和类别
    ques = question_jsons[i]["text"]
    cat = question_jsons[i]["category"]

    # 根据条件获取第一个答案的内容
    if 'generation_truncated' in answer1_jsons[i]:
        ans1 = answer1_jsons[i]["generation_truncated"].split(
            'A chat between a curious human and an artificial intelligence')[0]
    elif 'generation' in answer1_jsons[i]:
        ans1 = answer1_jsons[i]["generation"].split(
            'A chat between a curious human and an artificial intelligence')[0]
    else:
        ans1 = answer1_jsons[i]["text"]
    # ans1 = answer1_jsons[i]["text"]

    # 根据条件获取第二个答案的内容
    if 'generation_truncated' in answer2_jsons[i]:
        ans2 = answer2_jsons[i]["generation_truncated"].split(
            'A chat between a curious human and an artificial intelligence')[0]
    elif 'generation' in answer2_jsons[i]:
# 如果答案2中包含特定字符串，则将其分割并取前半部分作为ans2，否则直接使用答案2中的文本
if "A chat between a curious human and an artificial intelligence" in answer2_jsons[i]["generation"]:
    ans2 = answer2_jsons[i]["generation"].split('A chat between a curious human and an artificial intelligence')[0]
else:
    ans2 = answer2_jsons[i]["text"]

# 生成提示、问题、答案1、答案2对应的reviewer_id
sys_prompt, prompt, reviewer_id = gen_prompt(reviewer_jsons, prompt_jsons, cat, ques, ans1, ans2)

# 生成一个短的review_id
review_id = shortuuid.uuid()

# 将生成的review信息添加到review_jsons列表中
review_jsons.append(
    {
        "review_id": review_id,
        args.id_key: question_jsons[i][args.id_key],
        "answer1_id": answer1_jsons[i]["answer_id"] if 'answer_id' in answer1_jsons[i] else shortuuid.uuid(ans1),
        "answer2_id": answer2_jsons[i]["answer_id"] if 'answer_id' in answer2_jsons[i] else shortuuid.uuid(ans2),
        "reviewer_id": reviewer_id,
        "metadata": {},
    }
)

# 为了避免OpenAI设置的速率限制，将远程处理的结果添加到handles列表中
handles.append(get_eval.remote(sys_prompt, prompt, args.max_tokens, args.model))
# 记录日志，等待指定时间间隔后发送下一个请求
logger.info(
    f"Waiting for {REQ_TIME_GAP} seconds before sending the next request."
)
# 等待指定时间间隔
time.sleep(REQ_TIME_GAP)

# 获取处理后的评论数据
reviews = ray.get(handles)
# 打开目标文件，准备写入处理后的评论数据
with open(dest, "w") as output_review_file:
    # 遍历处理后的评论数据
    for idx, review in enumerate(reviews):
        # 根据参数文件类型选择不同的评分解析方法
        if 'threeclass' in args.prompt_file:
            scores = parse_three_class_score(review)
        else:
            scores = parse_score(review)
        # 更新评论数据和评分到 JSON 对象
        review_jsons[idx]["text"] = review
        review_jsons[idx]["score"] = scores
        # 将 JSON 对象转换为字符串并写入文件
        output_review_file.write(json.dumps(review_jsons[idx]) + "\n")
        # 刷新文件缓冲区
        output_review_file.flush()
```