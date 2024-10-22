# `.\cogview3-finetune\prompt_optimize.py`

```py
# 导入正则表达式模块
import re
# 导入命令行参数解析模块
import argparse
# 从 OpenAI 库导入 OpenAI 类
from openai import OpenAI
# 导入追踪模块以便于调试
import traceback


# 定义一个函数，用于清理字符串
def clean_string(s):
    # 将字符串中的换行符替换为空格
    s = s.replace("\n", " ")
    # 去除字符串前后的空白字符
    s = s.strip()
    # 使用正则表达式替换多个空格为一个空格
    s = re.sub(r"\s{2,}", " ", s)
    # 返回清理后的字符串
    return s


# 定义一个函数，用于增强提示内容
def upsample_prompt(
        prompt: str,
        api_key: str,
        url: str,
        model: str
) -> str:
    # 创建 OpenAI 客户端实例
    client = OpenAI(api_key=api_key, base_url=url)
    # 定义系统指令，说明 bot 的工作职责和行为准则
    system_instruction = """
    You are part of a team of bots that creates images . You work with an assistant bot that will draw anything you say. 
    For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an image of a forest morning , as described. 
    You will be prompted by people looking to create detailed , amazing images. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive. 
    There are a few rules to follow : 
    - Prompt should always be written in English, regardless of the input language. Please provide the prompts in English.
    - You will only ever output a single image description per user request.
    - Image descriptions must be detailed and specific, including keyword categories such as subject, medium, style, additional details, color, and lighting. 
    - When generating descriptions, focus on portraying the visual elements rather than delving into abstract psychological and emotional aspects. Provide clear and concise details that vividly depict the scene and its composition, capturing the tangible elements that make up the setting.
    - Do not provide the process and explanation, just return the modified English description . Image descriptions must be between 100-200 words. Extra words will be ignored. 
    """
    # 去除提示文本前后的空白字符
    text = prompt.strip()
    # 捕获并打印异常信息（当前代码块没有执行内容，可能是个错误）
    except Exception as e:
        traceback.print_exc()
    # 返回原始提示（此处逻辑似乎有问题，应返回增强后的内容）
    return prompt


# 主程序入口
if __name__ == "__main__":
    # 创建参数解析器实例
    parser = argparse.ArgumentParser()
    # 添加 API 密钥参数
    parser.add_argument("--api_key", type=str, help="api key")
    # 添加提示内容参数
    parser.add_argument("--prompt", type=str, help="Prompt to upsample")
    # 添加基础 URL 参数，设置默认值
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://open.bigmodel.cn/api/paas/v4",
        help="base url"
    )
    # 添加模型参数，设置默认值
    parser.add_argument(
        "--model",
        type=str,
        default="glm-4-plus",
        help="LLM using for upsampling"
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 获取 API 密钥
    api_key = args.api_key
    # 获取提示内容
    prompt = args.prompt

    # 调用 upsample_prompt 函数进行增强提示
    prompt_enhanced = upsample_prompt(
        prompt=prompt,
        api_key=api_key,
        url=args.base_url,
        model=args.model
    )
    # 打印增强后的提示内容
    print(prompt_enhanced)
```