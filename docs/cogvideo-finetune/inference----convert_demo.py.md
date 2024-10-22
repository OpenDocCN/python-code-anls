# `.\cogvideo-finetune\inference\convert_demo.py`

```py
"""
该CogVideoX模型旨在根据详细且高度描述性的提示生成高质量的视频。
当提供精细的、细致的提示时，模型表现最佳，这能提高视频生成的质量。
该脚本旨在帮助将简单的用户输入转换为适合CogVideoX的详细提示。
它可以处理文本到视频（t2v）和图像到视频（i2v）的转换。

- 对于文本到视频，只需提供提示。
- 对于图像到视频，提供图像文件的路径和可选的用户输入。
图像将被编码并作为请求的一部分发送给Azure OpenAI。

### 如何运行：
运行脚本进行**文本到视频**：
    $ python convert_demo.py --prompt "一个女孩骑自行车。" --type "t2v"

运行脚本进行**图像到视频**：
    $ python convert_demo.py --prompt "猫在跑" --type "i2v" --image_path "/path/to/your/image.jpg"
"""

# 导入argparse库以处理命令行参数
import argparse
# 从openai库导入OpenAI和AzureOpenAI类
from openai import OpenAI, AzureOpenAI
# 导入base64库以进行数据编码
import base64
# 从mimetypes库导入guess_type函数以推测文件类型
from mimetypes import guess_type

# 定义文本到视频的系统提示
sys_prompt_t2v = """您是一个创建视频的机器人团队的一部分。您与一个助手机器人合作，助手会绘制您所说的方括号中的任何内容。

例如，输出“一个阳光穿过树木的美丽清晨”将触发您的伙伴机器人输出如描述的森林早晨的视频。您将被希望创建详细、精彩视频的人所提示。完成此任务的方法是将他们的简短提示转化为极其详细和描述性的内容。
需要遵循一些规则：

您每次用户请求只能输出一个视频描述。

当请求修改时，您不应简单地将描述变得更长。您应重构整个描述，以整合建议。
有时用户不想要修改，而是希望得到一个新图像。在这种情况下，您应忽略与用户的先前对话。

视频描述必须与以下示例的单词数量相同。多余的单词将被忽略。
"""

# 定义图像到视频的系统提示
sys_prompt_i2v = """
**目标**：**根据输入图像和用户输入给出高度描述性的视频说明。** 作为专家，深入分析图像，运用丰富的创造力和细致的思考。在描述图像的细节时，包含适当的动态信息，以确保视频说明包含合理的动作和情节。如果用户输入不为空，则说明应根据用户的输入进行扩展。

**注意**：输入图像是视频的第一帧，输出视频说明应描述从当前图像开始的运动。用户输入是可选的，可以为空。

**注意**：不要包含相机转场！！！不要包含画面切换！！！不要包含视角转换！！！

**回答风格**：
# 定义将图像文件转换为 URL 的函数
def image_to_url(image_path):
    # 根据图像路径猜测其 MIME 类型，第二个返回值忽略
    mime_type, _ = guess_type(image_path)
    # 如果无法猜测 MIME 类型，则设置为通用二进制流类型
    if mime_type is None:
        mime_type = "application/octet-stream"
    # 以二进制模式打开图像文件
    with open(image_path, "rb") as image_file:
        # 读取图像文件内容并进行 Base64 编码，解码为 UTF-8 格式
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")
    # 返回格式化的 Base64 数据 URL 字符串
    return f"data:{mime_type};base64,{base64_encoded_data}"


# 定义将提示转换为可用于模型推理的格式的函数
def convert_prompt(prompt: str, retry_times: int = 3, type: str = "t2v", image_path: str = None):
    """
    将提示转换为可用于模型推理的格式
    """

    # 创建 OpenAI 客户端实例
    client = OpenAI()
    ## 如果使用 Azure OpenAI，请取消注释下面一行并注释上面一行
    # client = AzureOpenAI(
    #     api_key="",
    #     api_version="",
    #     azure_endpoint=""
    # )

    # 去除提示字符串两端的空白
    text = prompt.strip()
    # 返回未处理的提示
    return prompt


# 如果当前脚本是主程序
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加提示参数，类型为字符串，必需
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to convert")
    # 添加重试次数参数，类型为整数，默认值为 3
    parser.add_argument("--retry_times", type=int, default=3, help="Number of times to retry the conversion")
    # 添加转换类型参数，类型为字符串，默认值为 "t2v"
    parser.add_argument("--type", type=str, default="t2v", help="Type of conversion (t2v or i2v)")
    # 添加图像路径参数，类型为字符串，默认值为 None
    parser.add_argument("--image_path", type=str, default=None, help="Path to the image file")
    # 解析命令行参数并存储在 args 中
    args = parser.parse_args()

    # 调用 convert_prompt 函数进行提示转换
    converted_prompt = convert_prompt(args.prompt, args.retry_times, args.type, args.image_path)
    # 打印转换后的提示
    print(converted_prompt)
```