# `.\AutoGPT\autogpts\autogpt\autogpt\commands\image_gen.py`

```py
"""Commands to generate images based on text input"""

import io  # 导入io模块
import json  # 导入json模块
import logging  # 导入logging模块
import time  # 导入time模块
import uuid  # 导入uuid模块
from base64 import b64decode  # 从base64模块导入b64decode函数
from pathlib import Path  # 从pathlib模块导入Path类

import requests  # 导入requests模块
from openai import OpenAI  # 从openai模块导入OpenAI类
from PIL import Image  # 从PIL模块导入Image类

from autogpt.agents.agent import Agent  # 从autogpt.agents.agent模块导入Agent类
from autogpt.command_decorator import command  # 从autogpt.command_decorator模块导入command装饰器
from autogpt.core.utils.json_schema import JSONSchema  # 从autogpt.core.utils.json_schema模块导入JSONSchema类

COMMAND_CATEGORY = "text_to_image"  # 定义命令类别为"text_to_image"
COMMAND_CATEGORY_TITLE = "Text to Image"  # 定义命令类别标题为"Text to Image"

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


@command(
    "generate_image",
    "Generates an Image",
    {
        "prompt": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The prompt used to generate the image",
            required=True,
        ),
    },
    lambda config: bool(config.image_provider),
    "Requires a image provider to be set.",
)
def generate_image(prompt: str, agent: Agent, size: int = 256) -> str:
    """Generate an image from a prompt.

    Args:
        prompt (str): The prompt to use
        size (int, optional): The size of the image. Defaults to 256.
            Not supported by HuggingFace.

    Returns:
        str: The filename of the image
    """
    filename = agent.workspace.root / f"{str(uuid.uuid4())}.jpg"  # 生成一个唯一的文件名

    # DALL-E
    if agent.legacy_config.image_provider == "dalle":  # 如果图像提供者是DALL-E
        return generate_image_with_dalle(prompt, filename, size, agent)  # 调用generate_image_with_dalle函数生成图像
    # HuggingFace
    elif agent.legacy_config.image_provider == "huggingface":  # 如果图像提供者是HuggingFace
        return generate_image_with_hf(prompt, filename, agent)  # 调用generate_image_with_hf函数生成图像
    # SD WebUI
    elif agent.legacy_config.image_provider == "sdwebui":  # 如果图像提供者是SD WebUI
        return generate_image_with_sd_webui(prompt, filename, agent, size)  # 调用generate_image_with_sd_webui函数生成图像
    return "No Image Provider Set"  # 如果没有设置图像提供者，则返回提示信息


def generate_image_with_hf(prompt: str, output_file: Path, agent: Agent) -> str:
    """Generate an image with HuggingFace's API.

    Args:
        prompt (str): The prompt to use
        filename (Path): The filename to save the image to

    Returns:
        str: The filename of the image
    """
    # 设置 Hugging Face 模型 API 的 URL
    API_URL = f"https://api-inference.huggingface.co/models/{agent.legacy_config.huggingface_image_model}"  # noqa: E501
    # 检查是否设置了 Hugging Face API token，如果没有则抛出数值错误
    if agent.legacy_config.huggingface_api_token is None:
        raise ValueError(
            "You need to set your Hugging Face API token in the config file."
        )
    # 设置请求头信息，包括 Authorization 和 X-Use-Cache
    headers = {
        "Authorization": f"Bearer {agent.legacy_config.huggingface_api_token}",
        "X-Use-Cache": "false",
    }

    # 初始化重试次数
    retry_count = 0
    # 循环执行最多 10 次
    while retry_count < 10:
        # 发送 POST 请求到 API_URL，传递请求头和输入数据
        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "inputs": prompt,
            },
        )

        # 如果响应成功
        if response.ok:
            try:
                # 尝试打开响应内容作为图像
                image = Image.open(io.BytesIO(response.content))
                logger.info(f"Image Generated for prompt:{prompt}")
                # 保存图像到输出文件
                image.save(output_file)
                return f"Saved to disk: {output_file}"
            except Exception as e:
                logger.error(e)
                break
        else:
            try:
                # 尝试解析响应内容为 JSON
                error = json.loads(response.text)
                # 如果错误信息中包含估计的时间，则等待一段时间后重试
                if "estimated_time" in error:
                    delay = error["estimated_time"]
                    logger.debug(response.text)
                    logger.info("Retrying in", delay)
                    time.sleep(delay)
                else:
                    break
            except Exception as e:
                logger.error(e)
                break

        # 增加重试次数
        retry_count += 1

    # 返回错误信息
    return "Error creating image."
# 使用 DALL-E 生成图像
def generate_image_with_dalle(
    prompt: str, output_file: Path, size: int, agent: Agent
) -> str:
    """Generate an image with DALL-E.

    Args:
        prompt (str): The prompt to use
        filename (Path): The filename to save the image to
        size (int): The size of the image

    Returns:
        str: The filename of the image
    """

    # 检查支持的图像尺寸
    if size not in [256, 512, 1024]:
        closest = min([256, 512, 1024], key=lambda x: abs(x - size))
        logger.info(
            "DALL-E only supports image sizes of 256x256, 512x512, or 1024x1024. "
            f"Setting to {closest}, was {size}."
        )
        size = closest

    # 使用 OpenAI 生成图像
    response = OpenAI(
        api_key=agent.legacy_config.openai_credentials.api_key.get_secret_value()
    ).images.generate(
        prompt=prompt,
        n=1,
        size=f"{size}x{size}",
        response_format="b64_json",
    )

    logger.info(f"Image Generated for prompt:{prompt}")

    # 解码图像数据并保存到文件
    image_data = b64decode(response.data[0].b64_json)

    with open(output_file, mode="wb") as png:
        png.write(image_data)

    return f"Saved to disk: {output_file}"


# 使用 Stable Diffusion webui 生成图像
def generate_image_with_sd_webui(
    prompt: str,
    output_file: Path,
    agent: Agent,
    size: int = 512,
    negative_prompt: str = "",
    extra: dict = {},
) -> str:
    """Generate an image with Stable Diffusion webui.
    Args:
        prompt (str): The prompt to use
        filename (str): The filename to save the image to
        size (int, optional): The size of the image. Defaults to 256.
        negative_prompt (str, optional): The negative prompt to use. Defaults to "".
        extra (dict, optional): Extra parameters to pass to the API. Defaults to {}.
    Returns:
        str: The filename of the image
    """
    # 创建会话并设置基本身份验证（如果需要）
    s = requests.Session()
    # 如果代理的旧配置中包含 sd_webui_auth，则解析出用户名和密码
    if agent.legacy_config.sd_webui_auth:
        username, password = agent.legacy_config.sd_webui_auth.split(":")
        s.auth = (username, password or "")

    # 生成图片
    response = requests.post(
        f"{agent.legacy_config.sd_webui_url}/sdapi/v1/txt2img",
        json={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "sampler_index": "DDIM",
            "steps": 20,
            "config_scale": 7.0,
            "width": size,
            "height": size,
            "n_iter": 1,
            **extra,
        },
    )

    # 记录生成的图片信息
    logger.info(f"Image Generated for prompt: '{prompt}'")

    # 将图片保存到磁盘
    response = response.json()
    b64 = b64decode(response["images"][0].split(",", 1)[0])
    image = Image.open(io.BytesIO(b64))
    image.save(output_file)

    # 返回保存到磁盘的图片文件路径
    return f"Saved to disk: {output_file}"
```