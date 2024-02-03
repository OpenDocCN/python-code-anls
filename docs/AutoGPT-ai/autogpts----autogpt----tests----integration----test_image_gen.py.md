# `.\AutoGPT\autogpts\autogpt\tests\integration\test_image_gen.py`

```py
# 导入必要的库
import functools
import hashlib
from pathlib import Path
from unittest.mock import patch
import pytest
from PIL import Image
from autogpt.agents.agent import Agent
from autogpt.commands.image_gen import generate_image, generate_image_with_sd_webui

# 定义一个 fixture，参数化图片大小
@pytest.fixture(params=[256, 512, 1024])
def image_size(request):
    """Parametrize image size."""
    return request.param

# 标记测试需要 OpenAI API 密钥
@pytest.mark.requires_openai_api_key
# 标记测试使用 VCR
@pytest.mark.vcr
def test_dalle(agent: Agent, workspace, image_size, cached_openai_client):
    """Test DALL-E image generation."""
    # 调用生成和验证函数，传入参数
    generate_and_validate(
        agent,
        workspace,
        image_provider="dalle",
        image_size=image_size,
    )

# 标记测试为预期失败，原因是图片太大无法放入 CI 管道的 cassette 中
@pytest.mark.xfail(
    reason="The image is too big to be put in a cassette for a CI pipeline. "
    "We're looking into a solution."
)
# 标记测试需要 HuggingFace API 密钥
@pytest.mark.requires_huggingface_api_key
# 参数化测试，传入不同的 image_model
@pytest.mark.parametrize(
    "image_model",
    ["CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2-1"],
)
def test_huggingface(agent: Agent, workspace, image_size, image_model):
    """Test HuggingFace image generation."""
    # 调用生成和验证函数，传入参数
    generate_and_validate(
        agent,
        workspace,
        image_provider="huggingface",
        image_size=image_size,
        hugging_face_image_model=image_model,
    )

# 标记测试为预期失败，原因是 SD WebUI 调用不起作用
@pytest.mark.xfail(reason="SD WebUI call does not work.")
def test_sd_webui(agent: Agent, workspace, image_size):
    """Test SD WebUI image generation."""
    # 调用生成和验证函数，传入参数
    generate_and_validate(
        agent,
        workspace,
        image_provider="sd_webui",
        image_size=image_size,
    )

# 标记测试为预期失败，原因是 SD WebUI 调用不起作用
@pytest.mark.xfail(reason="SD WebUI call does not work.")
def test_sd_webui_negative_prompt(agent: Agent, workspace, image_size):
    # 使用 functools.partial 创建一个部分函数，传入参数
    gen_image = functools.partial(
        generate_image_with_sd_webui,
        prompt="astronaut riding a horse",
        agent=agent,
        size=image_size,
        extra={"seed": 123},
    )

    # 生成一个带有负面提示的图片
    # 生成一张带有负面提示的图片，并将路径存储在列表中
    image_path = lst(
        gen_image(negative_prompt="horse", output_file=Path("negative.jpg"))
    )
    # 打开图片文件，计算其 MD5 哈希值
    with Image.open(image_path) as img:
        neg_image_hash = hashlib.md5(img.tobytes()).hexdigest()

    # 生成一张没有负面提示的图片，并将路径存储在列表中
    image_path = lst(gen_image(output_file=Path("positive.jpg")))
    # 打开图片文件，计算其 MD5 哈希值
    with Image.open(image_path) as img:
        image_hash = hashlib.md5(img.tobytes()).hexdigest()

    # 断言两张图片的哈希值不相等
    assert image_hash != neg_image_hash
# 从 `generate_image()` 的输出中提取文件路径
def lst(txt):
    return Path(txt.split(": ", maxsplit=1)[1].strip())

# 生成图像并验证输出
def generate_and_validate(
    agent: Agent,
    workspace,
    image_size,
    image_provider,
    hugging_face_image_model=None,
    **kwargs,
):
    # 设置代理的图像提供者
    agent.legacy_config.image_provider = image_provider
    # 如果有 Hugging Face 图像模型，则设置代理的 Hugging Face 图像模型
    if hugging_face_image_model:
        agent.legacy_config.huggingface_image_model = hugging_face_image_model
    # 设置提示文本
    prompt = "astronaut riding a horse"

    # 生成图像并获取图像路径
    image_path = lst(generate_image(prompt, agent, image_size, **kwargs))
    # 断言图像路径存在
    assert image_path.exists()
    # 打开图像并断言图像尺寸符合预期
    with Image.open(image_path) as img:
        assert img.size == (image_size, image_size)

# 参数化测试用例
@pytest.mark.parametrize(
    "return_text",
    [
        # 延迟
        '{"error":"Model [model] is currently loading","estimated_time": [delay]}',
        '{"error":"Model [model] is currently loading"}',  # 无延迟
        '{"error:}',  # 错误的 JSON
        "",  # 错误的图像
    ],
)
@pytest.mark.parametrize(
    "image_model",
    ["CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2-1"],
)
@pytest.mark.parametrize("delay", [10, 0])
# 测试 Hugging Face 请求失败的情况
def test_huggingface_fail_request_with_delay(
    agent: Agent, workspace, image_size, image_model, return_text, delay
):
    # 替换返回文本中的模型和延迟信息
    return_text = return_text.replace("[model]", image_model).replace(
        "[delay]", str(delay)
    )
    # 使用 patch 函数模拟 requests.post 方法
    with patch("requests.post") as mock_post:
        # 如果 return_text 为空字符串
        if return_text == "":
            # 测试错误的图片
            mock_post.return_value.status_code = 200
            mock_post.return_value.ok = True
            mock_post.return_value.content = b"bad image"
        else:
            # 测试延迟和错误的 JSON
            mock_post.return_value.status_code = 500
            mock_post.return_value.ok = False
            mock_post.return_value.text = return_text

        # 设置 legacy_config 的一些属性
        agent.legacy_config.image_provider = "huggingface"
        agent.legacy_config.huggingface_api_token = "mock-api-key"
        agent.legacy_config.huggingface_image_model = image_model
        prompt = "astronaut riding a horse"

        # 使用 patch 函数模拟 time.sleep 方法
        with patch("time.sleep") as mock_sleep:
            # 验证请求失败
            result = generate_image(prompt, agent, image_size)
            assert result == "Error creating image."

            # 如果 return_text 中包含 estimated_time，则验证是否调用了 mock_sleep 方法
            if "estimated_time" in return_text:
                mock_sleep.assert_called_with(delay)
            else:
                mock_sleep.assert_not_called()
# 测试当请求失败且没有延迟时的情况
def test_huggingface_fail_request_no_delay(mocker, agent: Agent):
    # 设置 Hugging Face API token
    agent.legacy_config.huggingface_api_token = "1"

    # 模拟 requests.post
    mock_post = mocker.patch("requests.post")
    mock_post.return_value.status_code = 500
    mock_post.return_value.ok = False
    mock_post.return_value.text = (
        '{"error":"Model CompVis/stable-diffusion-v1-4 is currently loading"}'
    )

    # 模拟 time.sleep
    mock_sleep = mocker.patch("time.sleep")

    # 设置图像提供者和模型
    agent.legacy_config.image_provider = "huggingface"
    agent.legacy_config.huggingface_image_model = "CompVis/stable-diffusion-v1-4"

    # 生成图像
    result = generate_image("astronaut riding a horse", agent, 512)

    # 断言生成图像失败
    assert result == "Error creating image."

    # 验证未调用重试
    mock_sleep.assert_not_called()


# 测试当请求返回的 JSON 格式不正确时的情况
def test_huggingface_fail_request_bad_json(mocker, agent: Agent):
    # 设置 Hugging Face API token
    agent.legacy_config.huggingface_api_token = "1"

    # 模拟 requests.post
    mock_post = mocker.patch("requests.post")
    mock_post.return_value.status_code = 500
    mock_post.return_value.ok = False
    mock_post.return_value.text = '{"error:}'

    # 模拟 time.sleep
    mock_sleep = mocker.patch("time.sleep")

    # 设置图像提供者和模型
    agent.legacy_config.image_provider = "huggingface"
    agent.legacy_config.huggingface_image_model = "CompVis/stable-diffusion-v1-4"

    # 生成图像
    result = generate_image("astronaut riding a horse", agent, 512)

    # 断言生成图像失败
    assert result == "Error creating image."

    # 验证未调用重试
    mock_sleep.assert_not_called()


# 测试当请求返回的图像不正确时的情况
def test_huggingface_fail_request_bad_image(mocker, agent: Agent):
    # 设置 Hugging Face API token
    agent.legacy_config.huggingface_api_token = "1"

    # 模拟 requests.post
    mock_post = mocker.patch("requests.post")
    mock_post.return_value.status_code = 200

    # 设置图像提供者和模型
    agent.legacy_config.image_provider = "huggingface"
    agent.legacy_config.huggingface_image_model = "CompVis/stable-diffusion-v1-4"

    # 生成图像
    result = generate_image("astronaut riding a horse", agent, 512)

    # 断言生成图像失败
    assert result == "Error creating image."
# 定义一个测试函数，测试在缺少 API 令牌的情况下 Hugging Face 失败的情况
def test_huggingface_fail_missing_api_token(mocker, agent: Agent):
    # 设置代理的图像提供者为 Hugging Face
    agent.legacy_config.image_provider = "huggingface"
    # 设置 Hugging Face 图像模型
    agent.legacy_config.huggingface_image_model = "CompVis/stable-diffusion-v1-4"

    # 使用 mocker.patch 来模拟 requests.post 抛出 ValueError 异常
    mocker.patch("requests.post", side_effect=ValueError)

    # 验证请求是否会引发错误
    with pytest.raises(ValueError):
        # 调用生成图像函数，传入描述、代理和图像大小
        generate_image("astronaut riding a horse", agent, 512)
```