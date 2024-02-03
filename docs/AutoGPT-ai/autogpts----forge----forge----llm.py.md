# `.\AutoGPT\autogpts\forge\forge\llm.py`

```py
# 从 pathlib 模块中导入 Path 类
from pathlib import Path

# 从 litellm 模块中导入 AuthenticationError, InvalidRequestError, ModelResponse, acompletion 函数
from litellm import AuthenticationError, InvalidRequestError, ModelResponse, acompletion
# 从 openai 模块中导入 OpenAI 类
from openai import OpenAI
# 从 openai.types 模块中导入 CreateEmbeddingResponse 类
from openai.types import CreateEmbeddingResponse
# 从 openai.types.audio 模块中导入 Transcription 类
from openai.types.audio import Transcription
# 从 tenacity 模块中导入 retry, stop_after_attempt, wait_random_exponential 函数
from tenacity import retry, stop_after_attempt, wait_random_exponential

# 从当前目录下的 sdk.forge_log 模块中导入 ForgeLogger 类
from .sdk.forge_log import ForgeLogger

# 创建一个 ForgeLogger 实例，传入当前模块的名称作为参数
LOG = ForgeLogger(__name__)

# 使用 tenacity 库的 retry 装饰器，设置重试策略
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
# 异步函数，生成对一系列消息的响应，使用 OpenAI 的 API
async def chat_completion_request(model, messages, **kwargs) -> ModelResponse:
    """Generate a response to a list of messages using OpenAI's API"""
    try:
        # 将 model 和 messages 添加到 kwargs 中
        kwargs["model"] = model
        kwargs["messages"] = messages

        # 调用 acompletion 函数，传入 kwargs，并等待结果
        resp = await acompletion(**kwargs)
        return resp
    except AuthenticationError as e:
        # 记录认证错误日志
        LOG.exception("Authentication Error")
        raise
    except InvalidRequestError as e:
        # 记录无效请求错误日志
        LOG.exception("Invalid Request Error")
        raise
    except Exception as e:
        # 记录无法生成 ChatCompletion 响应的错误日志
        LOG.error("Unable to generate ChatCompletion response")
        LOG.error(f"Exception: {e}")
        raise

# 使用 tenacity 库的 retry 装饰器，设置重试策略
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
# 异步函数，为一系列消息生成嵌入，使用 OpenAI 的 API
async def create_embedding_request(
    messages, model="text-embedding-ada-002"
) -> CreateEmbeddingResponse:
    """Generate an embedding for a list of messages using OpenAI's API"""
    try:
        # 调用 OpenAI 实例的 embeddings.create 方法，传入消息列表的嵌入参数和模型
        return OpenAI().embeddings.create(
            input=[f"{m['role']}: {m['content']}" for m in messages],
            model=model,
        )
    except Exception as e:
        # 记录无法生成 ChatCompletion 响应的错误日志
        LOG.error("Unable to generate ChatCompletion response")
        LOG.error(f"Exception: {e}")
        raise

# 使用 tenacity 库的 retry 装饰器，设置重试策略
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
# 异步函数，使用 OpenAI 的 API 转录音频文件
async def transcribe_audio(audio_file: Path) -> Transcription:
    """Transcribe an audio file using OpenAI's API"""
    try:
        # 调用 OpenAI 实例的 audio.transcriptions.create 方法，传入模型和打开的音频文件
        return OpenAI().audio.transcriptions.create(
            model="whisper-1", file=audio_file.open(mode="rb")
        )
    # 捕获任何异常并记录错误信息
    except Exception as e:
        # 记录无法生成 ChatCompletion 响应的错误信息
        LOG.error("Unable to generate ChatCompletion response")
        # 记录异常信息
        LOG.error(f"Exception: {e}")
        # 抛出异常
        raise
```