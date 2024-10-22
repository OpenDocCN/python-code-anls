# `.\chatglm4-finetune\basic_demo\glm4v_server.py`

```py
# 导入垃圾回收模块
import gc
# 导入线程模块
import threading
# 导入时间模块
import time
# 导入 base64 编码模块
import base64
# 导入系统模块
import sys
# 从上下文管理器导入异步上下文管理器
from contextlib import asynccontextmanager
# 导入类型提示相关模块
from typing import List, Literal, Union, Tuple, Optional
# 导入 PyTorch 库
import torch
# 导入 uvicorn 作为 ASGI 服务器
import uvicorn
# 导入请求库
import requests
# 导入 FastAPI 框架
from fastapi import FastAPI, HTTPException
# 导入 CORS 中间件
from fastapi.middleware.cors import CORSMiddleware
# 导入 Pydantic 基础模型和字段定义
from pydantic import BaseModel, Field
# 导入 SSE 事件源响应
from sse_starlette.sse import EventSourceResponse
# 导入 Transformers 模型相关模块
from transformers import (
    AutoTokenizer,
    AutoModel,
    TextIteratorStreamer
)
# 导入 PEFT 模型
from peft import PeftModelForCausalLM
# 导入图像处理库
from PIL import Image
# 导入字节流模块
from io import BytesIO
# 导入路径处理模块
from pathlib import Path

# 设置设备为 CUDA，如果不可用则为 CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# 根据 GPU 能力选择适当的 PyTorch 数据类型
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# 定义异步上下文管理器以管理 FastAPI 应用的生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    一个异步上下文管理器，用于管理 FastAPI 应用的生命周期。
    确保在应用生命周期结束后清理 GPU 内存，这是 GPU 环境中高效资源管理的关键。
    """
    yield
    # 如果可用，清空 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 进行 CUDA IPC 垃圾回收
        torch.cuda.ipc_collect()

# 创建 FastAPI 应用，并传入生命周期管理器
app = FastAPI(lifespan=lifespan)

# 添加 CORS 中间件，以允许跨域请求
app.add_middleware(
    CORSMiddleware,
    # 允许所有来源
    allow_origins=["*"],
    # 允许凭证
    allow_credentials=True,
    # 允许所有方法
    allow_methods=["*"],
    # 允许所有请求头
    allow_headers=["*"],
)

# 定义表示模型卡的 Pydantic 模型
class ModelCard(BaseModel):
    """
    表示模型卡的 Pydantic 模型，提供机器学习模型的元数据。
    包括模型 ID、所有者和创建时间等字段。
    """
    id: str
    object: str = "model"  # 模型对象类型
    created: int = Field(default_factory=lambda: int(time.time()))  # 创建时间戳
    owned_by: str = "owner"  # 所有者信息
    root: Optional[str] = None  # 根模型信息（可选）
    parent: Optional[str] = None  # 父模型信息（可选）
    permission: Optional[list] = None  # 权限信息（可选）

# 定义表示模型列表的 Pydantic 模型
class ModelList(BaseModel):
    object: str = "list"  # 对象类型为列表
    data: List[ModelCard] = []  # 包含模型卡的列表

# 定义表示图像 URL 的 Pydantic 模型
class ImageUrl(BaseModel):
    url: str  # 图像 URL 字符串

# 定义表示文本内容的 Pydantic 模型
class TextContent(BaseModel):
    type: Literal["text"]  # 类型为文本
    text: str  # 文本内容

# 定义表示图像 URL 内容的 Pydantic 模型
class ImageUrlContent(BaseModel):
    type: Literal["image_url"]  # 类型为图像 URL
    image_url: ImageUrl  # 图像 URL 对象

# 定义内容项的联合类型
ContentItem = Union[TextContent, ImageUrlContent]

# 定义聊天消息输入的 Pydantic 模型
class ChatMessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]  # 消息角色
    content: Union[str, List[ContentItem]]  # 消息内容，可以是字符串或内容项列表
    name: Optional[str] = None  # 消息发送者名称（可选）

# 定义聊天消息响应的 Pydantic 模型
class ChatMessageResponse(BaseModel):
    role: Literal["assistant"]  # 消息角色为助手
    content: str = None  # 消息内容
    name: Optional[str] = None  # 消息发送者名称（可选）

# 定义增量消息的 Pydantic 模型
class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None  # 可选角色
    content: Optional[str] = None  # 可选内容

# 定义聊天完成请求的 Pydantic 模型
class ChatCompletionRequest(BaseModel):
    model: str  # 使用的模型名称
    messages: List[ChatMessageInput]  # 消息输入列表
    temperature: Optional[float] = 0.8  # 温度参数（控制生成随机性）
    top_p: Optional[float] = 0.8  # Top-p 采样参数
    max_tokens: Optional[int] = None  # 最大生成标记数（可选）
    stream: Optional[bool] = False  # 是否流式返回结果
    # 附加参数
    repetition_penalty: Optional[float] = 1.0  # 重复惩罚参数

# 定义聊天完成响应选择的 Pydantic 模型
class ChatCompletionResponseChoice(BaseModel):
    index: int  # 选择的索引
    message: ChatMessageResponse  # 消息内容
# 定义一个模型类，用于表示聊天完成响应流中的选择
class ChatCompletionResponseStreamChoice(BaseModel):
    # 选择的索引
    index: int
    # 存储差异信息的消息
    delta: DeltaMessage


# 定义一个模型类，用于表示使用信息
class UsageInfo(BaseModel):
    # 提示的 token 数量，默认为 0
    prompt_tokens: int = 0
    # 总 token 数量，默认为 0
    total_tokens: int = 0
    # 完成的 token 数量，默认为 0
    completion_tokens: Optional[int] = 0


# 定义一个模型类，用于表示聊天完成的响应
class ChatCompletionResponse(BaseModel):
    # 使用的模型名称
    model: str
    # 对象的字面量类型
    object: Literal["chat.completion", "chat.completion.chunk"]
    # 选择列表，包含聊天完成的选择
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    # 创建时间，默认为当前时间
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    # 使用信息，可选
    usage: Optional[UsageInfo] = None


# 定义一个 GET 请求的端点，列出可用模型
@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """
    一个用于列出可用模型的端点。返回模型卡的列表。
    这对客户端查询和了解可用模型非常有用。
    """
    # 创建一个模型卡实例
    model_card = ModelCard(id="GLM-4v-9b")
    # 返回包含模型卡的模型列表
    return ModelList(data=[model_card])


# 定义一个 POST 请求的端点，用于创建聊天完成
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    # 声明全局变量，模型和分词器
    global model, tokenizer

    # 验证请求是否有效，确保消息数量大于 0 且最后一条消息不是助手
    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    # 设置生成参数的字典
    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        repetition_penalty=request.repetition_penalty
    )

    # 如果请求要求流式生成
    if request.stream:
        # 调用预测函数生成内容
        generate = predict(request.model, gen_params)
        # 返回事件源响应
        return EventSourceResponse(generate, media_type="text/event-stream")
    # 生成聊天完成响应
    response = generate_glm4v(model, tokenizer, gen_params)

    # 创建使用信息的实例
    usage = UsageInfo()

    # 创建聊天消息响应，角色为助手
    message = ChatMessageResponse(
        role="assistant",
        content=response["text"],
    )
    # 创建聊天完成响应选择数据
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
    )
    # 验证和累加任务使用信息
    task_usage = UsageInfo.model_validate(response["usage"])
    for usage_key, usage_value in task_usage.model_dump().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)
    # 返回聊天完成响应
    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion", usage=usage)


# 定义一个预测函数，生成聊天完成的响应
def predict(model_id: str, params: dict):
    # 声明全局变量，模型和分词器
    global model, tokenizer

    # 创建聊天完成响应流选择数据
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    # 创建聊天完成响应的片段
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    # 生成响应的 JSON 格式
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    # 初始化之前的文本为空
    previous_text = ""
    # 遍历生成的响应流，获取每个新的响应
        for new_response in generate_stream_glm4v(model, tokenizer, params):
            # 从新响应中提取解码后的文本内容
            decoded_unicode = new_response["text"]
            # 计算与之前文本的差异部分
            delta_text = decoded_unicode[len(previous_text):]
            # 更新之前的文本为当前的解码文本
            previous_text = decoded_unicode
            # 创建一个 DeltaMessage 对象，包含差异文本和角色信息
            delta = DeltaMessage(content=delta_text, role="assistant")
            # 创建选择数据对象，表示响应中的一个选项
            choice_data = ChatCompletionResponseStreamChoice(index=0, delta=delta)
            # 创建一个聊天完成响应对象，包含模型ID和选择数据
            chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
            # 生成并返回 JSON 格式的聊天完成响应，排除未设置的值
            yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    
    # 创建一个空的 DeltaMessage 对象，用于最后的响应
        choice_data = ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage())
        # 创建最后的聊天完成响应对象
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        # 生成并返回 JSON 格式的聊天完成响应，排除未设置的值
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))
# 生成基于 GLM-4v-9b 模型的响应，处理聊天历史和图像数据（如果有），并调用模型生成响应
def generate_glm4v(model: AutoModel, tokenizer: AutoTokenizer, params: dict):
    # 初始化响应变量为 None
    response = None

    # 通过生成器函数生成模型响应，使用 for 循环遍历每个生成的响应
    for response in generate_stream_glm4v(model, tokenizer, params):
        pass  # 在此处不做任何处理，仅用于消耗生成器
    # 返回生成的响应
    return response


# 处理历史消息以提取文本，识别最后的用户查询，并将 base64 编码的图像 URL 转换为 PIL 图像
def process_history_and_images(messages: List[ChatMessageInput]) -> Tuple[
    Optional[str], Optional[List[Tuple[str, str]]], Optional[List[Image.Image]]]:
    """
    Args:
        messages(List[ChatMessageInput]): ChatMessageInput 对象的列表。
    return: 包含三个元素的元组：
             - 最后一个用户查询的字符串。
             - 格式化为模型所需的元组列表的文本历史。
             - 从消息中提取的 PIL 图像对象的列表。
    """

    # 初始化格式化历史、图像列表和最后用户查询
    formatted_history = []
    image_list = []
    last_user_query = ''

    # 遍历每条消息及其索引
    for i, message in enumerate(messages):
        role = message.role  # 获取消息的角色
        content = message.content  # 获取消息内容

        if isinstance(content, list):  # 如果内容是列表，处理文本
            # 将内容中的文本连接为一个字符串
            text_content = ' '.join(item.text for item in content if isinstance(item, TextContent))
        else:
            text_content = content  # 否则直接使用内容

        if isinstance(content, list):  # 如果内容是列表，处理图像
            for item in content:
                if isinstance(item, ImageUrlContent):  # 如果项是图像 URL 内容
                    image_url = item.image_url.url  # 获取图像 URL
                    if image_url.startswith("data:image/jpeg;base64,"):  # 如果是 base64 编码图像
                        # 解码 base64 编码的图像数据
                        base64_encoded_image = image_url.split("data:image/jpeg;base64,")[1]
                        image_data = base64.b64decode(base64_encoded_image)  # 解码
                        image = Image.open(BytesIO(image_data)).convert('RGB')  # 转换为 RGB 图像
                    else:  # 如果是常规 URL
                        response = requests.get(image_url, verify=False)  # 获取图像数据
                        image = Image.open(BytesIO(response.content)).convert('RGB')  # 转换为 RGB 图像
                    image_list.append(image)  # 将图像添加到图像列表中

        if role == 'user':  # 如果角色是用户
            if i == len(messages) - 1:  # 如果是最后一条用户消息
                last_user_query = text_content  # 更新最后用户查询
            else:
                formatted_history.append((text_content, ''))  # 将文本内容添加到格式化历史中
        elif role == 'assistant':  # 如果角色是助手
            if formatted_history:  # 如果格式化历史不为空
                if formatted_history[-1][1] != '':  # 检查最后的查询是否已经有回答
                    assert False, f"the last query is answered. answer again. {formatted_history[-1][0]}, {formatted_history[-1][1]}, {text_content}"
                formatted_history[-1] = (formatted_history[-1][0], text_content)  # 更新助手的回答
            else:
                assert False, f"assistant reply before user"  # 如果助手在用户之前回复，触发错误
        else:
            assert False, f"unrecognized role: {role}"  # 如果角色不被识别，触发错误

    # 返回最后用户查询、格式化历史和图像列表
    return last_user_query, formatted_history, image_list


# 使用 PyTorch 的推理模式定义生成流
@torch.inference_mode()
def generate_stream_glm4v(model: AutoModel, tokenizer: AutoTokenizer, params: dict):
    # 从参数中提取消息列表
        messages = params["messages"]
        # 获取温度参数，默认值为1.0
        temperature = float(params.get("temperature", 1.0))
        # 获取重复惩罚参数，默认值为1.0
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        # 获取top_p参数，默认值为1.0
        top_p = float(params.get("top_p", 1.0))
        # 获取最大新标记数，默认值为256
        max_new_tokens = int(params.get("max_tokens", 256))
        # 处理历史消息和图片，返回查询、历史记录和图片列表
        query, history, image_list = process_history_and_images(messages)
    
        # 初始化输入列表
        inputs = []
        # 遍历历史记录，按索引和消息对进行迭代
        for idx, (user_msg, model_msg) in enumerate(history):
            # 如果是最后一条历史记录且没有模型消息
            if idx == len(history) - 1 and not model_msg:
                # 将用户消息添加到输入列表
                inputs.append({"role": "user", "content": user_msg})
                # 如果有图片且未上传
                if image_list and not uploaded:
                    # 更新输入列表中的最后一条，添加图片
                    inputs[-1].update({"image": image_list[0]})
                    uploaded = True
                # 跳出循环
                break
            # 如果有用户消息
            if user_msg:
                # 添加用户消息到输入列表
                inputs.append({"role": "user", "content": user_msg})
            # 如果有模型消息
            if model_msg:
                # 添加模型消息到输入列表
                inputs.append({"role": "assistant", "content": model_msg})
        # 将查询和图片添加到输入列表
        inputs.append({"role": "user", "content": query, "image": image_list[0]})
    
        # 使用tokenizer处理输入，应用聊天模板
        model_inputs = tokenizer.apply_chat_template(
            inputs,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(next(model.parameters()).device)  # 将模型输入移动到模型所在设备
        # 计算输入回声长度
        input_echo_len = len(model_inputs["input_ids"][0])
        # 初始化文本流迭代器
        streamer = TextIteratorStreamer(
            tokenizer=tokenizer,
            timeout=60.0,
            skip_prompt=True,
            skip_special_tokens=True
        )
        # 准备生成文本的参数
        gen_kwargs = {
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "do_sample": True if temperature > 1e-5 else False,  # 根据温度判断是否采样
            "top_p": top_p if temperature > 1e-5 else 0,  # 根据温度调整top_p
            "top_k": 1,
            'streamer': streamer,  # 使用文本流迭代器
        }
        # 如果温度大于阈值，添加温度参数
        if temperature > 1e-5:
            gen_kwargs["temperature"] = temperature
    
        # 初始化生成文本的变量
        generated_text = ""
    
        # 定义生成文本的函数
        def generate_text():
            with torch.no_grad():  # 禁用梯度计算以节省内存
                model.generate(**model_inputs, **gen_kwargs)  # 调用模型生成文本
    
        # 启动生成文本的线程
        generation_thread = threading.Thread(target=generate_text)
        generation_thread.start()
    
        # 记录输入回声的总长度
        total_len = input_echo_len
        # 从流中获取下一个文本片段
        for next_text in streamer:
            # 将下一个文本片段添加到生成文本中
            generated_text += next_text
            # 更新总长度
            total_len = len(tokenizer.encode(generated_text))
            # 生成并返回当前文本和使用情况
            yield {
                "text": generated_text,
                "usage": {
                    "prompt_tokens": input_echo_len,  # 输入提示的标记数
                    "completion_tokens": total_len - input_echo_len,  # 生成文本的标记数
                    "total_tokens": total_len,  # 总标记数
                },
            }
        # 等待生成线程完成
        generation_thread.join()
    
        # 最终返回生成的文本和使用情况
        yield {
            "text": generated_text,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": total_len - input_echo_len,
                "total_tokens": total_len,
            },
        }
# 垃圾回收，释放未使用的内存
gc.collect()
# 清空 CUDA 缓存以释放显存
torch.cuda.empty_cache()

# 判断是否为主程序入口
if __name__ == "__main__":
    # 从命令行参数获取模型路径
    MODEL_PATH = sys.argv[1]
    # 解析模型路径，处理用户目录符号并获取绝对路径
    model_dir = Path(MODEL_PATH).expanduser().resolve()
    # 检查配置文件是否存在
    if (model_dir / 'adapter_config.json').exists():
        # 导入 JSON 库以读取配置文件
        import json
        # 打开配置文件并以 UTF-8 编码读取内容
        with open(model_dir / 'adapter_config.json', 'r', encoding='utf-8') as file:
            # 加载 JSON 配置为字典
            config = json.load(file)
        # 从预训练模型中加载基础模型，配置自动选择设备
        model = AutoModel.from_pretrained(
            config.get('base_model_name_or_path'),
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=TORCH_TYPE
        )
        # 从预训练模型中加载适配模型
        model = PeftModelForCausalLM.from_pretrained(
            model=model,
            model_id=model_dir,
            trust_remote_code=True,
        )
        # 从预训练模型中加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            config.get('base_model_name_or_path'),
            trust_remote_code=True,
            encode_special_tokens=True
        )
        # 将模型设置为评估模式并移动到指定设备
        model.eval().to(DEVICE)
    else:
        # 从模型路径加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        encode_special_tokens=True
        )
        # 从模型路径加载预训练模型，设置数据类型和设备
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=TORCH_TYPE,
            trust_remote_code=True,
            device_map="auto",
        ).eval().to(DEVICE)

    # 启动 Uvicorn 服务器，监听指定的 IP 和端口
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
```