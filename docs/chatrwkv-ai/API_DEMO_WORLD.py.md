# `ChatRWKV\API_DEMO_WORLD.py`

```py
# 导入 RWKV 语言模型相关的库
import os, re

# 设置环境变量，开启 RWKV 的即时编译功能
os.environ["RWKV_JIT_ON"] = "1"
# 设置环境变量，开启 RWKV 的 CUDA 加速功能，需要安装 C++ 编译器和 CUDA 库
os.environ["RWKV_CUDA_ON"] = "0"

# 导入 RWKV 模型和工具库
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

# 定义 RWKV 模型文件路径
MODEL_FILE = "/fsx/BlinkDL/HF-MODEL/rwkv-4-world/RWKV-4-World-7B-v1-20230626-ctx4096"

# 创建 RWKV 模型对象，选择 CUDA 加速和使用半精度浮点数
model = RWKV(model=MODEL_FILE, strategy="cuda fp16")
# 创建 RWKV 模型的管道对象，指定词汇表
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

# 打印演示标题
print("\n#### Demo 1: free generation ####\n")

# 定义上下文
ctx = "Assistant: Sure! Here is a Python function to find Elon Musk's current location:"
print(ctx, end="")

# 定义自定义打印函数
def my_print(s):
    print(s, end="", flush=True)

# 定义生成参数
args = PIPELINE_ARGS(
    temperature=1.5,
    top_p=0.3,
    top_k=0,  # top_k = 0 -> ignore top_k
    alpha_frequency=0.2,  # frequency penalty - see https://platform.openai.com/docs/api-reference/parameter-details
    alpha_presence=0.2,  # presence penalty - see https://platform.openai.com/docs/api-reference/parameter-details
    token_ban=[],  # ban the generation of some tokens
    token_stop=[],  # stop generation at these tokens
    chunk_len=256,  # split input into chunks to save VRAM (shorter -> less VRAM, but slower)
)

# 生成文本并调用自定义打印函数
pipeline.generate(ctx, token_count=200, args=args, callback=my_print)
print("\n")
# 打印演示2的标题
print("\n#### Demo 2: single-round Q & A ####\n")

# 定义一个生成器函数，用于生成问题和回答
def my_qa_generator(ctx):
    # 初始化变量
    out_tokens = []
    out_len = 0
    out_str = ""
    occurrence = {}
    state = None
    # 循环999次，生成回答
    for i in range(999):
        # 如果是第一次循环，使用pipeline.model.forward()方法生成输出和状态
        if i == 0:
            out, state = pipeline.model.forward(pipeline.encode(ctx), state)
        # 如果不是第一次循环，使用pipeline.model.forward()方法生成输出和状态
        else:
            out, state = pipeline.model.forward([token], state)

        # 根据出现次数调整输出概率
        for n in occurrence:
            out[n] -= (
                0.4 + occurrence[n] * 0.4
            )  # 由于top_p较低，因此重复惩罚较高

        # 从输出中采样下一个token
        token = pipeline.sample_logits(
            out, temperature=1.0, top_p=0.2
        )  # 采样下一个token

        # 如果token为0，退出循环
        if token == 0:
            break  # 在token [0] = 
    # 创建一个包含对话内容的列表
    chat_rounds = [
        "User: hi",  # 用户发起对话
        "Assistant: Hi. I am your assistant and I will provide expert full response in full details.",  # 助手回复
        "User: "  # 用户发起对话
        + re.sub(r"\n{2,}", "\n", question)  # 使用正则表达式替换多个换行符为一个换行符
        .strip()  # 去除字符串两端的空白字符
        .replace("\r\n", "\n"),  # 替换所有的 \r\n 为 \n
        "Assistant:",  # 助手回复
    ]  # 对话内容列表结束

    # 打印最后两个对话内容，并以空格结尾
    print("\n\n".join(chat_rounds[-2:]), end="")

    # 调用 my_qa_generator 函数，传入整个对话内容
    my_qa_generator("\n\n".join(chat_rounds))
    # 打印一行分隔符
    print("\n" + "=" * 80)
# 打印演示标题：CFG解码
print("\n#### Demo 3: CFG decoding ####\n")

# 定义一个函数，用于生成上下文和非上下文的CFG（上下文自由语法）因子
# 参数with_ctx：包含上下文的因子
# 参数without_ctx：不包含上下文的因子
# 参数cfg_factor：CFG因子，默认为1
def cfg_generator(with_ctx, without_ctx, cfg_factor=1):
    # 初始化输出的标记列表
    out_tokens = []
    # 初始化输出的长度
    out_len = 0
    # 初始化输出的字符串
    out_str = ""
    # 初始化出现次数的字典
    occurrence = {}
    # 初始化包含上下文的状态
    with_ctx_state = None
    # 初始化不包含上下文的状态
    without_ctx_state = None
    # 循环999次，进行文本生成
    for i in range(999):
        # 如果是第一次循环
        if i == 0:
            # 使用上下文进行模型前向传播，获取输出和状态
            with_ctx_out, with_ctx_state = pipeline.model.forward(
                pipeline.encode(with_ctx), with_ctx_state
            )
            # 不使用上下文进行模型前向传播，获取输出和状态
            without_ctx_out, without_ctx_state = pipeline.model.forward(
                pipeline.encode(without_ctx), without_ctx_state
            )
        # 如果不是第一次循环
        else:
            # 使用上下文进行模型前向传播，获取输出和状态
            with_ctx_out, with_ctx_state = pipeline.model.forward(
                [token], with_ctx_state
            )
            # 不使用上下文进行模型前向传播，获取输出和状态
            without_ctx_out, without_ctx_state = pipeline.model.forward(
                [token], without_ctx_state
            )
        # 根据上下文输出和非上下文输出以及配置因子进行解码
        out = with_ctx_out * cfg_factor + without_ctx_out * (1 - cfg_factor)
        # 对出现次数进行惩罚
        for n in occurrence:
            out[n] -= (
                0.4 + occurrence[n] * 0.4
            )  #### 因为 top_p 较低，所以增加更高的重复惩罚
        # 从输出中采样下一个标记
        token = pipeline.sample_logits(
            out, temperature=1.0, top_p=0.2
        )  #### 采样下一个标记
        # 如果标记为0，则退出循环
        if token == 0:
            break  #### 在标记 [0] = 
# 定义包含两个问题的上下文列表
context_list = [ 
    "RWKV (Recurrent Weighted Key-Value) is a type of recurrent neural network that does not require an attention mechanism, resulting in faster speed and lower memory consumption. It also supports GPT (Generative Pre-trained Transformer) mode for parallel training. RWKV was designed and further developed by Peng Bo, inspired by language models like AFT (Attention-Free Transformer), as a large language model.",
    "RWKV 是一种无需注意力机制的循环神经网络，因此速度更快且更省显存。 它还支持GPT 模式并行训练。 RWKV 是由Peng Bo 受AFT（Attention-Free Transformer）等语言模型启发，设计并进一步开发的一种大型语言模型（Large Language Model）",
]
# 定义问题列表
question_list = ["what is RWKV?", "RWKV是什么?"]
# 遍历问题列表和上下文列表，生成对话
for q, ctx in zip(question_list, context_list):
    # 包含上下文的对话
    chat_with_context = [
        "User: hi",
        "Assistant: Hi. I am your assistant and I will provide expert full response in full details.",
        "User: "
        + re.sub(r"\n{2,}", "\n", ctx) + "\n"
        + re.sub(r"\n{2,}", "\n", q)
        .strip()
        .replace("\r\n", "\n"),  #### replace all \n\n and \r\n by \n
        "Assistant:",
    ]  #### dont add space after this final ":"
    # 不包含上下文的对话
    chat_without_context = [
        "User: " + re.sub(r"\n{2,}", "\n", q).strip().replace("\r\n", "\n"),
        "Assistant:",
    ]

    # 输出不包含上下文的自由生成对话
    print("free generation without context, cfg_factor=0\n")

    print("\n\n".join(chat_without_context[-2:]), end="")

    # 调用 cfg_generator 函数，生成对话
    cfg_generator(
        "\n\n".join(chat_with_context),
        "\n\n".join(chat_without_context),
        cfg_factor=0,
    )
    print("\n" + "=" * 80)

    # 输出包含上下文的生成对话，不包含分类器的自由引导，cfg_factor=1
    print(
        "generation with context, without classifier-free guidance, cfg_factor=1\n"
    )

    print("\n\n".join(chat_with_context[-2:]), end="")

    # 调用 cfg_generator 函数，生成对话
    cfg_generator(
        "\n\n".join(chat_with_context),
        "\n\n".join(chat_without_context),
        cfg_factor=1,
    )
    print("\n" + "=" * 80)

    # 输出包含上下文的生成对话，包含分类器的自由引导，cfg_factor=1.5
    print(
        "generation with context, with classifier-free guidance, cfg_factor=1.5\n"
    )

    print("\n\n".join(chat_with_context[-2:]), end="")
    # 调用 cfg_generator 函数，将带上下文的聊天内容和不带上下文的聊天内容合并成一个字符串，并传入 cfg_factor 参数为 1.5
    cfg_generator(
        "\n\n".join(chat_with_context),
        "\n\n".join(chat_without_context),
        cfg_factor=1.5,
    )
    # 打印分隔线
    print("\n" + "=" * 80)
```