# `.\DB-GPT-src\dbgpt\model\proxy\llms\claude.py`

```py
# 导入 ProxyModel 类，该类来自 dbgpt.model.proxy.llms.proxy_model 模块
from dbgpt.model.proxy.llms.proxy_model import ProxyModel

# 定义一个生成器函数 claude_generate_stream，接受以下参数：
# - model: ProxyModel 类型，表示使用的模型
# - tokenizer: 表示分词器对象，未具体指定类型
# - params: 表示参数对象，未具体指定类型
# - device: 表示设备对象，未具体指定类型
# - context_len: 整数类型，默认为 2048，表示上下文长度

def claude_generate_stream(
    model: ProxyModel, tokenizer, params, device, context_len=2048
):
    # 生成器函数内部仅生成一个字符串 "claude LLM was not supported!"，并返回
    yield "claude LLM was not supported!"
```