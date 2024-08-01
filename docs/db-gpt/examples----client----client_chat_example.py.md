# `.\DB-GPT-src\examples\client\client_chat_example.py`

```py
# 引入 asyncio 库，用于异步编程
import asyncio

# 从 dbgpt 客户端模块中导入 Client 类
from dbgpt.client import Client

# 定义异步函数 main，作为程序的入口点
async def main():
    # 设定调试服务的 API 密钥
    DBGPT_API_KEY = "dbgpt"
    # 创建一个 Client 对象，使用指定的 API 密钥
    client = Client(api_key=DBGPT_API_KEY)
    
    # 调用客户端的 chat 方法，与 chatgpt_proxyllm 模型进行对话，并等待返回结果
    data = await client.chat(model="chatgpt_proxyllm", messages="hello")
    print(data)  # 打印对话结果

    # 注释掉的部分是示例代码，展示了如何使用 chat_stream 方法
    # async for data in client.chat_stream(
    #     model="chatgpt_proxyllm",
    #     messages="hello",
    # ):
    # print(data)

    # 注释掉的部分是另一种与模型交互的示例，使用 chat 方法并打印返回的 JSON 结果
    # res = await client.chat(model="chatgpt_proxyllm", messages="hello")
    # print(res)

# 如果当前脚本作为主程序运行，则调用 asyncio 库运行 main 函数
if __name__ == "__main__":
    asyncio.run(main())
```