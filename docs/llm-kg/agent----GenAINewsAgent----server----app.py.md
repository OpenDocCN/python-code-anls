# `.\agent\GenAINewsAgent\server\app.py`

```
# 导入 FastAPI 框架
from fastapi import FastAPI
# 导入 StreamingResponse 用于返回流式响应
from fastapi.responses import StreamingResponse
# 导入 CORS 中间件处理跨域请求
from fastapi.middleware.cors import CORSMiddleware
# 导入 uvicorn 用于运行 ASGI 应用
import uvicorn
# 导入 Union 和 Literal 用于类型注解
from typing import Union, Literal

# 从 agent 模块中导入 newsAgent 类或函数
from agent import newsAgent

# 创建 FastAPI 应用实例
app = FastAPI()

# 设置允许的跨域来源
origins = ["*"]

# 添加 CORS 中间件到应用，允许所有来源、凭证、所有 HTTP 方法和所有 HTTP 头部
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义根路径的异步处理函数，返回一个简单的 JSON 响应
@app.get("/")
async def index():
    return {"ok": True}

# 定义 /api/news 路径的异步处理函数，接收查询参数 query，并返回一个流式响应
@app.get("/api/news")
async def api_news(query: str):
    return StreamingResponse(newsAgent(query), media_type="text/event-stream")

# 当该脚本直接运行时，使用 uvicorn 启动 FastAPI 应用
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8899, reload=True)
```