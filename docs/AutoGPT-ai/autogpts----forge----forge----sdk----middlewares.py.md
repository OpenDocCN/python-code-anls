# `.\AutoGPT\autogpts\forge\forge\sdk\middlewares.py`

```py
from fastapi import FastAPI

# 定义一个中间件类，用于将代理实例注入到请求范围内
class AgentMiddleware:
    """
    Middleware that injects the agent instance into the request scope.
    """

    def __init__(self, app: FastAPI, agent: "Agent"):
        """
        初始化方法，接收 FastAPI 应用实例和代理实例作为参数

        Args:
            app: The FastAPI app - automatically injected by FastAPI.
            agent: The agent instance to inject into the request scope.

        Examples:
            用法示例
            >>> from fastapi import FastAPI, Request
            >>> from agent_protocol.agent import Agent
            >>> from agent_protocol.middlewares import AgentMiddleware
            >>> app = FastAPI()
            >>> @app.get("/")
            >>> async def root(request: Request):
            >>>     agent = request["agent"]
            >>>     task = agent.db.create_task("Do something.")
            >>>     return {"task_id": a.task_id}
            >>> agent = Agent()
            >>> app.add_middleware(AgentMiddleware, agent=agent)
        """
        self.app = app
        self.agent = agent

    async def __call__(self, scope, receive, send):
        # 将代理实例注入到请求范围内
        scope["agent"] = self.agent
        # 调用 FastAPI 应用实例处理请求
        await self.app(scope, receive, send)
```