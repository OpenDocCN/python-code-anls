# `.\chatglm4-finetune\composite_demo\browser\src\server.ts`

```py
# 导入 express 和相关类型
import express, { Express, Request, Response } from 'express';

# 导入自定义浏览器类
import { SimpleBrowser } from './browser';
# 导入配置文件
import config from './config';
# 导入日志工具
import { logger } from './utils';

# 初始化一个记录会话历史的对象
const session_history: Record<string, SimpleBrowser> = {};

# 创建一个 Express 应用实例
const app: Express = express();

# 中间件，解析 JSON 格式的请求体
app.use(express.json());

# 定义 POST 请求的根路由
app.post('/', async (req: Request, res: Response) => {
  # 从请求体中解构出 session_id 和 action
  const {
    session_id,
    action,
  }: {
    session_id: string;
    action: string;
  } = req.body;
  # 记录会话 ID 到日志
  logger.info(`session_id: ${session_id}`);
  # 记录动作到日志
  logger.info(`action: ${action}`);
  
  # 如果 session_history 中没有该 session_id，创建新的 SimpleBrowser 实例
  if (!session_history[session_id]) {
    session_history[session_id] = new SimpleBrowser();
  }

  # 获取对应 session_id 的浏览器实例
  const browser = session_history[session_id];

  try {
    # 执行浏览器动作并返回 JSON 响应
    res.json(await browser.action(action));
  } catch (err) {
    # 记录错误到日志
    logger.error(err);
    # 返回 400 状态码和错误信息
    res.status(400).json(err);
  }
})

# 处理 SIGINT 信号以优雅退出进程
process.on('SIGINT', () => {
  process.exit(0);
});

# 处理未捕获的异常并记录到日志
process.on('uncaughtException', e => {
  logger.error(e);
});

# 从配置中解构出主机和端口
const { HOST, PORT } = config;

# 创建一个自执行的异步函数以启动服务器
(async () => {
  # 监听指定的端口和主机
  app.listen(PORT, HOST, () => {
    # 记录服务器启动信息
    logger.info(`⚡️[server]: Server is running at http://${HOST}:${PORT}`);
    try {
      # 发送 "ready" 信号给进程
      (<any>process).send('ready');
    } catch (err) {}
  });
})();
```