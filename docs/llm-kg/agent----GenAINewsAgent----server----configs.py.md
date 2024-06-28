# `.\agent\GenAINewsAgent\server\configs.py`

```
# 导入操作系统相关的模块
import os
# 从dotenv模块中导入load_dotenv函数，用于加载环境变量
from dotenv import load_dotenv

# 加载环境变量文件中的环境变量
load_dotenv()

# 从环境变量中获取GROQ_API_KEY，用于GROQ API的认证
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# 从环境变量中获取GROQ_MODEL_NAME，用于指定GROQ模型的名称
GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL_NAME")

# 从环境变量中获取NEWS_API_KEY，用于新闻API的认证
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
# 从环境变量中获取NEWS_BASE_URL，用于指定新闻API的基本URL
NEWS_BASE_URL = os.environ.get("NEWS_BASE_URL")

# 从环境变量中获取BRAVE_API_KEY，用于Brave浏览器API的认证
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY")
```