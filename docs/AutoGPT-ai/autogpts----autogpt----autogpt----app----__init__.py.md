# `.\AutoGPT\autogpts\autogpt\autogpt\app\__init__.py`

```py
# 导入 load_dotenv 函数从 dotenv 模块中
from dotenv import load_dotenv

# 将用户的 .env 文件加载到环境变量中，verbose=True 表示输出详细信息，override=True 表示覆盖已存在的环境变量
load_dotenv(verbose=True, override=True)

# 删除 load_dotenv 函数，清理命名空间
del load_dotenv
```