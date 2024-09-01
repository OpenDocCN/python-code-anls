# `.\flux\src\flux\__main__.py`

```py
# 从同一目录下的 cli 模块导入 app 函数
from .cli import app

# 如果当前模块是主程序，则执行 app 函数
if __name__ == "__main__":
    app()
```