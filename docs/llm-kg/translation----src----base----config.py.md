# `.\translation\src\base\config.py`

```
# 导入Path类，用于处理路径操作
# 导入yaml模块，用于读取和解析YAML格式的配置文件

class Config:
    def __init__(self):
        # 获取当前脚本文件的父目录的绝对路径
        self.parent_path = Path().parent.absolute()
        # 打开并安全加载配置文件config.yaml，将其内容解析为Python对象
        self.config = yaml.safe_load(open(f"{self.parent_path}/config.yaml"))
        # 打开并安全加载语言映射文件lang_map.yaml，将其内容解析为Python对象
        self.lang_map = yaml.safe_load(open(f"{self.parent_path}/lang_map.yaml"))
```