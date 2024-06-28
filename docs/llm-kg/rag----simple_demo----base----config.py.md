# `.\rag\simple_demo\base\config.py`

```
# 导入路径操作模块 Path
from pathlib import Path
# 导入 YAML 解析模块
import yaml

# 定义 Config 类，用于处理配置信息
class Config:
    # 初始化方法，初始化配置文件路径和加载配置数据
    def __init__(self):
        # 获取当前脚本的父路径的绝对路径
        self.parent_path = Path().parent.absolute()
        # 打开并安全加载配置文件 config.yaml，将其内容存储在 self.config 中
        self.config = yaml.safe_load(open(f"{self.parent_path}/config.yaml"))
```