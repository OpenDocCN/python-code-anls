# `.\transcription-and-semantic-search\base\config.py`

```
# 导入必要的模块：Path 用于处理路径操作，yaml 用于加载和解析 YAML 文件
from pathlib import Path
import yaml

# 定义一个 Config 类，用于加载和管理配置信息
class Config:
    # 初始化方法，创建 Config 对象时执行
    def __init__(self):
        # 获取当前文件的父目录的绝对路径
        self.parent_path = Path().parent.absolute()
        # 使用 yaml.safe_load 方法加载并解析指定路径下的 config.yaml 文件，将其内容存储在 self.config 中
        self.config = yaml.safe_load(open(f"{self.parent_path}/config.yaml"))
```