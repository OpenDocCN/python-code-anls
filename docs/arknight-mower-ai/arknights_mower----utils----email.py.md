# `arknights-mower\arknights_mower\utils\email.py`

```
# 导入所需的模块
from jinja2 import Environment, FileSystemLoader, select_autoescape
import os
import sys

# 检查是否是打包后的可执行文件，并获取模板目录路径
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    # 如果是打包后的可执行文件，则使用打包后的模板目录路径
    template_dir = os.path.join(
        sys._MEIPASS,
        "arknights_mower",
        "__init__",
        "templates",
        "email",
    )
else:
    # 如果不是打包后的可执行文件，则使用当前工作目录下的模板目录路径
    template_dir = os.path.join(
        os.getcwd(),
        "arknights_mower",
        "templates",
        "email",
    )

# 创建模板环境
env = Environment(
    loader=FileSystemLoader(template_dir),  # 设置模板加载器的目录为模板目录路径
    autoescape=select_autoescape(),  # 设置自动转义
)

# 获取各个模板对象
task_template = env.get_template("task.html")
maa_template = env.get_template("maa.html")
recruit_template = env.get_template("recruit_template.html")
recruit_rarity = env.get_template("recruit_rarity.html")
```