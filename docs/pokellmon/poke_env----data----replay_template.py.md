# `.\PokeLLMon\poke_env\data\replay_template.py`

```py
# 导入 os 模块
import os

# 打开 replay_template.html 文件，使用绝对路径拼接得到文件路径
with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "static", "replay_template.html"
    )
) as f:
    # 读取文件内容并赋值给 REPLAY_TEMPLATE 变量
    REPLAY_TEMPLATE = f.read()
```