# `.\marker\marker\cleaners\bullets.py`

```
# 导入正则表达式模块
import re

# 定义函数，用于替换文本中的特殊符号为 -
def replace_bullets(text):
    # 定义匹配特殊符号的正则表达式模式
    bullet_pattern = r"(^|[\n ])[•●○■▪▫–—]( )"
    # 使用正则表达式替换特殊符号为 -
    replaced_string = re.sub(bullet_pattern, r"\1-\2", text)
    # 返回替换后的文本
    return replaced_string
```