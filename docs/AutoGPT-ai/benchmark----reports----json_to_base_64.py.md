# `.\AutoGPT\benchmark\reports\json_to_base_64.py`

```py
# 导入 base64 和 json 模块
import base64
import json

# 从文件中加载 JSON 数据
with open("secrets.json", "r") as f:
    data = json.load(f)

# 将 JSON 对象转换为字符串
json_string = json.dumps(data)

# 将字符串编码为字节
json_bytes = json_string.encode("utf-8")

# 将字节转换为 base64 字符串
base64_string = base64.b64encode(json_bytes).decode("utf-8")

# 打印 base64 字符串
print(base64_string)
```