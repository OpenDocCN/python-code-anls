# `D:\src\scipysrc\sympy\sympy\functions\elementary\benchmarks\__init__.py`

```
# 导入标准库中的 hashlib 模块，用于生成哈希值
import hashlib

# 定义函数 calculate_hash，接收一个字符串参数 s
def calculate_hash(s):
    # 创建一个名为 m 的新的哈希对象
    m = hashlib.sha256()
    # 将参数 s 编码为 UTF-8 格式并更新哈希对象 m
    m.update(s.encode('utf-8'))
    # 返回哈希对象 m 的十六进制表示的哈希值
    return m.hexdigest()
```