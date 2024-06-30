# `D:\src\scipysrc\sympy\sympy\functions\special\tests\__init__.py`

```
# 定义一个函数 parse_config，用于解析给定配置文件的内容并返回解析结果
def parse_config(filename):
    # 初始化一个空字典用于存储配置项和对应的值
    config = {}
    # 打开配置文件，使用 'r' 模式读取文件内容
    with open(filename, 'r') as f:
        # 逐行读取配置文件内容
        for line in f:
            # 去除每行首尾的空白字符，确保清除换行符等不可见字符
            line = line.strip()
            # 如果该行为空或者以 '#' 开头（注释行），则跳过
            if not line or line.startswith('#'):
                continue
            # 使用 '=' 进行分割，得到配置项和对应的值
            key, value = line.split('=')
            # 去除配置项和值两端的空白字符，确保数据的纯净性
            key = key.strip()
            value = value.strip()
            # 将配置项及其对应值存入配置字典中
            config[key] = value
    # 返回解析完成的配置字典
    return config
```