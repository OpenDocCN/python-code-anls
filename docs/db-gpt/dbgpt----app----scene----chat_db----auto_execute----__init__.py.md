# `.\DB-GPT-src\dbgpt\app\scene\chat_db\auto_execute\__init__.py`

```py
# 定义一个名为 parse_config 的函数，接受一个文件名参数
def parse_config(filename):
    # 打开文件，使用 'r' 模式读取文件内容，并使用 with 语句管理文件资源
    with open(filename, 'r') as f:
        # 读取文件内容，并以每行作为列表的元素
        lines = f.readlines()
    
    # 初始化一个空字典，用于存储配置信息
    config = {}
    
    # 遍历文件的每一行
    for line in lines:
        # 去除每行两端的空白字符，并按 '=' 进行分割，将键值对分别存入列表
        key_value = line.strip().split('=')
        
        # 如果列表长度不为 2，则跳过当前循环
        if len(key_value) != 2:
            continue
        
        # 分别获取键和值，并去除空白字符
        key = key_value[0].strip()
        value = key_value[1].strip()
        
        # 将键值对存入配置字典中
        config[key] = value
    
    # 返回解析得到的配置字典
    return config
```