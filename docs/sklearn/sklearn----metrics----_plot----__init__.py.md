# `D:\src\scipysrc\scikit-learn\sklearn\metrics\_plot\__init__.py`

```
# 定义一个名为 read_config 的函数，接收一个文件名参数 fname
def read_config(fname):
    # 打开文件 fname 作为只读模式，并使用 with 语句保证文件操作安全关闭
    with open(fname, 'r') as f:
        # 读取文件的所有行，并存储在列表 lines 中
        lines = f.readlines()
    
    # 初始化一个空字典 config，用于存储配置项和对应的值
    config = {}
    
    # 遍历 lines 中的每一行内容
    for line in lines:
        # 去除每行两端的空白字符（空格、制表符、换行符等）
        line = line.strip()
        
        # 跳过空行
        if not line:
            continue
        
        # 如果行以 '#' 开头，则为注释行，跳过
        if line.startswith('#'):
            continue
        
        # 使用等号 '=' 进行分割，获取配置项和对应的值
        key_value = line.split('=')
        
        # 如果分割结果不符合预期（不包含等号或者等号左侧为空），跳过处理
        if len(key_value) != 2 or not key_value[0]:
            continue
        
        # 分别获取配置项名和对应的值，去除两侧空白字符后存储到 config 字典中
        key = key_value[0].strip()
        value = key_value[1].strip()
        config[key] = value
    
    # 返回解析后的配置字典
    return config
```