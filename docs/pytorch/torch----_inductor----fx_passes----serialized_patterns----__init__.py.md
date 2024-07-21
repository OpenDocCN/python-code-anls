# `.\pytorch\torch\_inductor\fx_passes\serialized_patterns\__init__.py`

```py
# 定义一个函数 parse_csv，接收一个文件名作为参数
def parse_csv(filename):
    # 打开文件，模式为只读文本模式，获取文件对象
    with open(filename, 'rt') as f:
        # 读取文件内容，按行分割成列表
        lines = f.readlines()
    
    # 获取第一行作为表头，移除末尾的换行符并按逗号分割得到列名列表
    headers = lines[0].strip().split(',')
    
    # 初始化一个空列表，用于存储解析后的每行数据
    data = []
    
    # 遍历除表头外的每一行数据
    for line in lines[1:]:
        # 移除末尾的换行符，按逗号分割得到每行的数据列表
        values = line.strip().split(',')
        # 将列名与数据以字典形式组合，添加到数据列表中
        row = {headers[i]: values[i] for i in range(len(headers))}
        # 将当前行的数据字典添加到数据列表中
        data.append(row)
    
    # 返回解析后的数据列表
    return data
```