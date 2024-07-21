# `.\pytorch\test\jit\_imported_class_test\__init__.py`

```
# 定义一个名为 read_csv 的函数，接受一个参数 filename
def read_csv(filename):
    # 打开文件 filename 作为 f，模式为只读模式（'r'）
    with open(filename, 'r') as f:
        # 从文件对象 f 中读取所有行，返回一个包含所有行的列表
        lines = f.readlines()
    
    # 初始化一个空列表，用于存储解析后的 CSV 数据
    csv_data = []
    
    # 遍历 lines 列表中的每一行
    for line in lines:
        # 去除行末尾的换行符，并以逗号为分隔符，将行拆分成一个字段列表
        fields = line.strip().split(',')
        # 将拆分后的字段列表添加到 csv_data 列表中
        csv_data.append(fields)
    
    # 返回解析后的 CSV 数据
    return csv_data
```