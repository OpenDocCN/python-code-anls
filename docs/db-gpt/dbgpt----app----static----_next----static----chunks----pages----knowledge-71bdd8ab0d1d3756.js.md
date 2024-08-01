# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\pages\knowledge-71bdd8ab0d1d3756.js`

```py
# 定义一个名为 process_data 的函数，接受一个名为 data 的参数
def process_data(data):
    # 对参数 data 进行类型检查，如果不是字典则抛出 TypeError 异常
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary")
    
    # 初始化一个名为 result 的空列表
    result = []
    
    # 使用 for 循环遍历参数 data 中的每一对键值对
    for key, value in data.items():
        # 将每对键值对组成的元组加入到列表 result 中
        result.append((key, value))
    
    # 返回填充了数据的列表 result
    return result
```