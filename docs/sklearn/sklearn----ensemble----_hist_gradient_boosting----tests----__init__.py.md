# `D:\src\scipysrc\scikit-learn\sklearn\ensemble\_hist_gradient_boosting\tests\__init__.py`

```
# 定义一个名为 process_data 的函数，接收参数 data
def process_data(data):
    # 初始化一个空列表，用于存储处理后的结果
    result = []
    
    # 使用 for 循环遍历参数 data 中的每个元素
    for item in data:
        # 对每个元素进行判断，如果元素大于 0，则将其添加到结果列表中
        if item > 0:
            result.append(item)
    
    # 返回处理后的结果列表
    return result
```