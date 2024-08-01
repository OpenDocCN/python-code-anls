# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\4835.da0dc28fd35c4aee.js`

```py
# 定义一个名为 read_csv 的函数，接受一个文件名参数 fname
def read_csv(fname):
    # 打开文件 fname 作为 f，并指定模式为只读模式
    with open(fname, 'r') as f:
        # 读取文件 f 的全部内容，并以逗号为分隔符将其分割成列表形式
        data = f.read().split(',')
        # 返回处理后的数据列表
        return data
```