# `arknights-mower\arknights_mower\utils\pipe.py`

```py
# 定义装饰器函数，用于向管道发送操作符数据
def push_operators(func):
    # 定义装饰器函数的内部函数，接收参数s和*args
    def func_dec(s,*args):
        # 调用原始函数func，并将结果保存到r中
        r=func(s,*args)
        # 如果Pipe不为空且Pipe的连接不为空
        if Pipe is not None and Pipe.conn is not None:
            # 向管道发送操作符数据
            Pipe.conn.send({'type':'operators','data':s.op_data.operators})
        # 返回原始函数的结果
        return r
    # 返回内部函数
    return func_dec

# 定义管道类
class Pipe:
    # 初始化连接属性为None
    conn = None
```