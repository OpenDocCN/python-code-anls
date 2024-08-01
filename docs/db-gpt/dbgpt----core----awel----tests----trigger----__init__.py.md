# `.\DB-GPT-src\dbgpt\core\awel\tests\trigger\__init__.py`

```py
# 定义一个名为 ultimate 的类
class ultimate:
    # 类的初始化方法，接受参数 x
    def __init__(self, x):
        # 使用参数 x 初始化对象的属性 x
        self.x = x

    # 定义一个方法 double_x，用于返回对象属性 x 的两倍
    def double_x(self):
        # 返回对象属性 x 的两倍
        return self.x * 2

# 创建一个 ultimate 类的实例对象，参数为 5
obj = ultimate(5)
# 调用实例对象的 double_x 方法，并打印其返回值
print(obj.double_x())
```