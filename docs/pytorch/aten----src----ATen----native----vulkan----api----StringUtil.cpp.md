# `.\pytorch\aten\src\ATen\native\vulkan\api\StringUtil.cpp`

```
# 定义一个类，表示简单的二维向量
class Vector2D:
    # 类的初始化方法，接受两个参数作为向量的初始坐标
    def __init__(self, x, y):
        # 使用参数 x 初始化对象的 x 坐标属性
        self.x = x
        # 使用参数 y 初始化对象的 y 坐标属性
        self.y = y

    # 方法用于计算两个向量的和
    def add(self, other):
        # 返回一个新的向量，其 x 和 y 坐标分别是当前向量与参数向量对应坐标的和
        return Vector2D(self.x + other.x, self.y + other.y)

    # 方法用于计算两个向量的点积
    def dot_product(self, other):
        # 返回两个向量对应坐标乘积的和，即点积结果
        return self.x * other.x + self.y * other.y

# 创建两个向量对象并进行测试
v1 = Vector2D(1, 2)
v2 = Vector2D(3, 4)
# 计算两个向量的和并打印结果
result_sum = v1.add(v2)
# 计算两个向量的点积并打印结果
result_dot = v1.dot_product(v2)
```