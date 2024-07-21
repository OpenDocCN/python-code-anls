# `.\pytorch\torch\fx\experimental\refinement_types.py`

```
# 定义一个名为 Equality 的类，用于表示等式
class Equality:
    # 初始化方法，接受左右两个操作数 lhs 和 rhs
    def __init__(self, lhs, rhs):
        self.lhs = lhs  # 将参数 lhs 存储在实例变量中
        self.rhs = rhs  # 将参数 rhs 存储在实例变量中

    # 返回对象的字符串表示形式，格式为 "lhs = rhs"
    def __str__(self):
        return f'{self.lhs} = {self.rhs}'

    # 返回对象的字符串表示形式，与 __str__ 方法相同
    def __repr__(self):
        return f'{self.lhs} = {self.rhs}'

    # 判断当前对象与另一个对象是否相等
    def __eq__(self, other):
        # 如果 other 是 Equality 类型的对象，则比较 lhs 和 rhs 是否相等
        if isinstance(other, Equality):
            return self.lhs == other.lhs and self.rhs == other.rhs
        else:
            return False
```