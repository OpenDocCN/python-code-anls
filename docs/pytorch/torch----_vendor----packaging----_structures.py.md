# `.\pytorch\torch\_vendor\packaging\_structures.py`

```
# 定义一个代表正无穷的类型
class InfinityType:
    # 返回对象的字符串表示为"Infinity"
    def __repr__(self) -> str:
        return "Infinity"

    # 返回对象的哈希值
    def __hash__(self) -> int:
        return hash(repr(self))

    # 小于比较方法，始终返回False
    def __lt__(self, other: object) -> bool:
        return False

    # 小于等于比较方法，始终返回False
    def __le__(self, other: object) -> bool:
        return False

    # 等于比较方法，只有与同一类型对象相等时返回True
    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    # 大于比较方法，始终返回True
    def __gt__(self, other: object) -> bool:
        return True

    # 大于等于比较方法，始终返回True
    def __ge__(self, other: object) -> bool:
        return True

    # 负号操作方法，返回负无穷对象NegativeInfinityType的实例
    def __neg__(self: object) -> "NegativeInfinityType":
        return NegativeInfinity


# 创建正无穷对象的全局实例
Infinity = InfinityType()


# 定义一个代表负无穷的类型
class NegativeInfinityType:
    # 返回对象的字符串表示为"-Infinity"
    def __repr__(self) -> str:
        return "-Infinity"

    # 返回对象的哈希值
    def __hash__(self) -> int:
        return hash(repr(self))

    # 小于比较方法，始终返回True
    def __lt__(self, other: object) -> bool:
        return True

    # 小于等于比较方法，始终返回True
    def __le__(self, other: object) -> bool:
        return True

    # 等于比较方法，只有与同一类型对象相等时返回True
    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    # 大于比较方法，始终返回False
    def __gt__(self, other: object) -> bool:
        return False

    # 大于等于比较方法，始终返回False
    def __ge__(self, other: object) -> bool:
        return False

    # 负号操作方法，返回正无穷对象InfinityType的实例
    def __neg__(self: object) -> InfinityType:
        return Infinity


# 创建负无穷对象的全局实例
NegativeInfinity = NegativeInfinityType()
```