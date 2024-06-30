# `D:\src\scipysrc\scikit-learn\sklearn\externals\_packaging\_structures.py`

```
"""Vendoered from
https://github.com/pypa/packaging/blob/main/packaging/_structures.py
"""
# 版权声明，声明版权所有，禁止未经授权的复制、修改和再发布

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.

#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


class InfinityType:
    # 定义 InfinityType 类，表示正无穷大
    def __repr__(self) -> str:
        return "Infinity"

    # 返回 Infinity 的字符串表示形式 "Infinity"

    def __hash__(self) -> int:
        return hash(repr(self))

    # 返回对象的哈希值，基于其字符串表示形式的哈希值

    def __lt__(self, other: object) -> bool:
        return False

    # 比较方法：当前对象与其他对象比较，永远返回 False

    def __le__(self, other: object) -> bool:
        return False

    # 比较方法：当前对象与其他对象比较，永远返回 False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    # 比较方法：判断当前对象是否与另一个对象类型相同

    def __ne__(self, other: object) -> bool:
        return not isinstance(other, self.__class__)

    # 比较方法：判断当前对象是否与另一个对象类型不同

    def __gt__(self, other: object) -> bool:
        return True

    # 比较方法：当前对象与其他对象比较，永远返回 True

    def __ge__(self, other: object) -> bool:
        return True

    # 比较方法：当前对象与其他对象比较，永远返回 True

    def __neg__(self: object) -> "NegativeInfinityType":
        return NegativeInfinity


    # 取反方法：返回 NegativeInfinityType 类型的对象 NegativeInfinity


Infinity = InfinityType()


class NegativeInfinityType:
    # 定义 NegativeInfinityType 类，表示负无穷大
    def __repr__(self) -> str:
        return "-Infinity"

    # 返回 NegativeInfinity 的字符串表示形式 "-Infinity"

    def __hash__(self) -> int:
        return hash(repr(self))

    # 返回对象的哈希值，基于其字符串表示形式的哈希值

    def __lt__(self, other: object) -> bool:
        return True

    # 比较方法：当前对象与其他对象比较，永远返回 True

    def __le__(self, other: object) -> bool:
        return True

    # 比较方法：当前对象与其他对象比较，永远返回 True

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    # 比较方法：判断当前对象是否与另一个对象类型相同

    def __ne__(self, other: object) -> bool:
        return not isinstance(other, self.__class__)

    # 比较方法：判断当前对象是否与另一个对象类型不同

    def __gt__(self, other: object) -> bool:
        return False

    # 比较方法：当前对象与其他对象比较，永远返回 False

    def __ge__(self, other: object) -> bool:
        return False

    # 比较方法：当前对象与其他对象比较，永远返回 False

    def __neg__(self: object) -> InfinityType:
        return Infinity


    # 取反方法：返回 InfinityType 类型的对象 Infinity


NegativeInfinity = NegativeInfinityType()
```