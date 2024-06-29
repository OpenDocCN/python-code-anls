# `D:\src\scipysrc\matplotlib\lib\matplotlib\_tri.pyi`

```
# 这是一个使用 C++ 实现的私有模块
# 因此，这些类型存根过于通用，但允许它们作为公共方法的返回类型
from typing import Any, final

# 定义一个 TrapezoidMapTriFinder 类，使用 @final 装饰器使其无法继承
@final
class TrapezoidMapTriFinder:
    # 初始化方法，接受任意位置参数和关键字参数，不返回任何值
    def __init__(self, *args, **kwargs) -> None: ...
    
    # find_many 方法，接受任意位置参数和关键字参数，返回 Any 类型的结果
    def find_many(self, *args, **kwargs) -> Any: ...
    
    # get_tree_stats 方法，接受任意位置参数和关键字参数，返回 Any 类型的结果
    def get_tree_stats(self, *args, **kwargs) -> Any: ...
    
    # initialize 方法，接受任意位置参数和关键字参数，返回 Any 类型的结果
    def initialize(self, *args, **kwargs) -> Any: ...
    
    # print_tree 方法，接受任意位置参数和关键字参数，返回 Any 类型的结果
    def print_tree(self, *args, **kwargs) -> Any: ...

# 定义一个 TriContourGenerator 类，使用 @final 装饰器使其无法继承
@final
class TriContourGenerator:
    # 初始化方法，接受任意位置参数和关键字参数，不返回任何值
    def __init__(self, *args, **kwargs) -> None: ...
    
    # create_contour 方法，接受任意位置参数和关键字参数，返回 Any 类型的结果
    def create_contour(self, *args, **kwargs) -> Any: ...
    
    # create_filled_contour 方法，接受任意位置参数和关键字参数，返回 Any 类型的结果
    def create_filled_contour(self, *args, **kwargs) -> Any: ...

# 定义一个 Triangulation 类，使用 @final 装饰器使其无法继承
@final
class Triangulation:
    # 初始化方法，接受任意位置参数和关键字参数，不返回任何值
    def __init__(self, *args, **kwargs) -> None: ...
    
    # calculate_plane_coefficients 方法，接受任意位置参数和关键字参数，返回 Any 类型的结果
    def calculate_plane_coefficients(self, *args, **kwargs) -> Any: ...
    
    # get_edges 方法，接受任意位置参数和关键字参数，返回 Any 类型的结果
    def get_edges(self, *args, **kwargs) -> Any: ...
    
    # get_neighbors 方法，接受任意位置参数和关键字参数，返回 Any 类型的结果
    def get_neighbors(self, *args, **kwargs) -> Any: ...
    
    # set_mask 方法，接受任意位置参数和关键字参数，返回 Any 类型的结果
    def set_mask(self, *args, **kwargs) -> Any: ...
```