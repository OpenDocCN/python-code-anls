# `D:\src\scipysrc\matplotlib\lib\matplotlib\stackplot.pyi`

```
# 从 matplotlib 库中导入 Axes 类，用于图表的坐标轴
# 从 matplotlib.collections 模块中导入 PolyCollection 类，用于管理多边形集合

# 从 collections.abc 模块中导入 Iterable 类型，用于标识可迭代对象
# 从 typing 模块中导入 Literal 类型，用于指定字面值类型
# 从 numpy.typing 模块中导入 ArrayLike 类型，用于指代类似数组的数据类型
# 从 matplotlib.typing 模块中导入 ColorType 类型，用于表示颜色的数据类型

# 定义 stackplot 函数，用于在给定的 Axes 上绘制堆叠的面积图
def stackplot(
    axes: Axes,                              # 参数 axes: 图表的坐标轴对象
    x: ArrayLike,                            # 参数 x: x 轴的数据，类似数组
    *args: ArrayLike,                        # 可变位置参数 args: 堆叠区域的数据，每个参数类似数组
    labels: Iterable[str] = ...,             # 关键字参数 labels: 区域的标签，可迭代的字符串
    colors: Iterable[ColorType] | None = ..., # 关键字参数 colors: 区域的颜色，可以是可迭代的颜色类型或 None
    hatch: Iterable[str] | str | None = ..., # 关键字参数 hatch: 区域的填充图案，可以是字符串、可迭代的字符串或 None
    baseline: Literal["zero", "sym",         # 关键字参数 baseline: 堆叠的基线类型，只能是 "zero"、"sym"、
                      "wiggle", "weighted_wiggle"] = ..., # "wiggle" 或 "weighted_wiggle"
    **kwargs                                # 其余的关键字参数，用于传递给底层函数
) -> list[PolyCollection]:                   # 返回值类型为 PolyCollection 对象的列表，用于表示绘制的多边形集合
    ...
```