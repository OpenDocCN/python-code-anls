# `D:\src\scipysrc\matplotlib\lib\matplotlib\projections\geo.pyi`

```py
# 从 matplotlib.axes 中导入 Axes 类，用于创建自定义的地理坐标轴
# 从 matplotlib.ticker 中导入 Formatter 类，用于格式化坐标轴上的刻度
# 从 matplotlib.transforms 中导入 Transform 类，用于坐标变换

from matplotlib.axes import Axes
from matplotlib.ticker import Formatter
from matplotlib.transforms import Transform

# 导入类型提示模块中的 Any 和 Literal 类型
from typing import Any, Literal

# 定义一个自定义的地理坐标轴类 GeoAxes，继承自 matplotlib 的 Axes 类
class GeoAxes(Axes):
    
    # 内部定义一个 ThetaFormatter 类，继承自 Formatter 类，用于角度格式化
    class ThetaFormatter(Formatter):
        def __init__(self, round_to: float = ...) -> None: ...
        def __call__(self, x: float, pos: Any | None = ...): ...
    
    # 类属性 RESOLUTION，表示地理坐标轴的分辨率
    RESOLUTION: float
    
    # 获取 X 轴变换对象的方法，根据参数 which 确定是刻度1、刻度2还是网格
    def get_xaxis_transform(
        self, which: Literal["tick1", "tick2", "grid"] = ...
    ) -> Transform: ...
    
    # 获取 X 轴文本1的变换对象及对齐方式的元组
    def get_xaxis_text1_transform(
        self, pad: float
    ) -> tuple[
        Transform,
        Literal["center", "top", "bottom", "baseline", "center_baseline"],
        Literal["center", "left", "right"],
    ]: ...
    
    # 获取 X 轴文本2的变换对象及对齐方式的元组
    def get_xaxis_text2_transform(
        self, pad: float
    ) -> tuple[
        Transform,
        Literal["center", "top", "bottom", "baseline", "center_baseline"],
        Literal["center", "left", "right"],
    ]: ...
    
    # 获取 Y 轴变换对象的方法，根据参数 which 确定是刻度1、刻度2还是网格
    def get_yaxis_transform(
        self, which: Literal["tick1", "tick2", "grid"] = ...
    ) -> Transform: ...
    
    # 获取 Y 轴文本1的变换对象及对齐方式的元组
    def get_yaxis_text1_transform(
        self, pad: float
    ) -> tuple[
        Transform,
        Literal["center", "top", "bottom", "baseline", "center_baseline"],
        Literal["center", "left", "right"],
    ]: ...
    
    # 获取 Y 轴文本2的变换对象及对齐方式的元组
    def get_yaxis_text2_transform(
        self, pad: float
    ) -> tuple[
        Transform,
        Literal["center", "top", "bottom", "baseline", "center_baseline"],
        Literal["center", "left", "right"],
    ]: ...
    
    # 设置 X 轴的限制范围，并返回限制的起始和结束值的元组
    def set_xlim(self, *args, **kwargs) -> tuple[float, float]: ...
    
    # 设置 Y 轴的限制范围，并返回限制的起始和结束值的元组
    def set_ylim(self, *args, **kwargs) -> tuple[float, float]: ...
    
    # 格式化坐标点的方法，根据给定的经度和纬度返回格式化后的字符串
    def format_coord(self, lon: float, lat: float) -> str: ...
    
    # 设置经度网格线的方法，参数为度数
    def set_longitude_grid(self, degrees: float) -> None: ...
    
    # 设置纬度网格线的方法，参数为度数
    def set_latitude_grid(self, degrees: float) -> None: ...
    
    # 设置经度网格线结束点的方法，参数为度数
    def set_longitude_grid_ends(self, degrees: float) -> None: ...
    
    # 获取数据比例的方法，返回地理坐标轴的数据比例
    def get_data_ratio(self) -> float: ...
    
    # 判断是否可以缩放的方法，返回布尔值
    def can_zoom(self) -> bool: ...
    
    # 判断是否可以平移的方法，返回布尔值
    def can_pan(self) -> bool: ...
    
    # 开始进行平移操作的方法，接收点击位置的坐标和按钮信息
    def start_pan(self, x, y, button) -> None: ...
    
    # 结束平移操作的方法
    def end_pan(self) -> None: ...
    
    # 执行拖动平移操作的方法，接收按钮、键盘、以及拖动的位置信息
    def drag_pan(self, button, key, x, y) -> None: ...

# 定义一个私有的地理转换类 _GeoTransform，继承自 Transform 类
class _GeoTransform(Transform):
    input_dims: int
    output_dims: int
    
    # 初始化方法，接收分辨率参数
    def __init__(self, resolution: int) -> None: ...

# Aitoff 投影地理坐标轴类，继承自 GeoAxes 类
class AitoffAxes(GeoAxes):
    name: str
    
    # Aitoff 投影的地理转换类 AitoffTransform，继承自 _GeoTransform
    class AitoffTransform(_GeoTransform):
        
        # 返回反转的 Aitoff 投影转换对象的方法
        def inverted(self) -> AitoffAxes.InvertedAitoffTransform: ...

    # 反转的 Aitoff 投影转换类，继承自 _GeoTransform
    class InvertedAitoffTransform(_GeoTransform):
        
        # 返回反转的 Aitoff 投影转换对象的方法
        def inverted(self) -> AitoffAxes.AitoffTransform: ...

# Hammer 投影地理坐标轴类，继承自 GeoAxes 类
class HammerAxes(GeoAxes):
    name: str
    
    # Hammer 投影的地理转换类 HammerTransform，继承自 _GeoTransform
    class HammerTransform(_GeoTransform):
        
        # 返回反转的 Hammer 投影转换对象的方法
        def inverted(self) -> HammerAxes.InvertedHammerTransform: ...

    # 反转的 Hammer 投影转换类，继承自 _GeoTransform
    class InvertedHammerTransform(_GeoTransform):
        
        # 返回反转的 Hammer 投影转换对象的方法
        def inverted(self) -> HammerAxes.HammerTransform: ...

# Mollweide 投影地理坐标轴类，继承自 GeoAxes 类
class MollweideAxes(GeoAxes):
    name: str
    
    # Mollweide 投影的地理转换类 MollweideTransform，继承自 _GeoTransform
    class MollweideTransform(_GeoTransform):
        
        # 返回反转的 Mollweide 投影转换对象的方法
        def inverted(self) -> MollweideAxes.InvertedMollweideTransform: ...
    # 定义一个类 InvertedMollweideTransform，继承自 _GeoTransform 类
    class InvertedMollweideTransform(_GeoTransform):
        # 声明一个方法 inverted，其返回类型为 MollweideAxes.MollweideTransform
        def inverted(self) -> MollweideAxes.MollweideTransform:
            # 此处省略具体的方法实现，使用 '...' 表示
            ...
# 自定义地理坐标系 LambertAxes，继承自 GeoAxes
class LambertAxes(GeoAxes):
    # 类属性：名称
    name: str

    # Lambert 变换类，继承自 _GeoTransform
    class LambertTransform(_GeoTransform):
        # 构造函数，初始化 Lambert 变换对象
        def __init__(
            self, center_longitude: float, center_latitude: float, resolution: int
        ) -> None: ...
        
        # 返回反转的 Lambert 变换对象
        def inverted(self) -> LambertAxes.InvertedLambertTransform: ...

    # 反转的 Lambert 变换类，继承自 _GeoTransform
    class InvertedLambertTransform(_GeoTransform):
        # 构造函数，初始化反转的 Lambert 变换对象
        def __init__(
            self, center_longitude: float, center_latitude: float, resolution: int
        ) -> None: ...
        
        # 返回反转的 Lambert 变换对象
        def inverted(self) -> LambertAxes.LambertTransform: ...

    # 构造函数，初始化 LambertAxes 对象
    def __init__(
        self,
        *args,
        center_longitude: float = ...,
        center_latitude: float = ...,
        **kwargs
    ) -> None: ...
```