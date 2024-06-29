# `D:\src\scipysrc\matplotlib\lib\matplotlib\patheffects.pyi`

```py
from collections.abc import Iterable, Sequence
from typing import Any

from matplotlib.backend_bases import RendererBase, GraphicsContextBase
from matplotlib.path import Path
from matplotlib.patches import Patch
from matplotlib.transforms import Transform

from matplotlib.typing import ColorType

class AbstractPathEffect:
    def __init__(self, offset: tuple[float, float] = ...) -> None: ...
    def draw_path(
        self,
        renderer: RendererBase,
        gc: GraphicsContextBase,
        tpath: Path,
        affine: Transform,
        rgbFace: ColorType | None = ...,
    ) -> None: ...

class PathEffectRenderer(RendererBase):
    def __init__(
        self, path_effects: Iterable[AbstractPathEffect], renderer: RendererBase
    ) -> None: ...
    def copy_with_path_effect(self, path_effects: Iterable[AbstractPathEffect]) -> PathEffectRenderer: ...
    def draw_path(
        self,
        gc: GraphicsContextBase,
        tpath: Path,
        affine: Transform,
        rgbFace: ColorType | None = ...,
    ) -> None: ...
    def draw_markers(
        self,
        gc: GraphicsContextBase,
        marker_path: Path,
        marker_trans: Transform,
        path: Path,
        *args,
        **kwargs
    ) -> None: ...
    def draw_path_collection(
        self,
        gc: GraphicsContextBase,
        master_transform: Transform,
        paths: Sequence[Path],
        *args,
        **kwargs
    ) -> None: ...
    def __getattribute__(self, name: str) -> Any: ...

class Normal(AbstractPathEffect): ...
# 描述：普通路径效果类，继承自抽象路径效果类

class Stroke(AbstractPathEffect):
    def __init__(self, offset: tuple[float, float] = ..., **kwargs) -> None: ...
    # 描述：设置笔划路径效果类，包括偏移量和其他关键字参数
    # 注意：覆盖方法中，rgbFace 参数变为非可选项

class withStroke(Stroke): ...
# 描述：继承自 Stroke 类，带有笔划的路径效果类

class SimplePatchShadow(AbstractPathEffect):
    def __init__(
        self,
        offset: tuple[float, float] = ...,
        shadow_rgbFace: ColorType | None = ...,
        alpha: float | None = ...,
        rho: float = ...,
        **kwargs
    ) -> None: ...
    # 描述：简单的补丁阴影路径效果类，包括偏移量、阴影颜色、透明度、rho 参数和其他关键字参数
    # 注意：覆盖方法中，rgbFace 参数变为非可选项

class withSimplePatchShadow(SimplePatchShadow): ...
# 描述：继承自 SimplePatchShadow 类，带有简单补丁阴影的路径效果类

class SimpleLineShadow(AbstractPathEffect):
    def __init__(
        self,
        offset: tuple[float, float] = ...,
        shadow_color: ColorType = ...,
        alpha: float = ...,
        rho: float = ...,
        **kwargs
    ) -> None: ...
    # 描述：简单的线条阴影路径效果类，包括偏移量、阴影颜色、透明度、rho 参数和其他关键字参数
    # 注意：覆盖方法中，rgbFace 参数变为非可选项

class PathPatchEffect(AbstractPathEffect):
    patch: Patch
    def __init__(self, offset: tuple[float, float] = ..., **kwargs) -> None: ...
    # 描述：路径补丁效果类，包括偏移量和其他关键字参数
    # 定义一个方法 draw_path，用于绘制路径
    # 参数说明：
    # - renderer: 渲染器对象，用于执行实际的绘制操作
    # - gc: 图形上下文对象，包含绘制时的各种参数和状态
    # - tpath: 要绘制的路径对象，类型为 Path
    # - affine: 变换矩阵，用于对路径进行变换
    # - rgbFace: 颜色类型对象，表示路径的填充颜色
    # 返回值为 None，表示该方法不返回任何结果
    
    def draw_path(self, renderer: RendererBase, gc: GraphicsContextBase, tpath: Path, affine: Transform, rgbFace: ColorType) -> None: ...  # type: ignore[override]
    # type: ignore[override] 是类型提示的一部分，用于告知类型检查器忽略对该方法覆盖性质的检查
class TickedStroke(AbstractPathEffect):
    # TickedStroke 类继承自 AbstractPathEffect 抽象类，用于实现一个特殊的路径效果

    def __init__(
        self,
        offset: tuple[float, float] = ...,
        spacing: float = ...,
        angle: float = ...,
        length: float = ...,
        **kwargs
    ) -> None:
        # 构造函数初始化 TickedStroke 实例
        # offset: 偏移量元组，指定效果的起始位置
        # spacing: 距离控制点之间的间距
        # angle: 控制点之间连线的角度
        # length: 控制点之间连线的长度
        pass
        # pass 语句表示此处暂无额外逻辑，保持结构完整性

    # rgbFace becomes non-optional
    def draw_path(self, renderer: RendererBase, gc: GraphicsContextBase, tpath: Path, affine: Transform, rgbFace: ColorType) -> None:
        # draw_path 方法用于绘制路径，覆盖了父类 AbstractPathEffect 的方法
        # renderer: 渲染器对象，用于将路径绘制到画布上
        # gc: 图形上下文对象，包含绘制路径所需的图形属性和状态
        # tpath: 要绘制的路径对象
        # affine: 变换对象，用于处理路径的变换
        # rgbFace: 要填充的颜色，已经变为非可选参数
        pass
        # pass 语句表示此处暂无额外逻辑，保持结构完整性

class withTickedStroke(TickedStroke):
    # withTickedStroke 类继承自 TickedStroke 类，扩展了其功能
    pass
    # pass 语句表示此处暂无额外逻辑，保持结构完整性
```