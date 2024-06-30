# `D:\src\scipysrc\sympy\sympy\plotting\pygletplot\plot_object.py`

```
# 定义一个名为 PlotObject 的类，用作可以在绘图中显示的对象的基类
class PlotObject:
    """
    Base class for objects which can be displayed in
    a Plot.
    """
    
    # 类属性，表示该对象是否可见，默认为 True
    visible = True

    # 私有方法，用于绘制对象，仅当对象可见时才调用实例方法 draw()
    def _draw(self):
        if self.visible:
            self.draw()

    # 实例方法，用于实际绘制对象的具体实现，应在子类中重写
    def draw(self):
        """
        OpenGL rendering code for the plot object.
        Override in base class.
        """
        pass
```