# `D:\src\scipysrc\sympy\sympy\physics\quantum\circuitplot.py`

```
"""Matplotlib based plotting of quantum circuits.

Todo:

* Optimize printing of large circuits.
* Get this to work with single gates.
* Do a better job checking the form of circuits to make sure it is a Mul of
  Gates.
* Get multi-target gates plotting.
* Get initial and final states to plot.
* Get measurements to plot. Might need to rethink measurement as a gate
  issue.
* Get scale and figsize to be handled in a better way.
* Write some tests/examples!
"""

# 引入必要的模块和函数
from __future__ import annotations

from sympy.core.mul import Mul
from sympy.external import import_module
from sympy.physics.quantum.gate import Gate, OneQubitGate, CGate, CGateS

# 定义公开的接口
__all__ = [
    'CircuitPlot',
    'circuit_plot',
    'labeller',
    'Mz',
    'Mx',
    'CreateOneQubitGate',
    'CreateCGate',
]

# 导入 numpy 和 matplotlib，处理可能的导入错误
np = import_module('numpy')
matplotlib = import_module(
    'matplotlib', import_kwargs={'fromlist': ['pyplot']},
    catch=(RuntimeError,))  # This is raised in environments that have no display.

# 如果 numpy 和 matplotlib 都成功导入，则设置 pyplot、Line2D 和 Circle
if np and matplotlib:
    pyplot = matplotlib.pyplot
    Line2D = matplotlib.lines.Line2D
    Circle = matplotlib.patches.Circle

# 类的定义
class CircuitPlot:
    """A class for managing a circuit plot."""

    # 类级别的默认参数设置
    scale = 1.0
    fontsize = 20.0
    linewidth = 1.0
    control_radius = 0.05
    not_radius = 0.15
    swap_delta = 0.05
    labels: list[str] = []
    inits: dict[str, str] = {}
    label_buffer = 0.5

    def __init__(self, c, nqubits, **kwargs):
        """Initialize the CircuitPlot instance.

        Args:
            c (sympy.core.mul.Mul): Quantum circuit to plot.
            nqubits (int): Number of qubits in the circuit.
            **kwargs: Additional parameters for customization.
        """
        # 检查 numpy 和 matplotlib 是否可用
        if not np or not matplotlib:
            raise ImportError('numpy or matplotlib not available.')
        
        # 初始化实例变量
        self.circuit = c
        self.ngates = len(self.circuit.args)
        self.nqubits = nqubits
        self.update(kwargs)
        self._create_grid()
        self._create_figure()
        self._plot_wires()
        self._plot_gates()
        self._finish()

    def update(self, kwargs):
        """Load the kwargs into the instance dict.

        Args:
            kwargs (dict): Keyword arguments for customization.
        """
        self.__dict__.update(kwargs)

    def _create_grid(self):
        """Create the grid of wires."""
        # 创建电路线网格和门网格
        scale = self.scale
        wire_grid = np.arange(0.0, self.nqubits*scale, scale, dtype=float)
        gate_grid = np.arange(0.0, self.ngates*scale, scale, dtype=float)
        self._wire_grid = wire_grid
        self._gate_grid = gate_grid

    def _create_figure(self):
        """Create the main matplotlib figure."""
        # 创建主要的 matplotlib 图形
        self._figure = pyplot.figure(
            figsize=(self.ngates*self.scale, self.nqubits*self.scale),
            facecolor='w',
            edgecolor='w'
        )
        ax = self._figure.add_subplot(
            1, 1, 1,
            frameon=True
        )
        ax.set_axis_off()
        offset = 0.5*self.scale
        ax.set_xlim(self._gate_grid[0] - offset, self._gate_grid[-1] + offset)
        ax.set_ylim(self._wire_grid[0] - offset, self._wire_grid[-1] + offset)
        ax.set_aspect('equal')
        self._axes = ax

    def _plot_wires(self):
        """Plot the wires on the created figure."""
        # 绘制电路的线
        pass  # Placeholder for actual wire plotting code

    def _plot_gates(self):
        """Plot the gates on the created figure."""
        # 绘制电路的门
        pass  # Placeholder for actual gate plotting code

    def _finish(self):
        """Finalize the plot."""
        # 完成绘图
        pass  # Placeholder for finalization code
    # 绘制电路图中的导线
    def _plot_wires(self):
        """Plot the wires of the circuit diagram."""
        # 确定导线的起始和终止位置
        xstart = self._gate_grid[0]
        xstop = self._gate_grid[-1]
        # 定义 x 轴数据范围
        xdata = (xstart - self.scale, xstop + self.scale)
        # 遍历每个量子比特的索引
        for i in range(self.nqubits):
            # 定义 y 轴数据范围
            ydata = (self._wire_grid[i], self._wire_grid[i])
            # 创建线条对象，表示一条导线
            line = Line2D(
                xdata, ydata,
                color='k',  # 线条颜色为黑色
                lw=self.linewidth  # 线条宽度
            )
            # 将线条添加到图形对象中
            self._axes.add_line(line)
            # 如果需要标签
            if self.labels:
                # 初始化标签缓冲
                init_label_buffer = 0
                # 如果量子比特有初始标签，则增加标签缓冲
                if self.inits.get(self.labels[i]):
                    init_label_buffer = 0.25
                # 在图上添加标签
                self._axes.text(
                    xdata[0] - self.label_buffer - init_label_buffer, ydata[0],
                    render_label(self.labels[i], self.inits),  # 渲染标签
                    size=self.fontsize,  # 字体大小
                    color='k',  # 字体颜色为黑色
                    ha='center', va='center'  # 水平和垂直对齐方式
                )
        # 绘制测量后的导线
        self._plot_measured_wires()

    # 绘制测量后的导线
    def _plot_measured_wires(self):
        """Plot the wires after measurements."""
        # 获取已测量的量子比特
        ismeasured = self._measurements()
        # 确定 x 轴终止位置
        xstop = self._gate_grid[-1]
        # 确定导线向上移动的量
        dy = 0.04  # 当导线被双倍时移动的量

        # 绘制测量后的双倍导线
        for im in ismeasured:
            xdata = (self._gate_grid[ismeasured[im]], xstop + self.scale)
            ydata = (self._wire_grid[im] + dy, self._wire_grid[im] + dy)
            line = Line2D(
                xdata, ydata,
                color='k',  # 线条颜色为黑色
                lw=self.linewidth  # 线条宽度
            )
            # 将线条添加到图形对象中
            self._axes.add_line(line)

        # 对于从这些导线出发的任何控制线，也进行双倍处理
        for i, g in enumerate(self._gates()):
            if isinstance(g, (CGate, CGateS)):
                wires = g.controls + g.targets
                for wire in wires:
                    if wire in ismeasured and \
                            self._gate_grid[i] > self._gate_grid[ismeasured[wire]]:
                        ydata = min(wires), max(wires)
                        xdata = self._gate_grid[i] - dy, self._gate_grid[i] - dy
                        line = Line2D(
                            xdata, ydata,
                            color='k',  # 线条颜色为黑色
                            lw=self.linewidth  # 线条宽度
                        )
                        # 将线条添加到图形对象中
                        self._axes.add_line(line)

    # 获取电路图中的所有门
    def _gates(self):
        """Create a list of all gates in the circuit plot."""
        gates = []
        # 如果电路是多重门，则逆序获取每个门
        if isinstance(self.circuit, Mul):
            for g in reversed(self.circuit.args):
                if isinstance(g, Gate):
                    gates.append(g)
        # 如果电路只有一个门，则直接添加
        elif isinstance(self.circuit, Gate):
            gates.append(self.circuit)
        return gates

    # 绘制电路图中的门
    def _plot_gates(self):
        """Iterate through the gates and plot each of them."""
        # 遍历所有门并绘制它们
        for i, gate in enumerate(self._gates()):
            gate.plot_gate(self, i)
    def _measurements(self):
        """Return a dict ``{i:j}`` where i is the index of the wire that has
        been measured, and j is the gate where the wire is measured.
        """
        # 初始化一个空字典，用于记录已测量的线的索引和其测量所在的门的索引
        ismeasured = {}
        
        # 枚举所有门的索引和门对象
        for i, g in enumerate(self._gates()):
            # 检查门对象是否有 measurement 属性
            if getattr(g, 'measurement', False):
                # 对于每个门的目标线
                for target in g.targets:
                    # 如果目标线已经在 ismeasured 中
                    if target in ismeasured:
                        # 更新目标线的索引，只保留最早的门索引
                        if ismeasured[target] > i:
                            ismeasured[target] = i
                    else:
                        # 将目标线添加到 ismeasured 中，记录当前门的索引
                        ismeasured[target] = i
        
        # 返回记录了测量信息的字典
        return ismeasured

    def _finish(self):
        # 禁用裁剪以确保大电路能正常显示
        for o in self._figure.findobj():
            o.set_clip_on(False)

    def one_qubit_box(self, t, gate_idx, wire_idx):
        """Draw a box for a single qubit gate."""
        # 获取单量子门框的坐标
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        
        # 在坐标 (x, y) 处绘制文本框，显示 t
        self._axes.text(
            x, y, t,
            color='k',
            ha='center',
            va='center',
            bbox={"ec": 'k', "fc": 'w', "fill": True, "lw": self.linewidth},
            size=self.fontsize
        )

    def two_qubit_box(self, t, gate_idx, wire_idx):
        """Draw a box for a two qubit gate. Does not work yet.
        """
        # 这个方法暂时不起作用，所以这里没有实际操作
        pass

    def control_line(self, gate_idx, min_wire, max_wire):
        """Draw a vertical control line."""
        # 控制线的 x 坐标
        xdata = (self._gate_grid[gate_idx], self._gate_grid[gate_idx])
        # 控制线的 y 坐标范围
        ydata = (self._wire_grid[min_wire], self._wire_grid[max_wire])
        
        # 创建一条控制线对象
        line = Line2D(
            xdata, ydata,
            color='k',
            lw=self.linewidth
        )
        
        # 将控制线添加到图形中
        self._axes.add_line(line)

    def control_point(self, gate_idx, wire_idx):
        """Draw a control point."""
        # 控制点的坐标
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        # 控制点的半径
        radius = self.control_radius
        
        # 创建一个控制点对象
        c = Circle(
            (x, y),
            radius*self.scale,
            ec='k',
            fc='k',
            fill=True,
            lw=self.linewidth
        )
        
        # 将控制点添加到图形中
        self._axes.add_patch(c)
    # 定义一个方法来绘制 NOT 门，其图形为一个中间带有加号的圆圈。
    def not_point(self, gate_idx, wire_idx):
        """Draw a NOT gates as the circle with plus in the middle."""
        # 获取门的水平位置和导线的垂直位置
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        # 设置圆圈的半径
        radius = self.not_radius
        # 创建一个圆形对象，表示 NOT 门的外观
        c = Circle(
            (x, y),  # 圆心的坐标
            radius,  # 圆的半径
            ec='k',  # 边缘颜色为黑色
            fc='w',  # 填充颜色为白色
            fill=False,  # 不填充圆形区域
            lw=self.linewidth  # 设置线宽
        )
        # 将圆形对象添加到图形坐标系中
        self._axes.add_patch(c)
        # 绘制一个垂直于导线的直线，表示 NOT 门的输出线
        l = Line2D(
            (x, x),  # 直线的 x 坐标范围（两个相同，表示竖直线）
            (y - radius, y + radius),  # 直线的 y 坐标范围
            color='k',  # 直线颜色为黑色
            lw=self.linewidth  # 设置线宽
        )
        # 将直线对象添加到图形坐标系中
        self._axes.add_line(l)

    # 定义一个方法来绘制交换点，其图形为一个十字形状。
    def swap_point(self, gate_idx, wire_idx):
        """Draw a swap point as a cross."""
        # 获取门的水平位置和导线的垂直位置
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        # 设置交换点的大小（十字的大小）
        d = self.swap_delta
        # 创建两条直线对象，表示交换点的外观（构成十字形）
        l1 = Line2D(
            (x - d, x + d),  # 第一条直线的 x 坐标范围
            (y - d, y + d),  # 第一条直线的 y 坐标范围
            color='k',  # 直线颜色为黑色
            lw=self.linewidth  # 设置线宽
        )
        l2 = Line2D(
            (x - d, x + d),  # 第二条直线的 x 坐标范围
            (y + d, y - d),  # 第二条直线的 y 坐标范围
            color='k',  # 直线颜色为黑色
            lw=self.linewidth  # 设置线宽
        )
        # 将两条直线对象添加到图形坐标系中，形成交换点的十字形状
        self._axes.add_line(l1)
        self._axes.add_line(l2)
# 根据给定的电路和量子比特数绘制电路图。

def circuit_plot(c, nqubits, **kwargs):
    """Draw the circuit diagram for the circuit with nqubits.

    Parameters
    ==========

    c : circuit
        The circuit to plot. Should be a product of Gate instances.
    nqubits : int
        The number of qubits to include in the circuit. Must be at least
        as big as the largest ``min_qubits`` of the gates.
    """
    # 调用 CircuitPlot 类来绘制电路图，传入电路 c、量子比特数 nqubits 和其他关键字参数 kwargs
    return CircuitPlot(c, nqubits, **kwargs)

# 用更加灵活的方式渲染标签。

def render_label(label, inits={}):
    """Slightly more flexible way to render labels.

    >>> from sympy.physics.quantum.circuitplot import render_label
    >>> render_label('q0')
    '$\\\\left|q0\\\\right\\\\rangle$'
    >>> render_label('q0', {'q0':'0'})
    '$\\\\left|q0\\\\right\\\\rangle=\\\\left|0\\\\right\\\\rangle$'
    """
    # 获取初始化的值
    init = inits.get(label)
    # 如果有初始化值，则返回带等号的标签格式
    if init:
        return r'$\left|%s\right\rangle=\left|%s\right\rangle$' % (label, init)
    # 否则返回普通标签格式
    return r'$\left|%s\right\rangle$' % label

# 自动为量子电路的线路生成标签。

def labeller(n, symbol='q'):
    """Autogenerate labels for wires of quantum circuits.

    Parameters
    ==========

    n : int
        number of qubits in the circuit.
    symbol : string
        A character string to precede all gate labels. E.g. 'q_0', 'q_1', etc.

    >>> from sympy.physics.quantum.circuitplot import labeller
    >>> labeller(2)
    ['q_1', 'q_0']
    >>> labeller(3,'j')
    ['j_2', 'j_1', 'j_0']
    """
    # 使用列表推导式生成标签列表，从 symbol_n-1 到 symbol_0
    return ['%s_%d' % (symbol,n-i-1) for i in range(n)]

# Mz 类，模拟 Z 测量门。

class Mz(OneQubitGate):
    """Mock-up of a z measurement gate.

    This is in circuitplot rather than gate.py because it's not a real
    gate, it just draws one.
    """
    measurement = True
    gate_name='Mz'
    gate_name_latex='M_z'

# Mx 类，模拟 X 测量门。

class Mx(OneQubitGate):
    """Mock-up of an x measurement gate.

    This is in circuitplot rather than gate.py because it's not a real
    gate, it just draws one.
    """
    measurement = True
    gate_name='Mx'
    gate_name_latex='M_x'

# CreateOneQubitGate 类，用于创建单量子门类。

class CreateOneQubitGate(type):
    def __new__(mcl, name, latexname=None):
        if not latexname:
            latexname = name
        # 返回一个新的类，继承自 OneQubitGate，具有指定的名称和 LaTeX 名称
        return type(name + "Gate", (OneQubitGate,),
            {'gate_name': name, 'gate_name_latex': latexname})

# 创建一个控制门的函数，使用词法闭包来实现。

def CreateCGate(name, latexname=None):
    """Use a lexical closure to make a controlled gate.
    """
    if not latexname:
        latexname = name
    # 创建一个单量子门类
    onequbitgate = CreateOneQubitGate(name, latexname)
    # 定义一个控制门函数，接受控制线路和目标线路作为参数，返回控制门的实例
    def ControlledGate(ctrls,target):
        return CGate(tuple(ctrls),onequbitgate(target))
    return ControlledGate
```