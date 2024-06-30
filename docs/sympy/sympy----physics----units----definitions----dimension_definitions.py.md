# `D:\src\scipysrc\sympy\sympy\physics\units\definitions\dimension_definitions.py`

```
# 导入 SymPy 中的 Dimensions 类，用于处理物理量的维度表示
from sympy.physics.units import Dimension

# 创建一个名为 "angle" 的维度对象，并指定其类型为 Dimension
angle = Dimension(name="angle")  # type: Dimension

# 创建基本维度 (MKS 系统)
# 长度维度，名称为 "length"，符号为 "L"
length = Dimension(name="length", symbol="L")
# 质量维度，名称为 "mass"，符号为 "M"
mass = Dimension(name="mass", symbol="M")
# 时间维度，名称为 "time"，符号为 "T"
time = Dimension(name="time", symbol="T")

# 创建基本维度 (MKSA 系统，不在 MKS 中)
# 电流维度，名称为 "current"，符号为 "I"，类型为 Dimension
current = Dimension(name='current', symbol='I')  # type: Dimension

# 其他基本维度:
# 温度维度，名称为 "temperature"，符号为 "T"，类型为 Dimension
temperature = Dimension("temperature", "T")  # type: Dimension
# 物质量维度，名称为 "amount_of_substance"，类型为 Dimension
amount_of_substance = Dimension("amount_of_substance")  # type: Dimension
# 发光强度维度，名称为 "luminous_intensity"，类型为 Dimension
luminous_intensity = Dimension("luminous_intensity")  # type: Dimension

# 衍生维度 (MKS 系统)
# 速度维度，名称为 "velocity"
velocity = Dimension(name="velocity")
# 加速度维度，名称为 "acceleration"
acceleration = Dimension(name="acceleration")
# 动量维度，名称为 "momentum"
momentum = Dimension(name="momentum")
# 力量维度，名称为 "force"，符号为 "F"
force = Dimension(name="force", symbol="F")
# 能量维度，名称为 "energy"，符号为 "E"
energy = Dimension(name="energy", symbol="E")
# 功率维度，名称为 "power"
power = Dimension(name="power")
# 压力维度，名称为 "pressure"
pressure = Dimension(name="pressure")
# 频率维度，名称为 "frequency"，符号为 "f"
frequency = Dimension(name="frequency", symbol="f")
# 动作维度，名称为 "action"，符号为 "A"
action = Dimension(name="action", symbol="A")
# 面积维度，名称为 "area"
area = Dimension("area")
# 体积维度，名称为 "volume"
volume = Dimension("volume")

# 衍生维度 (MKSA 系统，不在 MKS 中)
# 电压维度，名称为 "voltage"，符号为 "U"，类型为 Dimension
voltage = Dimension(name='voltage', symbol='U')  # type: Dimension
# 阻抗维度，名称为 "impedance"，符号为 "Z"，类型为 Dimension
impedance = Dimension(name='impedance', symbol='Z')  # type: Dimension
# 电导维度，名称为 "conductance"，符号为 "G"，类型为 Dimension
conductance = Dimension(name='conductance', symbol='G')  # type: Dimension
# 电容维度，名称为 "capacitance"，类型为 Dimension
capacitance = Dimension(name='capacitance')  # type: Dimension
# 电感维度，名称为 "inductance"，类型为 Dimension
inductance = Dimension(name='inductance')  # type: Dimension
# 电荷维度，名称为 "charge"，符号为 "Q"，类型为 Dimension
charge = Dimension(name='charge', symbol='Q')  # type: Dimension
# 磁场密度维度，名称为 "magnetic_density"，符号为 "B"，类型为 Dimension
magnetic_density = Dimension(name='magnetic_density', symbol='B')  # type: Dimension
# 磁通量维度，名称为 "magnetic_flux"，类型为 Dimension
magnetic_flux = Dimension(name='magnetic_flux')  # type: Dimension

# 信息论中的维度:
# 信息维度，名称为 "information"，类型为 Dimension
information = Dimension(name='information')  # type: Dimension
```