# `D:\src\scipysrc\sympy\sympy\physics\units\systems\cgs.py`

```
# 导入必要的模块和类
from sympy.core.singleton import S  # 导入SymPy库中的S类，用于创建符号变量
from sympy.functions.elementary.miscellaneous import sqrt  # 导入SymPy库中的平方根函数
from sympy.physics.units import (  # 导入SymPy物理单位相关模块和单位
    UnitSystem, centimeter, gram, second, coulomb, charge, speed_of_light, current, mass,
    length, voltage, magnetic_density, magnetic_flux
)
from sympy.physics.units.definitions import coulombs_constant  # 导入库中库伦常数
from sympy.physics.units.definitions.unit_definitions import (  # 导入单位定义相关模块和单位
    statcoulomb, statampere, statvolt, volt, tesla, gauss,
    weber, maxwell, debye, oersted, ohm, farad, henry, erg, ampere, coulomb_constant
)
from sympy.physics.units.systems.mks import dimsys_length_weight_time  # 导入单位系统模块中的长度、重量、时间的系统

# 创建一个表示数字1的符号变量
One = S.One

# 创建一个新的CGS高斯单位系统，并扩展长度、重量、时间的单位系统
dimsys_cgs = dimsys_length_weight_time.extend(
    [],
    new_dim_deps={
        # 派生维度的单位依赖关系
        "impedance": {"time": 1, "length": -1},
        "conductance": {"time": -1, "length": 1},
        "capacitance": {"length": 1},
        "inductance": {"time": 2, "length": -1},
        "charge": {"mass": S.Half, "length": S(3)/2, "time": -1},
        "current": {"mass": One/2, "length": 3*One/2, "time": -2},
        "voltage": {"length": -One/2, "mass": One/2, "time": -1},
        "magnetic_density": {"length": -One/2, "mass": One/2, "time": -1},
        "magnetic_flux": {"length": 3*One/2, "mass": One/2, "time": -1},
    }
)

# 创建一个新的CGS高斯单位系统对象
cgs_gauss = UnitSystem(
    base_units=[centimeter, gram, second],  # 设置基本单位为厘米、克、秒
    units=[],  # 暂未定义额外的单位
    name="cgs_gauss",  # 设定单位系统名称为cgs_gauss
    dimension_system=dimsys_cgs  # 使用上面扩展的长度、重量、时间的单位系统
)

# 设置库伦常数在CGS高斯单位系统中的比例因子为1
cgs_gauss.set_quantity_scale_factor(coulombs_constant, 1)

# 设置静电单位电荷statcoulomb在CGS高斯单位系统中的维度为电荷
cgs_gauss.set_quantity_dimension(statcoulomb, charge)
# 设置静电单位电荷statcoulomb在CGS高斯单位系统中的比例因子
cgs_gauss.set_quantity_scale_factor(statcoulomb, centimeter**(S(3)/2)*gram**(S.Half)/second)

# 设置库伦在CGS高斯单位系统中的维度为电荷
cgs_gauss.set_quantity_dimension(coulomb, charge)

# 设置静电单位电流statampere在CGS高斯单位系统中的维度为电流
cgs_gauss.set_quantity_dimension(statampere, current)
# 设置静电单位电流statampere在CGS高斯单位系统中的比例因子
cgs_gauss.set_quantity_scale_factor(statampere, statcoulomb/second)

# 设置静电单位电压statvolt在CGS高斯单位系统中的维度为电压
cgs_gauss.set_quantity_dimension(statvolt, voltage)
# 设置静电单位电压statvolt在CGS高斯单位系统中的比例因子
cgs_gauss.set_quantity_scale_factor(statvolt, erg/statcoulomb)

# 设置标准单位电压volt在CGS高斯单位系统中的维度为电压
cgs_gauss.set_quantity_dimension(volt, voltage)

# 设置高斯单位磁密度gauss在CGS高斯单位系统中的维度为磁密度
cgs_gauss.set_quantity_dimension(gauss, magnetic_density)
# 设置高斯单位磁密度gauss在CGS高斯单位系统中的比例因子
cgs_gauss.set_quantity_scale_factor(gauss, sqrt(gram/centimeter)/second)

# 设置特斯拉单位在CGS高斯单位系统中的维度为磁密度
cgs_gauss.set_quantity_dimension(tesla, magnetic_density)

# 设置麦克斯韦单位在CGS高斯单位系统中的维度为磁通量
cgs_gauss.set_quantity_dimension(maxwell, magnetic_flux)
# 设置麦克斯韦单位在CGS高斯单位系统中的比例因子
cgs_gauss.set_quantity_scale_factor(maxwell, sqrt(centimeter**3*gram)/second)

# 将SI单位表达为CGS高斯单位系统中的比例因子
cgs_gauss.set_quantity_scale_factor(coulomb, 10*speed_of_light*statcoulomb)
cgs_gauss.set_quantity_scale_factor(ampere, 10*speed_of_light*statcoulomb/second)
cgs_gauss.set_quantity_scale_factor(volt, 10**6/speed_of_light*statvolt)
cgs_gauss.set_quantity_scale_factor(weber, 10**8*maxwell)
cgs_gauss.set_quantity_scale_factor(tesla, 10**4*gauss)
cgs_gauss.set_quantity_scale_factor(debye, One/10**18*statcoulomb*centimeter)
cgs_gauss.set_quantity_scale_factor(oersted, sqrt(gram/centimeter)/second)
cgs_gauss.set_quantity_scale_factor(ohm, 10**5/speed_of_light**2*second/centimeter)
# 设置 CGS 高斯单位制的电容量量纲的比例因子
cgs_gauss.set_quantity_scale_factor(farad, One/10**5*speed_of_light**2*centimeter)

# 设置 CGS 高斯单位制的电感量量纲的比例因子
cgs_gauss.set_quantity_scale_factor(henry, 10**5/speed_of_light**2/centimeter*second**2)

# 设置库仑常数的量纲为 1
cgs_gauss.set_quantity_dimension(coulomb_constant, 1)

# 设置库仑常数的比例因子为 1
cgs_gauss.set_quantity_scale_factor(coulomb_constant, 1)

# __all__ 列表定义，指定在模块中导入时可用的符号列表
__all__ = [
    'ohm', 'tesla', 'maxwell', 'speed_of_light', 'volt', 'second', 'voltage',
    'debye', 'dimsys_length_weight_time', 'centimeter', 'coulomb_constant',
    'farad', 'sqrt', 'UnitSystem', 'current', 'charge', 'weber', 'gram',
    'statcoulomb', 'gauss', 'S', 'statvolt', 'oersted', 'statampere',
    'dimsys_cgs', 'coulomb', 'magnetic_density', 'magnetic_flux', 'One',
    'length', 'erg', 'mass', 'coulombs_constant', 'henry', 'ampere',
    'cgs_gauss',
]
```