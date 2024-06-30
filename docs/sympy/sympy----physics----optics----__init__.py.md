# `D:\src\scipysrc\sympy\sympy\physics\optics\__init__.py`

```
# 定义一个列表，包含所有将在模块中公开的符号名称
__all__ = [
    'TWave',  # 用于波动模块的波对象

    'RayTransferMatrix', 'FreeSpace', 'FlatRefraction', 'CurvedRefraction',
    'FlatMirror', 'CurvedMirror', 'ThinLens', 'GeometricRay', 'BeamParameter',
    'waist2rayleigh', 'rayleigh2waist', 'geometric_conj_ab',
    'geometric_conj_af', 'geometric_conj_bf', 'gaussian_conj',
    'conjugate_gauss_beams',  # 光学系统和高斯光束操作的类和函数

    'Medium',  # 介质类，处理光学介质的性质和行为

    'refraction_angle', 'deviation', 'fresnel_coefficients', 'brewster_angle',
    'critical_angle', 'lens_makers_formula', 'mirror_formula', 'lens_formula',
    'hyperfocal_distance', 'transverse_magnification',  # 光学公式和参数计算的函数

    'jones_vector', 'stokes_vector', 'jones_2_stokes', 'linear_polarizer',
    'phase_retarder', 'half_wave_retarder', 'quarter_wave_retarder',
    'transmissive_filter', 'reflective_filter', 'mueller_matrix',
    'polarizing_beam_splitter',  # 极化光学和光学滤波器的相关函数和类
]

# 导入模块中的波对象
from .waves import TWave

# 导入模块中的光学设计和高斯光束相关的类和函数
from .gaussopt import (RayTransferMatrix, FreeSpace, FlatRefraction,
        CurvedRefraction, FlatMirror, CurvedMirror, ThinLens, GeometricRay,
        BeamParameter, waist2rayleigh, rayleigh2waist, geometric_conj_ab,
        geometric_conj_af, geometric_conj_bf, gaussian_conj,
        conjugate_gauss_beams)

# 导入模块中的介质类
from .medium import Medium

# 导入模块中的光学公式和参数计算的函数
from .utils import (refraction_angle, deviation, fresnel_coefficients,
        brewster_angle, critical_angle, lens_makers_formula, mirror_formula,
        lens_formula, hyperfocal_distance, transverse_magnification)

# 导入模块中的极化光学相关的函数和类
from .polarization import (jones_vector, stokes_vector, jones_2_stokes,
        linear_polarizer, phase_retarder, half_wave_retarder,
        quarter_wave_retarder, transmissive_filter, reflective_filter,
        mueller_matrix, polarizing_beam_splitter)
```