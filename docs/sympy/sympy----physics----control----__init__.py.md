# `D:\src\scipysrc\sympy\sympy\physics\control\__init__.py`

```
# 导入模块中的一系列类和函数，用于控制系统理论和绘图功能
from .lti import (TransferFunction, Series, MIMOSeries, Parallel, MIMOParallel,
    Feedback, MIMOFeedback, TransferFunctionMatrix, StateSpace, gbt, bilinear, forward_diff,
    backward_diff, phase_margin, gain_margin)
# 导入模块中的一系列函数，用于控制系统的各种响应的数值数据和绘图
from .control_plots import (pole_zero_numerical_data, pole_zero_plot, step_response_numerical_data,
    step_response_plot, impulse_response_numerical_data, impulse_response_plot, ramp_response_numerical_data,
    ramp_response_plot, bode_magnitude_numerical_data, bode_phase_numerical_data, bode_magnitude_plot,
    bode_phase_plot, bode_plot)

# __all__ 是一个列表，用于定义在使用 from module import * 时应该导入的名字
__all__ = ['TransferFunction', 'Series', 'MIMOSeries', 'Parallel',
    'MIMOParallel', 'Feedback', 'MIMOFeedback', 'TransferFunctionMatrix', 'StateSpace',
    'gbt', 'bilinear', 'forward_diff', 'backward_diff', 'phase_margin', 'gain_margin',
    'pole_zero_numerical_data', 'pole_zero_plot', 'step_response_numerical_data',
    'step_response_plot', 'impulse_response_numerical_data', 'impulse_response_plot',
    'ramp_response_numerical_data', 'ramp_response_plot',
    'bode_magnitude_numerical_data', 'bode_phase_numerical_data',
    'bode_magnitude_plot', 'bode_phase_plot', 'bode_plot']
```