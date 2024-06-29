# `D:\src\scipysrc\matplotlib\galleries\examples\style_sheets\plot_solarizedlight2.py`

```
"""
==========================
Solarized Light stylesheet
==========================

This shows an example of "Solarized_Light" styling, which
tries to replicate the styles of:

- https://ethanschoonover.com/solarized/
- https://github.com/jrnold/ggthemes
- http://www.pygal.org/en/stable/documentation/builtin_styles.html#light-solarized

and work of:

- https://github.com/tonysyu/mpltools

using all 8 accents of the color palette - starting with blue

Still TODO:

- Create alpha values for bar and stacked charts. .33 or .5
- Apply Layout Rules
"""

# 引入 matplotlib.pyplot 和 numpy 库
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以保证结果可重现性
np.random.seed(19680801)

# 生成 x 轴数据
x = np.linspace(0, 10)

# 使用 'Solarize_Light2' 风格绘制图表
with plt.style.context('Solarize_Light2'):
    # 绘制多条随机波动的曲线
    plt.plot(x, np.sin(x) + x + np.random.randn(50))
    plt.plot(x, np.sin(x) + 2 * x + np.random.randn(50))
    plt.plot(x, np.sin(x) + 3 * x + np.random.randn(50))
    plt.plot(x, np.sin(x) + 4 + np.random.randn(50))
    plt.plot(x, np.sin(x) + 5 * x + np.random.randn(50))
    plt.plot(x, np.sin(x) + 6 * x + np.random.randn(50))
    plt.plot(x, np.sin(x) + 7 * x + np.random.randn(50))
    plt.plot(x, np.sin(x) + 8 * x + np.random.randn(50))
    
    # 设置图表标题
    plt.title('8 Random Lines - Line')
    # 设置 x 轴标签和字体大小
    plt.xlabel('x label', fontsize=14)
    # 设置 y 轴标签和字体大小
    plt.ylabel('y label', fontsize=14)

# 显示图表
plt.show()
```