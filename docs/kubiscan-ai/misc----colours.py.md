# `KubiScan\misc\colours.py`

```
# 定义 ANSI 转义码，用于控制终端输出的颜色和样式
'''
\033[  转义码，始终是这个
0 = 样式，0 代表正常
32 = 文本颜色，32 代表亮绿色
40m = 背景颜色，40 代表黑色


文本颜色    代码    文本样式    代码    背景颜色    代码
  黑色         30        无效果    0         黑色         40
  红色         31        粗体        1         红色         41
  绿色         32        下划线    2         绿色         42
  黄色         33        反色1    3         黄色         43
  蓝色         34        反色2    5         蓝色         44
  紫色         35                                  紫色         45
  青色         36                                  青色         46
  白色         37                                  白色         47
'''


# 定义 ANSI 转义码常量，用于控制终端输出的颜色和样式
RED = '\033[0;31;49m'
LIGHTRED = '\033[0;91;49m'
YELLOW = '\033[0;33;49m'
LIGHTYELLOW = '\033[0;93;49m'

WHITE = '\033[0;47;49m'

# 导入命名元组和有序字典
from collections import namedtuple, OrderedDict

# 定义 RGB 颜色常量和颜色字典
Color = namedtuple('RGB', 'red, green, blue')
colors = {}  # 颜色字典


# 定义 RGB 类，用于表示颜色，并提供十六进制格式的方法
class RGB(Color):
    def hex_format(self):
        '''返回十六进制格式的颜色'''
        return '#{:02X}{:02X}{:02X}'.format(self.red, self.green, self.blue)


# 定义几种红色的 RGB 颜色常量
RED1 = RGB(255, 0, 0)
RED2 = RGB(238, 0, 0)
RED3 = RGB(205, 0, 0)
RED4 = RGB(139, 0, 0)

# 将红色常量添加到颜色字典中
colors['red1'] = RED1
colors['red2'] = RED2
colors['red3'] = RED3
colors['red4'] = RED4

# 对颜色字典按键进行排序
colors = OrderedDict(sorted(colors.items(), key=lambda t: t[0]))
```