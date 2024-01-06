# `KubiScan\misc\colours.py`

```
# 可以使用colorama模块，但需要安装该模块
'''
\033[  转义码，始终相同
0 = 样式，0表示正常
32 = 文本颜色，32表示亮绿色
40m = 背景颜色，40表示黑色

文本颜色	代码	文本样式	代码	背景颜色	代码
  黑色	     30	    无效果	0	     黑色	             40
  红色	     31	    粗体	    1	     红色	             41
  绿色	     32	    下划线	2	     绿色	             42
  黄色	 33	    负片1	3	     黄色	             43
  蓝色	     34	    负片2	5	     蓝色	             44
  紫色	 35			  	       		 紫色	             45
  青色	     36			  	       		 青色	             46
  白色	     37			  	       		 白色	             47
'''
# 定义 ANSI 转义码，用于控制终端输出颜色
RED = '\033[0;31;49m'  # 红色
LIGHTRED = '\033[0;91;49m'  # 亮红色
YELLOW = '\033[0;33;49m'  # 黄色
LIGHTYELLOW = '\033[0;93;49m'  # 亮黄色

WHITE = '\033[0;47;49m'  # 白色

# 提供 RGB 颜色常量和一个格式为 colors[colorname] = CONSTANT 的颜色字典
from collections import namedtuple, OrderedDict

# 定义一个命名元组 Color，包含红、绿、蓝三个属性
Color = namedtuple('RGB', 'red, green, blue')
colors = {}  # 颜色字典，用于存储颜色对象

# 定义 RGB 类，继承自 Color，用于表示颜色，并提供颜色转换方法
class RGB(Color):
    def hex_format(self):
        '''Returns color in hex format'''  # 返回颜色的十六进制格式
# 返回一个表示颜色的十六进制字符串，格式为#RRGGBB
return '#{:02X}{:02X}{:02X}'.format(self.red, self.green, self.blue)

# 创建表示红色的RGB对象
RED1 = RGB(255, 0, 0)
RED2 = RGB(238, 0, 0)
RED3 = RGB(205, 0, 0)
RED4 = RGB(139, 0, 0)

# 将不同红色对象添加到颜色字典中
colors['red1'] = RED1
colors['red2'] = RED2
colors['red3'] = RED3
colors['red4'] = RED4

# 对颜色字典按照键进行排序，返回一个有序字典
colors = OrderedDict(sorted(colors.items(), key=lambda t: t[0]))
```