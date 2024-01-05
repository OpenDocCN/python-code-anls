# `Code\Debug.py`

```
# 导入pygame和os模块
import pygame
import os

# 设置当前工作目录为项目所在的目录（用于导入文件，特别是图片）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 初始化pygame
pygame.init()

# 创建字体对象，设置字体大小为30
font = pygame.font.Font(None, 30)

# 定义一个用于调试显示信息的函数，参数info为要显示的信息，y和x为显示位置的坐标
def debug(info, y = 10, x = 10):
    # 获取当前显示窗口的表面对象
    display_surface = pygame.display.get_surface()
    # 创建一个包含调试信息的表面对象
    debug_surf = font.render(str(info), True, "White")
    # 获取调试信息表面对象的矩形区域
    debug_rect = debug_surf.get_rect(topleft = (x, y))
    # 在显示窗口上绘制一个黑色矩形
    pygame.draw.rect(display_surface, "Black", debug_rect)
    # 在显示窗口上显示调试信息
    display_surface.blit(debug_surf, debug_rect)
```