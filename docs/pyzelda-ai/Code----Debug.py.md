# `.\Zelda-with-Python\Code\Debug.py`

```

# 导入pygame和os模块
import pygame
import os

# 设置当前工作目录为项目所在的目录，用于导入文件（特别是图片）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 初始化pygame
pygame.init()

# 创建字体对象，用于在屏幕上显示调试信息
font = pygame.font.Font(None, 30)

# 定义一个函数用于在屏幕上显示调试信息
def debug(info, y = 10, x = 10):
    # 获取当前显示的表面
    display_surface = pygame.display.get_surface()
    # 创建一个包含调试信息的表面
    debug_surf = font.render(str(info), True, "White")
    # 获取调试信息表面的矩形
    debug_rect = debug_surf.get_rect(topleft = (x, y))
    # 在屏幕上绘制一个黑色矩形
    pygame.draw.rect(display_surface, "Black", debug_rect)
    # 在屏幕上显示调试信息
    display_surface.blit(debug_surf, debug_rect)

```