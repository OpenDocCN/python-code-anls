# `.\Zelda-with-Python\Code\Main.py`

```

# 导入pygame和sys模块
import pygame, sys
# 从Settings模块中导入所有内容
from Settings import *
# 从Level模块中导入Level类
from Level import Level
# 导入os模块
import os

# 更改工作目录到项目所在的目录
os.chdir(os.path.dirname(os.path.abspath(__file__))

# 创建游戏屏幕并调用一些类
class Game:
    def __init__(self):
        # 初始化pygame
        pygame.init()
        # 创建屏幕并设置宽度和高度
        self.screen = pygame.display.set_mode((WIDTH, HEIGTH))
        # 设置窗口标题
        pygame.display.set_caption("Zelda with Python")
        # 加载游戏图标
        pygame_icon = pygame.image.load("../Graphics/Test/Player.png")
        pygame.display.set_icon(pygame_icon)
        # 创建时钟对象
        self.clock = pygame.time.Clock()

        # 创建Level对象
        self.level = Level()

        # 加载音乐
        main_sound = pygame.mixer.Sound("../Audio/Main.ogg")
        main_sound.set_volume(0.5)
        main_sound.play(loops = -1)

    # 游戏运行循环
    def run(self):
        while True:
            # 处理事件
            for event in pygame.event.get():
                # 如果事件类型为退出，则退出游戏
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                # 如果按下键盘上的m键，则切换菜单状态
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m:
                        self.level.toggle_menu()
            
            # 填充屏幕颜色
            self.screen.fill(WATER_COLOR)
            # 运行关卡
            self.level.run()
            # 更新屏幕
            pygame.display.update()
            # 控制帧率
            self.clock.tick(FPS)

# 如果该文件被直接运行，则创建Game对象并运行游戏
if __name__ == "__main__":
    game = Game()
    game.run()

```