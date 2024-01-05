# `Code\Main.py`

```
# 导入pygame和sys模块
import pygame, sys
# 从Settings模块中导入设置
from Settings import *
# 从Level模块中导入Level类
from Level import Level
# 导入os模块
import os

# 用于导入文件（特别是图片）的文件导入（这一行将目录更改为项目保存的位置）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 创建一个屏幕并调用一些类
class Game:
    def __init__(self):
        # 通用设置
        pygame.init()
        # 设置屏幕大小
        self.screen = pygame.display.set_mode((WIDTH, HEIGTH))
# 设置窗口标题为 "Zelda with Python"
pygame.display.set_caption("Zelda with Python")
# 加载游戏图标
pygame_icon = pygame.image.load("../Graphics/Test/Player.png")
# 设置窗口图标
pygame.display.set_icon(pygame_icon)
# 创建一个时钟对象，用于控制游戏帧率
self.clock = pygame.time.Clock()

# 创建一个关卡对象
self.level = Level()

# 加载并播放背景音乐
main_sound = pygame.mixer.Sound("../Audio/Main.ogg")
# 设置音量
main_sound.set_volume(0.5)
# 循环播放背景音乐
main_sound.play(loops = -1)

# 游戏主循环
def run(self):
    while True:
        # 处理事件
        for event in pygame.event.get():
            # 如果点击了关闭窗口按钮，则退出游戏
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # 如果按下了键盘按键
            if event.type == pygame.KEYDOWN:
                # 如果按下了 "m" 键
                if event.key == pygame.K_m:
# 如果当前的游戏状态是菜单状态，切换菜单
self.level.toggle_menu()

# 用水的颜色填充屏幕
self.screen.fill(WATER_COLOR)

# 运行当前关卡
self.level.run()

# 更新屏幕显示
pygame.display.update()

# 控制游戏循环的速度
self.clock.tick(FPS)

# 如果当前脚本被直接执行，则创建游戏对象并运行游戏
if __name__ == "__main__":
    game = Game()
    game.run()
```