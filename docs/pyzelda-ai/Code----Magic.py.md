# `.\Zelda-with-Python\Code\Magic.py`

```py

# 导入pygame模块
import pygame
# 从Settings模块中导入所有内容
from Settings import *
# 从random模块中导入randint函数
from random import randint
# 导入os模块
import os

# 这是用于文件（特别是图像）导入的（这一行将目录更改为项目保存的位置）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 创建魔法玩家类
class MagicPlayer:
    def __init__(self, animation_player):
        # 初始化动画播放器
        self.animation_player = animation_player
        # 初始化声音字典
        self.sounds = {
            "heal": pygame.mixer.Sound("../Audio/Heal.wav"), 
            "flame": pygame.mixer.Sound("../Audio/Fire.wav")
        }

    # 治愈方法
    def heal(self, player, strength, cost, groups):
        # 如果玩家能量大于等于治愈消耗
        if player.energy >= cost:
            # 播放治愈声音
            self.sounds["heal"].play()
            # 增加玩家生命值
            player.health += strength
            # 减少玩家能量
            player.energy -= cost
            # 如果玩家生命值大于等于最大生命值
            if player.health >= player.stats["health"]:
                player.health = player.stats["health"]
            # 创建光环粒子效果
            self.animation_player.create_particles("aura", player.rect.center, groups)
            # 创建治愈粒子效果
            self.animation_player.create_particles("heal", player.rect.center + pygame.math.Vector2(0, -60), groups)

    # 火焰方法
    def flame(self, player, cost, groups):
        # 如果玩家能量大于等于火焰消耗
        if player.energy >= cost:
            # 减少玩家能量
            player.energy -= cost
            # 播放火焰声音
            self.sounds["flame"].play()

            # 根据玩家状态确定火焰方向
            if player.status.split("_")[0] == "right": direction = pygame.math.Vector2(1, 0)
            elif player.status.split("_")[0] == "left": direction = pygame.math.Vector2(-1, 0)
            elif player.status.split("_")[0] == "up": direction = pygame.math.Vector2(0, -1)
            else: direction = pygame.math.Vector2(0, 1)

            # 创建火焰粒子效果
            for i in range(1, 6):
                if direction.x: # 水平
                    offset_x = (direction.x * i) * TILESIZE
                    x = player.rect.centerx + offset_x + randint(-TILESIZE // 3, TILESIZE // 3)
                    y = player.rect.centery + randint(-TILESIZE // 3, TILESIZE // 3)
                    self.animation_player.create_particles("flame", (x, y), groups)
                else: # 垂直
                    offset_y = (direction.y * i) * TILESIZE
                    x = player.rect.centerx + randint(-TILESIZE // 3, TILESIZE // 3)
                    y = player.rect.centery + offset_y + randint(-TILESIZE // 3, TILESIZE // 3)
                    self.animation_player.create_particles("flame", (x, y), groups)

```