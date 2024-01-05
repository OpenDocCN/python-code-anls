# `Code\Magic.py`

```
# 导入pygame模块
import pygame
# 从Settings模块中导入所有内容
from Settings import *
# 从random模块中导入randint函数
from random import randint
# 导入os模块
import os

# 这是用于导入文件（特别是图片）的代码（这行代码将目录更改为项目保存的位置）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 定义魔法玩家类
class MagicPlayer:
    # 初始化方法，接受动画播放器作为参数
    def __init__(self, animation_player):
        # 设置动画播放器属性
        self.animation_player = animation_player
        # 设置声音字典属性，包括"heal"和"flame"两种声音
        self.sounds = {
            "heal": pygame.mixer.Sound("../Audio/Heal.wav"), 
            "flame": pygame.mixer.Sound("../Audio/Fire.wav")
        }

    # 治愈方法，接受玩家、强度、消耗和组作为参数
    def heal(self, player, strength, cost, groups):
        # 如果玩家的能量大于等于消耗
        if player.energy >= cost:
# 播放“治疗”音效
self.sounds["heal"].play()
# 增加玩家的健康值
player.health += strength
# 减少玩家的能量值
player.energy -= cost
# 如果玩家的健康值超过了最大健康值，则将其设置为最大健康值
if player.health >= player.stats["health"]:
    player.health = player.stats["health"]
# 创建“光环”粒子效果
self.animation_player.create_particles("aura", player.rect.center, groups)
# 创建“治疗”粒子效果
self.animation_player.create_particles("heal", player.rect.center + pygame.math.Vector2(0, -60), groups)

# 火焰技能
def flame(self, player, cost, groups):
    # 如果玩家的能量值足够
    if player.energy >= cost:
        # 减少玩家的能量值
        player.energy -= cost
        # 播放“火焰”音效

        self.sounds["flame"].play()

        # 根据玩家的状态确定火焰的方向
        if player.status.split("_")[0] == "right": 
            direction = pygame.math.Vector2(1, 0)
        elif player.status.split("_")[0] == "left": 
            direction = pygame.math.Vector2(-1, 0)
        elif player.status.split("_")[0] == "up": 
            direction = pygame.math.Vector2(0, -1)
        else: 
            direction = pygame.math.Vector2(0, 1)

        # 循环创建火焰效果
        for i in range(1, 6):
            # 如果方向为水平方向
            if direction.x: 
# 根据方向和偏移量计算火焰粒子的 x 坐标
offset_x = (direction.x * i) * TILESIZE
# 根据玩家的中心点位置和偏移量随机生成 x 坐标
x = player.rect.centerx + offset_x + randint(-TILESIZE // 3, TILESIZE // 3)
# 根据玩家的中心点位置随机生成 y 坐标
y = player.rect.centery + randint(-TILESIZE // 3, TILESIZE // 3)
# 使用动画播放器创建火焰粒子，传入位置和粒子组
self.animation_player.create_particles("flame", (x, y), groups)
# 如果是垂直方向
else: 
    # 根据方向和偏移量计算火焰粒子的 y 坐标
    offset_y = (direction.y * i) * TILESIZE
    # 根据玩家的中心点位置和偏移量随机生成 y 坐标
    x = player.rect.centerx + randint(-TILESIZE // 3, TILESIZE // 3)
    y = player.rect.centery + offset_y + randint(-TILESIZE // 3, TILESIZE // 3)
    # 使用动画播放器创建火焰粒子，传入位置和粒子组
    self.animation_player.create_particles("flame", (x, y), groups)
```