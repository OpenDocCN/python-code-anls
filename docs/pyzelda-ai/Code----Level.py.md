# `.\Zelda-with-Python\Code\Level.py`

```

# 导入pygame模块
import pygame
# 从Settings模块中导入所有内容
from Settings import *
# 从Tile模块中导入Tile类
from Tile import Tile
# 从Player模块中导入Player类
from Player import Player
# 从Debug模块中导入debug函数
from Debug import debug
# 从Support模块中导入所有内容
from Support import *
# 从random模块中导入choice和randint函数
from random import choice, randint
# 从Weapon模块中导入Weapon类
from Weapon import Weapon
# 从UI模块中导入UI类
from UI import UI
# 从Enemy模块中导入Enemy类
from Enemy import Enemy
# 从Particles模块中导入AnimationPlayer类
from Particles import AnimationPlayer
# 从Magic模块中导入MagicPlayer类
from Magic import MagicPlayer
# 从Upgrade模块中导入Upgrade类
from Upgrade import Upgrade
# 导入os模块
import os

# 用于文件（特别是图片）导入的代码（这行代码将目录更改为项目保存的位置）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 创建YSortCameraGroup类，继承自pygame.sprite.Group类
class YSortCameraGroup(pygame.sprite.Group):
    def __init__(self):
        # 通用设置
        super().__init__()
        self.display_surface = pygame.display.get_surface()
        self.half_width = self.display_surface.get_size()[0] // 2
        self.half_height = self.display_surface.get_size()[1] // 2
        self.offset = pygame.math.Vector2()

        # 创建地板
        self.floor_surf = pygame.image.load("../Graphics/Tilemap/Ground.png").convert()
        self.floor_rect = self.floor_surf.get_rect(topleft = (0, 0))

    # 自定义绘制方法
    def custom_draw(self, player):
        # 获取偏移量
        self.offset.x = player.rect.centerx - self.half_width
        self.offset.y = player.rect.centery - self.half_height

        # 绘制地板
        floor_offset_pos = self.floor_rect.topleft - self.offset
        self.display_surface.blit(self.floor_surf, floor_offset_pos)

        # 对精灵进行排序绘制
        for sprite in sorted(self.sprites(), key = lambda sprite: sprite.rect.centery):
            offset_pos = sprite.rect.topleft - self.offset
            self.display_surface.blit(sprite.image, offset_pos)

    # 更新敌人位置
    def enemy_update(self, player):
        enemy_sprites = [sprite for sprite in self.sprites() if hasattr(sprite, "sprite_type") and sprite.sprite_type == "enemy"]
        for enemy in enemy_sprites:
            enemy.enemy_update(player)

```