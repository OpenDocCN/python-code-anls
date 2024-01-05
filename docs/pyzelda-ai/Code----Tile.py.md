# `.\Zelda-with-Python\Code\Tile.py`

```
# 导入pygame模块
import pygame
# 从Settings模块中导入所有内容
from Settings import *
# 导入os模块
import os

# 将当前工作目录更改为Main.py所在的目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 创建Tile类，继承自pygame.sprite.Sprite类
class Tile(pygame.sprite.Sprite):
    # 初始化方法，接受位置、组、精灵类型和表面作为参数
    def __init__(self, pos, groups, sprite_type, surface = pygame.Surface((TILESIZE, TILESIZE))):
        # 调用父类的初始化方法
        super().__init__(groups)

        # 设置精灵类型
        self.sprite_type = sprite_type
        # 根据精灵类型获取y轴偏移量
        y_offset = HITBOX_OFFSET[sprite_type]
        # 设置精灵的图像
        self.image = surface
        
        # 根据精灵类型设置精灵的矩形位置
        if sprite_type == "object":
            self.rect = self.image.get_rect(topleft = (pos[0], pos[1] - TILESIZE))
        else:
            self.rect = self.image.get_rect(topleft = pos)
        # 设置精灵的碰撞框
        self.hitbox = self.rect.inflate(0, y_offset)
```