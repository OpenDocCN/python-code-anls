# `.\Zelda-with-Python\Code\Weapon.py`

```py

# 导入必要的模块
import os
import pygame

# 这是用于导入文件（特别是图片）的部分（这一行将目录更改为项目保存的位置）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 创建武器类，继承自 pygame.sprite.Sprite 类
class Weapon(pygame.sprite.Sprite):
    def __init__(self, player, groups):
        super().__init__(groups)
        self.sprite_type = "weapon"
        # 获取玩家的状态（方向）
        direction = player.status.split("_")[0]

        # 图形
        full_path = f"../Graphics/Weapons/{player.weapon}/{direction}.png"
        # 加载武器图片并转换为透明度格式
        self.image = pygame.image.load(full_path).convert_alpha()

        # 放置
        if direction == "right":
            # 如果方向向右，则将武器放置在玩家矩形的右侧中间位置
            self.rect = self.image.get_rect(midleft = player.rect.midright + pygame.math.Vector2(0, 16))
        elif direction == "left":
            # 如果方向向左，则将武器放置在玩家矩形的左侧中间位置
            self.rect = self.image.get_rect(midright = player.rect.midleft + pygame.math.Vector2(0, 16))
        elif direction == "down":
            # 如果方向向下，则将武器放置在玩家矩形的底部中间位置
            self.rect = self.image.get_rect(midtop = player.rect.midbottom + pygame.math.Vector2(-10, 0))
        else:
            # 如果方向向上，则将武器放置在玩家矩形的顶部中间位置
            self.rect = self.image.get_rect(midbottom = player.rect.midtop + pygame.math.Vector2(-10, 0))

```