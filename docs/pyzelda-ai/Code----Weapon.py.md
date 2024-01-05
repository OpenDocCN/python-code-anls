# `.\Zelda-with-Python\Code\Weapon.py`

```
# 导入 os 模块和 pygame 模块
import os
import pygame

# 这是用于导入文件（特别是图片）的代码（这行代码将目录更改为项目保存的位置）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 创建 Weapon 类，继承自 pygame.sprite.Sprite 类
class Weapon(pygame.sprite.Sprite):
    def __init__(self, player, groups):
        # 调用父类的初始化方法
        super().__init__(groups)
        # 设置精灵类型为 "weapon"
        self.sprite_type = "weapon"
        # 获取玩家的状态（方向）
        direction = player.status.split("_")[0]

        # 图形
        # 拼接图片的完整路径
        full_path = f"../Graphics/Weapons/{player.weapon}/{direction}.png"
        # 加载图片并转换为透明度格式
        self.image = pygame.image.load(full_path).convert_alpha()

        # 放置
        if direction == "right":
# 如果方向是向右，则设置当前对象的矩形位置为玩家对象矩形的中左侧加上偏移量(0, 16)
self.rect = self.image.get_rect(midleft = player.rect.midright + pygame.math.Vector2(0, 16))

# 如果方向是向左，则设置当前对象的矩形位置为玩家对象矩形的中右侧加上偏移量(0, 16)
elif direction == "left":
    self.rect = self.image.get_rect(midright = player.rect.midleft + pygame.math.Vector2(0, 16)

# 如果方向是向下，则设置当前对象的矩形位置为玩家对象矩形的中底部加上偏移量(-10, 0)
elif direction == "down":
    self.rect = self.image.get_rect(midtop = player.rect.midbottom + pygame.math.Vector2(-10, 0)

# 如果方向是向上，则设置当前对象的矩形位置为玩家对象矩形的中顶部加上偏移量(-10, 0)
else:
    self.rect = self.image.get_rect(midbottom = player.rect.midtop + pygame.math.Vector2(-10, 0)
```