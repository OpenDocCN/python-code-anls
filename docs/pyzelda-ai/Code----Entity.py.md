# `.\Zelda-with-Python\Code\Entity.py`

```
# 从 cmath 模块中导入 rect 函数
from cmath import rect
# 从 pygame 模块中导入所有内容
import pygame
# 从 math 模块中导入 sin 函数
from math import sin
# 从 os 模块中导入所有内容
import os

# 这是用于文件（特别是图片）导入的代码（这行将目录更改为项目保存的位置）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 创建一个名为 Entity 的类，继承自 pygame.sprite.Sprite 类
class Entity(pygame.sprite.Sprite):
    # 初始化方法，接受一个 groups 参数
    def __init__(self, groups):
        # 调用父类的初始化方法
        super().__init__(groups)
        # 设置帧索引为 0
        self.frame_index = 0
        # 设置动画速度为 0.15
        self.animation_speed = 0.15
        # 创建一个名为 direction 的 pygame.math.Vector2 对象
        self.direction = pygame.math.Vector2()
    
    # 移动方法，接受一个 speed 参数
    def move(self, speed):
        # 如果方向向量的大小不为 0，则将方向向量归一化
        if self.direction.magnitude() != 0:
            self.direction = self.direction.normalize()
        
        # 根据方向向量和速度移动 hitbox
        self.hitbox.x += self.direction.x * speed
# 检测水平方向的碰撞
self.collision("Horizontal")
# 根据垂直方向的速度更新角色的位置
self.hitbox.y += self.direction.y * speed
# 检测垂直方向的碰撞
self.collision("Vertical")
# 更新角色的矩形位置
self.rect.center = self.hitbox.center

# 处理碰撞
def collision(self, direction):
    # 如果是水平方向的碰撞
    if direction == "Horizontal":
        # 遍历障碍物精灵组
        for sprite in self.obstacle_sprites:
            # 如果角色的hitbox与障碍物的hitbox相交
            if sprite.hitbox.colliderect(self.hitbox):
                # 如果角色正在向右移动
                if self.direction.x > 0: # Moving Right
                    # 调整角色的hitbox位置，使其与障碍物的左边界对齐
                    self.hitbox.right = sprite.hitbox.left
                # 如果角色正在向左移动
                if self.direction.x < 0: # Moving Left
                    # 调整角色的hitbox位置，使其与障碍物的右边界对齐
                    self.hitbox.left = sprite.hitbox.right
                        
    # 如果是垂直方向的碰撞
    if direction == "Vertical":
        # 遍历障碍物精灵组
        for sprite in self.obstacle_sprites:
            # 如果角色的hitbox与障碍物的hitbox相交
            if sprite.hitbox.colliderect(self.hitbox):
                # 如果角色正在向下移动
                if self.direction.y > 0: # Moving Down
                    # 调整角色的hitbox位置，使其与障碍物的上边界对齐
                    self.hitbox.bottom = sprite.hitbox.top
                # 如果角色正在向上移动
                if self.direction.y < 0: # Moving Up
                    # 调整角色的hitbox位置，使其与障碍物的下边界对齐
# 将自身的碰撞框的顶部位置设置为另一个精灵的碰撞框的底部位置
self.hitbox.top = sprite.hitbox.bottom

# 返回一个正弦波函数值，根据当前时间获取毫秒数计算
def wave_value(self):
    # 获取当前时间的毫秒数，并计算其正弦值
    value = sin(pygame.time.get_ticks())
    # 如果正弦值大于等于0，则返回255
    if value >= 0:
        return 255
    # 否则返回0
    else:
        return 0
```