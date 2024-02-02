# `.\Zelda-with-Python\Code\Entity.py`

```py

# 从 cmath 模块中导入 rect 函数
from cmath import rect
# 导入 pygame 模块
import pygame
# 从 math 模块中导入 sin 函数
from math import sin
# 导入 os 模块
import os

# 这是用于文件（特别是图片）导入的代码（这行代码将目录更改为项目保存的位置）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 创建实体类，继承自 pygame.sprite.Sprite 类
class Entity(pygame.sprite.Sprite):
    # 初始化方法
    def __init__(self, groups):
        # 调用父类的初始化方法
        super().__init__(groups)
        # 设置帧索引初始值为 0
        self.frame_index = 0
        # 设置动画速度为 0.15
        self.animation_speed = 0.15
        # 创建一个方向向量
        self.direction = pygame.math.Vector2()
    
    # 移动方法
    def move(self, speed):
        # 如果方向向量的大小不为 0，则将其归一化
        if self.direction.magnitude() != 0:
            self.direction = self.direction.normalize()
        
        # 根据方向向量的值移动碰撞框
        self.hitbox.x += self.direction.x * speed
        self.collision("Horizontal")
        self.hitbox.y += self.direction.y * speed
        self.collision("Vertical")
        self.rect.center = self.hitbox.center

    # 碰撞检测方法
    def collision(self, direction):
        # 如果是水平方向的碰撞检测
        if direction == "Horizontal":
            # 遍历障碍物精灵组中的精灵
            for sprite in self.obstacle_sprites:
                # 如果碰撞框发生碰撞
                if sprite.hitbox.colliderect(self.hitbox):
                    # 根据移动方向调整碰撞框的位置
                    if self.direction.x > 0: # 向右移动
                        self.hitbox.right = sprite.hitbox.left
                    if self.direction.x < 0: # 向左移动
                        self.hitbox.left = sprite.hitbox.right
                        
        # 如果是垂直方向的碰撞检测
        if direction == "Vertical":
            # 遍历障碍物精灵组中的精灵
            for sprite in self.obstacle_sprites:
                # 如果碰撞框发生碰撞
                if sprite.hitbox.colliderect(self.hitbox):
                    # 根据移动方向调整碰撞框的位置
                    if self.direction.y > 0: # 向下移动
                        self.hitbox.bottom = sprite.hitbox.top
                    if self.direction.y < 0: # 向上移动
                        self.hitbox.top = sprite.hitbox.bottom

    # 返回正弦波值的方法
    def wave_value(self):
        # 获取当前时间的正弦值
        value = sin(pygame.time.get_ticks())
        # 如果值大于等于 0，则返回 255，否则返回 0
        if value >= 0:
            return 255
        else:
            return 0

```