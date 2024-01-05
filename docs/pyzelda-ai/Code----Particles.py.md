# `.\Zelda-with-Python\Code\Particles.py`

```
# 导入pygame模块
import pygame
# 从Support模块中导入import_folder函数
from Support import import_folder
# 从random模块中导入choice函数
from random import choice
# 导入os模块
import os

# 这是用于文件（特别是图片）导入的（这一行将目录更改为项目保存的位置）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 定义动画播放器类
class AnimationPlayer:
    def __init__(self):
        # 初始化帧字典
        self.frames = {
            # 魔法
            "flame": import_folder("../Graphics/Particles/Flame/Frames"),
            "aura": import_folder("../Graphics/Particles/Aura"),
            "heal": import_folder("../Graphics/Particles/Heal/Frames"),
            # 攻击
            "claw": import_folder("../Graphics/Particles/Claw"),
```

# 导入不同粒子效果的文件夹，将其存储在对应的键值对中
"slash": import_folder("../Graphics/Particles/Slash"),
"sparkle": import_folder("../Graphics/Particles/Sparkle"),
"leaf_attack": import_folder("../Graphics/Particles/leaf_attack"),
"thunder": import_folder("../Graphics/Particles/Thunder"),

# 怪物死亡效果
"squid": import_folder("../Graphics/Particles/smoke_orange"),
"raccoon": import_folder("../Graphics/Particles/Raccoon"),
"spirit": import_folder("../Graphics/Particles/Nova"),
"bamboo": import_folder("../Graphics/Particles/Bamboo"),

# 叶子效果
"leaf":(
    import_folder("../Graphics/Particles/Leaf1"),
    import_folder("../Graphics/Particles/Leaf2"),
    import_folder("../Graphics/Particles/Leaf3"),
    import_folder("../Graphics/Particles/Leaf4"),
    import_folder("../Graphics/Particles/Leaf5"),
    import_folder("../Graphics/Particles/Leaf6"),
    self.reflect_images(import_folder("../Graphics/Particles/Leaf1")),
```
这段代码是在导入不同粒子效果的文件夹，并将它们存储在对应的键值对中。同时也包括了怪物死亡效果和叶子效果的导入。
# 调用 reflect_images 方法，导入 Leaf2 文件夹中的图片，并将结果反射到当前对象
self.reflect_images(import_folder("../Graphics/Particles/Leaf2")),
# 调用 reflect_images 方法，导入 Leaf3 文件夹中的图片，并将结果反射到当前对象
self.reflect_images(import_folder("../Graphics/Particles/Leaf3")),
# 调用 reflect_images 方法，导入 Leaf4 文件夹中的图片，并将结果反射到当前对象
self.reflect_images(import_folder("../Graphics/Particles/Leaf4")),
# 调用 reflect_images 方法，导入 Leaf5 文件夹中的图片，并将结果反射到当前对象
self.reflect_images(import_folder("../Graphics/Particles/Leaf5")),
# 调用 reflect_images 方法，导入 Leaf6 文件夹中的图片，并将结果反射到当前对象
self.reflect_images(import_folder("../Graphics/Particles/Leaf6"))
)

# 定义 reflect_images 方法，用于对传入的帧列表进行水平翻转
def reflect_images(self, frames):
    # 创建一个新的帧列表
    new_frames = []

    # 遍历传入的帧列表
    for frame in frames:
        # 对每一帧进行水平翻转，并添加到新的帧列表中
        flipped_frame = pygame.transform.flip(frame, True, False)
        new_frames.append(flipped_frame)
    # 返回翻转后的新帧列表
    return new_frames

# 定义 create_grass_particles 方法，用于创建草粒子效果
def create_grass_particles(self, pos, groups):
    # 从 leaf 关键字对应的帧列表中随机选择一组动画帧
    animation_frames = choice(self.frames["leaf"])
    # 创建一个草粒子效果对象，传入位置、动画帧和粒子组
    ParticleEffect(pos, animation_frames, groups)
# 创建粒子效果的方法，接受动画类型、位置和精灵组作为参数
def create_particles(self, animation_type, pos, groups):
    # 获取指定动画类型的动画帧
    animation_frames = self.frames[animation_type]
    # 创建粒子效果对象，传入位置、动画帧和精灵组
    ParticleEffect(pos, animation_frames, groups)

# 粒子效果类，继承自pygame.sprite.Sprite类
class ParticleEffect(pygame.sprite.Sprite):
    # 初始化方法，接受位置、动画帧和精灵组作为参数
    def __init__(self, pos, animation_frames, groups):
        # 调用父类的初始化方法，将粒子效果对象加入精灵组
        super().__init__(groups)
        # 设置精灵类型为"magic"
        self.sprite_type = "magic"
        # 设置当前帧索引为0
        self.frame_index = 0
        # 设置动画播放速度为0.15
        self.animation_speed = 0.15
        # 保存动画帧
        self.frames = animation_frames
        # 设置精灵的图像为当前帧的图像
        self.image = self.frames[self.frame_index]
        # 设置精灵的矩形范围为以pos为中心的矩形
        self.rect = self.image.get_rect(center = pos)

    # 动画方法，用于更新帧索引并检查是否播放完所有帧
    def animate(self):
        # 更新帧索引
        self.frame_index += self.animation_speed
        # 如果帧索引超过动画帧数，则销毁精灵
        if self.frame_index >= len(self.frames):
            self.kill()
        else:
            # 否则继续播放下一帧
# 将self.frames中索引为self.frame_index的帧赋值给self.image
self.image = self.frames[int(self.frame_index)]

# 调用animate方法，用于更新帧的索引
def update(self):
    self.animate()
```