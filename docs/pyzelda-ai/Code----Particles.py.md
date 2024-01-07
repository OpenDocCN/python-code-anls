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

# 改变当前工作目录到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 定义动画播放器类
class AnimationPlayer:
    def __init__(self):
        # 定义frames属性，包含不同动画类型的帧
        self.frames = {
            # Magic
            "flame": import_folder("../Graphics/Particles/Flame/Frames"),
            "aura": import_folder("../Graphics/Particles/Aura"),
            "heal": import_folder("../Graphics/Particles/Heal/Frames"),
            # Attacks 
            "claw": import_folder("../Graphics/Particles/Claw"),
            "slash": import_folder("../Graphics/Particles/Slash"),
            "sparkle": import_folder("../Graphics/Particles/Sparkle"),
            "leaf_attack": import_folder("../Graphics/Particles/leaf_attack"),
            "thunder": import_folder("../Graphics/Particles/Thunder"),
            # Monster Deaths
            "squid": import_folder("../Graphics/Particles/smoke_orange"),
            "raccoon": import_folder("../Graphics/Particles/Raccoon"),
            "spirit": import_folder("../Graphics/Particles/Nova"),
            "bamboo": import_folder("../Graphics/Particles/Bamboo"),
            # Leafs
            "leaf":(
                import_folder("../Graphics/Particles/Leaf1"),
                import_folder("../Graphics/Particles/Leaf2"),
                import_folder("../Graphics/Particles/Leaf3"),
                import_folder("../Graphics/Particles/Leaf4"),
                import_folder("../Graphics/Particles/Leaf5"),
                import_folder("../Graphics/Particles/Leaf6"),
                self.reflect_images(import_folder("../Graphics/Particles/Leaf1")),
                self.reflect_images(import_folder("../Graphics/Particles/Leaf2")),
                self.reflect_images(import_folder("../Graphics/Particles/Leaf3")),
                self.reflect_images(import_folder("../Graphics/Particles/Leaf4")),
                self.reflect_images(import_folder("../Graphics/Particles/Leaf5")),
                self.reflect_images(import_folder("../Graphics/Particles/Leaf6"))
                )
            }

    # 反转图像
    def reflect_images(self, frames):
        new_frames = []
        for frame in frames:
            flipped_frame = pygame.transform.flip(frame, True, False)
            new_frames.append(flipped_frame)
        return new_frames

    # 创建草粒子效果
    def create_grass_particles(self, pos, groups):
        animation_frames = choice(self.frames["leaf"])
        ParticleEffect(pos, animation_frames, groups)

    # 创建粒子效果
    def create_particles(self, animation_type, pos, groups):
        animation_frames = self.frames[animation_type]
        ParticleEffect(pos, animation_frames, groups)

# 粒子效果类
class ParticleEffect(pygame.sprite.Sprite):
    def __init__(self, pos, animation_frames, groups):
        super().__init__(groups)
        self.sprite_type = "magic"
        self.frame_index = 0
        self.animation_speed = 0.15
        self.frames = animation_frames
        self.image = self.frames[self.frame_index]
        self.rect = self.image.get_rect(center = pos)

    # 动画
    def animate(self):
        self.frame_index += self.animation_speed
        if self.frame_index >= len(self.frames):
            self.kill()
        else:
            self.image = self.frames[int(self.frame_index)]

    # 更新
    def update(self):
        self.animate()

```