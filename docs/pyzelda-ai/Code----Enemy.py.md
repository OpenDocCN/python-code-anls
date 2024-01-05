# `Code\Enemy.py`

```
# 导入pygame模块
import pygame
# 从Settings模块中导入所有内容
from Settings import *
# 从Entity模块中导入Entity类
from Entity import Entity
# 从Support模块中导入所有内容
from Support import *
# 导入os模块
import os

# 设置当前工作目录为Main.py所在的目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 定义Enemy类，继承自Entity类
class Enemy(Entity):
    # 初始化方法，接受怪物名称、位置、组、障碍物精灵、对玩家造成的伤害、触发死亡粒子效果、增加经验值作为参数
    def __init__(self, monster_name, pos, groups, obstacle_sprites, damage_player, trigger_death_particles, add_exp):
        
        # 调用父类Entity的初始化方法
        super().__init__(groups)
        # 设置精灵类型为"enemy"
        self.sprite_type = "enemy"

        # 导入怪物的图像资源
        self.import_graphics(monster_name)
# 设置角色状态为"空闲"
self.status = "idle"
# 根据当前状态和帧索引获取角色的图像
self.image = self.animations[self.status][self.frame_index]

# 设置角色的矩形位置
self.rect = self.image.get_rect(topleft = pos)
# 设置角色的碰撞箱
self.hitbox = self.rect.inflate(0, -10)
# 设置角色的障碍物精灵
self.obstacle_sprites = obstacle_sprites

# 设置角色的属性
self.monster_name = monster_name
# 获取怪物信息
monster_info = monster_data[self.monster_name]
self.health = monster_info["health"]
self.exp = monster_info["exp"]
self.speed = monster_info["speed"]
self.attack_damage = monster_info["damage"]
self.resistance = monster_info["resistance"]
self.attack_radius = monster_info["attack_radius"]
self.notice_radius = monster_info["notice_radius"]
self.attack_type = monster_info["attack_type"]
# 玩家交互
# 设置玩家是否可以攻击的标志
self.can_attack = True
# 设置攻击时间
self.attack_time = None
# 设置攻击冷却时间
self.attack_cooldown = 400
# 设置对玩家造成伤害的函数
self.damage_player = damage_player
# 设置触发死亡粒子效果的函数
self.trigger_death_particles = trigger_death_particles
# 设置增加经验值的函数
self.add_exp = add_exp

# 无敌计时器
# 设置怪物是否易受攻击的标志
self.vulnerable = True
# 设置受击时间
self.hit_time = None
# 设置无敌持续时间
self.invincibility_duration = 300

# 声音
# 加载死亡音效
self.death_sound = pygame.mixer.Sound("../Audio/Death.wav")
# 加载受击音效
self.hit_sound = pygame.mixer.Sound("../Audio/Hit.wav")
# 加载攻击音效
self.attack_sound = pygame.mixer.Sound(monster_info["attack_sound"])
# 设置死亡音效音量
self.death_sound.set_volume(0.6)
# 设置受击音效音量
self.hit_sound.set_volume(0.6)
# 设置攻击音效音量
self.attack_sound.set_volume(0.3)
# 导入角色动画资源
def import_graphics(self, name):
    # 初始化动画字典
    self.animations = {"idle": [], "move": [], "attack": []}
    # 设置主路径
    main_path = f"../Graphics/Monsters/{name}/"
    # 遍历动画字典的键，导入对应文件夹的动画资源
    for animation in self.animations.keys():
        self.animations[animation] = import_folder(main_path + animation)

# 获取与玩家的距离和方向
def get_player_distance_direction(self, player):
    # 获取敌人和玩家的位置向量
    enemy_vec = pygame.math.Vector2(self.rect.center)
    player_vec = pygame.math.Vector2(player.rect.center)
    # 计算距离
    distance = (player_vec - enemy_vec).magnitude()
    # 如果距离大于0，计算方向向量
    if distance > 0:
        direction = (player_vec - enemy_vec).normalize()
    # 如果距离等于0，方向向量为0
    else:
        direction = pygame.math.Vector2()
    # 返回距离和方向
    return(distance, direction)

# 获取状态
def get_status(self, player):
# 获取玩家距离和方向的信息，取距离
distance = self.get_player_distance_direction(player)[0]

# 如果玩家距离小于等于攻击半径并且可以攻击，则改变状态为攻击
if distance <= self.attack_radius and self.can_attack:
    if self.status != "attack":
        self.frame_index = 0
    self.status = "attack"
# 如果玩家距离小于等于警戒半径，则改变状态为移动
elif distance <= self.notice_radius:
    self.status = "move"
# 否则，改变状态为待机
else:
    self.status = "idle"

# 定义角色的行为
def actions(self, player):
    # 如果状态为攻击，则记录攻击时间，造成伤害，播放攻击音效
    if self.status == "attack":
        self.attack_time = pygame.time.get_ticks()
        self.damage_player(self.attack_damage, self.attack_type)
        self.attack_sound.play()
    # 如果状态为移动，则根据玩家位置改变角色方向
    elif self.status == "move":
        self.direction = self.get_player_distance_direction(player)[1]
    # 否则，角色方向为零向量
    else:
        self.direction = pygame.math.Vector2()
# 定义一个动画方法，根据角色当前状态选择对应的动画
def animate(self):
    # 获取当前状态对应的动画列表
    animation = self.animations[self.status]
    
    # 更新帧索引
    self.frame_index += self.animation_speed
    # 如果帧索引超过了动画帧数，根据状态进行处理
    if self.frame_index >= len(animation):
        if self.status == "attack":
            self.can_attack = False
        self.frame_index = 0

    # 更新角色图像和位置
    self.image = animation[int(self.frame_index)]
    self.rect = self.image.get_rect(center = self.hitbox.center)

    # 如果角色不可受伤，根据波动值设置图像透明度
    if not self.vulnerable:
        alpha = self.wave_value()
        self.image.set_alpha(alpha)
    else:
        self.image.set_alpha(255)

# 定义一个冷却方法
def cooldowns(self):
# 获取当前时间
current_time = pygame.time.get_ticks()

# 如果不能攻击
if not self.can_attack:
    # 如果距离上次攻击时间超过攻击冷却时间，可以攻击
    if current_time - self.attack_time >= self.attack_cooldown:
        self.can_attack = True

# 如果不易受伤
if not self.vulnerable:
    # 如果距离上次受伤时间超过无敌持续时间，变为易受伤状态
    if current_time - self.hit_time >= self.invincibility_duration:
        self.vulnerable = True

# 受到伤害
def get_damage(self, player, attack_type):
    # 如果易受伤
    if self.vulnerable:
        # 播放受伤音效
        self.hit_sound.play()
        # 根据玩家距离确定方向
        self.direction = self.get_player_distance_direction(player)[1]
        # 根据攻击类型减少生命值
        if attack_type == "weapon":
            self.health -= player.get_full_weapon_damage()
        else:
            self.health -= player.get_full_magic_damage()
        # 记录受伤时间，变为不易受伤状态
        self.hit_time = pygame.time.get_ticks()
        self.vulnerable = False
# 检查角色是否死亡，如果健康值小于等于0，则执行死亡操作，触发死亡粒子效果，增加经验值，播放死亡音效
def check_death(self):
    if self.health <= 0:
        self.kill()  # 移除角色
        self.trigger_death_particles(self.rect.center, self.monster_name)  # 触发死亡粒子效果
        self.add_exp(self.exp)  # 增加经验值
        self.death_sound.play()  # 播放死亡音效

# 角色受到攻击时的反应，如果不易受伤，则改变方向
def hit_reaction(self):
    if not self.vulnerable:
        self.direction *= -self.resistance  # 改变方向

# 更新角色状态，包括受击反应、移动、动画、冷却和死亡检查
def update(self):
    self.hit_reaction()  # 角色受击反应
    self.move(self.speed)  # 移动
    self.animate()  # 播放动画
    self.cooldowns()  # 冷却
    self.check_death()  # 检查死亡

# 更新敌人角色状态，获取与玩家角色的状态
def enemy_update(self, player):
    self.get_status(player)  # 获取玩家角色状态
# 调用名为"actions"的方法，传入参数"player"。
```