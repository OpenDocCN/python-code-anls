# `.\Zelda-with-Python\Code\Player.py`

```
# 导入 pygame 模块
import pygame
# 从 Support 模块中导入 import_folder 函数
from Support import import_folder
# 从 Settings 模块中导入所有内容
from Settings import *
# 从 Entity 模块中导入 Entity 类
from Entity import Entity
# 导入 os 和 sys 模块
import os, sys

# 设置当前工作目录为 Main.py 所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 定义 Player 类，继承自 Entity 类
class Player(Entity):
    # 初始化方法，接受位置、组、障碍物精灵、创建攻击、销毁攻击、创建魔法作为参数
    def __init__(self, pos, groups, obstacle_sprites, create_attack, destroy_attack, create_magic):
        # 调用父类的初始化方法
        super().__init__(groups)
        # 加载玩家图片并转换为透明度格式
        self.image = pygame.image.load("../Graphics/Test/Player.png").convert_alpha()
        # 获取图片的矩形对象并设置位置为 pos
        self.rect = self.image.get_rect(topleft = pos)
        # 根据玩家矩形对象创建碰撞箱对象
        self.hitbox = self.rect.inflate(-6, HITBOX_OFFSET["player"])

        # 图形设置
        # 导入玩家资源
        self.import_player_assets()
        # 设置玩家状态为 "down"
        self.status = "down"
# 定义角色的攻击状态为未攻击
self.attacking = False
# 设置攻击冷却时间为400毫秒
self.attack_cooldown = 400
# 初始化攻击时间为空
self.attack_time = None

# 设置障碍物精灵组
self.obstacle_sprites = obstacle_sprites

# 设置创建攻击的函数
self.create_attack = create_attack
# 设置销毁攻击的函数
self.destroy_attack = destroy_attack
# 初始化武器索引为0
self.weapon_index = 0
# 从武器数据中获取第一个武器，并设置为当前武器
self.weapon = list(weapon_data.keys())[self.weapon_index]
# 设置可以切换武器的标志为True
self.can_switch_weapon = True
# 初始化武器切换时间为空
self.weapon_switch_time = None
# 设置切换武器的冷却时间为200毫秒

# 设置创建魔法的函数
self.create_magic = create_magic
# 初始化魔法索引为0
self.magic_index = 0
# 从魔法数据中获取第一个魔法，并设置为当前魔法
self.magic = list(magic_data.keys())[self.magic_index]
# 设置一个开关，表示是否可以切换魔法
self.can_switch_magic = True
# 初始化魔法切换时间为None
self.magic_switch_time = None

# 初始化角色的基本属性
self.stats = {"health": 100, "energy": 60, "attack": 10, "magic": 4, "speed": 5}
# 初始化角色的最大属性值
self.max_stats = {"health": 300, "energy": 140, "attack": 20, "magic": 10, "speed": 10}
# 初始化升级属性所需的成本
self.upgrade_cost = {"health": 100, "energy": 100, "attack": 100, "magic": 100, "speed": 100}
# 初始化角色的健康值为初始健康值
self.health = self.stats["health"]
# 初始化角色的能量值为初始能量值
self.energy = self.stats["energy"]
# 初始化角色的经验值为0
self.exp = 0
# 初始化角色的速度为初始速度值
self.speed = self.stats["speed"]

# 初始化伤害计时器
self.vulnerable = True
# 初始化受伤时间为None
self.hurt_time = None
# 初始化无敌持续时间为500
self.invulnerability_duration = 500

# 导入武器攻击音效
self.weapon_attack_sound = pygame.mixer.Sound("../Audio/Sword.wav")
# 设置武器攻击音效的音量
self.weapon_attack_sound.set_volume(0.2)
# 导入玩家角色的资源
def import_player_assets(self):
    # 玩家角色资源文件夹路径
    character_path = "../Graphics/Player/"
    # 初始化动画字典
    self.animations = {
        "up": [], "down": [], "left": [], "right": [],
        "right_idle": [], "left_idle": [], "up_idle": [], "down_idle":[],
        "right_attack": [], "left_attack": [], "up_attack": [], "down_attack": []
    }

    # 遍历动画字典的键
    for animation in self.animations.keys():
        # 获取动画文件夹的完整路径
        full_path = character_path + animation
        # 导入动画文件夹中的资源
        self.animations[animation] = import_folder(full_path)

# 处理玩家输入
def input(self):
    # 如果玩家没有在攻击
    if not self.attacking:
        # 获取当前按下的键
        keys = pygame.key.get_pressed()

        # 处理移动输入
        if keys[pygame.K_UP]:
            self.direction.y = -1
            # 如果按下上方向键，设置角色的y方向为-1，表示向上移动，并更新状态为“up”
            if keys[pygame.K_UP]:
                self.direction.y = -1
                self.status = "up"
            # 如果按下下方向键，设置角色的y方向为1，表示向下移动，并更新状态为“down”
            elif keys[pygame.K_DOWN]:
                self.direction.y = 1
                self.status = "down"
            # 如果未按下上下方向键，将角色的y方向设为0

            # 如果按下右方向键，设置角色的x方向为1，表示向右移动，并更新状态为“right”
            if keys[pygame.K_RIGHT]:
                self.direction.x = 1
                self.status = "right"
            # 如果按下左方向键，设置角色的x方向为-1，表示向左移动，并更新状态为“left”
            elif keys[pygame.K_LEFT]:
                self.direction.x = -1
                self.status = "left"
            # 如果未按下左右方向键，将角色的x方向设为0

            # 攻击输入
            # 如果按下空格键，设置角色的攻击状态为True，并记录攻击时间
            if keys [pygame.K_SPACE]:
                self.attacking = True
                self.attack_time = pygame.time.get_ticks()
            # 调用 create_attack 方法，创建角色的攻击动作
            self.create_attack()
            # 播放武器攻击音效
            self.weapon_attack_sound.play()

            # 魔法输入
            if keys [pygame.K_LCTRL]:
                # 设置角色正在进行攻击
                self.attacking = True
                # 获取当前时间，用于计算攻击持续时间
                self.attack_time = pygame.time.get_ticks()
                # 获取当前魔法的类型
                style = list(magic_data.keys())[self.magic_index]
                # 计算魔法的强度，包括基础强度和角色的魔法属性加成
                strength = list(magic_data.values())[self.magic_index]["strength"] + self.stats["magic"]
                # 获取当前魔法的消耗
                cost = list(magic_data.values())[self.magic_index]["cost"]
                # 创建魔法，传入魔法类型、强度和消耗
                self.create_magic(style, strength, cost)

            # 切换武器
            if keys [pygame.K_q] and self.can_switch_weapon:
                # 设置不能立即切换武器
                self.can_switch_weapon = False
                # 获取当前时间，用于计算切换武器的持续时间
                self.weapon_switch_time = pygame.time.get_ticks()
                
                # 如果当前武器索引小于武器数据列表的长度减一，则增加索引，否则重置为0
                if self.weapon_index < len(list(weapon_data.keys())) - 1:
                    self.weapon_index += 1
                else:
                    self.weapon_index = 0
# 设置角色的武器为武器数据字典的第一个键
self.weapon = list(weapon_data.keys())[self.weapon_index]

# 检查是否按下了 E 键并且可以切换魔法
if keys[pygame.K_e] and self.can_switch_magic:
    # 设置不能切换魔法，并记录切换魔法的时间
    self.can_switch_magic = False
    self.magic_switch_time = pygame.time.get_ticks()
    
    # 如果当前魔法索引小于魔法数据字典的键的数量减一，则增加魔法索引，否则重置为0
    if self.magic_index < len(list(magic_data.keys())) - 1:
        self.magic_index += 1
    else:
        self.magic_index = 0
    # 设置角色的魔法为魔法数据字典的第一个键
    self.magic = list(magic_data.keys())[self.magic_index]

# 获取角色的状态
def get_status(self):

    # 如果角色的方向 x 和 y 都为0
    if self.direction.x == 0 and self.direction.y == 0:
        # 如果状态不包含 "idle" 并且不包含 "attack"，则将状态设置为状态加上 "_idle"
        if not "idle" in self.status and not "attack" in self.status:
            self.status = self.status + "_idle"

    # 如果正在攻击
    if self.attacking:
# 将角色的x方向和y方向设为0，即停止移动
self.direction.x = 0
self.direction.y = 0
# 如果角色不处于攻击状态
if not "attack" in self.status:
    # 如果角色处于空闲状态
    if "idle" in self.status:
        # 将状态从空闲改为攻击
        self.status = self.status.replace("_idle", "_attack")
    else:
        # 将状态改为攻击状态
        self.status = self.status + "_attack"
# 如果角色处于攻击状态
else:
    # 将状态从攻击改为非攻击
    if "attack" in self.status:
        self.status = self.status.replace("_attack", "")

# 角色的冷却时间
def cooldowns(self):
    # 获取当前时间
    current_time = pygame.time.get_ticks()

    # 如果正在攻击
    if self.attacking:
        # 如果当前时间减去攻击时间大于攻击冷却时间加上武器数据中对应武器的冷却时间
        if current_time - self.attack_time >= self.attack_cooldown + weapon_data[self.weapon]["cooldown"]:
            # 停止攻击
            self.attacking = False
            # 销毁攻击

        # 如果不能切换武器
        if not self.can_switch_weapon:
# 如果距离上次切换武器的时间超过了切换冷却时间，那么可以切换武器
if current_time - self.weapon_switch_time >= self.switch_duration_cooldown:
    self.can_switch_weapon = True

# 如果不能切换魔法，并且距离上次切换魔法的时间超过了切换冷却时间，那么可以切换魔法
if not self.can_switch_magic:
    if current_time - self.magic_switch_time >= self.switch_duration_cooldown:
        self.can_switch_magic = True

# 如果不处于无敌状态，并且距离上次受伤的时间超过了无敌持续时间，那么变为可受伤状态
if not self.vulnerable:
    if current_time - self.hurt_time >= self.invulnerability_duration:
        self.vulnerable = True

# 根据当前状态选择对应的动画
def animate(self):
    animation = self.animations[self.status]

    # 循环播放动画帧
    self.frame_index += self.animation_speed
    if self.frame_index >= len(animation):
        self.frame_index = 0

    # 设置角色的图像
# 设置角色的图像为动画中的某一帧
self.image = animation[int(self.frame_index)]
# 设置角色的矩形范围为图像的矩形范围
self.rect = self.image.get_rect(center = self.hitbox.center)

# 闪烁效果
if not self.vulnerable:
    # 根据波动值设置图像的透明度
    alpha = self.wave_value()
    self.image.set_alpha(alpha)
else:
    # 设置图像完全不透明
    self.image.set_alpha(255)

# 获取角色的全武器伤害
def get_full_weapon_damage(self):
    # 获取基础伤害
    base_damage = self.stats["attack"]
    # 获取武器的伤害
    weapon_damage = weapon_data[self.weapon]["damage"]
    return base_damage + weapon_damage

# 获取角色的全魔法伤害
def get_full_magic_damage(self):
    # 获取基础伤害
    base_damage = self.stats["magic"]
    # 获取法术的伤害
    spell_damage = magic_data[self.magic]["strength"]
    return base_damage + spell_damage
# 根据索引获取字典中值的函数
def get_value_by_index(self, index):
    return list(self.stats.values())[index]

# 玩家死亡时退出游戏
def player_death(self):
    if self.health <= 0:
        sys.exit()

# 根据索引获取升级成本的函数
def get_cost_by_index(self, index):
    return list(self.upgrade_cost.values())[index]

# 能量恢复函数，如果能量小于最大能量值，则每次恢复0.01倍魔法值的能量
def energy_recovery(self):
    if self.energy < self.stats["energy"]:
        self.energy += 0.01 * self.stats["magic"]
    else:
        self.energy = self.stats["energy"]

# 更新函数，包括处理输入、冷却时间和获取状态
def update(self):
    self.input()
    self.cooldowns()
    self.get_status()
# 调用动画方法，用于播放角色的动画效果
self.animate()
# 移动角色，根据角色的速度属性进行移动
self.move(self.stats["speed"])
# 角色能量恢复，用于角色能量的自动恢复
self.energy_recovery()
# 检测角色是否死亡，如果死亡则执行相应的操作
self.player_death()
```