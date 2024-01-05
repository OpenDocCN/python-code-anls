# `.\Zelda-with-Python\Code\Level.py`

```
# 导入 pygame 模块
import pygame
# 从 Settings 模块中导入所有内容
from Settings import *
# 从 Tile 模块中导入 Tile 类
from Tile import Tile
# 从 Player 模块中导入 Player 类
from Player import Player
# 从 Debug 模块中导入 debug 函数
from Debug import debug
# 从 Support 模块中导入所有内容
from Support import *
# 从 random 模块中导入 choice 和 randint 函数
from random import choice, randint
# 从 Weapon 模块中导入 Weapon 类
from Weapon import Weapon
# 从 UI 模块中导入 UI 类
from UI import UI
# 从 Enemy 模块中导入 Enemy 类
from Enemy import Enemy
# 从 Particles 模块中导入 AnimationPlayer 类
from Particles import AnimationPlayer
# 从 Magic 模块中导入 MagicPlayer 类
from Magic import MagicPlayer
# 从 Upgrade 模块中导入 Upgrade 类
from Upgrade import Upgrade
# 导入 os 模块
import os

# 这行代码用于导入文件（特别是图片），并将当前目录更改为项目所在的目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# 定义一个名为Level的类
class Level:
    # 初始化方法
    def __init__(self):
        # 获取显示表面
        self.display_surface = pygame.display.get_surface()
        # 游戏是否暂停的标志
        self.game_paused = False

        # 精灵组设置
        self.visible_sprites = YSortCameraGroup()  # 可见精灵组
        self.obstacle_sprites = pygame.sprite.Group()  # 障碍物精灵组

        # 攻击精灵
        self.current_attack = None  # 当前攻击
        self.attack_sprites = pygame.sprite.Group()  # 攻击精灵组
        self.attackable_sprites = pygame.sprite.Group()  # 可攻击的精灵组

        # 精灵设置
        self.create_map()  # 创建地图

        # 用户界面
# 创建用户界面对象
self.ui = UI()
# 创建升级对象，传入玩家对象
self.upgrade = Upgrade(self.player)

# 创建粒子效果播放器
self.animation_player = AnimationPlayer()
# 创建魔法播放器，传入粒子效果播放器
self.magic_player = MagicPlayer(self.animation_player)

# 创建地图
def create_map(self):
    # 从CSV文件导入地图布局数据，存储在字典中
    layouts = {
        "boundary": import_csv_layout("../Map/map_FloorBlocks.csv"),
        "grass": import_csv_layout("../Map/map_Grass.csv"),
        "object": import_csv_layout("../Map/map_Objects.csv"),
        "entities": import_csv_layout("../Map/map_Entities.csv")
    }

    # 从文件夹导入地图图形数据，存储在字典中
    graphics = {
        "grass": import_folder("../Graphics/Grass"),
        "objects": import_folder("../Graphics/Objects")
    }
# 遍历layouts字典，获取每个style和对应的layout
for style, layout in layouts.items():
    # 遍历layout中的每一行
    for row_index, row in enumerate(layout):
        # 遍历每一行中的每个col
        for col_index, col in enumerate(row):
            # 如果col不等于"-1"
            if col != "-1":
                # 根据col_index和row_index计算出x和y的坐标
                x = col_index * TILESIZE
                y = row_index * TILESIZE
                # 如果style是"boundary"
                if style == "boundary":
                    # 创建一个障碍物Tile对象，放入obstacle_sprites组中，类型为"invisible"
                    Tile((x, y), [self.obstacle_sprites], "invisible")
                # 如果style是"grass"
                if style == "grass":
                    # 从graphics["grass"]中随机选择一个grass图片
                    random_grass_image = choice(graphics["grass"])
                    # 创建一个草地Tile对象，放入visible_sprites、obstacle_sprites和attackable_sprites组中，类型为"grass"，使用随机选择的grass图片
                    Tile(
                        (x, y), 
                        [self.visible_sprites, self.obstacle_sprites, self.attackable_sprites], 
                        "grass", 
                        random_grass_image
                        )
# 如果样式为"object"，则从图形对象中获取对应的表面，并创建一个Tile对象
if style == "object":
    surf = graphics["objects"][int(col)]  # 从图形对象中获取对应的表面
    Tile((x, y), [self.visible_sprites, self.obstacle_sprites], "object", surf)  # 创建一个Tile对象

# 如果样式为"entities"
if style == "entities":
    # 如果颜色为"394"，则创建一个Player对象
    if col == "394":
        self.player = Player(
            (x, y), 
            [self.visible_sprites], 
            self.obstacle_sprites, 
            self.create_attack, 
            self.destroy_attack,
            self.create_magic
        )
    else:
        # 如果颜色为"390"，则怪物名称为"bamboo"
        if col == "390": monster_name = "bamboo"
        # 如果颜色为"391"，则怪物名称为"spirit"
        elif col == "391": monster_name = "spirit"
        # 如果颜色为"392"，则怪物名称为"raccoon"
        elif col == "392": monster_name = "raccoon"
        # 否则，怪物名称为"squid"
        else: monster_name = "squid"
# 创建一个敌人对象，传入怪物名称、位置、可见精灵组、可攻击精灵组、障碍物精灵组、伤害玩家方法、触发死亡粒子方法、增加经验方法
Enemy(
    monster_name, 
    (x, y), 
    [self.visible_sprites, self.attackable_sprites], 
    self.obstacle_sprites,
    self.damage_player,
    self.trigger_death_particles,
    self.add_exp
)

# 创建一个攻击对象，传入玩家对象和可见精灵组、攻击精灵组
def create_attack(self):
    self.current_attack = Weapon(self.player, [self.visible_sprites, self.attack_sprites])

# 创建一个魔法对象，传入类型、强度、消耗
def create_magic(self, style, strength, cost):
    # 如果类型是治疗，则调用玩家对象的治疗方法，传入玩家对象、强度、消耗、可见精灵组
    if style == "heal":
        self.magic_player.heal(self.player, strength, cost, [self.visible_sprites])
    
    # 如果类型是火焰，则调用玩家对象的火焰方法，传入玩家对象、消耗、可见精灵组、攻击精灵组
    if style == "flame":
        self.magic_player.flame(self.player, cost, [self.visible_sprites, self.attack_sprites])
# 销毁当前攻击对象
def destroy_attack(self):
    # 如果存在当前攻击对象，则销毁它
    if self.current_attack:
        self.current_attack.kill()
    # 将当前攻击对象设为 None
    self.current_attack = None

# 玩家攻击逻辑
def player_attack_logic(self):
    # 如果存在攻击精灵
    if self.attack_sprites:
        # 遍历攻击精灵
        for attack_sprite in self.attack_sprites:
            # 检测攻击精灵与可攻击精灵的碰撞
            collision_sprites = pygame.sprite.spritecollide(attack_sprite, self.attackable_sprites, False)
            # 如果有碰撞发生
            if collision_sprites:
                # 遍历碰撞到的精灵
                for target_sprite in collision_sprites:
                    # 如果目标精灵类型为 "grass"
                    if target_sprite.sprite_type == "grass":
                        # 获取目标精灵的中心位置
                        pos = target_sprite.rect.center
                        # 设置偏移量
                        offset = pygame.math.Vector2(0, 75)
                        # 创建草粒子效果
                        for leaf in range(randint(3, 6)):
                            self.animation_player.create_grass_particles(pos - offset, [self.visible_sprites])
                        # 销毁目标精灵
                        target_sprite.kill()
                    # 如果目标精灵类型不为 "grass"
                    else:
                        # 使目标精灵受到伤害
                        target_sprite.get_damage(self.player, attack_sprite.sprite_type)
    # 造成伤害给玩家角色，根据攻击类型进行不同的伤害处理
    def damage_player(self, amount, attack_type):
        # 如果玩家角色处于易受伤状态
        if self.player.vulnerable:
            # 减少玩家角色的健康值
            self.player.health -= amount
            # 取消易受伤状态
            self.player.vulnerable = False
            # 记录玩家受伤的时间
            self.player.hurt_time = pygame.time.get_ticks()
            # 创建粒子效果来表示受到攻击
            self.animation_player.create_particles(attack_type, self.player.rect.center, [self.visible_sprites])

    # 触发死亡粒子效果
    def trigger_death_particles(self, pos, particle_type):
        # 创建指定类型的粒子效果
        self.animation_player.create_particles(particle_type, pos, self.visible_sprites)

    # 增加经验值给玩家角色
    def add_exp(self, amount):
        # 增加玩家角色的经验值
        self.player.exp += amount

    # 切换游戏菜单的显示状态
    def toggle_menu(self):
        # 反转游戏暂停状态
        self.game_paused = not self.game_paused

    # 运行游戏
    def run(self):
# 调用visible_sprites对象的custom_draw方法，将玩家角色绘制到屏幕上
self.visible_sprites.custom_draw(self.player)
# 调用ui对象的display方法，显示玩家的界面
self.ui.display(self.player)

# 如果游戏暂停
if self.game_paused:
    # 显示升级界面
    self.upgrade.display()
# 如果游戏没有暂停
else:
    # 更新可见精灵
    self.visible_sprites.update()
    # 更新敌人的位置
    self.visible_sprites.enemy_update(self.player)
    # 玩家攻击逻辑
    self.player_attack_logic()


class YSortCameraGroup(pygame.sprite.Group):
    def __init__(self):

        # 通用设置
        super().__init__()
        # 获取显示表面
        self.display_surface = pygame.display.get_surface()
        # 获取显示表面宽度的一半
        self.half_width = self.display_surface.get_size()[0] // 2
        # 获取显示表面高度的一半
        self.half_height = self.display_surface.get_size()[1] // 2
        # 设置偏移量为零
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

    # 遍历精灵并按照纵坐标排序绘制
    for sprite in sorted(self.sprites(), key = lambda sprite: sprite.rect.centery):
        offset_pos = sprite.rect.topleft - self.offset
        self.display_surface.blit(sprite.image, offset_pos)
# 更新敌人精灵的状态
def enemy_update(self, player):
    # 从所有精灵中筛选出敌人精灵
    enemy_sprites = [sprite for sprite in self.sprites() if hasattr(sprite, "sprite_type") and sprite.sprite_type == "enemy"]
    # 遍历所有敌人精灵，更新它们的状态
    for enemy in enemy_sprites:
        enemy.enemy_update(player)
```