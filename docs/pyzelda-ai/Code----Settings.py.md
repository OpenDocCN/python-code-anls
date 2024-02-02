# `.\Zelda-with-Python\Code\Settings.py`

```py

# 导入 os 模块
import os

# 更改当前工作目录为当前文件所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 游戏设置
WIDTH = 1280  # 游戏窗口宽度
HEIGTH = 720  # 游戏窗口高度
FPS = 60  # 游戏帧率
TILESIZE = 64  # 地图瓦片大小
HITBOX_OFFSET = {  # 不同对象的碰撞框偏移量
	"player": -26,
	"object": -40,
	"grass": -10,
	"invisible": 0
	}

# 用户界面设置
BAR_HEIGHT = 20  # 柱状图高度
HEALTH_BAR_WIDTH = 200  # 生命值柱状图宽度
ENERGY_BAR_WIDTH = 140  # 能量值柱状图宽度
ITEM_BOX_SIZE = 80  # 物品框大小
UI_FONT = "../Graphics/Font/Joystix.ttf"  # 用户界面字体
UI_FONT_SIZE = 18  # 用户界面字体大小

# 通用颜色
WATER_COLOR = "#71ddee"  # 水的颜色
UI_BG_COLOR = "#222222"  # 用户界面背景颜色
UI_BORDER_COLOR = "#111111"  # 用户界面边框颜色
TEXT_COLOR = "#EEEEEE"  # 文本颜色

# 用户界面颜色
HEALTH_COLOR = "Red"  # 生命值颜色
ENERGY_COLOR = "Blue"  # 能量值颜色
UI_BORDER_COLOR_ACTIVE = "Gold"  # 用户界面激活时的边框颜色

# 升级菜单
TEXT_COLOR_SELECTED = "#111111"  # 选中文本颜色
BAR_COLOR = "#EEEEEE"  # 柱状图颜色
BAR_COLOR_SELECTED = "#111111"  # 选中柱状图颜色
UPGRADE_BG_COLOR_SELECTED = "#EEEEEE"  # 选中升级菜单背景颜色

# 武器
weapon_data = {
	"sword": {"cooldown": 100, "damage": 15, "graphic": "../Graphics/Weapons/Sword/Full.png"},  # 剑的冷却时间、伤害和图像路径
	"lance": {"cooldown": 400, "damage": 30, "graphic": "../Graphics/Weapons/Lance/Full.png"},  # 枪的冷却时间、伤害和图像路径
	"axe": {"cooldown": 300, "damage": 20, "graphic": "../Graphics/Weapons/Axe/Full.png"},  # 斧头的冷却时间、伤害和图像路径
	"rapier": {"cooldown": 50, "damage": 8, "graphic": "../Graphics/Weapons/Rapier/Full.png"},  # 细剑的冷却时间、伤害和图像路径
	"sai": {"cooldown": 80, "damage": 10, "graphic": "../Graphics/Weapons/Sai/Full.png"}  # 菜刀的冷却时间、伤害和图像路径
    }

# 魔法
magic_data = {
	"flame": {"strength": 5, "cost": 20, "graphic": "../Graphics/Particles/Flame/Fire.png"},  # 火焰的强度、消耗和图像路径
	"heal": {"strength": 20, "cost": 10, "graphic": "../Graphics/Particles/Heal/Heal.png"}  # 治疗的强度、消耗和图像路径
	}

# 敌人
monster_data = {
	"squid": {"health": 100, "exp": 180, "damage": 20, "attack_type": "slash", "attack_sound": "../Audio/Attack/Slash.wav", "speed": 3, "resistance": 3, "attack_radius": 80, "notice_radius": 360},  # 鱿鱼的生命值、经验值、伤害等信息
	"raccoon": {"health": 300, "exp": 300, "damage": 40, "attack_type": "claw", "attack_sound": "../Audio/Attack/Claw.wav", "speed": 2, "resistance": 3, "attack_radius": 120, "notice_radius": 400},  # 浣熊的生命值、经验值、伤害等信息
	"spirit": {"health": 100, "exp": 200, "damage": 8, "attack_type": "thunder", "attack_sound": "../Audio/Attack/Fireball.wav", "speed": 4, "resistance": 3, "attack_radius": 60, "notice_radius": 350},  # 精灵的生命值、经验值、伤害等信息
	"bamboo": {"health": 70, "exp": 150, "damage": 6, "attack_type": "leaf_attack", "attack_sound": "../Audio/Attack/Slash.wav", "speed": 3, "resistance": 3, "attack_radius": 50, "notice_radius": 300}  # 竹子的生命值、经验值、伤害等信息
	}

```