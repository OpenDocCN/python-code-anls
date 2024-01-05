# `Code\Settings.py`

```
# 导入 os 模块
import os

# 将当前工作目录更改为脚本文件所在的目录（用于导入文件，特别是图片）
os.chdir(os.path.dirname(os.path.abspath(__file__))

# 游戏设置
WIDTH = 1280  # 游戏窗口宽度
HEIGTH = 720  # 游戏窗口高度（应为 HEIGHT）
FPS = 60  # 游戏帧率
TILESIZE = 64  # 地图瓦片大小
HITBOX_OFFSET = {  # 不同对象的碰撞框偏移量
	"player": -26,  # 玩家
	"object": -40,  # 物体
	"grass": -10,  # 草地
	"invisible": 0  # 不可见对象
	}

# 用户界面设置
BAR_HEIGHT = 20  # 柱状图高度
HEALTH_BAR_WIDTH = 200  # 生命值条宽度
# 定义能量条的宽度
ENERGY_BAR_WIDTH = 140
# 定义物品框的大小
ITEM_BOX_SIZE = 80
# 定义UI字体的路径
UI_FONT = "../Graphics/Font/Joystix.ttf"
# 定义UI字体的大小
UI_FONT_SIZE = 18

# 一般颜色
WATER_COLOR = "#71ddee"
UI_BG_COLOR = "#222222"
UI_BORDER_COLOR = "#111111"
TEXT_COLOR = "#EEEEEE"

# UI颜色
HEALTH_COLOR = "Red"
ENERGY_COLOR = "Blue"
UI_BORDER_COLOR_ACTIVE = "Gold"

# 升级菜单
TEXT_COLOR_SELECTED = "#111111"
BAR_COLOR = "#EEEEEE"
BAR_COLOR_SELECTED = "#111111"
# 选定升级背景颜色
UPGRADE_BG_COLOR_SELECTED = "#EEEEEE"

# 武器数据字典，包含不同武器的冷却时间、伤害和图像路径
weapon_data = {
	"sword": {"cooldown": 100, "damage": 15, "graphic": "../Graphics/Weapons/Sword/Full.png"},
	"lance": {"cooldown": 400, "damage": 30, "graphic": "../Graphics/Weapons/Lance/Full.png"},
	"axe": {"cooldown": 300, "damage": 20, "graphic": "../Graphics/Weapons/Axe/Full.png"},
	"rapier": {"cooldown": 50, "damage": 8, "graphic": "../Graphics/Weapons/Rapier/Full.png"},
	"sai": {"cooldown": 80, "damage": 10, "graphic": "../Graphics/Weapons/Sai/Full.png"}
    }

# 魔法数据字典，包含不同魔法的强度、消耗和图像路径
magic_data = {
	"flame": {"strength": 5, "cost": 20, "graphic": "../Graphics/Particles/Flame/Fire.png"},
	"heal": {"strength": 20, "cost": 10, "graphic": "../Graphics/Particles/Heal/Heal.png"}
	}

# 敌人数据字典，包含不同敌人的生命值、经验值、伤害、攻击类型、攻击声音、速度、抗性、攻击半径和发现半径
monster_data = {
	"squid": {"health": 100, "exp": 180, "damage": 20, "attack_type": "slash", "attack_sound": "../Audio/Attack/Slash.wav", "speed": 3, "resistance": 3, "attack_radius": 80, "notice_radius": 360},
# 定义了三种不同的怪物，每种怪物都有不同的属性
"raccoon": {"health": 300, "exp": 300, "damage": 40, "attack_type": "claw", "attack_sound": "../Audio/Attack/Claw.wav", "speed": 2, "resistance": 3, "attack_radius": 120, "notice_radius": 400},
"spirit": {"health": 100, "exp": 200, "damage": 8, "attack_type": "thunder", "attack_sound": "../Audio/Attack/Fireball.wav", "speed": 4, "resistance": 3, "attack_radius": 60, "notice_radius": 350},
"bamboo": {"health": 70, "exp": 150, "damage": 6, "attack_type": "leaf_attack", "attack_sound": "../Audio/Attack/Slash.wav", "speed": 3, "resistance": 3, "attack_radius": 50, "notice_radius": 300}
}
```