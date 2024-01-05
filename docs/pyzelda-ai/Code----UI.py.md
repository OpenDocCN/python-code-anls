# `.\Zelda-with-Python\Code\UI.py`

```
# 导入pygame模块
import pygame
# 从Settings模块中导入所有内容
from Settings import *
# 导入os模块
import os

# 这是用于导入文件（特别是图片）的代码（这行代码将目录更改为项目保存的位置）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 创建UI类
class UI:
    def __init__(self):
        
        # General
        # 获取显示表面
        self.display_surface = pygame.display.get_surface()
        # 创建字体对象
        self.font = pygame.font.Font(UI_FONT, UI_FONT_SIZE)

        # Bar Setup
        # 创建健康条矩形对象
        self.health_bar_rect = pygame.Rect(10, 10, HEALTH_BAR_WIDTH, BAR_HEIGHT)
        # 创建能量条矩形对象
        self.energy_bar_rect = pygame.Rect(10, 34, ENERGY_BAR_WIDTH, BAR_HEIGHT)

        # Convert Weapon Dictionary
        # 创建武器图形列表
        self.weapon_graphics = []
# 遍历武器数据字典中的值，获取武器的图像路径，加载图像并转换为带透明度的图像，然后添加到武器图像列表中
for weapon in weapon_data.values():
    path = weapon["graphic"]
    weapon = pygame.image.load(path).convert_alpha()
    self.weapon_graphics.append(weapon)

# 转换魔法字典中的图像路径为带透明度的图像，并添加到魔法图像列表中
self.magic_graphics = []
for magic in magic_data.values():
    magic = pygame.image.load(magic["graphic"]).convert_alpha()
    self.magic_graphics.append(magic)

# 显示状态条
def show_bar(self, current, max_amount, bg_rect, color):
    # 绘制背景
    pygame.draw.rect(self.display_surface, UI_BG_COLOR, bg_rect)

    # 将状态转换为像素
    ratio = current / max_amount
    current_width = bg_rect.width * ratio
    current_rect = bg_rect.copy()
    current_rect.width = current_width
# 绘制一个矩形条
pygame.draw.rect(self.display_surface, color, current_rect)
# 绘制一个带有边框的矩形条
pygame.draw.rect(self.display_surface, UI_BORDER_COLOR, bg_rect, 3)

# 显示经验值
text_surf = self.font.render(str(int(exp)), False, TEXT_COLOR)
x = self.display_surface.get_size()[0] - 20
y = self.display_surface.get_size()[1] - 20
text_rect = text_surf.get_rect(bottomright = (x, y))

# 绘制一个带有背景的矩形条
pygame.draw.rect(self.display_surface, UI_BG_COLOR, text_rect.inflate(20, 20))
# 在矩形条上显示文本
self.display_surface.blit(text_surf, text_rect)
# 绘制一个带有边框的矩形条
pygame.draw.rect(self.display_surface, UI_BORDER_COLOR, text_rect.inflate(20, 20), 3)

# 绘制一个选择框
bg_rect = pygame.Rect(left, top, ITEM_BOX_SIZE, ITEM_BOX_SIZE)
pygame.draw.rect(self.display_surface, UI_BG_COLOR, bg_rect)
# 如果已经切换，绘制一个带有活动边框颜色的矩形条
if has_switched:
    pygame.draw.rect(self.display_surface, UI_BORDER_COLOR_ACTIVE, bg_rect, 3)
        else:
            # 如果条件不满足，绘制一个带有边框的矩形
            pygame.draw.rect(self.display_surface, UI_BORDER_COLOR, bg_rect, 3)
        # 返回背景矩形
        return bg_rect

    def weapon_overlay(self, weapon_index, has_switched):
        # 获取武器框的背景矩形
        bg_rect = self.selection_box(10, 630, has_switched) # Weapon Box
        # 获取武器图像并设置其位置
        weapon_surf = self.weapon_graphics[weapon_index]
        weapon_rect = weapon_surf.get_rect(center = bg_rect.center)

        # 在显示表面上绘制武器图像
        self.display_surface.blit(weapon_surf, weapon_rect)

    def magic_overlay(self, magic_index, has_switched):
        # 获取魔法框的背景矩形
        bg_rect = self.selection_box(100, 630, has_switched) # Magix Box (80, 635) in Tutorial
        # 获取魔法图像并设置其位置
        magic_surf = self.magic_graphics[magic_index]
        magic_rect = magic_surf.get_rect(center = bg_rect.center)

        # 在显示表面上绘制魔法图像
        self.display_surface.blit(magic_surf, magic_rect)

    def display(self, player):
        # 显示玩家的健康条
        self.show_bar(player.health, player.stats["health"], self.health_bar_rect, HEALTH_COLOR)
# 显示玩家能量条，参数分别为玩家当前能量值、玩家最大能量值、能量条的位置和颜色
self.show_bar(player.energy, player.stats["energy"], self.energy_bar_rect, ENERGY_COLOR)

# 显示玩家经验值
self.show_exp(player.exp)

# 显示武器覆盖层，参数分别为玩家当前武器索引和是否可以切换武器的布尔值
self.weapon_overlay(player.weapon_index, not player.can_switch_weapon)

# 显示魔法覆盖层，参数分别为玩家当前魔法索引和是否可以切换魔法的布尔值
self.magic_overlay(player.magic_index, not player.can_switch_magic)
```