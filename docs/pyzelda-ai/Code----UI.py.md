# `.\Zelda-with-Python\Code\UI.py`

```py

# 导入pygame模块
import pygame
# 从Settings模块中导入所有内容
from Settings import *
# 导入os模块
import os

# 改变当前工作目录到项目所在的目录
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
        # 创建血条矩形对象
        self.health_bar_rect = pygame.Rect(10, 10, HEALTH_BAR_WIDTH, BAR_HEIGHT)
        # 创建能量条矩形对象
        self.energy_bar_rect = pygame.Rect(10, 34, ENERGY_BAR_WIDTH, BAR_HEIGHT)

        # Convert Weapon Dictionary
        # 将武器字典中的图像转换为pygame图像对象
        self.weapon_graphics = []
        for weapon in weapon_data.values():
            path = weapon["graphic"]
            weapon = pygame.image.load(path).convert_alpha()
            self.weapon_graphics.append(weapon)

        # Convert Magic Dictionary
        # 将魔法字典中的图像转换为pygame图像对象
        self.magic_graphics = []
        for magic in magic_data.values():
            magic = pygame.image.load(magic["graphic"]).convert_alpha()
            self.magic_graphics.append(magic)

    # 显示条形图
    def show_bar(self, current, max_amount, bg_rect, color):
        # 绘制背景
        pygame.draw.rect(self.display_surface, UI_BG_COLOR, bg_rect)

        # 将状态转换为像素
        ratio = current / max_amount
        current_width = bg_rect.width * ratio
        current_rect = bg_rect.copy()
        current_rect.width = current_width

        # 绘制条形图
        pygame.draw.rect(self.display_surface, color, current_rect)
        pygame.draw.rect(self.display_surface, UI_BORDER_COLOR, bg_rect, 3)

    # 显示经验值
    def show_exp(self, exp):
        text_surf = self.font.render(str(int(exp)), False, TEXT_COLOR)
        x = self.display_surface.get_size()[0] - 20
        y = self.display_surface.get_size()[1] - 20
        text_rect = text_surf.get_rect(bottomright = (x, y))

        pygame.draw.rect(self.display_surface, UI_BG_COLOR, text_rect.inflate(20, 20))
        self.display_surface.blit(text_surf, text_rect)
        pygame.draw.rect(self.display_surface, UI_BORDER_COLOR, text_rect.inflate(20, 20), 3)

    # 选择框
    def selection_box(self, left, top, has_switched):
        bg_rect = pygame.Rect(left, top, ITEM_BOX_SIZE, ITEM_BOX_SIZE)
        pygame.draw.rect(self.display_surface, UI_BG_COLOR, bg_rect)
        if has_switched:
            pygame.draw.rect(self.display_surface, UI_BORDER_COLOR_ACTIVE, bg_rect, 3)
        else:
            pygame.draw.rect(self.display_surface, UI_BORDER_COLOR, bg_rect, 3)
        return bg_rect

    # 武器叠加
    def weapon_overlay(self, weapon_index, has_switched):
        bg_rect = self.selection_box(10, 630, has_switched) # 武器框
        weapon_surf = self.weapon_graphics[weapon_index]
        weapon_rect = weapon_surf.get_rect(center = bg_rect.center)

        self.display_surface.blit(weapon_surf, weapon_rect)

    # 魔法叠加
    def magic_overlay(self, magic_index, has_switched):
        bg_rect = self.selection_box(100, 630, has_switched) # 魔法框
        magic_surf = self.magic_graphics[magic_index]
        magic_rect = magic_surf.get_rect(center = bg_rect.center)

        self.display_surface.blit(magic_surf, magic_rect)

    # 显示
    def display(self, player):
        self.show_bar(player.health, player.stats["health"], self.health_bar_rect, HEALTH_COLOR)
        self.show_bar(player.energy, player.stats["energy"], self.energy_bar_rect, ENERGY_COLOR)

        self.show_exp(player.exp)

        self.weapon_overlay(player.weapon_index, not player.can_switch_weapon)
        self.magic_overlay(player.magic_index, not player.can_switch_magic)

```