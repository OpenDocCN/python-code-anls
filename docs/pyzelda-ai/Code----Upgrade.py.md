# `.\Zelda-with-Python\Code\Upgrade.py`

```
# 导入模块
import imp
# 导入模块中的指定成员
from traceback import print_tb
# 导入 pygame 模块
import pygame
# 从 Settings 模块中导入所有成员
from Settings import *
# 导入 os 模块
import os

# 这是用于导入文件（特别是图片）的代码（这行代码将目录更改为项目保存的位置）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 定义 Upgrade 类
class Upgrade:
    def __init__(self, player):

        # 通用设置
        # 获取当前显示的表面
        self.display_surface = pygame.display.get_surface()
        # 设置玩家属性
        self.player = player
        # 玩家属性数量
        self.attribute_nr = len(player.stats)
        # 玩家属性名称列表
        self.attribute_names = list(player.stats.keys())
        # 玩家最大属性值列表
        self.max_values = list(player.max_stats.values())
# 使用指定的字体和字体大小创建一个字体对象
self.font = pygame.font.Font(UI_FONT, UI_FONT_SIZE)

# 创建项目
# 计算高度为显示表面高度的80%，宽度为显示表面宽度的1/6
self.height = self.display_surface.get_size()[1] * 0.8
self.width = self.display_surface.get_size()[0] // 6
# 调用create_items方法创建项目

# 选择系统
# 初始化选择索引为0，选择时间为空，可以移动为True
self.selection_index = 0
self.selection_time = None
self.can_move = True

# 输入方法
# 获取当前按下的键
keys = pygame.key.get_pressed()

# 如果可以移动
if self.can_move:
    # 如果按下右箭头键并且选择索引小于属性数量减1
    if keys[pygame.K_RIGHT] and self.selection_index < self.attribute_nr - 1:
        # 选择索引加1
        self.selection_index += 1
        # 不能移动设为False
        self.can_move = False
        # 记录选择时间
        self.selection_time = pygame.time.get_ticks()
            # 如果按下左箭头键并且选择索引大于等于1，则将选择索引减1
            elif keys[pygame.K_LEFT] and self.selection_index >= 1:
                self.selection_index -= 1
                # 不能移动，用于控制选择的频率
                self.can_move = False
                # 记录选择的时间
                self.selection_time = pygame.time.get_ticks()

            # 如果按下空格键
            if keys[pygame.K_SPACE]:
                # 不能移动，用于控制选择的频率
                self.can_move = False
                # 记录选择的时间
                self.selection_time = pygame.time.get_ticks()
                # 触发选择的物品
                self.item_list[self.selection_index].trigger(self.player)

    # 选择冷却
    def selection_cooldown(self):
        # 如果不能移动
        if not self.can_move:
            # 获取当前时间
            current_time = pygame.time.get_ticks()
            # 如果当前时间减去选择的时间大于等于300毫秒，则可以移动
            if current_time - self.selection_time >= 300:
                self.can_move = True

    # 创建物品
    def create_items(self):
        # 初始化物品列表
        self.item_list = []

        # 遍历属性数量的范围
        for item, index in enumerate(range(self.attribute_nr)):
```

            # 计算水平位置
            # 获取显示表面的宽度
            full_width = self.display_surface.get_size()[0]
            # 计算每个属性的水平间距
            increment = full_width // self.attribute_nr
            # 计算每个属性的左边界位置
            left = (item * increment) + (increment - self.width) // 2

            # 计算垂直位置
            # 获取显示表面的高度，并计算垂直位置
            top = self.display_surface.get_size()[1] * 0.1

            # 创建对象
            # 使用计算得到的左边界和顶部位置，以及指定的宽度、高度、索引和字体创建一个项目对象
            item = Item(left, top, self.width, self.height, index, self.font)
            # 将项目对象添加到项目列表中
            self.item_list.append(item)

    def display(self):
        # 调用input方法
        self.input()
        # 调用selection_cooldown方法
        self.selection_cooldown()
        
        # 遍历项目列表
        for index, item in enumerate(self.item_list):

            # 获取属性
            # 根据索引获取属性名称
            name = self.attribute_names[index]
# 通过索引获取玩家的数值
value = self.player.get_value_by_index(index)
# 获取指定索引的最大数值
max_value = self.max_values[index]
# 通过索引获取玩家的消耗
cost = self.player.get_cost_by_index(index)
# 在界面上显示物品的信息，包括名称、数值、最大数值和消耗
item.display(self.display_surface, self.selection_index, name, value, max_value, cost)

# 物品类，初始化时设置位置、索引和字体
class Item:
    def __init__(self, l, t, w, h, index, font):
        self.rect = pygame.Rect(l, t, w, h)
        self.index = index
        self.font = font

    # 在界面上显示物品名称和消耗
    def display_names(self, surface, name, cost, selected):
        # 根据是否被选中设置显示颜色
        color = TEXT_COLOR_SELECTED if selected else TEXT_COLOR

        # 渲染物品名称
        title_surf = self.font.render(name, False, color)
        # 设置物品名称的位置
        title_rect = title_surf.get_rect(midtop = self.rect.midtop + pygame.math.Vector2(0, 20)

        # 显示物品消耗
        # 使用字体渲染成文本图像，表示成本值，设置颜色
        cost_surf = self.font.render(f"{int(cost)}", False, color)
        # 获取文本图像的矩形区域，设置在按钮矩形区域的中下方偏移20个像素
        cost_rect = cost_surf.get_rect(midbottom = self.rect.midbottom - pygame.math.Vector2(0, 20))

        # 绘制标题文本图像和成本文本图像
        surface.blit(title_surf, title_rect)
        surface.blit(cost_surf, cost_rect)

    def display_bar(self, surface, value, max_value, selected):

        # 绘制设置
        top = self.rect.midtop + pygame.math.Vector2(0, 60)
        bottom = self.rect.midbottom - pygame.math.Vector2(0, 60)
        color = BAR_COLOR_SELECTED if selected else BAR_COLOR

        # 进度条设置
        full_height = bottom[1] - top[1]
        relative_number = (value / max_value) * full_height
        # 创建表示进度条值的矩形区域
        value_rect = pygame.Rect(top[0] - 15, bottom[1] - relative_number, 30, 10)
# 绘制元素
pygame.draw.line(surface, color, top, bottom, 5)  # 在表面上绘制一条线段
pygame.draw.rect(surface, color, value_rect)  # 在表面上绘制一个矩形

# 触发升级
def trigger(self, player):
    upgrade_attribute = list(player.stats.keys())[self.index]  # 获取玩家属性字典中的键值列表中的第self.index个属性

    # 如果玩家经验值大于升级所需经验值，并且玩家当前属性值小于最大属性值
    if player.exp >= player.upgrade_cost[upgrade_attribute] and player.stats[upgrade_attribute] < player.max_stats[upgrade_attribute]:
        player.exp -= player.upgrade_cost[upgrade_attribute]  # 减去升级所需经验值
        player.stats[upgrade_attribute] *= 1.2  # 将玩家属性值增加20%
        player.upgrade_cost[upgrade_attribute] *= 1.4  # 增加升级所需经验值

    # 如果玩家属性值大于最大属性值
    if player.stats[upgrade_attribute] > player.max_stats[upgrade_attribute]:
        player.stats[upgrade_attribute] = player.max_stats[upgrade_attribute]  # 将玩家属性值设为最大属性值

# 显示
def display(self, surface, selection_num, name, value, max_value, cost):
    # 如果当前索引等于选择的索引号
    if self.index == selection_num:
        pygame.draw.rect(surface, UPGRADE_BG_COLOR_SELECTED, self.rect)  # 在表面上绘制一个选中的背景颜色的矩形
        pygame.draw.rect(surface, UI_BORDER_COLOR, self.rect, 4)  # 在表面上绘制一个边框颜色的矩形
        # 如果不是选中状态，绘制背景色矩形
        else:
            pygame.draw.rect(surface, UI_BG_COLOR, self.rect)
            # 绘制带边框的矩形
            pygame.draw.rect(surface, UI_BORDER_COLOR, self.rect, 4)

        # 在界面上显示名称和成本
        self.display_names(surface, name, cost, self.index == selection_num)
        # 在界面上显示数值条
        self.display_bar(surface, value, max_value, self.index == selection_num)
```