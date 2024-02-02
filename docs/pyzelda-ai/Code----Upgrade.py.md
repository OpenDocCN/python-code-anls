# `.\Zelda-with-Python\Code\Upgrade.py`

```py

# 导入必要的模块
import imp  # 引入模块
from traceback import print_tb  # 从模块中引入特定函数
import pygame  # 导入pygame模块
from Settings import *  # 从Settings模块中导入所有内容
import os  # 导入os模块

# 更改工作目录到项目所在的目录
os.chdir(os.path.dirname(os.path.abspath(__file__))

# 定义Upgrade类
class Upgrade:
    def __init__(self, player):
        # 通用设置
        self.display_surface = pygame.display.get_surface()  # 获取显示表面
        self.player = player  # 设置玩家对象
        self.attribute_nr = len(player.stats)  # 获取玩家属性数量
        self.attribute_names = list(player.stats.keys())  # 获取玩家属性名称列表
        self.max_values = list(player.max_stats.values())  # 获取玩家最大属性值列表
        self.font = pygame.font.Font(UI_FONT, UI_FONT_SIZE)  # 设置字体

        # 创建物品
        self.height = self.display_surface.get_size()[1] * 0.8  # 设置高度
        self.width = self.display_surface.get_size()[0] // 6  # 设置宽度
        self.create_items()  # 创建物品

        # 选择系统
        self.selection_index = 0  # 初始化选择索引
        self.selection_time = None  # 初始化选择时间
        self.can_move = True  # 初始化可移动状态

    # 输入处理
    def input(self):
        keys = pygame.key.get_pressed()  # 获取按键状态

        if self.can_move:
            if keys[pygame.K_RIGHT] and self.selection_index < self.attribute_nr - 1:  # 如果按下右键且选择索引小于属性数量减1
                self.selection_index += 1  # 选择索引加1
                self.can_move = False  # 设置为不可移动
                self.selection_time = pygame.time.get_ticks()  # 获取当前时间
            elif keys[pygame.K_LEFT] and self.selection_index >= 1:  # 如果按下左键且选择索引大于等于1
                self.selection_index -= 1  # 选择索引减1
                self.can_move = False  # 设置为不可移动
                self.selection_time = pygame.time.get_ticks()  # 获取当前时间

            if keys[pygame.K_SPACE]:  # 如果按下空格键
                self.can_move = False  # 设置为不可移动
                self.selection_time = pygame.time.get_ticks()  # 获取当前时间
                self.item_list[self.selection_index].trigger(self.player)  # 触发物品效果

    # 选择冷却
    def selection_cooldown(self):
        if not self.can_move:  # 如果可以移动
            current_time = pygame.time.get_ticks()  # 获取当前时间
            if current_time - self.selection_time >= 300:  # 如果当前时间减去选择时间大于等于300
                self.can_move = True  # 设置为可移动

    # 创建物品
    def create_items(self):
        self.item_list = []  # 初始化物品列表

        for item, index in enumerate(range(self.attribute_nr)):  # 遍历属性数量
            # 水平位置
            full_width = self.display_surface.get_size()[0]  # 获取显示表面宽度
            increment = full_width // self.attribute_nr  # 计算增量
            left = (item * increment) + (increment - self.width) // 2  # 计算左边距

            # 垂直位置
            top = self.display_surface.get_size()[1] * 0.1  # 计算顶部距离

            # 创建对象
            item = Item(left, top, self.width, self.height, index, self.font)  # 创建物品对象
            self.item_list.append(item)  # 将物品对象添加到列表中

    # 显示
    def display(self):
        self.input()  # 处理输入
        self.selection_cooldown()  # 处理选择冷却

        for index, item in enumerate(self.item_list):  # 遍历物品列表
            # 获取属性
            name = self.attribute_names[index]  # 获取属性名称
            value = self.player.get_value_by_index(index)  # 获取属性值
            max_value = self.max_values[index]  # 获取最大属性值
            cost = self.player.get_cost_by_index(index)  # 获取花费
            item.display(self.display_surface, self.selection_index, name, value, max_value, cost)  # 显示物品


# 物品类
class Item:
    def __init__(self, l, t, w, h, index, font):
        self.rect = pygame.Rect(l, t, w, h)  # 设置矩形
        self.index = index  # 设置索引
        self.font = font  # 设置字体

    # 显示名称
    def display_names(self, surface, name, cost, selected):
        color = TEXT_COLOR_SELECTED if selected else TEXT_COLOR  # 根据选择状态设置颜色

        # 标题
        title_surf = self.font.render(name, False, color)  # 渲染标题
        title_rect = title_surf.get_rect(midtop=self.rect.midtop + pygame.math.Vector2(0, 20))  # 设置标题位置

        # 花费
        cost_surf = self.font.render(f"{int(cost)}", False, color)  # 渲染花费
        cost_rect = cost_surf.get_rect(midbottom=self.rect.midbottom - pygame.math.Vector2(0, 20))  # 设置花费位置

        # 绘制
        surface.blit(title_surf, title_rect)  # 绘制标题
        surface.blit(cost_surf, cost_rect)  # 绘制花费

    # 显示进度条
    def display_bar(self, surface, value, max_value, selected):
        # 绘制设置
        top = self.rect.midtop + pygame.math.Vector2(0, 60)  # 设置顶部位置
        bottom = self.rect.midbottom - pygame.math.Vector2(0, 60)  # 设置底部位置
        color = BAR_COLOR_SELECTED if selected else BAR_COLOR  # 根据选择状态设置颜色

        # 进度条设置
        full_height = bottom[1] - top[1]  # 计算总高度
        relative_number = (value / max_value) * full_height  # 计算相对数值
        value_rect = pygame.Rect(top[0] - 15, bottom[1] - relative_number, 30, 10)  # 设置数值矩形

        # 绘制元素
        pygame.draw.line(surface, color, top, bottom, 5)  # 绘制线条
        pygame.draw.rect(surface, color, value_rect)  # 绘制矩形

    # 触发效果
    def trigger(self, player):
        upgrade_attribute = list(player.stats.keys())[self.index]  # 获取升级属性

        if player.exp >= player.upgrade_cost[upgrade_attribute] and player.stats[upgrade_attribute] < player.max_stats[upgrade_attribute]:  # 如果玩家经验大于等于升级花费且属性小于最大属性
            player.exp -= player.upgrade_cost[upgrade_attribute]  # 减去升级花费
            player.stats[upgrade_attribute] *= 1.2  # 属性增加20%
            player.upgrade_cost[upgrade_attribute] *= 1.4  # 升级花费增加40%

        if player.stats[upgrade_attribute] > player.max_stats[upgrade_attribute]:  # 如果属性大于最大属性
            player.stats[upgrade_attribute] = player.max_stats[upgrade_attribute]  # 将属性设置为最大属性

    # 显示
    def display(self, surface, selection_num, name, value, max_value, cost):
        if self.index == selection_num:  # 如果索引等于选择索引
            pygame.draw.rect(surface, UPGRADE_BG_COLOR_SELECTED, self.rect)  # 绘制选中背景
            pygame.draw.rect(surface, UI_BORDER_COLOR, self.rect, 4)  # 绘制边框
        else:
            pygame.draw.rect(surface, UI_BG_COLOR, self.rect)  # 绘制背景
            pygame.draw.rect(surface, UI_BORDER_COLOR, self.rect, 4)  # 绘制边框

        self.display_names(surface, name, cost, self.index == selection_num)  # 显示名称
        self.display_bar(surface, value, max_value, self.index == selection_num)  # 显示进度条

```