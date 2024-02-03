# `.\AutoGPT\benchmark\agbenchmark\reports\processing\graphs.py`

```py
# 导入所需的模块
from pathlib import Path
from typing import Any
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

# 定义函数，保存组合雷达图
def save_combined_radar_chart(categories: dict[str, Any], save_path: str | Path) -> None:
    # 过滤掉值为空的键值对
    categories = {k: v for k, v in categories.items() if v}
    # 如果所有值都为空，则抛出异常
    if not all(categories.values()):
        raise Exception("No data to plot")
    # 获取标签
    labels = np.array(list(next(iter(categories.values())).keys())  # 使用第一个类别获取键
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 将第一个角度添加到列表末尾以确保多边形闭合

    # 创建雷达图
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)  # 设置极坐标的偏移
    ax.set_theta_direction(-1)  # 设置极坐标的方向
    ax.spines["polar"].set_visible(False)  # 移除边框

    # 定义自定义归一化，从中间开始颜色
    norm = Normalize(
        vmin=0, vmax=max([max(val.values()) for val in categories.values()])
    )  # 使用所有类别的最大值进行归一化

    cmap = plt.cm.get_cmap("nipy_spectral", len(categories))  # 获取颜色映射

    colors = [cmap(i) for i in range(len(categories))]

    # 遍历类别并绘制雷达图
    for i, (cat_name, cat_values) in enumerate(categories.items()):
    ):  # 遍历每个类别（系列）
        values = np.array(list(cat_values.values()))
        values = np.concatenate((values, values[:1]))  # 确保多边形闭合

        ax.fill(angles, values, color=colors[i], alpha=0.25)  # 绘制填充多边形
        ax.plot(angles, values, color=colors[i], linewidth=2)  # 绘制多边形
        ax.plot(
            angles,
            values,
            "o",
            color="white",
            markersize=7,
            markeredgecolor=colors[i],
            markeredgewidth=2,
        )  # 绘制点

        # 绘制图例
        legend = ax.legend(
            handles=[
                mpatches.Patch(color=color, label=cat_name, alpha=0.25)
                for cat_name, color in zip(categories.keys(), colors)
            ],
            loc="upper left",
            bbox_to_anchor=(0.7, 1.3),
        )

        # 调整布局以腾出空间给图例
        plt.tight_layout()

    lines, labels = plt.thetagrids(
        np.degrees(angles[:-1]), (list(next(iter(categories.values())).keys()))
    )  # 使用第一个类别获取键

    highest_score = 7

    # 将 y 轴限制设置为 7
    ax.set_ylim(top=highest_score)

    # 将标签移开离图表
    for label in labels:
        label.set_position(
            (label.get_position()[0], label.get_position()[1] + -0.05)
        )  # 根据需要调整 0.1

    # 将径向标签移开离图表
    ax.set_rlabel_position(180)  # 类型：忽略

    ax.set_yticks([])  # 移除默认的 y 轴刻度

    # 手动创建网格线
    # 遍历从 0 到最高分数的整数，步长为 1
    for y in np.arange(0, highest_score + 1, 1):
        # 如果当前的 y 不是最高分数
        if y != highest_score:
            # 在雷达图上绘制灰色虚线，表示网格线
            ax.plot(
                angles, [y] * len(angles), color="gray", linewidth=0.5, linestyle=":"
            )
        # 为手动创建的网格线添加标签
        ax.text(
            angles[0],
            y + 0.2,
            str(int(y)),
            color="black",
            size=9,
            horizontalalignment="center",
            verticalalignment="center",
        )

    # 将图形保存为 PNG 文件
    plt.savefig(save_path, dpi=300)
    # 关闭图形以释放内存
    plt.close()
def save_single_radar_chart(
    category_dict: dict[str, int], save_path: str | Path
) -> None:
    # 将字典的键转换为数组，作为雷达图的标签
    labels = np.array(list(category_dict.keys()))
    # 将字典的值转换为数组，作为雷达图的数据
    values = np.array(list(category_dict.values()))

    # 获取标签数量
    num_vars = len(labels)

    # 计算每个标签对应的角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # 将第一个角度复制到最后一个角度，使雷达图闭合
    angles += angles[:1]
    # 将第一个值复制到最后一个值，使雷达图闭合
    values = np.concatenate((values, values[:1]))

    # 设置雷达图的颜色
    colors = ["#1f77b4"]

    # 创建一个雷达图的图形和坐标轴
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    # 设置极坐标的偏移角度
    ax.set_theta_offset(np.pi / 2)  # type: ignore
    # 设置极坐标的方向
    ax.set_theta_direction(-1)  # type: ignore

    # 隐藏极坐标的边框
    ax.spines["polar"].set_visible(False)

    # 在雷达图上显示标签
    lines, labels = plt.thetagrids(
        np.degrees(angles[:-1]), (list(category_dict.keys()))
    )

    # 设置y轴的最大值为7
    highest_score = 7
    ax.set_ylim(top=highest_score)

    # 调整标签的位置
    for label in labels:
        label.set_position((label.get_position()[0], label.get_position()[1] + -0.05))

    # 填充雷达图的区域
    ax.fill(angles, values, color=colors[0], alpha=0.25)
    # 绘制雷达图的线条
    ax.plot(angles, values, color=colors[0], linewidth=2)

    # 在雷达图上显示数值
    for i, (angle, value) in enumerate(zip(angles, values)):
        ha = "left"
        if angle in {0, np.pi}:
            ha = "center"
        elif np.pi < angle < 2 * np.pi:
            ha = "right"
        ax.text(
            angle,
            value - 0.5,
            f"{value}",
            size=10,
            horizontalalignment=ha,
            verticalalignment="center",
            color="black",
        )

    # 隐藏y轴的标签和刻度
    ax.set_yticklabels([])
    ax.set_yticks([])

    # 如果数据为空，则直接返回
    if values.size == 0:
        return

    # 绘制水平虚线
    for y in np.arange(0, highest_score, 1):
        ax.plot(angles, [y] * len(angles), color="gray", linewidth=0.5, linestyle=":")

    # 在雷达图上显示数据点
    for angle, value in zip(angles, values):
        ax.plot(
            angle,
            value,
            "o",
            color="white",
            markersize=7,
            markeredgecolor=colors[0],
            markeredgewidth=2,
        )
    plt.savefig(save_path, dpi=300)  # 将图形保存为 PNG 文件，设置分辨率为 300
    plt.close()  # 关闭图形以释放内存
# 保存组合柱状图
def save_combined_bar_chart(categories: dict[str, Any], save_path: str | Path) -> None:
    # 检查是否所有类别都有数据，如果没有数据则抛出异常
    if not all(categories.values()):
        raise Exception("No data to plot")

    # 将字典转换为 DataFrame
    df = pd.DataFrame(categories)

    # 创建一个分组柱状图
    df.plot(kind="bar", figsize=(10, 7))

    # 设置图表标题
    plt.title("Performance by Category for Each Agent")
    # 设置 x 轴标签
    plt.xlabel("Category")
    # 设置 y 轴标签
    plt.ylabel("Performance")

    # 将图表保存为 PNG 文件
    plt.savefig(save_path, dpi=300)
    # 关闭图表以释放内存
    plt.close()
```