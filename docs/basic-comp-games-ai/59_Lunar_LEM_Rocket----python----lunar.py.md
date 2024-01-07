# `basic-computer-games\59_Lunar_LEM_Rocket\python\lunar.py`

```

"""
LUNAR

Lunar landing simulation

Ported by Dave LeCompte
"""

import math  # 导入数学库
from dataclasses import dataclass  # 导入dataclass模块
from typing import Any, NamedTuple  # 导入类型提示模块

# 定义常量
PAGE_WIDTH = 64
COLUMN_WIDTH = 2
SECONDS_WIDTH = 4
MPH_WIDTH = 6
ALT_MI_WIDTH = 6
ALT_FT_WIDTH = 4
FUEL_WIDTH = 8
BURN_WIDTH = 10

# 定义列的位置
SECONDS_LEFT = 0
SECONDS_RIGHT = SECONDS_LEFT + SECONDS_WIDTH
ALT_LEFT = SECONDS_RIGHT + COLUMN_WIDTH
ALT_MI_RIGHT = ALT_LEFT + ALT_MI_WIDTH
ALT_FT_RIGHT = ALT_MI_RIGHT + COLUMN_WIDTH + ALT_FT_WIDTH
MPH_LEFT = ALT_FT_RIGHT + COLUMN_WIDTH
MPH_RIGHT = MPH_LEFT + MPH_WIDTH
FUEL_LEFT = MPH_RIGHT + COLUMN_WIDTH
FUEL_RIGHT = FUEL_LEFT + FUEL_WIDTH
BURN_LEFT = FUEL_RIGHT + COLUMN_WIDTH
BURN_RIGHT = BURN_LEFT + BURN_WIDTH

# 定义一个名为PhysicalState的命名元组
class PhysicalState(NamedTuple):
    velocity: float
    altitude: float

# 定义一个打印居中文本的函数
def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)

# 定义一个打印标题的函数
def print_header(title: str) -> None:
    print_centered(title)
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

# 定义一个右对齐添加字段的函数
def add_rjust(line: str, s: Any, pos: int) -> str:
    # ...

# 定义一个左对齐添加字段的函数
def add_ljust(line: str, s: str, pos: int) -> str:
    # ...

# 打印说明
def print_instructions() -> None:
    # ...

# 打印介绍
def print_intro() -> None:
    # ...

# 格式化行以供报告
def format_line_for_report(
    t: Any,
    miles: Any,
    feet: Any,
    velocity: Any,
    fuel: Any,
    burn_rate: str,
    is_header: bool,
) -> str:
    # ...

# 定义一个模拟时钟类
class SimulationClock:
    # ...

# 定义一个名为Capsule的数据类
@dataclass
class Capsule:
    # ...

# 显示着陆
def show_landing(sim_clock: SimulationClock, capsule: Capsule) -> None:
    # ...

# 显示燃料耗尽
def show_out_of_fuel(sim_clock: SimulationClock, capsule: Capsule) -> None:
    # ...

# 处理最终时刻
def process_final_tick(
    delta_t: float, sim_clock: SimulationClock, capsule: Capsule
) -> None:
    # ...

# 处理飞离
def handle_flyaway(sim_clock: SimulationClock, capsule: Capsule) -> bool:
    # ...

# 结束模拟
def end_sim() -> None:
    # ...

# 运行模拟
def run_simulation() -> None:
    # ...

# 主函数
def main() -> None:
    # ...

# 如果是主程序，则执行主函数
if __name__ == "__main__":
    main()

```