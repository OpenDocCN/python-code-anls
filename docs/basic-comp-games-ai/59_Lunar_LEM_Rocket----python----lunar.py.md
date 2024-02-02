# `basic-computer-games\59_Lunar_LEM_Rocket\python\lunar.py`

```py
"""
LUNAR

Lunar landing simulation

Ported by Dave LeCompte
"""

# 导入 math 模块
import math
# 导入 dataclasses 模块中的 dataclass 类和 typing 模块中的 Any 类和 NamedTuple 类
from dataclasses import dataclass
from typing import Any, NamedTuple

# 定义页面宽度常量
PAGE_WIDTH = 64

# 定义各列宽度常量
COLUMN_WIDTH = 2
SECONDS_WIDTH = 4
MPH_WIDTH = 6
ALT_MI_WIDTH = 6
ALT_FT_WIDTH = 4
MPH_WIDTH = 6
FUEL_WIDTH = 8
BURN_WIDTH = 10

# 定义各列的位置常量
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

# 定义名为 PhysicalState 的命名元组，包含 velocity 和 altitude 两个字段
class PhysicalState(NamedTuple):
    velocity: float
    altitude: float

# 定义函数 print_centered，用于打印居中的消息
def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)

# 定义函数 print_header，用于打印标题
def print_header(title: str) -> None:
    print_centered(title)
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

# 定义函数 add_rjust，用于在指定位置右对齐添加新字段到行
def add_rjust(line: str, s: Any, pos: int) -> str:
    """Add a new field to a line right justified to end at pos"""
    s_str = str(s)
    slen = len(s_str)
    if len(line) + slen > pos:
        new_len = pos - slen
        line = line[:new_len]
    if len(line) + slen < pos:
        spaces = " " * (pos - slen - len(line))
        line = line + spaces
    return line + s_str

# 定义函数 add_ljust，用于在指定位置左对齐添加新字段到行
def add_ljust(line: str, s: str, pos: int) -> str:
    """Add a new field to a line left justified starting at pos"""
    s = str(s)
    if len(line) > pos:
        line = line[:pos]
    if len(line) < pos:
        spaces = " " * (pos - len(line))
        line = line + spaces
    return line + s

# 定义函数 print_instructions，用于打印指令
def print_instructions() -> None:
    """Somebody had a bad experience with Xerox."""
    print("THIS IS A COMPUTER SIMULATION OF AN APOLLO LUNAR")
    print("LANDING CAPSULE.\n\n")
    print("THE ON-BOARD COMPUTER HAS FAILED (IT WAS MADE BY")
    # 打印字符串 "XEROX) SO YOU HAVE TO LAND THE CAPSULE MANUALLY.\n"
# 打印程序介绍信息
def print_intro() -> None:
    print("SET BURN RATE OF RETRO ROCKETS TO ANY VALUE BETWEEN")
    print("0 (FREE FALL) AND 200 (MAXIMUM BURN) POUNDS PER SECOND.")
    print("SET NEW BURN RATE EVERY 10 SECONDS.\n")
    print("CAPSULE WEIGHT 32,500 LBS; FUEL WEIGHT 16,000 LBS.\n\n\n")
    print("GOOD LUCK\n")


# 为报告格式化行
def format_line_for_report(
    t: Any,
    miles: Any,
    feet: Any,
    velocity: Any,
    fuel: Any,
    burn_rate: str,
    is_header: bool,
) -> str:
    line = add_rjust("", t, SECONDS_RIGHT)
    line = add_rjust(line, miles, ALT_MI_RIGHT)
    line = add_rjust(line, feet, ALT_FT_RIGHT)
    line = add_rjust(line, velocity, MPH_RIGHT)
    line = add_rjust(line, fuel, FUEL_RIGHT)
    if is_header:
        line = add_rjust(line, burn_rate, BURN_RIGHT)
    else:
        line = add_ljust(line, burn_rate, BURN_LEFT)
    return line


# 模拟时钟类
class SimulationClock:
    def __init__(self, elapsed_time: float, time_until_next_prompt: float) -> None:
        self.elapsed_time = elapsed_time
        self.time_until_next_prompt = time_until_next_prompt

    def time_for_prompt(self) -> bool:
        return self.time_until_next_prompt < 1e-3

    def advance(self, delta_t: float) -> None:
        self.elapsed_time += delta_t
        self.time_until_next_prompt -= delta_t


# 胶囊数据类
@dataclass
class Capsule:
    altitude: float = 120  # in miles above the surface
    velocity: float = 1  # downward
    m: float = 32500  # mass_with_fuel
    n: float = 16500  # mass_without_fuel
    g: float = 1e-3
    z: float = 1.8
    fuel_per_second: float = 0

    def remaining_fuel(self) -> float:
        return self.m - self.n

    def is_out_of_fuel(self) -> bool:
        return self.remaining_fuel() < 1e-3

    def update_state(
        self, sim_clock: SimulationClock, delta_t: float, new_state: PhysicalState
    def advance(self, delta_t: float) -> None:
        # 推进模拟时钟
        sim_clock.advance(delta_t)
        # 更新剩余燃料量
        self.m = self.m - delta_t * self.fuel_per_second
        # 更新高度和速度
        self.altitude = new_state.altitude
        self.velocity = new_state.velocity

    def fuel_time_remaining(self) -> float:
        # 推算当前燃料燃烧速率下剩余燃料可以支持的时间
        assert self.fuel_per_second > 0
        return self.remaining_fuel() / self.fuel_per_second

    def predict_motion(self, delta_t: float) -> PhysicalState:
        # 使用欧拉方法对运动方程进行数值积分

        q = delta_t * self.fuel_per_second / self.m

        # 新速度
        new_velocity = (
            self.velocity
            + self.g * delta_t
            + self.z * (-q - q**2 / 2 - q**3 / 3 - q**4 / 4 - q**5 / 5)
        )

        # 新高度
        new_altitude = (
            self.altitude
            - self.g * delta_t**2 / 2
            - self.velocity * delta_t
            + self.z
            * delta_t
            * (q / 2 + q**2 / 6 + q**3 / 12 + q**4 / 20 + q**5 / 30)
        )

        return PhysicalState(altitude=new_altitude, velocity=new_velocity)

    def make_state_display_string(self, sim_clock: SimulationClock) -> str:
        # 生成状态显示字符串
        seconds = sim_clock.elapsed_time
        miles = int(self.altitude)
        feet = int(5280 * (self.altitude - miles))
        velocity = int(3600 * self.velocity)
        fuel = int(self.remaining_fuel())
        burn_rate = " ? "

        return format_line_for_report(
            seconds, miles, feet, velocity, fuel, burn_rate, False
        )

    def prompt_for_burn(self, sim_clock: SimulationClock) -> None:
        # 提示用户输入燃烧速率
        msg = self.make_state_display_string(sim_clock)

        self.fuel_per_second = float(input(msg))
        sim_clock.time_until_next_prompt = 10
# 显示着陆情况，根据模拟时钟和太空舱对象
def show_landing(sim_clock: SimulationClock, capsule: Capsule) -> None:
    # 计算着陆速度
    w = 3600 * capsule.velocity
    # 打印着陆信息
    print(
        f"ON MOON AT {sim_clock.elapsed_time:.2f} SECONDS - IMPACT VELOCITY {w:.2f} MPH"
    )
    # 根据着陆速度判断着陆情况
    if w < 1.2:
        print("PERFECT LANDING!")
    elif w < 10:
        print("GOOD LANDING (COULD BE BETTER)")
    elif w <= 60:
        print("CRAFT DAMAGE... YOU'RE STRANDED HERE UNTIL A RESCUE")
        print("PARTY ARRIVES. HOPE YOU HAVE ENOUGH OXYGEN!")
    else:
        print("SORRY THERE WERE NO SURVIVORS. YOU BLEW IT!")
        print(f"IN FACT, YOU BLASTED A NEW LUNAR CRATER {w*.227:.2f} FEET DEEP!")
    # 结束模拟
    end_sim()


# 处理燃料耗尽情况，根据模拟时钟和太空舱对象
def show_out_of_fuel(sim_clock: SimulationClock, capsule: Capsule) -> None:
    # 打印燃料耗尽信息
    print(f"FUEL OUT AT {sim_clock.elapsed_time} SECONDS")
    # 计算下降速度
    delta_t = (
        -capsule.velocity
        + math.sqrt(capsule.velocity**2 + 2 * capsule.altitude * capsule.g)
    ) / capsule.g
    # 更新速度和模拟时钟，然后显示着陆情况
    capsule.velocity += capsule.g * delta_t
    sim_clock.advance(delta_t)
    show_landing(sim_clock, capsule)


# 处理最后一个时间间隔，根据时间间隔、模拟时钟和太空舱对象
def process_final_tick(
    delta_t: float, sim_clock: SimulationClock, capsule: Capsule
) -> None:
    # 当我们根据速度和时间间隔推测位置时，我们超过了表面。为了更好的准确性，我们将后退并进行更短的时间推进。

    while True:
        if delta_t < 5e-3:
            # 显示着陆情况并返回
            show_landing(sim_clock, capsule)
            return
        # 计算平均速度
        average_vel = (
            capsule.velocity
            + math.sqrt(
                capsule.velocity**2
                + 2
                * capsule.altitude
                * (capsule.g - capsule.z * capsule.fuel_per_second / capsule.m)
            )
        ) / 2
        # 根据平均速度计算新的时间间隔
        delta_t = capsule.altitude / average_vel
        # 预测新的状态并更新太空舱状态
        new_state = capsule.predict_motion(delta_t)
        capsule.update_state(sim_clock, delta_t, new_state)


# 处理飞离情况，根据模拟时钟和太空舱对象，返回布尔值
def handle_flyaway(sim_clock: SimulationClock, capsule: Capsule) -> bool:
    """
    # 用户开始离开月球。由于这是一个月球着陆模拟，我们等待直到太空舱的速度为正（向下）才提示输入更多信息。
    # 如果成功着陆，返回True；如果模拟应该继续，返回False。
    """

    while True:
        # 计算重力加速度和推进器产生的加速度之比
        w = (1 - capsule.m * capsule.g / (capsule.z * capsule.fuel_per_second)) / 2
        # 计算时间间隔
        delta_t = (
            capsule.m
            * capsule.velocity
            / (
                capsule.z
                * capsule.fuel_per_second
                * math.sqrt(w**2 + capsule.velocity / capsule.z)
            )
        ) + 0.05

        # 预测新的状态
        new_state = capsule.predict_motion(delta_t)

        # 如果高度小于等于0，表示着陆
        if new_state.altitude <= 0:
            # 已着陆
            return True

        # 更新太空舱状态
        capsule.update_state(sim_clock, delta_t, new_state)

        # 如果新状态的速度大于0，或者太空舱速度小于等于0，则返回到正常模拟
        if (new_state.velocity > 0) or (capsule.velocity <= 0):
            # 返回到正常模拟
            return False
# 定义结束模拟的函数，不返回任何结果
def end_sim() -> None:
    # 打印提示信息
    print("\n\n\nTRY AGAIN??\n\n\n")

# 运行模拟的函数，不返回任何结果
def run_simulation() -> None:
    # 打印空行
    print()
    # 打印格式化后的报告表头
    print(
        format_line_for_report("SEC", "MI", "FT", "MPH", "LB FUEL", "BURN RATE", True)
    )

    # 创建模拟时钟对象
    sim_clock = SimulationClock(0, 10)
    # 创建太空舱对象
    capsule = Capsule()

    # 提示用户输入燃烧信息
    capsule.prompt_for_burn(sim_clock)

    while True:
        # 如果太空舱燃料用尽
        if capsule.is_out_of_fuel():
            # 显示燃料用尽信息
            show_out_of_fuel(sim_clock, capsule)
            return

        # 如果到了提示时间
        if sim_clock.time_for_prompt():
            # 提示用户输入燃烧信息
            capsule.prompt_for_burn(sim_clock)
            continue

        # 计算时钟前进的时间，取最短的时间
        if capsule.fuel_per_second > 0:
            delta_t = min(
                sim_clock.time_until_next_prompt, capsule.fuel_time_remaining()
            )
        else:
            delta_t = sim_clock.time_until_next_prompt

        # 预测新的状态
        new_state = capsule.predict_motion(delta_t)

        # 如果高度小于等于0
        if new_state.altitude <= 0:
            # 处理最后一个时间片
            process_final_tick(delta_t, sim_clock, capsule)
            return

        # 如果速度大于0且新速度小于0
        if capsule.velocity > 0 and new_state.velocity < 0:
            # 远离月球
            landed = handle_flyaway(sim_clock, capsule)
            if landed:
                # 处理最后一个时间片
                process_final_tick(delta_t, sim_clock, capsule)
                return
        else:
            # 更新状态
            capsule.update_state(sim_clock, delta_t, new_state)

# 主函数
def main() -> None:
    # 打印标题
    print_header("LUNAR")
    # 打印指令
    print_instructions()
    while True:
        # 打印介绍
        print_intro()
        # 运行模拟
        run_simulation()

# 如果是主程序
if __name__ == "__main__":
    # 运行主函数
    main()
```