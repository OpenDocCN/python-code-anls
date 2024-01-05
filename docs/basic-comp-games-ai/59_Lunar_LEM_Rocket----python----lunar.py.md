# `59_Lunar_LEM_Rocket\python\lunar.py`

```
"""
LUNAR

Lunar landing simulation

Ported by Dave LeCompte
"""

import math  # 导入 math 模块，用于数学运算
from dataclasses import dataclass  # 导入 dataclass 模块，用于创建数据类
from typing import Any, NamedTuple  # 导入 Any 和 NamedTuple 类型，用于类型提示

PAGE_WIDTH = 64  # 定义页面宽度为 64

COLUMN_WIDTH = 2  # 定义列宽为 2
SECONDS_WIDTH = 4  # 定义秒数宽度为 4
MPH_WIDTH = 6  # 定义英里/小时宽度为 6
ALT_MI_WIDTH = 6  # 定义海里高度宽度为 6
ALT_FT_WIDTH = 4  # 定义英尺高度宽度为 4
MPH_WIDTH = 6  # 定义英里/小时宽度为 6（此处 MPH_WIDTH 被重复定义，可能是笔误）
FUEL_WIDTH = 8  # 设置燃料宽度为8
BURN_WIDTH = 10  # 设置燃烧宽度为10

SECONDS_LEFT = 0  # 设置秒数左边界为0
SECONDS_RIGHT = SECONDS_LEFT + SECONDS_WIDTH  # 设置秒数右边界为SECONDS_LEFT加上SECONDS_WIDTH
ALT_LEFT = SECONDS_RIGHT + COLUMN_WIDTH  # 设置高度左边界为SECONDS_RIGHT加上COLUMN_WIDTH
ALT_MI_RIGHT = ALT_LEFT + ALT_MI_WIDTH  # 设置高度英里右边界为ALT_LEFT加上ALT_MI_WIDTH
ALT_FT_RIGHT = ALT_MI_RIGHT + COLUMN_WIDTH + ALT_FT_WIDTH  # 设置高度英尺右边界为ALT_MI_RIGHT加上COLUMN_WIDTH再加上ALT_FT_WIDTH
MPH_LEFT = ALT_FT_RIGHT + COLUMN_WIDTH  # 设置速度左边界为ALT_FT_RIGHT加上COLUMN_WIDTH
MPH_RIGHT = MPH_LEFT + MPH_WIDTH  # 设置速度右边界为MPH_LEFT加上MPH_WIDTH
FUEL_LEFT = MPH_RIGHT + COLUMN_WIDTH  # 设置燃料左边界为MPH_RIGHT加上COLUMN_WIDTH
FUEL_RIGHT = FUEL_LEFT + FUEL_WIDTH  # 设置燃料右边界为FUEL_LEFT加上FUEL_WIDTH
BURN_LEFT = FUEL_RIGHT + COLUMN_WIDTH  # 设置燃烧左边界为FUEL_RIGHT加上COLUMN_WIDTH
BURN_RIGHT = BURN_LEFT + BURN_WIDTH  # 设置燃烧右边界为BURN_LEFT加上BURN_WIDTH

class PhysicalState(NamedTuple):  # 定义一个名为PhysicalState的类，继承自NamedTuple
    velocity: float  # 定义velocity属性为浮点型
    altitude: float  # 定义altitude属性为浮点型
def print_centered(msg: str) -> None:
    # 计算需要添加的空格数，使得消息居中显示
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    # 在屏幕上打印居中显示的消息
    print(spaces + msg)


def print_header(title: str) -> None:
    # 调用print_centered函数打印居中显示的标题
    print_centered(title)
    # 打印固定格式的页眉
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")


def add_rjust(line: str, s: Any, pos: int) -> str:
    """Add a new field to a line right justified to end at pos"""
    # 将输入的s转换为字符串
    s_str = str(s)
    # 计算s_str的长度
    slen = len(s_str)
    # 如果line加上s_str的长度超过了pos，则截取line使其长度为pos-slen
    if len(line) + slen > pos:
        new_len = pos - slen
        line = line[:new_len]
    # 如果line加上s_str的长度小于pos，则添加足够的空格使其长度为pos-slen
    if len(line) + slen < pos:
        spaces = " " * (pos - slen - len(line))
def add_ljust(line: str, s: str, pos: int) -> str:
    """Add a new field to a line left justified starting at pos"""
    # 将 s 转换为字符串类型
    s = str(s)
    # 如果 line 的长度大于 pos，则截取 line 到 pos 的部分
    if len(line) > pos:
        line = line[:pos]
    # 如果 line 的长度小于 pos，则在 line 后面添加足够的空格使其长度达到 pos
    if len(line) < pos:
        spaces = " " * (pos - len(line))
        line = line + spaces
    # 返回 line 和 s 拼接后的结果
    return line + s


def print_instructions() -> None:
    """Somebody had a bad experience with Xerox."""
    # 打印模拟阿波罗登月舱的计算机故障信息
    print("THIS IS A COMPUTER SIMULATION OF AN APOLLO LUNAR")
    print("LANDING CAPSULE.\n\n")
    print("THE ON-BOARD COMPUTER HAS FAILED (IT WAS MADE BY")
    print("XEROX) SO YOU HAVE TO LAND THE CAPSULE MANUALLY.\n")
    # 打印提示信息，要求手动着陆太空舱

def print_intro() -> None:
    print("SET BURN RATE OF RETRO ROCKETS TO ANY VALUE BETWEEN")
    print("0 (FREE FALL) AND 200 (MAXIMUM BURN) POUNDS PER SECOND.")
    print("SET NEW BURN RATE EVERY 10 SECONDS.\n")
    print("CAPSULE WEIGHT 32,500 LBS; FUEL WEIGHT 16,000 LBS.\n\n\n")
    print("GOOD LUCK\n")
    # 打印游戏介绍信息

def format_line_for_report(
    t: Any,
    miles: Any,
    feet: Any,
    velocity: Any,
    fuel: Any,
    burn_rate: str,
    is_header: bool,
) -> str:
    # 格式化报告行的函数，接受时间、英里、英尺、速度、燃料、燃烧速率和是否为标题的参数
    line = add_rjust("", t, SECONDS_RIGHT)  # 使用 add_rjust 函数将 t 右对齐并添加到 line 中
    line = add_rjust(line, miles, ALT_MI_RIGHT)  # 使用 add_rjust 函数将 miles 右对齐并添加到 line 中
    line = add_rjust(line, feet, ALT_FT_RIGHT)  # 使用 add_rjust 函数将 feet 右对齐并添加到 line 中
    line = add_rjust(line, velocity, MPH_RIGHT)  # 使用 add_rjust 函数将 velocity 右对齐并添加到 line 中
    line = add_rjust(line, fuel, FUEL_RIGHT)  # 使用 add_rjust 函数将 fuel 右对齐并添加到 line 中
    if is_header:  # 如果是标题行
        line = add_rjust(line, burn_rate, BURN_RIGHT)  # 使用 add_rjust 函数将 burn_rate 右对齐并添加到 line 中
    else:  # 如果不是标题行
        line = add_ljust(line, burn_rate, BURN_LEFT)  # 使用 add_ljust 函数将 burn_rate 左对齐并添加到 line 中
    return line  # 返回处理后的 line


class SimulationClock:
    def __init__(self, elapsed_time: float, time_until_next_prompt: float) -> None:
        self.elapsed_time = elapsed_time  # 初始化实例变量 elapsed_time
        self.time_until_next_prompt = time_until_next_prompt  # 初始化实例变量 time_until_next_prompt

    def time_for_prompt(self) -> bool:
        return self.time_until_next_prompt < 1e-3  # 返回判断结果，判断是否到了提示时间
    def advance(self, delta_t: float) -> None:
        # 更新经过的时间
        self.elapsed_time += delta_t
        # 更新距离下一个提示的时间
        self.time_until_next_prompt -= delta_t


@dataclass
class Capsule:
    altitude: float = 120  # 飞船离地面的高度，单位为英里
    velocity: float = 1  # 速度，向下为正
    m: float = 32500  # 带燃料的质量
    n: float = 16500  # 不带燃料的质量
    g: float = 1e-3  # 重力加速度
    z: float = 1.8  # 未知参数
    fuel_per_second: float = 0  # 每秒消耗的燃料量

    def remaining_fuel(self) -> float:
        # 返回剩余燃料量
        return self.m - self.n

    def is_out_of_fuel(self) -> bool:
        # 判断是否耗尽燃料
        return self.remaining_fuel() < 1e-3
    def update_state(
        self, sim_clock: SimulationClock, delta_t: float, new_state: PhysicalState
    ) -> None:
        # 更新模拟时钟
        sim_clock.advance(delta_t)
        # 更新剩余燃料量
        self.m = self.m - delta_t * self.fuel_per_second
        # 更新飞行器的高度
        self.altitude = new_state.altitude
        # 更新飞行器的速度
        self.velocity = new_state.velocity

    def fuel_time_remaining(self) -> float:
        # 推算当前燃料燃烧速率下剩余燃料可以支持的时间
        assert self.fuel_per_second > 0
        return self.remaining_fuel() / self.fuel_per_second

    def predict_motion(self, delta_t: float) -> PhysicalState:
        # 使用欧拉方法对运动方程进行数值积分

        # 计算燃料燃烧速率对质量的影响
        q = delta_t * self.fuel_per_second / self.m

        # 计算新的速度
        # 计算新的速度
        new_velocity = (
            self.velocity  # 当前速度
            + self.g * delta_t  # 加上重力加速度乘以时间间隔
            + self.z * (-q - q**2 / 2 - q**3 / 3 - q**4 / 4 - q**5 / 5)  # 加上阻尼系数乘以阻尼项
        )

        # 新的高度
        new_altitude = (
            self.altitude  # 当前高度
            - self.g * delta_t**2 / 2  # 减去重力加速度乘以时间间隔的平方再除以2
            - self.velocity * delta_t  # 减去速度乘以时间间隔
            + self.z  # 加上阻尼系数
            * delta_t  # 乘以时间间隔
            * (q / 2 + q**2 / 6 + q**3 / 12 + q**4 / 20 + q**5 / 30)  # 乘以阻尼项
        )

        # 返回新的物理状态
        return PhysicalState(altitude=new_altitude, velocity=new_velocity)

    def make_state_display_string(self, sim_clock: SimulationClock) -> str:
        seconds = sim_clock.elapsed_time  # 获取模拟时钟的已经过去的时间
        miles = int(self.altitude)  # 将self.altitude转换为整数，表示高度的整数部分
        feet = int(5280 * (self.altitude - miles))  # 计算高度的小数部分对应的英尺数
        velocity = int(3600 * self.velocity)  # 将速度转换为整数，表示每小时的速度
        fuel = int(self.remaining_fuel())  # 获取剩余燃料的整数值
        burn_rate = " ? "  # 初始化burn_rate为问号字符串

        return format_line_for_report(
            seconds, miles, feet, velocity, fuel, burn_rate, False
        )  # 调用format_line_for_report函数，返回格式化后的报告行数据

    def prompt_for_burn(self, sim_clock: SimulationClock) -> None:
        msg = self.make_state_display_string(sim_clock)  # 生成包含模拟时钟状态的消息字符串

        self.fuel_per_second = float(input(msg))  # 从用户输入中获取每秒燃料消耗率，并转换为浮点数
        sim_clock.time_until_next_prompt = 10  # 设置下一次提示的时间为10秒后


def show_landing(sim_clock: SimulationClock, capsule: Capsule) -> None:
    w = 3600 * capsule.velocity  # 计算着陆舱的速度
    print(  # 打印着陆信息
    f"ON MOON AT {sim_clock.elapsed_time:.2f} SECONDS - IMPACT VELOCITY {w:.2f} MPH"
)
```
这行代码是一个字符串格式化操作，用来打印模拟时钟的已经过去的时间和着陆时的速度。

```
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
end_sim()
```
这段代码是一个条件语句，根据着陆速度的不同情况打印不同的提示信息，最后调用end_sim()函数结束模拟。

```
def show_out_of_fuel(sim_clock: SimulationClock, capsule: Capsule) -> None:
    print(f"FUEL OUT AT {sim_clock.elapsed_time} SECONDS")
    delta_t = (
        -capsule.velocity
        + math.sqrt(capsule.velocity**2 + 2 * capsule.altitude * capsule.g)
```
这段代码定义了一个函数show_out_of_fuel()，用来在燃料耗尽时打印提示信息。在函数内部，使用字符串格式化打印燃料耗尽时的模拟时钟已经过去的时间。接着计算delta_t的值。
    ) / capsule.g  # 计算速度变化
    capsule.velocity += capsule.g * delta_t  # 更新速度
    sim_clock.advance(delta_t)  # 模拟时钟推进
    show_landing(sim_clock, capsule)  # 显示着陆情况


def process_final_tick(
    delta_t: float, sim_clock: SimulationClock, capsule: Capsule
) -> None:
    # 当我们根据速度和时间增量推算位置时，我们超过了地表。为了更好的准确性，我们将后退并进行更短的时间推进。

    while True:
        if delta_t < 5e-3:  # 如果时间增量小于5e-3
            show_landing(sim_clock, capsule)  # 显示着陆情况
            return
        # line 35
        average_vel = (  # 计算平均速度
            capsule.velocity
# 计算飞船离开月球表面后的状态
def handle_flyaway(sim_clock: SimulationClock, capsule: Capsule) -> bool:
    """
    The user has started flying away from the moon. Since this is a
    lunar LANDING simulation, we wait until the capsule's velocity is
    positive (downward) before prompting for more input.

    Returns True if landed, False if simulation should continue.
    """
    while True:  # 进入无限循环
        w = (1 - capsule.m * capsule.g / (capsule.z * capsule.fuel_per_second)) / 2  # 计算 w 值
        delta_t = (  # 计算时间增量 delta_t
            capsule.m  # 胶囊质量
            * capsule.velocity  # 胶囊速度
            / (
                capsule.z  # 燃料流速
                * capsule.fuel_per_second  # 燃料消耗速率
                * math.sqrt(w**2 + capsule.velocity / capsule.z)  # 计算平方根
            )
        ) + 0.05  # 加上固定时间增量

        new_state = capsule.predict_motion(delta_t)  # 预测新的状态

        if new_state.altitude <= 0:  # 如果新状态的高度小于等于0
            # have landed  # 已经着陆
            return True  # 返回真值表示着陆

        capsule.update_state(sim_clock, delta_t, new_state)  # 更新胶囊状态
        if (new_state.velocity > 0) or (capsule.velocity <= 0):
            # 如果新状态的速度大于0或者胶囊的速度小于等于0，则返回到正常模拟
            return False


def end_sim() -> None:
    # 打印提示信息
    print("\n\n\nTRY AGAIN??\n\n\n")


def run_simulation() -> None:
    # 打印空行
    print()
    # 打印格式化后的报告标题
    print(
        format_line_for_report("SEC", "MI", "FT", "MPH", "LB FUEL", "BURN RATE", True)
    )

    # 创建模拟时钟对象
    sim_clock = SimulationClock(0, 10)
    # 创建胶囊对象
    capsule = Capsule()

    # 提示用户输入燃烧信息
    capsule.prompt_for_burn(sim_clock)
    while True:  # 进入无限循环
        if capsule.is_out_of_fuel():  # 如果太空舱耗尽燃料
            show_out_of_fuel(sim_clock, capsule)  # 显示燃料耗尽的提示信息
            return  # 结束函数

        if sim_clock.time_for_prompt():  # 如果是提示时间
            capsule.prompt_for_burn(sim_clock)  # 提示太空舱进行燃烧
            continue  # 继续下一次循环

        # 计算时钟前进的时间，取决于下一个提示的时间和燃料耗尽的时间中较短的一个
        if capsule.fuel_per_second > 0:  # 如果每秒燃料消耗大于0
            delta_t = min(
                sim_clock.time_until_next_prompt, capsule.fuel_time_remaining()
            )
        else:  # 如果每秒燃料消耗等于0
            delta_t = sim_clock.time_until_next_prompt

        new_state = capsule.predict_motion(delta_t)  # 预测太空舱的运动状态
        if new_state.altitude <= 0:  # 如果新状态的高度小于等于0
            process_final_tick(delta_t, sim_clock, capsule)  # 处理最终的时钟，返回
            return  # 返回

        if capsule.velocity > 0 and new_state.velocity < 0:  # 如果太空舱速度大于0且新状态速度小于0
            # moving away from the moon  # 远离月球

            landed = handle_flyaway(sim_clock, capsule)  # 处理飞离，返回是否着陆
            if landed:  # 如果着陆
                process_final_tick(delta_t, sim_clock, capsule)  # 处理最终的时钟，返回
                return  # 返回

        else:  # 否则
            capsule.update_state(sim_clock, delta_t, new_state)  # 更新太空舱状态

def main() -> None:  # 主函数
    print_header("LUNAR")  # 打印标题
    print_instructions()  # 打印指令
    while True:  # 无限循环，持续执行下面的代码
        print_intro()  # 调用 print_intro() 函数，打印程序介绍信息
        run_simulation()  # 调用 run_simulation() 函数，运行模拟程序


if __name__ == "__main__":  # 如果当前脚本被直接执行，而不是被导入其他模块
    main()  # 调用 main() 函数，作为程序的入口点
```