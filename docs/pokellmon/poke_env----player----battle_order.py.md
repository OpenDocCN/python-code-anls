# `.\PokeLLMon\poke_env\player\battle_order.py`

```
# 从 dataclasses 模块中导入 dataclass 装饰器
# 从 typing 模块中导入 Any, List, Optional, Union 类型
# 从 poke_env.environment.double_battle 模块中导入 DoubleBattle 类
# 从 poke_env.environment.move 模块中导入 Move 类
# 从 poke_env.environment.pokemon 模块中导入 Pokemon 类
from dataclasses import dataclass
from typing import Any, List, Optional, Union

# 定义一个名为 BattleOrder 的数据类
@dataclass
class BattleOrder:
    # order 属性可以是 Move 或 Pokemon 类型，初始值为 None
    order: Optional[Union[Move, Pokemon]]
    # mega, z_move, dynamax, terastallize, move_target 属性的默认值
    mega: bool = False
    z_move: bool = False
    dynamax: bool = False
    terastallize: bool = False
    move_target: int = DoubleBattle.EMPTY_TARGET_POSITION

    # 默认的指令字符串
    DEFAULT_ORDER = "/choose default"

    # 返回对象的字符串表示形式
    def __str__(self) -> str:
        return self.message

    # 返回消息字符串
    @property
    def message(self) -> str:
        # 如果 order 是 Move 类型
        if isinstance(self.order, Move):
            # 如果 order 的 id 是 "recharge"
            if self.order.id == "recharge":
                return "/choose move 1"

            # 构建消息字符串
            message = f"/choose move {self.order.id}"
            if self.mega:
                message += " mega"
            elif self.z_move:
                message += " zmove"
            elif self.dynamax:
                message += " dynamax"
            elif self.terastallize:
                message += " terastallize"

            # 如果 move_target 不是空目标位置
            if self.move_target != DoubleBattle.EMPTY_TARGET_POSITION:
                message += f" {self.move_target}"
            return message
        # 如果 order 是 Pokemon 类型
        elif isinstance(self.order, Pokemon):
            return f"/choose switch {self.order.species}"
        else:
            return ""

# 定义一个名为 DefaultBattleOrder 的类，继承自 BattleOrder 类
class DefaultBattleOrder(BattleOrder):
    # 初始化方法，不执行任何操作
    def __init__(self, *args: Any, **kwargs: Any):
        pass

    # 返回默认指令字符串
    @property
    def message(self) -> str:
        return self.DEFAULT_ORDER

# 定义一个名为 DoubleBattleOrder 的数据类，继承自 BattleOrder 类
@dataclass
class DoubleBattleOrder(BattleOrder):
    # 初始化方法，接受两个可选的 BattleOrder 参数
    def __init__(
        self,
        first_order: Optional[BattleOrder] = None,
        second_order: Optional[BattleOrder] = None,
    ):
        self.first_order = first_order
        self.second_order = second_order

    # 返回消息字符串
    @property
    # 返回合并后的消息字符串
    def message(self) -> str:
        # 如果存在第一和第二指令，则返回两者消息的组合
        if self.first_order and self.second_order:
            return (
                self.first_order.message
                + ", "
                + self.second_order.message.replace("/choose ", "")
            )
        # 如果只存在第一指令，则返回第一指令消息和默认消息的组合
        elif self.first_order:
            return self.first_order.message + ", default"
        # 如果只存在第二指令，则返回第二指令消息和默认消息的组合
        elif self.second_order:
            return self.second_order.message + ", default"
        # 如果都不存在指令，则返回默认指令消息
        else:
            return self.DEFAULT_ORDER

    # 静态方法，用于合并第一和第二指令列表生成双重战斗指令列表
    @staticmethod
    def join_orders(first_orders: List[BattleOrder], second_orders: List[BattleOrder]):
        # 如果第一和第二指令列表都存在
        if first_orders and second_orders:
            # 生成双重战斗指令列表，排除特定条件下的指令
            orders = [
                DoubleBattleOrder(first_order=first_order, second_order=second_order)
                for first_order in first_orders
                for second_order in second_orders
                if not first_order.mega or not second_order.mega
                if not first_order.z_move or not second_order.z_move
                if not first_order.dynamax or not second_order.dynamax
                if not first_order.terastallize or not second_order.terastallize
                if first_order.order != second_order.order
            ]
            # 如果生成了双重战斗指令列表，则返回该列表
            if orders:
                return orders
        # 如果只存在第一指令列表，则生成只包含第一指令的双重战斗指令列表
        elif first_orders:
            return [DoubleBattleOrder(first_order=order) for order in first_orders]
        # 如果只存在第二指令列表，则生成只包含第二指令的双重战斗指令列表
        elif second_orders:
            return [DoubleBattleOrder(first_order=order) for order in second_orders]
        # 如果两个指令列表都不存在，则返回只包含默认指令的双重战斗指令列表
        return [DefaultBattleOrder()]
# 定义一个名为ForfeitBattleOrder的类，继承自BattleOrder类
class ForfeitBattleOrder(BattleOrder):
    # 初始化方法，接受任意数量的位置参数和关键字参数
    def __init__(self, *args: Any, **kwargs: Any):
        # pass表示不做任何操作，保持方法的结构完整
        pass

    # 定义一个名为message的属性，返回字符串"/forfeit"
    @property
    def message(self) -> str:
        return "/forfeit"
```