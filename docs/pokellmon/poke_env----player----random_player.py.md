# `.\PokeLLMon\poke_env\player\random_player.py`

```py
# 定义一个随机玩家基线的模块
"""This module defines a random players baseline
"""

# 从 poke_env.environment 模块中导入 AbstractBattle 类
from poke_env.environment import AbstractBattle
# 从 poke_env.player.battle_order 模块中导入 BattleOrder 类
from poke_env.player.battle_order import BattleOrder
# 从 poke_env.player.player 模块中导入 Player 类
from poke_env.player.player import Player

# 定义一个 RandomPlayer 类，继承自 Player 类
class RandomPlayer(Player):
    # 定义 choose_move 方法，接受一个 AbstractBattle 对象作为参数，返回一个 BattleOrder 对象
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        # 调用 Player 类的 choose_random_move 方法选择一个随机的行动
        return self.choose_random_move(battle)
```