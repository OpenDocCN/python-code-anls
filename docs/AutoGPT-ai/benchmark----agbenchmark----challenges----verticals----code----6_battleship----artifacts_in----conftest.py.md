# `.\AutoGPT\benchmark\agbenchmark\challenges\verticals\code\6_battleship\artifacts_in\conftest.py`

```py
# 导入 pytest 模块
import pytest
# 从 abstract_class 模块中导入 ShipPlacement 和 Turn 类
from abstract_class import ShipPlacement, Turn
# 从 battleship 模块中导入 Battleship 类
from battleship import Battleship

# 创建一个名为 battleship_game 的 fixture，返回一个 Battleship 类的实例
@pytest.fixture
def battleship_game():
    return Battleship()

# 创建一个名为 initialized_game_id 的 fixture，依赖于 battleship_game fixture
@pytest.fixture
def initialized_game_id(battleship_game):
    # 创建一个游戏实例
    game_id = battleship_game.create_game()

    # 使用 battleship_game 的方法放置所有的船只
    sample_ship_placements = [
        ShipPlacement(
            ship_type="carrier", start={"row": 1, "column": "A"}, direction="horizontal"
        ),
        ShipPlacement(
            ship_type="battleship",
            start={"row": 2, "column": "A"},
            direction="horizontal",
        ),
        ShipPlacement(
            ship_type="cruiser", start={"row": 3, "column": "A"}, direction="horizontal"
        ),
        ShipPlacement(
            ship_type="submarine",
            start={"row": 4, "column": "A"},
            direction="horizontal",
        ),
        ShipPlacement(
            ship_type="destroyer",
            start={"row": 5, "column": "A"},
            direction="horizontal",
        ),
    ]

    for ship_placement in sample_ship_placements:
        # 使用 battleship_game 的方法放置船只
        battleship_game.create_ship_placement(game_id, ship_placement)

    return game_id

# 创建一个名为 game_over_fixture 的 fixture，依赖于 battleship_game 和 initialized_game_id fixtures
@pytest.fixture
def game_over_fixture(battleship_game, initialized_game_id):
    # 假设有一个 10x10 的网格，对所有可能的位置进行目标选择
    for row in range(1, 11):
        for column in list("ABCDEFGHIJ"):
            # 玩家 1 进行一次回合
            turn = Turn(target={"row": row, "column": column})
            battleship_game.create_turn(initialized_game_id, turn)

            # 玩家 2 进行一次回合，目标选择与玩家 1 相同
            battleship_game.create_turn(initialized_game_id, turn)

    # 在这个 fixture 结束时，游戏应该已经结束
    return initialized_game_id
```