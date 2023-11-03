# AutoGPT源码解析 31

# `benchmark/agbenchmark/challenges/verticals/code/6_battleship/artifacts_out/conftest.py`

这段代码是用于测试一个名为Battleship的游戏类。在这个游戏中，玩家需要创建一台属于不同类型的舰艇，然后放置这些舰艇在一个网格中。每个舰艇都有其特定的移动方向和范围。

首先，我们定义了一个抽象类ShipPlacement，所有继承它的类都必须实现这个方法。在这个方法中，我们可以创建一个将一个舰艇放置到游戏地图上的位置，并指定其移动方向（水平或垂直）。

接着，我们定义了一个具体的Battleship类，继承自抽象类ShipPlacement。这个类实现了创建和放置舰艇到游戏地图的具体方法。在创建游戏实例时，它需要调用抽象类中的方法来生成游戏地图，并创建一个将所有舰艇放置在游戏地图上的game_id。

最后，我们定义了一个测试Fixture类，用于在测试过程中创建并返回一个Battleship游戏实例。在测试中，我们可以使用这个Fixture来放置各种类型的舰艇，并运行游戏。


```py
import pytest
from abstract_class import ShipPlacement, Turn
from battleship import Battleship


@pytest.fixture
def battleship_game():
    return Battleship()


@pytest.fixture
def initialized_game_id(battleship_game):
    # Create a game instance
    game_id = battleship_game.create_game()

    # Place all the ships using battleship_game's methods
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
        # Place ship using battleship_game's methods
        battleship_game.create_ship_placement(game_id, ship_placement)

    return game_id


```

This code defines a fixture named `game_over_fixture` that takes two arguments: `battleship_game` and `initialized_game_id`.

The `game_over_fixture` fixture creates a temporary instance of the `Turn` class, which represents a player's turn in the game.

The fixture then uses a for loop to iterate through every possible position on the game board (assuming a 10x10 grid).

In the for loop, the `target` attribute of the `Turn` object is set to a dictionary containing the current row and column values. This means that on each iteration, the fixture creates a new `Turn` object with the given row and column values, and sets the `target` attribute to a dictionary containing the current state of the game.

The next line of code creates a new `Turn` object for player 1, with the `target` attribute set to the same dictionary as the `target` attribute of the `Turn` object. This is done with the `create_turn` method of the `battleship_game` object, passing in the `initialized_game_id` argument and the `turn` object.

The same is done for player 2, with the `create_turn` method of the `battleship_game` object, passing in the `initialized_game_id` argument and the `turn` object.

Finally, the fixture returns the `initialized_game_id` argument that was passed in when the fixture was defined.


```py
@pytest.fixture
def game_over_fixture(battleship_game, initialized_game_id):
    # Assuming 10x10 grid, target all possible positions
    for row in range(1, 11):
        for column in list("ABCDEFGHIJ"):
            # Player 1 takes a turn
            turn = Turn(target={"row": row, "column": column})
            battleship_game.create_turn(initialized_game_id, turn)

            # Player 2 takes a turn, targeting the same position as Player 1
            battleship_game.create_turn(initialized_game_id, turn)

    # At the end of this fixture, the game should be over
    return initialized_game_id

```

# `benchmark/agbenchmark/challenges/verticals/code/6_battleship/artifacts_out/test_negative.py`

这段代码是用来测试一个名为BattleshipGame的抽象类，其中包含一个名为ShipPlacement的类，以及一个名为Turn的类。主要目的是使用pytest库来运行测试，并验证BattleshipGame和ShipPlacement类是否符合预期行为。

具体来说，代码会创建一个名为"test_ship_placement_out_of_bounds"的测试函数。在该函数中，使用BattleshipGame类的create_game方法创建了一个游戏对象，并创建了一个ShipPlacement对象。然后，使用ShipPlacement对象的ship_type属性设置游戏对象的类型为"battleship"，并设置其开始位置为(11, "Z")，即游戏中的第11行，从左上角开始旋转。最后，使用ShipPlacement类的方法创建一个out_of_bounds_ship对象。

在测试函数中，使用try-except语句来捕获任何ValidationError异常，并跳过该异常的警告。然而，在创建out_of_bounds_ship对象时，使用pytest.raises方法抛出了一个ValueError异常，该异常的参数是游戏对象和ShipPlacement对象。这个异常被用于在测试函数中验证BattleshipGame类是否可以创建该类型的游戏对象。


```py
import pytest
from abstract_class import ShipPlacement, Turn
from pydantic import ValidationError


def test_ship_placement_out_of_bounds(battleship_game):
    game_id = battleship_game.create_game()

    try:
        out_of_bounds_ship = ShipPlacement(
            ship_type="battleship",
            start={"row": 11, "column": "Z"},
            direction="horizontal",
        )
    except ValidationError:  # Use the directly imported ValidationError class
        pass
    else:
        with pytest.raises(ValueError, match="Placement out of bounds"):
            battleship_game.create_ship_placement(game_id, out_of_bounds_ship)


```

这两个测试函数是用于测试Battleship游戏的逻辑的。

第一个测试函数 `test_no_ship_overlap` 的目的是验证在同一位置不能有战舰之间的重叠。具体来说，该函数会创建一个游戏对象 `battleship_game`，然后创建一个 `ShipPlacement` 对象 `placement1`，设置其船型为 "battleship"，位置为 "row 1, column A"，朝向为 "horizontal"(即水平方向)。然后，它尝试创建另一个 `ShipPlacement` 对象 `placement2`，设置其船型为 "cruiser"，位置为 "row 1, column A"，朝向为 "horizontal"(即水平方向)。然而，该函数会使用 `pytest.raises()` 函数 raise 一个 `ValueError`，匹配到 "All ships must be placed before starting turns"(即所有战舰必须在 placement 之前放置)。这个错误信息表明，在同一位置放置多个船会报错，因为在该位置已经有战舰了。

第二个测试函数 `test_cant_hit_before_ships_placed` 的目的是验证在放置了多个船之后，不能在同一位置继续放置更多船。具体来说，该函数会创建一个游戏对象 `battleship_game`，然后创建一个 `ShipPlacement` 对象 `placement1`，设置其船型为 "battleship"，位置为 "row 1, column A"，朝向为 "horizontal"(即水平方向)。然后，它创建了两个 `ShipPlacement` 对象 `placement2` 和 `placement3`，设置其船型为 "cruiser"，位置为 "row 4, column D"，朝向为 "horizontal"(即水平方向)。然后，它创建了一个 `Turn` 对象 `turn`，设置其目标为 `{row: 1, column: A}`(即在 "row 1, column A" 的位置)。最后，它使用 `pytest.raises()` 函数 raise 一个 `ValueError`，匹配到 "All ships must be placed before starting turns"(即所有战舰必须在 placement 之前放置)。这个错误信息表明，在同一位置放置了多个船之后，不能在同一位置继续放置更多船。


```py
def test_no_ship_overlap(battleship_game):
    game_id = battleship_game.create_game()
    placement1 = ShipPlacement(
        ship_type="battleship", start={"row": 1, "column": "A"}, direction="horizontal"
    )
    battleship_game.create_ship_placement(game_id, placement1)
    placement2 = ShipPlacement(
        ship_type="cruiser", start={"row": 1, "column": "A"}, direction="horizontal"
    )
    with pytest.raises(ValueError):
        battleship_game.create_ship_placement(game_id, placement2)


def test_cant_hit_before_ships_placed(battleship_game):
    game_id = battleship_game.create_game()
    placement1 = ShipPlacement(
        ship_type="battleship", start={"row": 1, "column": "A"}, direction="horizontal"
    )
    battleship_game.create_ship_placement(game_id, placement1)
    placement2 = ShipPlacement(
        ship_type="cruiser", start={"row": 4, "column": "D"}, direction="horizontal"
    )
    battleship_game.create_ship_placement(game_id, placement2)
    turn = Turn(target={"row": 1, "column": "A"})
    with pytest.raises(
        ValueError, match="All ships must be placed before starting turns"
    ):
        battleship_game.create_turn(game_id, turn)


```

这两行代码是针对一个名为 `test_cant_place_ship_after_all_ships_placed` 的测试函数。

首先，这两行导入了 `battleship_game` 和 `ShipPlacement` 类。

然后，这两行代码创建了一个 `game` 对象，其中 `initialized_game_id` 参数是在创建 `game` 对象时传递的。

接下来，这两行代码创建了一个 `ShipPlacement` 对象，其中 `ship_type` 属性设置为 "carrier", `start` 属性设置为 `{"row": 2, "column": "E"}` , `direction` 属性设置为 "horizontal"。

接着，这两行代码使用 `with pytest.raises(ValueError, match="All ships are already placed. Cannot place more ships.")` 来捕获 `ValueError` 异常的异常。这个异常是当尝试创建一个已经放置了所有船只的游戏对象时发生的。

最后，这两行代码使用 `battleship_game.create_ship_placement(initialized_game_id, additional_ship)` 来尝试创建一个新船只的位置。如果这个操作成功，它将添加一个新的船只到游戏对象中。


```py
def test_cant_place_ship_after_all_ships_placed(battleship_game, initialized_game_id):
    game = battleship_game.get_game(initialized_game_id)
    additional_ship = ShipPlacement(
        ship_type="carrier", start={"row": 2, "column": "E"}, direction="horizontal"
    )

    with pytest.raises(
        ValueError, match="All ships are already placed. Cannot place more ships."
    ):
        battleship_game.create_ship_placement(initialized_game_id, additional_ship)


def test_ship_placement_invalid_direction(battleship_game):
    game_id = battleship_game.create_game()

    with pytest.raises(ValueError, match="Invalid ship direction"):
        invalid_direction_ship = ShipPlacement(
            ship_type="battleship",
            start={"row": 1, "column": "A"},
            direction="diagonal",
        )
        battleship_game.create_ship_placement(game_id, invalid_direction_ship)


```



这三条测试用例函数旨在测试Battleship游戏中的两个功能：创建游戏和创建船位。这两个测试用例函数分别测试创建具有不同类型船位和超出游戏边界位置的船位，旨在验证游戏是否可以正确地处理这些情况。

具体来说，test_invalid_ship_type函数测试创建具有无效船位的游戏，具体来说，测试创建具有 "spacecraft" 船位的游戏，该船位的位置为(1,1)。如果测试成功，游戏应该会抛出一个ValueError异常，匹配诸如 "Invalid ship type" 这样的错误信息。

test_ship_placement_extends_beyond_boundaries函数测试创建超出游戏边界位置的船位，具体来说，测试创建具有 "cruiser" 船位的游戏位置，该位置超越了游戏中的列边界。如果测试成功，游戏应该会抛出一个ValueError异常，匹配诸如 "Ship extends beyond board boundaries" 这样的错误信息。

这两个测试用例函数使用pytest库调用Battleship游戏中的create_game和create_ship_placement函数，然后使用pytest的raises函数来捕获异常并验证游戏是否可以正确地处理这些情况。


```py
def test_invalid_ship_type(battleship_game):
    game_id = battleship_game.create_game()
    invalid_ship = ShipPlacement(
        ship_type="spacecraft", start={"row": 1, "column": "A"}, direction="horizontal"
    )
    with pytest.raises(ValueError, match="Invalid ship type"):
        battleship_game.create_ship_placement(game_id, invalid_ship)


def test_ship_placement_extends_beyond_boundaries(battleship_game):
    game_id = battleship_game.create_game()

    with pytest.raises(ValueError, match="Ship extends beyond board boundaries"):
        ship_extending_beyond = ShipPlacement(
            ship_type="battleship",
            start={"row": 1, "column": "H"},
            direction="horizontal",
        )
        battleship_game.create_ship_placement(game_id, ship_extending_beyond)

    with pytest.raises(ValueError, match="Ship extends beyond board boundaries"):
        ship_extending_beyond = ShipPlacement(
            ship_type="cruiser", start={"row": 9, "column": "A"}, direction="vertical"
        )
        battleship_game.create_ship_placement(game_id, ship_extending_beyond)

```

# `benchmark/agbenchmark/challenges/verticals/code/6_battleship/artifacts_out/test_positive.py`

该代码的主要目的是测试两个函数：`test_turns_and_results` 和 `test_game_status_and_winner`。这两个函数都是基于抽象类 `Turn` 和 `Game` 进行的。

具体来说，`test_turns_and_results` 函数会创建一个包含两个决策（目标位置）的 `Turn` 对象，然后将其传递给 `battleship_game` 类的 `create_turn` 函数。接着，函数会获取返回的响应（结果），并检查其结果是否为“命中”或“未命中”。如果结果为“命中”，则需要检查返回的决策的船类是否为“航母”。最后，函数会使用 `get_game` 函数获取游戏对象，并检查当前的 `Turn` 对象是否属于该游戏对象。

`test_game_status_and_winner` 函数会创建一个初始游戏对象，然后使用 `create_game` 函数创建一个新的游戏。接着，函数使用 `get_game_status` 函数获取游戏对象的状态，并使用自己的 `assertIs` 函数检查是否已经结束的游戏状态是否为真。如果是，函数使用 `get_winner` 函数获取获胜者，并检查该获胜者是否为 `None`。最后，函数会使用 `get_game` 函数获取游戏对象，并使用自己的 `assertIs` 函数检查获胜者是否为游戏对象。


```py
from abstract_class import ShipPlacement, Turn


def test_turns_and_results(battleship_game, initialized_game_id):
    turn = Turn(target={"row": 1, "column": "A"})
    response = battleship_game.create_turn(initialized_game_id, turn)

    assert response.result in ["hit", "miss"]
    if response.result == "hit":
        assert response.ship_type == "carrier"
    game = battleship_game.get_game(initialized_game_id)
    assert turn in game.turns


def test_game_status_and_winner(battleship_game):
    game_id = battleship_game.create_game()
    status = battleship_game.get_game_status(game_id)
    assert isinstance(status.is_game_over, bool)
    if status.is_game_over:
        winner = battleship_game.get_winner(game_id)
        assert winner is not None


```



这两段代码是对游戏进行测试用例函数。

1. `test_delete_game`函数的作用是测试是否可以从游戏对象中删除游戏。具体的测试逻辑是先创建一个游戏对象，然后调用该游戏对象的 `delete_game` 方法，并检查是否成功。如果成功，则游戏对象中应该没有任何关于该游戏的信息，因为所有游戏中包含的游戏对象都应该被删除。

2. `test_ship_rotation`函数的作用是测试是否可以更改游戏中的舰艇位置。具体的测试逻辑是创建一个游戏对象，然后创建两个舰艇位置，并将它们的相关信息设置为游戏中的舰艇类型和位置，最后检查是否可以成功更改它们。如果可以成功更改它们，则说明游戏中的舰艇位置可以被修改。


```py
def test_delete_game(battleship_game):
    game_id = battleship_game.create_game()
    battleship_game.delete_game(game_id)
    assert battleship_game.get_game(game_id) is None


def test_ship_rotation(battleship_game):
    game_id = battleship_game.create_game()
    placement_horizontal = ShipPlacement(
        ship_type="battleship", start={"row": 1, "column": "B"}, direction="horizontal"
    )
    battleship_game.create_ship_placement(game_id, placement_horizontal)
    placement_vertical = ShipPlacement(
        ship_type="cruiser", start={"row": 3, "column": "D"}, direction="vertical"
    )
    battleship_game.create_ship_placement(game_id, placement_vertical)
    game = battleship_game.get_game(game_id)
    assert placement_horizontal in game.ships
    assert placement_vertical in game.ships


```

这两段代码是针对一个名为"battleship_game"的游戏进行的测试。它们的目的是测试游戏的一些功能。

首先，是"test_game_state_updates"函数。它的作用是创建一个包含两个子游戏的的游戏对象，并调用这个游戏对象的"create_turn"方法来更新游戏状态。在这个函数中，使用了一个名为"Turn"的类来创建游戏状态的每个元素。然后，将创建的游戏对象传递给一个名为"battleship_game"的实例，并调用它的"create_turn"方法来更新游戏状态。

接下来是"test_ship_sinking_feedback"函数。它的作用是模拟船沉没的过程，并输出结果。在这个函数中，首先创建一个包含四个元素的列表，其中每个元素都是{"row": 2, "column": "A", "hit": "A"}。然后，使用两层循环来遍历这个列表，并对于每个元素，创建一个"Turn"对象，并使用"create_turn"方法来更新游戏状态。然后，在两层循环内部，分别调用"battleship_game"的"create_turn"方法，并传递不同层的静态移动。最后，比较结果，如果模拟成功，则输出"sunk"。


```py
def test_game_state_updates(battleship_game, initialized_game_id):
    turn = Turn(target={"row": 3, "column": "A"})
    battleship_game.create_turn(initialized_game_id, turn)

    game = battleship_game.get_game(initialized_game_id)

    target_key = (3, ord("A") - ord("A"))
    assert target_key in game.board and game.board[target_key] == "hit"


def test_ship_sinking_feedback(battleship_game, initialized_game_id):
    hits = ["A", "B", "C", "D"]
    static_moves = [
        {"row": 1, "column": "E"},
        {"row": 1, "column": "F"},
        {"row": 1, "column": "G"},
        {"row": 1, "column": "H"},
    ]

    for index, hit in enumerate(hits):
        turn = Turn(target={"row": 2, "column": hit})
        response = battleship_game.create_turn(initialized_game_id, turn)
        assert response.ship_type == "battleship"

        static_turn = Turn(target=static_moves[index])
        battleship_game.create_turn(initialized_game_id, static_turn)

    assert response.result == "sunk"


```

这两段代码是针对一个名为`battleship_game`的游戏进行测试的函数，主要的目的是验证游戏中的战斗舰是否具有正确的结构和位置。

首先，`test_restart_game`函数的作用是验证在创建和删除游戏之后，能否正确地重新创建和使用游戏。具体来说，该函数创建一个新的游戏，使用游戏ID输出新的游戏，然后在输出的游戏ID上创建新的游戏并使用它来访问游戏。通过对比输出和实际输出的游戏ID，可以验证游戏创建和删除操作的成功和失败。

其次，`test_ship_edge_overlapping`函数的作用是验证两个战斗舰的位置是否正确。具体来说，该函数使用`ShipPlacement`类创建两个不同的战斗舰（分别命名为`first_ship`和`next_ship`），并使用`battleship_game.create_ship_placement`函数将它们插入到游戏地图中的两个不同的位置。最后，使用`battleship_game.get_game`函数获取游戏ID，并使用`assert`语句验证战斗舰是否存在于游戏中。通过这种方法，可以验证战斗舰在游戏中的正确位置。


```py
def test_restart_game(battleship_game):
    game_id = battleship_game.create_game()
    battleship_game.delete_game(game_id)
    game_id = (
        battleship_game.create_game()
    )  # Use the returned game_id after recreating the game
    game = battleship_game.get_game(game_id)
    assert game is not None


def test_ship_edge_overlapping(battleship_game):
    game_id = battleship_game.create_game()

    first_ship = ShipPlacement(
        ship_type="battleship", start={"row": 1, "column": "A"}, direction="horizontal"
    )
    battleship_game.create_ship_placement(game_id, first_ship)

    next_ship = ShipPlacement(
        ship_type="cruiser", start={"row": 1, "column": "E"}, direction="horizontal"
    )
    battleship_game.create_ship_placement(game_id, next_ship)

    game = battleship_game.get_game(game_id)
    assert first_ship in game.ships
    assert next_ship in game.ships


```

这两段代码是针对一个游戏（battleship game）的测试用例。它们的作用是验证游戏在不同的状态下是否按照预期工作。

1. `test_game_state_after_ship_placement` 函数测试游戏在船只部署后的状态。它创建一个新的游戏，然后使用 `ShipPlacement` 类将一个 battleship 部署到游戏中的指定位置。最后，它使用 `get_game` 函数获取部署后的游戏，并检查位置 `ship_placement` 是否存在于游戏中的所有船只中。这个测试用例证明了船只部署后，游戏状态可以被正确地访问和检查。

2. `test_game_state_after_turn` 函数测试游戏在转了一圈（一个 round）后的状态。它创建一个 turnship，使用 `create_turn` 函数将 turnship 移动到游戏中的指定位置，然后使用 `get_game` 函数获取游戏并检查 turnship 是否击中目标。如果击中目标，游戏板上的格子显示 "hit"。否则，游戏板上的格子显示 "miss"。这个测试用例证明了 turnship 移动后，游戏状态可以被正确地访问和检查。


```py
def test_game_state_after_ship_placement(battleship_game):
    game_id = battleship_game.create_game()

    ship_placement = ShipPlacement(
        ship_type="battleship", start={"row": 1, "column": "A"}, direction="horizontal"
    )
    battleship_game.create_ship_placement(game_id, ship_placement)

    game = battleship_game.get_game(game_id)
    assert ship_placement in game.ships


def test_game_state_after_turn(initialized_game_id, battleship_game):
    turn = Turn(target={"row": 1, "column": "A"})
    response = battleship_game.create_turn(initialized_game_id, turn)

    game = battleship_game.get_game(initialized_game_id)

    if response.result == "hit":
        assert game.board[(1, 0)] == "hit"
    else:
        assert game.board[1][0] == "miss"


```

这两段代码是针对一个名为"test_multiple_hits_on_ship"的函数和"test_game_over_condition"函数，它们的目的是测试一个名为"battleship_game"的游戏是否在同一位置多次受到攻击时能够正确处理，并在游戏状态检测中检查游戏是否已经结束。

具体来说，这两段代码分别模拟了以下情况：

在"test_multiple_hits_on_ship"中，我们模拟了游戏中的攻击行为。我们定义了一个包含五个位置的列表"hit_positions"，然后使用for循环遍历这些位置。在每次循环中，我们创建一个名为Turn的类实例，并设置其目标为{"row": 1, "column": pos}。然后，我们使用battleship_game.create_turn方法来模拟玩家进行的一轮攻击。我们使用for循环来模拟同一位置多次攻击的行为，最后，我们检查如果攻击次数已经等于该位置的攻击次数上限，则游戏是否已经被击沉。否则，我们检查攻击是否成功，并检查结果。

在"test_game_over_condition"中，我们模拟了游戏结束的情况。我们使用for循环遍历每一行，然后使用Turn类创建多次攻击。每次攻击都会尝试改变游戏状态，并使用battleship_game.get_game_status方法获取游戏状态。我们检查游戏状态是否已经结束，如果游戏已经结束，则说明游戏已经被击沉，游戏状态为True，否则游戏状态为False。


```py
def test_multiple_hits_on_ship(battleship_game, initialized_game_id):
    hit_positions = ["A", "B", "C", "D", "E"]

    for index, pos in enumerate(hit_positions):
        turn = Turn(target={"row": 1, "column": pos})
        response = battleship_game.create_turn(initialized_game_id, turn)

        if index == len(hit_positions) - 1:
            assert response.result == "sunk"
        else:
            assert response.result == "hit"


def test_game_over_condition(battleship_game, initialized_game_id):
    for row in range(1, 11):
        for column in list("ABCDEFGHIJ"):
            turn = Turn(target={"row": row, "column": column})
            battleship_game.create_turn(initialized_game_id, turn)

            battleship_game.create_turn(initialized_game_id, turn)

    status = battleship_game.get_game_status(initialized_game_id)
    assert status.is_game_over

```

# `benchmark/agbenchmark/challenges/verticals/code/6_battleship/artifacts_out/__init__.py`

我需要更具体的上下文来回答你的问题。可以请你提供更多背景信息或者详细描述一下你想要了解的解释是什么？这样我才能够帮助你更好地。


```py

```

# `benchmark/agbenchmark/reports/agent_benchmark_config.py`

这段代码的作用是读取一个名为 "agbenchmark_config.json" 的JSON文件中的基准配置，并将其存储为类型的 AgentBenchmarkConfig 对象。该对象然后被存储在当前目录下的 "agbenchmark_config" 目录中的 "config.json" 文件中。

具体来说，代码首先导入了 agbenchmark_utils.data_types 命名空间中的 AgentBenchmarkConfig 类。接着，定义了一个名为 get_agent_benchmark_config 的函数，该函数从当前目录下的 "agbenchmark_config" 目录中读取基准配置文件，并将其存储为类型的 AgentBenchmarkConfig 对象。

在函数内部，使用了一组来自 agbenchmark_utils.data_types 的辅助函数，包括 json.load() 函数来读取基准配置文件中的 JSON 数据和 str() 函数来获取文件路径。然后，使用这些函数读取到了基准配置文件并将其转换为了 AgentBenchmarkConfig 对象。

接着，使用 try-except 语句来处理 JSON decode 错误。如果文件读取成功，则返回 AgentBenchmarkConfig 对象。如果文件名包含非法字符或者文件不存在，则输出错误信息并引发异常。


```py
import json
from pathlib import Path

from agbenchmark.utils.data_types import AgentBenchmarkConfig


def get_agent_benchmark_config() -> AgentBenchmarkConfig:
    agent_benchmark_config_path = str(Path.cwd() / "agbenchmark_config" / "config.json")
    try:
        with open(agent_benchmark_config_path, "r") as f:
            agent_benchmark_config = AgentBenchmarkConfig(**json.load(f))
            agent_benchmark_config.agent_benchmark_config_path = (
                agent_benchmark_config_path
            )
            return agent_benchmark_config
    except json.JSONDecodeError:
        print("Error: benchmark_config.json is not a valid JSON file.")
        raise

```

# `benchmark/agbenchmark/reports/ReportManager.py`

The `SingletonReportManager` class is a singleton class that manages the Singleman Report system. It is responsible for fetching the Singleman reports from the respective manager classes (`ReportManager` class for the regression reports, `ReportManager` class for the info reports, and `ReportManager` class for the internal info reports) and combining them into one report.

The Singleman Report Manager class has the following methods:

* `__new__(cls)`: The constructor method, which is responsible for creating a new instance of the class. It does the same as the `__init__` method in Python, but it is defined as a class method.
* `__init__(self, config):`: The constructor method, which takes an `AgentBenchmarkConfig` object


```py
import copy
import json
import os
import sys
import time
from datetime import datetime, timezone

from agbenchmark.reports.processing.graphs import save_single_radar_chart
from agbenchmark.reports.processing.process_report import get_agent_category
from agbenchmark.reports.processing.report_types import Report
from agbenchmark.utils.data_types import AgentBenchmarkConfig
from agbenchmark.utils.utils import get_highest_success_difficulty


class SingletonReportManager:
    instance = None

    def __new__(cls):
        from agbenchmark.reports.agent_benchmark_config import (
            get_agent_benchmark_config,
        )

        if not cls.instance:
            cls.instance = super(SingletonReportManager, cls).__new__(cls)

            agent_benchmark_config = get_agent_benchmark_config()
            benchmark_start_time_dt = datetime.now(
                timezone.utc
            )  # or any logic to fetch the datetime

            # Make the Managers class attributes
            cls.REGRESSION_MANAGER = ReportManager(
                agent_benchmark_config.get_regression_reports_path(),
                benchmark_start_time_dt,
            )
            cls.INFO_MANAGER = ReportManager(
                str(
                    agent_benchmark_config.get_reports_path(benchmark_start_time_dt)
                    / "report.json"
                ),
                benchmark_start_time_dt,
            )
            cls.INTERNAL_INFO_MANAGER = ReportManager(
                agent_benchmark_config.get_success_rate_path(), benchmark_start_time_dt
            )

        return cls.instance

    @classmethod
    def clear_instance(cls):
        cls.instance = None
        cls.REGRESSION_MANAGER = None
        cls.INFO_MANAGER = None
        cls.INTERNAL_INFO_MANAGER = None


```

This appears to be a Python class that generates a Radar Chart for agroup of tests. Here's a summary of how the class works:

1. It reads the tests from a file specified in the `config` parameter.
2. It gets the completion time (i.e., the time when the test suite was completed) and benchmarks start time from the tests.
3. It creates a dictionary called `metrics` that includes the run time, highest difficulty, and total costs for each test.
4. It imports the Report class to parse the metrics of the tests.
5. It imports the json.loads function to parse the config file.
6. It parses the tests and generates a Report object.
7. It radars the Report object and saves it in a specified directory.

It appears that the class also imports some other classes, such as `time` and `copy`, as well as a function called `get_highest_success_difficulty`, but these are not defined in the class's source code.


```py
class ReportManager:
    """Abstracts interaction with the regression tests file"""

    def __init__(self, filename: str, benchmark_start_time: str):
        self.filename = filename
        self.start_time = time.time()
        self.benchmark_start_time = benchmark_start_time

        self.load()

    def load(self) -> None:
        if not os.path.exists(self.filename):
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            with open(self.filename, "w") as f:
                pass

        try:
            with open(self.filename, "r") as f:
                file_content = (
                    f.read().strip()
                )  # read the content and remove any leading/trailing whitespace
                if file_content:  # if file is not empty, load the json
                    data = json.loads(file_content)
                    self.tests = {k: data[k] for k in sorted(data)}
                else:  # if file is empty, assign an empty dictionary
                    self.tests = {}
        except FileNotFoundError:
            self.tests = {}
        except json.decoder.JSONDecodeError:  # If JSON is invalid
            self.tests = {}
        self.save()

    def save(self) -> None:
        with open(self.filename, "w") as f:
            json.dump(self.tests, f, indent=4)

    def add_test(self, test_name: str, test_details: dict | list) -> None:
        if test_name.startswith("Test"):
            test_name = test_name[4:]
        self.tests[test_name] = test_details

        self.save()

    def remove_test(self, test_name: str) -> None:
        if test_name in self.tests:
            del self.tests[test_name]
            self.save()

    def reset(self) -> None:
        self.tests = {}
        self.save()

    def end_info_report(self, config: AgentBenchmarkConfig) -> None:
        command = " ".join(sys.argv)

        self.tests = {
            "command": command.split(os.sep)[-1],
            "benchmark_git_commit_sha": "---",
            "agent_git_commit_sha": "---",
            "completion_time": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S+00:00"
            ),
            "benchmark_start_time": self.benchmark_start_time.strftime(
                "%Y-%m-%dT%H:%M:%S+00:00"
            ),
            "metrics": {
                "run_time": str(round(time.time() - self.start_time, 2)) + " seconds",
                "highest_difficulty": get_highest_success_difficulty(self.tests),
                "total_cost": self.get_total_costs(),
            },
            "tests": copy.copy(self.tests),
            "config": {
                k: v for k, v in json.loads(config.json()).items() if v is not None
            },
        }
        Report.parse_obj(self.tests)

        converted_data = Report.parse_obj(self.tests)

        agent_categories = get_agent_category(converted_data)
        if len(agent_categories) > 1:
            save_single_radar_chart(
                agent_categories,
                config.get_reports_path(self.benchmark_start_time) / "radar_chart.png",
            )

        self.save()

    def get_total_costs(self):
        total_cost = 0
        all_costs_none = True
        for test_name, test_data in self.tests.items():
            cost = test_data["metrics"].get(
                "cost", 0
            )  # gets the cost or defaults to 0 if cost is missing

            if cost is not None:  # check if cost is not None
                all_costs_none = False
                total_cost += cost  # add cost to total
        if all_costs_none:
            total_cost = None
        return total_cost

```

# `benchmark/agbenchmark/reports/reports.py`

这段代码的主要作用是获取之前进行过的测试的结果，并输出一个包含成功率百分比的信息字典。

具体来说，代码首先导入了需要使用的库，包括`json`,`os`,`sys`，以及一个名为`Any`的泛型类型，一个名为`Dict`的键类型。

接下来，定义了一个名为`get_previous_test_results`的函数，它接受一个名为`test_name`的参数和一个名为`info_details`的信息字典作为输入。函数使用`SingletonReportManager`类的一个名为`tests.get`的方法，获取之前进行过的测试的编号，然后检查是否是一个模拟测试，如果是，就添加模拟测试的结果。否则，将之前测试的结果添加到信息字典中，并使用`add_test`方法将测试添加到信息字典中。最后，使用`calculate_success_percentage`函数计算成功率百分比，并将结果显示为信息字典中的`metrics`键中的`success_%`值。

整段代码的逻辑可以被解释为一个代理程序，它在运行时获取之前进行的所有测试的结果，并输出一个包含成功率百分比的信息字典。这个代理程序可以在实际应用中用于比较新的测试结果和之前测试的结果，以及比较测试结果和预期结果。


```py
import json
import os
import sys
from typing import Any, Dict

from agbenchmark.__main__ import CHALLENGES_ALREADY_BEATEN
from agbenchmark.reports.agent_benchmark_config import get_agent_benchmark_config
from agbenchmark.reports.ReportManager import SingletonReportManager
from agbenchmark.utils.data_types import DifficultyLevel
from agbenchmark.utils.get_data_from_helicone import get_data_from_helicone
from agbenchmark.utils.utils import calculate_success_percentage


def get_previous_test_results(
    test_name: str, info_details: dict[str, Any]
) -> list[bool]:
    agent_tests: dict[str, list[bool]] = {}
    mock = os.getenv("IS_MOCK")  # Check if --mock is in sys.argv

    prev_test_results = SingletonReportManager().INTERNAL_INFO_MANAGER.tests.get(
        test_name, []
    )

    if not mock:
        # only add if it's an actual test
        prev_test_results.append(info_details["metrics"]["success"])
        SingletonReportManager().INTERNAL_INFO_MANAGER.add_test(
            test_name, prev_test_results
        )

    # can calculate success rate regardless of mock
    info_details["metrics"]["success_%"] = calculate_success_percentage(
        prev_test_results
    )

    return prev_test_results


```

This appears to be a Python class that manages the `::failing_reason` attribute in an ` item` object.  The attribute is a dictionary that contains information about the failed test, including the difficulty of the test, the data path for the challenge location, and any additional details about the failure, such as whether the failure was a regression test or not.

It appears that the `challenge_location` attribute is a getter that retrieves the `CHALLENGE_LOCATION` attribute from the `cls` attribute of the `item.cls` object.  If this attribute is not present in the `cls` object, it defaults to an empty string.

The `test_name` attribute is a property that is generated based on the name of the node ID in the `item.nodeid` attribute.

The `info_details` attribute is another dictionary that is generated based on the information in the `challenge_location` attribute.  It contains information about the failed test, including the difficulty level, the data path for the challenge location, and any additional details about the failure, such as whether the failure was a regression test or not.

It appears that the `call` method is used to invoke the `metadata` attribute of the `challenge_location` object.  If the `call` method raises an exception, it sets the `metrics` attribute of the `info_details` dictionary to `None`.  If the `call` method does not raise an exception and the `execute` method of the `item.cls` object is called, it updates the `prev_test_results` list with the results of the test and sets the `info_details` attribute of the `item` object to the `info_details` dictionary.

It appears that the `execute` method of the `item.cls` object is responsible for updating the `prev_test_results` list with the results of the test and setting the `info_details` attribute of the `item` object to the `info_details` dictionary.

I'm sorry, but I'm unable to browse the internet and don't have any specific knowledge about the `item` object or the context in which this class is being used.  I'm also unable to see the definition of the `cls` attribute, so I'm unable to provide any additional information about the `item.cls` object.


```py
def update_regression_tests(
    prev_test_results: list[bool],
    info_details: dict,
    test_name: str,
    test_details: dict,
) -> None:
    if len(prev_test_results) >= 3 and prev_test_results[-3:] == [True, True, True]:
        # if the last 3 tests were successful, add to the regression tests
        info_details["is_regression"] = True
        SingletonReportManager().REGRESSION_MANAGER.add_test(test_name, test_details)


def generate_single_call_report(
    item: Any,
    call: Any,
    challenge_data: dict[str, Any],
    answers: dict[str, Any],
    challenge_location,
    test_name,
) -> None:
    try:
        difficulty = challenge_data["info"]["difficulty"]
    except KeyError:
        return None

    if isinstance(difficulty, DifficultyLevel):
        difficulty = difficulty.value

    # Extract the challenge_location from the class
    # challenge_location: str = getattr(item.cls, "CHALLENGE_LOCATION", "")
    # test_name = item.nodeid.split("::")[1]
    # item.test_name = test_name

    test_details = {
        "difficulty": difficulty,
        "data_path": challenge_location,
    }

    info_details: Any = {
        "data_path": challenge_location,
        "is_regression": False,
        "category": challenge_data["category"],
        "task": challenge_data["task"],
        "answer": challenge_data["ground"]["answer"],
        "description": challenge_data["info"]["description"],
        "metrics": {
            "difficulty": difficulty,
            "success": False,
            "attempted": True,
        },
        # "answers": answers,
    }
    if answers:
        info_details["answers"] = answers

    if "metadata" in challenge_data:
        info_details["metadata"] = challenge_data["metadata"]

    mock = os.getenv("IS_MOCK")  # Check if --mock is in sys.argv
    if call:
        if call.excinfo is None:
            info_details["metrics"]["success"] = True
        else:
            if not mock:  # don't remove if it's a mock test
                SingletonReportManager().REGRESSION_MANAGER.remove_test(test_name)
            info_details["metrics"]["fail_reason"] = str(call.excinfo.value)
            if call.excinfo.typename == "Skipped":
                info_details["metrics"]["attempted"] = False

    prev_test_results: list[bool] = get_previous_test_results(test_name, info_details)

    update_regression_tests(prev_test_results, info_details, test_name, test_details)

    # user facing reporting
    if item:
        item.info_details = info_details

    return info_details


```

这段代码是一个 Python 函数，名为 `finalize_reports`，功能是帮助用户统计测试用例的运行时和结果。它接受两个参数：`item` 和 `challenge_data`。

首先，它获取 `item` 对象的 `user_properties` 属性中的 `run_time` 字段。然后，获取 `item` 对象中 `info_details` 属性的值，或者默认值为 `{}`。同时，它还获取 `item` 对象中 `test_name` 属性的值，默认值为空字符串。

接下来，代码根据这两个值中的一个或同时来判断是否可以获取到 `run_time` 的值。如果 `run_time` 存在，那么它会被添加到 `info_details` 字典中，其中键为 `"metrics"`，值为 `"run_time"`。然后，代码会根据 `info_details` 和 `test_name` 值中的一个或同时来判断是否可以访问到一些元数据。如果 `"attempted"` 存在，那么它的值将为 `False`，否则它的值将永远不会成为 `False`。

接下来，代码会根据 `run_time` 值来判断是否已经超过了 `challenge_data` 对象中 `cutoff` 值的 threshold。如果 `run_time` 大于或等于 `challenge_data` 中的 `cutoff` 值，那么代码将添加结果到已完成的报告中。最后，代码会将报告添加到 `SingletonReportManager` 类中的 `INFO_MANAGER` 方法中，以便将结果添加到该报告中。


```py
def finalize_reports(item: Any, challenge_data: dict[str, Any]) -> None:
    run_time = dict(item.user_properties).get("run_time")

    info_details = getattr(item, "info_details", {})
    test_name = getattr(item, "test_name", "")

    if info_details and test_name:
        if run_time is not None:
            cost = None
            if "--mock" not in sys.argv and os.environ.get("HELICONE_API_KEY"):
                print("Getting cost from Helicone")
                cost = get_data_from_helicone(test_name)

            info_details["metrics"]["cost"] = cost

            if info_details["metrics"].get("success", None) is None:
                info_details["metrics"]["attempted"] = False
                info_details["metrics"]["success"] = False
            elif (
                info_details["metrics"].get("success") is False
                and "attempted" not in info_details["metrics"]
            ):
                info_details["metrics"]["attempted"] = False

            info_details["metrics"]["run_time"] = f"{str(round(run_time, 3))} seconds"

            info_details["reached_cutoff"] = float(run_time) > challenge_data["cutoff"]

            if "--mock" not in sys.argv:
                update_challenges_already_beaten(info_details, test_name)
                if info_details.get("tests") is not None:
                    for nested_test_name, nested_test_info in info_details[
                        "tests"
                    ].items():
                        update_challenges_already_beaten(
                            nested_test_info, nested_test_name
                        )

        SingletonReportManager().INFO_MANAGER.add_test(test_name, info_details)


```

这段代码的作用是更新一个名为"CHALLENGES_ALREADY_BEATEN"的文件中 TestCafe 应用程序的挑战是否已被击败。

具体来说，代码首先检查是否存在名为"test_name"的挑战，如果存在，则检查当前运行是否成功，如果不存在，则创建一个新的挑战。如果当前运行成功，则检查是否存在名为"test_name"的挑战，如果不存在，则创建一个新的挑战并将其标记为已击败。最后，将更新后的挑战数据写入名为"CHALLENGES_ALREADY_BEATEN"的文件中。

如果挑战已经被击败，则代码将挑战数据中的"test_name"键设置为False，并将其写入名为"CHALLENGES_ALREADY_BEATEN"的文件中。如果当前运行不成功，则代码将挑战数据中的"test_name"键设置为True，并将其写入名为"CHALLENGES_ALREADY_BEATEN"的文件中。


```py
def update_challenges_already_beaten(
    info_details: Dict[str, Any], test_name: str
) -> None:
    current_run_successful = info_details["metrics"]["success"]
    try:
        with open(CHALLENGES_ALREADY_BEATEN, "r") as f:
            challenge_data = json.load(f)
    except:
        challenge_data = {}
    challenge_beaten_in_the_past = challenge_data.get(test_name)

    challenge_data[test_name] = True
    if challenge_beaten_in_the_past is None and not current_run_successful:
        challenge_data[test_name] = False

    with open(CHALLENGES_ALREADY_BEATEN, "w") as f:
        json.dump(challenge_data, f, indent=4)


```

这段代码定义了一个名为 `session_finish` 的函数，接受一个名为 `suite_reports` 的字典参数。

首先，函数中调用了 `get_agent_benchmark_config` 函数，这个函数获取了 benchmark 配置信息。然后，使用 `SingletonReportManager` 类的一个 `save` 方法将配置信息保存到该类的 `INTERNAL_INFO_MANAGER` 和 `INFO_MANAGER` 成员变量中。

接着，使用 `SingletonReportManager` 类的另一个 `end_info_report` 方法将信息报告的配置信息保存到 `END_INFO_REPORT_CONFIG` 和 `AGENT_BENCHMARK_CONFIG` 配置信息中。

最后，使用 `SingletonReportManager` 类的 `save` 方法将基准报告保存到该类的 `REGRESSION_MANAGER` 成员变量中。

整个函数的作用是将基准报告的配置信息保存到指定位置，以供以后的报告和分析使用。


```py
def session_finish(suite_reports: dict) -> None:
    agent_benchmark_config = get_agent_benchmark_config()

    SingletonReportManager().INTERNAL_INFO_MANAGER.save()
    SingletonReportManager().INFO_MANAGER.end_info_report(agent_benchmark_config)
    SingletonReportManager().REGRESSION_MANAGER.save()

```

# `benchmark/agbenchmark/reports/processing/gen_combined_chart.py`

这段代码的作用是生成Combined Bar和Combined Radar Chart。具体来说，它将所有报告的数据存储在一个名为"reports"的目录中，并从该目录中读取所有报告的数据。然后，它将所有报告的数据存储在一个名为"combined_charts"的目录中，并从该目录中读取所有报告的数据。接下来，它将根据报告的数据计算出各种类别，并生成Combined Bar和Combined Radar Chart。最后，它将Combined Bar和Combined Radar Chart保存到运行结果目录中。


```py
import json
import os
from pathlib import Path

from agbenchmark.reports.processing.graphs import (
    save_combined_bar_chart,
    save_combined_radar_chart,
)
from agbenchmark.reports.processing.process_report import (
    all_agent_categories,
    get_reports_data,
)


def generate_combined_chart() -> None:
    all_agents_path = Path(__file__).parent.parent.parent.parent / "reports"

    combined_charts_folder = all_agents_path / "combined_charts"

    reports_data = get_reports_data(str(all_agents_path))

    categories = all_agent_categories(reports_data)

    # Count the number of directories in this directory
    num_dirs = len([f for f in combined_charts_folder.iterdir() if f.is_dir()])

    run_charts_folder = combined_charts_folder / f"run{num_dirs + 1}"

    if not os.path.exists(run_charts_folder):
        os.makedirs(run_charts_folder)

    info_data = {
        report_name: data.benchmark_start_time
        for report_name, data in reports_data.items()
        if report_name in categories
    }
    with open(Path(run_charts_folder) / "run_info.json", "w") as f:
        json.dump(info_data, f)

    save_combined_radar_chart(categories, Path(run_charts_folder) / "radar_chart.png")
    save_combined_bar_chart(categories, Path(run_charts_folder) / "bar_chart.png")


```

这段代码是一个 if 语句，它有以下几种作用：

1. 如果当前脚本（不是 __main__ 文件的话，`__main__` 是该脚本的主函数，也就是 `__main__` 函数）被运行，那么它将执行 if 语句块内的语句。
2. if 语句块内的语句将永远被视为有效，即使该脚本被其它脚本或程序所调用，它们也将始终执行 if 语句块内的语句。
3. if 语句块内的语句执行后，将返回一个布尔值，如果返回 True，则 if 语句块内的语句将被视为有效，否则他们将被视为无效。

在这段代码中，`generate_combined_chart()` 将始终被视为有效，只有当前脚本被正确地引用时，它才会编译并执行。


```py
if __name__ == "__main__":
    generate_combined_chart()

```

# `benchmark/agbenchmark/reports/processing/get_files.py`

这段代码定义了一个名为 `get_last_subdirectory` 的函数，它接受一个只包含纯字符串路径的参数 `directory_path`。这个函数返回目录中最后一个子目录的路径，如果目录中不存在子目录，则返回 `None`。

函数的实现主要分为以下几个步骤：

1. 通过 `os.listdir` 函数获取目录中的所有子目录名称，并将它们存储在一个列表中。
2. 通过 `os.path.isdir` 函数检查子目录是否存在，如果存在，则将其时间戳 `os.path.getctime` 记录下来。
3. 通过 `sorted` 函数按照创建时间对子目录进行排序，并将结果存储在一个新的列表中。
4. 通过 `len` 函数获取排好序的子目录列表的长度，如果子目录列表为空，则返回 `None`。
5. 通过 `sys.exit` 函数输出排好序的子目录列表，并使用列表的 `[-1]` 索引获取最后一个子目录的路径。如果子目录列表中没有最后一个子目录，则函数不会输出任何内容，并返回 `None`。

函数的实现主要依赖于操作系统，特别是文件系统的文件夹和子目录结构。函数将一个目录中的所有子目录按照创建时间进行排序，并将最后一个创建的子目录返回，如果目录中不存在子目录，则返回 `None`。


```py
import os


def get_last_subdirectory(directory_path: str) -> str | None:
    # Get all subdirectories in the directory
    subdirs = [
        os.path.join(directory_path, name)
        for name in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, name))
    ]

    # Sort the subdirectories by creation time
    subdirs.sort(key=os.path.getctime)

    # Return the last subdirectory in the list
    return subdirs[-1] if subdirs else None


```

这段代码定义了一个名为 `get_latest_report_from_agent_directories` 的函数，它接受一个目录路径参数 `directory_path`。该函数返回一个列表，其中包含每个客户端目录中最新创建的报告文件和该文件的路径。

函数首先使用 `os.scandir` 函数遍历传入的目录路径。如果目录路径中的子目录是目录，函数将使用 `get_last_subdirectory` 函数获取该目录中最近创建的子目录。如果 `get_last_subdirectory` 函数返回的子目录不是空目录，函数将使用 `os.path.isfile` 函数检查最近创建的报告文件是否存在于该子目录中。如果是，函数将该子目录和报告文件路径存储到 `latest_reports` 列表中。

函数最终返回 `latest_reports` 列表，其中包含每个客户端目录中的最新报告文件和该文件的路径。


```py
def get_latest_report_from_agent_directories(
    directory_path: str,
) -> list[tuple[os.DirEntry[str], str]]:
    latest_reports = []

    for subdir in os.scandir(directory_path):
        if subdir.is_dir():
            # Get the most recently created subdirectory within this agent's directory
            latest_subdir = get_last_subdirectory(subdir.path)
            if latest_subdir is not None:
                # Look for 'report.json' in the subdirectory
                report_file = os.path.join(latest_subdir, "report.json")
                if os.path.isfile(report_file):
                    latest_reports.append((subdir, report_file))

    return latest_reports

```

# `benchmark/agbenchmark/reports/processing/graphs.py`

0


```py
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


def save_combined_radar_chart(
    categories: dict[str, Any], save_path: str | Path
) -> None:
    categories = {k: v for k, v in categories.items() if v}
    if not all(categories.values()):
        raise Exception("No data to plot")
    labels = np.array(
        list(next(iter(categories.values())).keys())
    )  # We use the first category to get the keys
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[
        :1
    ]  # Add the first angle to the end of the list to ensure the polygon is closed

    # Create radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)  # type: ignore
    ax.set_theta_direction(-1)  # type: ignore
    ax.spines["polar"].set_visible(False)  # Remove border

    # Define a custom normalization to start the color from the middle
    norm = Normalize(
        vmin=0, vmax=max([max(val.values()) for val in categories.values()])
    )  # We use the maximum of all categories for normalization

    cmap = plt.cm.get_cmap("nipy_spectral", len(categories))  # type: ignore

    colors = [cmap(i) for i in range(len(categories))]

    for i, (cat_name, cat_values) in enumerate(
        categories.items()
    ):  # Iterating through each category (series)
        values = np.array(list(cat_values.values()))
        values = np.concatenate((values, values[:1]))  # Ensure the polygon is closed

        ax.fill(angles, values, color=colors[i], alpha=0.25)  # Draw the filled polygon
        ax.plot(angles, values, color=colors[i], linewidth=2)  # Draw polygon
        ax.plot(
            angles,
            values,
            "o",
            color="white",
            markersize=7,
            markeredgecolor=colors[i],
            markeredgewidth=2,
        )  # Draw points

        # Draw legend
        legend = ax.legend(
            handles=[
                mpatches.Patch(color=color, label=cat_name, alpha=0.25)
                for cat_name, color in zip(categories.keys(), colors)
            ],
            loc="upper left",
            bbox_to_anchor=(0.7, 1.3),
        )

        # Adjust layout to make room for the legend
        plt.tight_layout()

    lines, labels = plt.thetagrids(
        np.degrees(angles[:-1]), (list(next(iter(categories.values())).keys()))
    )  # We use the first category to get the keys

    highest_score = 7

    # Set y-axis limit to 7
    ax.set_ylim(top=highest_score)

    # Move labels away from the plot
    for label in labels:
        label.set_position(
            (label.get_position()[0], label.get_position()[1] + -0.05)
        )  # adjust 0.1 as needed

    # Move radial labels away from the plot
    ax.set_rlabel_position(180)  # type: ignore

    ax.set_yticks([])  # Remove default yticks

    # Manually create gridlines
    for y in np.arange(0, highest_score + 1, 1):
        if y != highest_score:
            ax.plot(
                angles, [y] * len(angles), color="gray", linewidth=0.5, linestyle=":"
            )
        # Add labels for manually created gridlines
        ax.text(
            angles[0],
            y + 0.2,
            str(int(y)),
            color="black",
            size=9,
            horizontalalignment="center",
            verticalalignment="center",
        )

    plt.savefig(save_path, dpi=300)  # Save the figure as a PNG file
    plt.close()  # Close the figure to free up memory


```

This code appears to plot the forces applied to a rigid rod subject to an external torque. The plot shows the torque and the distance from the center of the rod for different angles of the torque. The x-axis represents the angle of the torque, and the y-axis represents the distance from the center of the rod. The torque is plotted as an arrow and the distance is plotted as a line. The arrow lengths are labeled as the values in the `values` array.


```py
def save_single_radar_chart(
    category_dict: dict[str, int], save_path: str | Path
) -> None:
    labels = np.array(list(category_dict.keys()))
    values = np.array(list(category_dict.values()))

    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    angles += angles[:1]
    values = np.concatenate((values, values[:1]))

    colors = ["#1f77b4"]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)  # type: ignore
    ax.set_theta_direction(-1)  # type: ignore

    ax.spines["polar"].set_visible(False)

    lines, labels = plt.thetagrids(
        np.degrees(angles[:-1]), (list(category_dict.keys()))
    )

    highest_score = 7

    # Set y-axis limit to 7
    ax.set_ylim(top=highest_score)

    for label in labels:
        label.set_position((label.get_position()[0], label.get_position()[1] + -0.05))

    ax.fill(angles, values, color=colors[0], alpha=0.25)
    ax.plot(angles, values, color=colors[0], linewidth=2)

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

    ax.set_yticklabels([])

    ax.set_yticks([])

    if values.size == 0:
        return

    for y in np.arange(0, highest_score, 1):
        ax.plot(angles, [y] * len(angles), color="gray", linewidth=0.5, linestyle=":")

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

    plt.savefig(save_path, dpi=300)  # Save the figure as a PNG file
    plt.close()  # Close the figure to free up memory


```

这段代码是一个名为`save_combined_bar_chart`的函数，其作用是接收一个字典`categories`和一个文件路径`save_path`，并输出一个组合 bar 图。

具体来说，代码首先检查`categories`字典中是否所有值都为`None`，如果是，则 raises一个异常。如果所有值都不是`None`，则将`categories`字典转换为DataFrame对象，并对其中的`category`列进行分组。然后，使用`plot`函数创建一个带有索引标签的bar chart，图表类型设置为'bar'，图例设置为'off'，单位为px。接下来，设置图表标题、坐标轴标签和图形大小，然后将图表保存为指定文件路径的PNG格式。最后，使用`close`函数关闭图表和图形环境，释放内存。


```py
def save_combined_bar_chart(categories: dict[str, Any], save_path: str | Path) -> None:
    if not all(categories.values()):
        raise Exception("No data to plot")

    # Convert dictionary to DataFrame
    df = pd.DataFrame(categories)

    # Create a grouped bar chart
    df.plot(kind="bar", figsize=(10, 7))

    plt.title("Performance by Category for Each Agent")
    plt.xlabel("Category")
    plt.ylabel("Performance")

    plt.savefig(save_path, dpi=300)  # Save the figure as a PNG file
    plt.close()  # Close the figure to free up memory

```

# `benchmark/agbenchmark/reports/processing/process_report.py`

这段代码的作用是定义了一个名为 `get_reports_data` 的函数，它接受一个报告文件路径参数。函数内部使用 `get_latest_report_from_agent_directories` 函数获取最新的报告文件，并使用 `json.load` 函数将JSON数据解析为Python对象。接着，从解析后的JSON对象中提取出最后一个目录名称作为报告的key，并使用 Python 的字典数据类型将报告数据存储在 `reports_data` 字典中。

注意，这个函数需要 `os` 和 `agbenchmark.reports.processing.get_files` 模块的支持，同时需要定义 `Report` 和 `Test` 类型的接口，以便将JSON数据转换为AG Benchmark可用的Report和Test对象。


```py
import json
import os
from pathlib import Path
from typing import Any

from agbenchmark.reports.processing.get_files import (
    get_latest_report_from_agent_directories,
)
from agbenchmark.reports.processing.report_types import Report, Test
from agbenchmark.utils.data_types import STRING_DIFFICULTY_MAP


def get_reports_data(report_path: str) -> dict[str, Any]:
    latest_files = get_latest_report_from_agent_directories(report_path)

    reports_data = {}

    if latest_files is None:
        raise Exception("No files found in the reports directory")

    # This will print the latest file in each subdirectory and add to the files_data dictionary
    for subdir, file in latest_files:
        subdir_name = os.path.basename(os.path.normpath(subdir))
        with open(Path(subdir) / file, "r") as f:
            # Load the JSON data from the file
            json_data = json.load(f)
            converted_data = Report.parse_obj(json_data)
            # get the last directory name in the path as key
            reports_data[subdir_name] = converted_data

    return reports_data


```

这段代码是一个 Python 函数，名为 `get_agent_category`，它接收一个名为 `report` 的报告对象作为参数，并返回一个名为 `categories` 的字典对象。

函数内部首先创建了一个空字典 `categories`，然后使用一个内部函数 `get_highest_category_difficulty` 来获取报告中各个类别的难度分数的最高值。这个内部函数接收一个名为 `data` 的测试对象作为参数，并遍历 `data` 的 `category` 属性。如果 `category` 等于 "interface" 或 "iterate" 或 "product_advisor"，则不会遍历。否则，将 `categories` 设置为键，值为 `0`。

接着，遍历 `data` 的 `metrics.success` 属性。如果 `data.metrics.success` 为 `True`，则执行内部函数 `get_highest_category_difficulty`。这个内部函数使用一个名为 `STRING_DIFFICULTY_MAP` 的字典，它将字符串的难度映射为整数。然后，用这个映射计算数据 `metrics.difficulty` 与最高难度之间的差异，并将这个差异值作为 `categories` 对应的值。

最终，返回 `categories` 对象。


```py
def get_agent_category(report: Report) -> dict[str, Any]:
    categories: dict[str, Any] = {}

    def get_highest_category_difficulty(data: Test) -> None:
        for category in data.category:
            if (
                category == "interface"
                or category == "iterate"
                or category == "product_advisor"
            ):
                continue
            categories.setdefault(category, 0)
            if data.metrics.success:
                num_dif = STRING_DIFFICULTY_MAP[data.metrics.difficulty]
                if num_dif > categories[category]:
                    categories[category] = num_dif

    for _, test_data in report.tests.items():
        get_highest_category_difficulty(test_data)

    return categories


```

这段代码定义了一个名为 `all_agent_categories` 的函数，它接受一个名为 `reports_data` 的字典类型参数，并返回一个名为 `all_categories` 的字典类型参数。

函数内部先创建了一个空字典 `all_categories`，然后遍历 `reports_data` 中的每个键值对，获取该键对应的 `report` 对象，并使用 `get_agent_category` 函数获取该 `report` 对象所属的类别。

如果获取到的类别不为空，则将该类别添加到 `all_categories` 字典中，并在添加时输出该类别名称。最后，函数返回 `all_categories` 字典。


```py
def all_agent_categories(reports_data: dict[str, Any]) -> dict[str, Any]:
    all_categories: dict[str, Any] = {}

    for name, report in reports_data.items():
        categories = get_agent_category(report)
        if categories:  # only add to all_categories if categories is not empty
            print(f"Adding {name}: {categories}")
            all_categories[name] = categories

    return all_categories

```

# `benchmark/agbenchmark/reports/processing/report_types.py`

这段代码定义了一个名为 ForbidOptionalMeta 的元类，继承自 Pydantic 的 BaseModel 类。这个元类用于定义在 Pydantic 的 Model 类中，禁止使用可选字段（Optional field）。

具体来说，这个 ForbidOptionalMeta 的作用是：

1. 定义一个名为 `__new__` 的方法，这个方法接收一个类的名称 `name`、一个包含元数据（metadata）的对象 `bases` 和一个字典 `dct`，它用于创建类的实例。
2. 在 `__new__` 方法中，遍历 `dct` 中的所有属性，并检查是否存在一个名为 `__origin__` 的属性，如果是 `Union` 类型，并且其元素都是原始类型的参数，那么就发出一个 `TypeError`，指出存在禁止使用可选字段的元数据。
3. 在__new__ 方法中，首先调用父类的 `__new__` 方法，如果没有异常，那么将子类的元数据初始化回 `dct` 中传递给父类的 `__new__` 方法，这样可以确保子类在创建实例时可以正常使用可选字段。

这段代码定义了一个用于创建 Pydantic 模型的 ForbidOptionalMeta 类，它用于防止在 Model 类中使用未定义的选项（Optional field）。


```py
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field

datetime_format = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$"
from pydantic import BaseModel, constr


class ForbidOptionalMeta(type(BaseModel)):  # metaclass to forbid optional fields
    def __new__(cls, name: str, bases: tuple, dct: Dict[str, Any]) -> Any:
        for attr_name, attr_value in dct.items():
            if (
                getattr(attr_value, "__origin__", None) == Union
                and type(None) in attr_value.__args__
            ):
                raise TypeError(
                    f"Optional fields are forbidden, but found in {attr_name}"
                )

        return super().__new__(cls, name, bases, dct)


```

这段代码定义了一个名为 "BaseModelBenchmark" 的类，该类包含一个名为 "Metrics" 的子类。

在定义 "Metrics" 类时，使用了 "BaseModelBenchmark" 类中的 "BaseModel" 和 "ForbidOptionalMeta" 类。其中，Metrics 类继承了 BaseModelBenchmark 类，并添加了一个名为 "Metrics" 的额外类。

在 Metrics 类的定义中，定义了几个变量，包括 difficulty、success、success_percentage、run_time、fail_reason、attempted 和 cost。其中，difficulty 变量表示模型难度，success 变量表示模型是否成功，success_percentage 变量表示成功率，run_time 变量表示运行时间，fail_reason 变量表示失败的原因，attempted 变量表示是否尝试过，cost 变量表示成本，这些变量都是字符串类型的变量。

在 Metrics 类中还定义了一个 Runtime class，该类可能包含方法来计算成功和失败的数量，以及成功和失败百分比。

最后，在定义 Metrics 类时，使用了 FromRoot攻击域，这意味着可以从子类继承 Metrics，并传递参数和元类定义。


```py
class BaseModelBenchmark(BaseModel, metaclass=ForbidOptionalMeta):
    class Config:
        extra = "forbid"


class Metrics(BaseModelBenchmark):
    difficulty: str
    success: bool
    success_percentage: float = Field(..., alias="success_%")
    run_time: str
    fail_reason: str | None
    attempted: bool
    cost: float | None


```

这段代码定义了两个类，一个是`MetricsOverall`，另一个是`Test`。`MetricsOverall`类继承自`BaseModelBenchmark`类，定义了以下字段：

* `run_time`：运行时间，类型为字符串。
* `highest_difficulty`：最高难度，类型为字符串。
* `percentage`：百分比，类型可以是浮点数或 None。
* `total_cost`：总成本，类型可以是浮点数或 None。

`Test`类也继承自`BaseModelBenchmark`类，定义了以下字段：

* `data_path`：数据文件路径，类型为字符串。
* `is_regression`：是否进行回归测试，类型为布尔。
* `answer`：答案，类型为字符串。
* `description`：描述，类型为字符串。
* `metrics`：Metrics 对象，类型为`Metrics`。
* `category`：属于该测试的类别，类型为列表。
* `task`：要测试的任务，类型为字符串。
* `reached_cutoff`：是否达到了预设的截止点，类型为布尔。

`MetricsOverall`类中定义的`percentage`字段可以用于计算其他字段的值，例如：

```py
percentage = metrics.percentage_of_total_time
``` 

`Test`类中定义的`answer`字段用于比较测试结果和预期结果，例如：

```py
if answer == 'target_answer':
   is_correct = True
else:
   is_correct = False
``` 

`Metrics`类可能包含计算其他指标的函数，例如平均时间、内存占用率等。具体实现可能因应用而异。


```py
class MetricsOverall(BaseModelBenchmark):
    run_time: str
    highest_difficulty: str
    percentage: float | None
    total_cost: float | None


class Test(BaseModelBenchmark):
    data_path: str
    is_regression: bool
    answer: str
    description: str
    metrics: Metrics
    category: List[str]
    task: str
    reached_cutoff: bool


```

这段代码定义了一个名为`ReportBase`的类，继承自另一个名为`BaseModelBenchmark`的类。在这个类的继承中，`ReportBase`类添加了一个`command`属性，一个`completion_time`属性，一个`benchmark_start_time`属性，一个`metrics`属性，一个`config`属性，一个`agent_git_commit_sha`属性，一个`benchmark_git_commit_sha`属性和一个`repo_url`属性。

`ReportBase`类继承自`BaseModelBenchmark`类，因此继承了`BaseModelBenchmark`类中定义的所有属性和方法。此外，`ReportBase`类还定义了自己的方法，包括`metrics`属性的值和名称以及`config`属性的值和名称。

`Report`类继承自`ReportBase`类，定义了自己的`tests`属性。

`ReportBase`类的`command`属性是一个字符串，表示该基准测试的命令行工具。`completion_time`属性是一个字符串，表示基准测试完成后输出报告的时间。`benchmark_start_time`属性是一个函数，以某种格式表示基准测试的开始时间。`metrics`属性是一个`MetricsOverall`对象，用于存储基准测试的度量。`config`属性是一个字典，用于存储传递给基准测试的选项。`agent_git_commit_sha`属性是一个字符串，表示代理的Git代码库的提交记录。`benchmark_git_commit_sha`属性是一个字符串，表示基准测试所在的Git代码库的提交记录。`repo_url`属性是一个字符串，表示基准测试所在的Git仓库的URL。


```py
class ReportBase(BaseModelBenchmark):
    command: str
    completion_time: str | None
    benchmark_start_time: constr(regex=datetime_format)
    metrics: MetricsOverall
    config: Dict[str, str | dict[str, str]]
    agent_git_commit_sha: str | None
    benchmark_git_commit_sha: str | None
    repo_url: str | None


class Report(ReportBase):
    tests: Dict[str, Test]


```

这段代码定义了一个名为 "ReportV2" 的类，继承自 "ReportBase" 类，包含三个实例变量：

- "test_name": 一个字符串类型的变量，用于表示测试的名称。
- "run_id": 一个字符串类型的变量，用于表示运行的唯一标识符。
- "team_name": 一个字符串类型的变量，用于表示团队的名称。

这个类的作用是创建一个报告的实例，包含了在测试过程中需要记录的一些信息，如测试的名称、运行的唯一标识符和团队的名称等，可以方便地在测试过程中的不同阶段进行访问和修改。


```py
class ReportV2(Test, ReportBase):
    test_name: str
    run_id: str | None
    team_name: str | None

```