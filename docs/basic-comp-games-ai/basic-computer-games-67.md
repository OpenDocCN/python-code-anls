# BasicComputerGames源码解析 67

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `72_Queen/python/queen.py`

该代码是一个用于在Python 3环境中实现宾士游戏（Queens Game）的程序。游戏的规则如下：

1. 游戏开始时，将游戏板划分为9x9格，将王者和棋子赋值为初始值。
2. 游戏从行1列1开始，玩家轮流移动棋子。
3. 玩家可以点击棋子，选择交换棋子和行1列1的格子。
4. 如果点击棋子，则提示玩家输入新的位置。
5. 如果玩家输入的位置不存在，则允许更正。
6. 每次移动棋子后，更新游戏板的状态。
7. 游戏结束的条件：同行的两个棋子颜色相同或者某一列的棋子为空。
8. 如果游戏结束，输出提示信息，并允许玩家继续挑战。

这个程序实现了Queens Game在Python 3中的具体实现。


```
#!/usr/bin/env python3
"""
Implementation of Queens game in Python 3.

Original game in BASIC by David Ahl in _BASIC Comuter Games_, published in 1978,
as reproduced here:
    https://www.atariarchives.org/basicgames/showpage.php?page=133

Port to Python 3 by Christopher L. Phan <https://chrisphan.com>

Supports Python version 3.8 or later.
"""

from random import random
from typing import Final, FrozenSet, Optional, Tuple

```

这段代码是一个用于游戏配置的Python代码。它允许玩家编辑一些游戏行为的参数。以下是该代码的功能：

1. 定义了一个名为FIX_BOARD_BUG的布尔变量，初始值为False。
2. 定义了一个名为FIX_BOARD_DEBUG的布尔变量，初始值为False。
3. 定义了一个名为SHOW_BOARD_ALWAYS的布尔变量，初始值为False。
4. 在全局范围内，将FIX_BOARD_DEBUG设置为False，以便在代码中多次提及时能够防止误用。
5. 将SHOW_BOARD_ALWAYS设置为True，以便在每次游戏运行时显示游戏板。

这段代码的主要作用是允许玩家在游戏中编辑行为参数，包括修改游戏板显示方式和禁止玩家在无据点的位置开始游戏等。通过设置这些变量的值，玩家可以更好地调整游戏体验。


```
########################################################################################
#                                  Optional configs
########################################################################################
# You can edit these variables to change the behavior of the game.
#
# The original implementation has a bug that allows a player to move off the board,
# e.g. start at the nonexistant space 91. Change the variable FIX_BOARD_BUG to ``True``
# to fix this behavior.
#

FIX_BOARD_BUG: Final[bool] = False

# In the original implementation, the board is only printed once. Change the variable
# SHOW_BOARD_ALWAYS to ``True`` to display the board every time.

```

这段代码的主要作用是定义了一个名为SHOW_BOARD_ALWAYS的布尔类型变量，并将其初始化为False。

接着，代码定义了一个名为ALIGNED_BOARD的布尔类型变量，并将其初始化为False。

然后，代码通过一个包含多个语句的段落来描述了游戏规则。这个描述包括了一些重要的信息，如游戏规则、初始状态等。

最后，代码通过一些条件语句来判断游戏是否结束，并输出了一些关于游戏的信息。


```
SHOW_BOARD_ALWAYS: Final[bool] = False

# In the original implementaiton, the board is printed a bit wonky because of the
# differing widths of the numbers. Change the variable ALIGNED_BOARD to ``True`` to
# fix this.

ALIGNED_BOARD: Final[bool] = False

########################################################################################

INSTR_TXT: Final[
    str
] = """WE ARE GOING TO PLAY A GAME BASED ON ONE OF THE CHESS
MOVES.  OUR QUEEN WILL BE ABLE TO MOVE ONLY TO THE LEFT,
DOWN, OR DIAGONALLY DOWN AND TO THE LEFT.

```

这段代码是一个游戏界面，它允许玩家通过交替移动来将皇后放在下方的左侧手数平方位置。游戏的目标是让玩家首先放置皇后，并在置换移动后获胜。玩家可以通过输入数字来选择移动，每次移动后需要按回键继续游戏。在游戏结束时，系统会输出胜者。


```
THE OBJECT OF THE GAME IS TO PLACE THE QUEEN IN THE LOWER
LEFT HAND SQUARE BY ALTERNATING MOVES BETWEEN YOU AND THE
COMPUTER.  THE FIRST ONE TO PLACE THE QUEEN THERE WINS.

YOU GO FIRST AND PLACE THE QUEEN IN ANY ONE OF THE SQUARES
ON THE TOP ROW OR RIGHT HAND COLUMN.
THAT WILL BE YOUR FIRST MOVE.
WE ALTERNATE MOVES.
YOU MAY FORFEIT BY TYPING '0' AS YOUR MOVE.
BE SURE TO PRESS THE RETURN KEY AFTER EACH RESPONSE.

"""


WIN_MSG: Final[
    str
] = """C O N G R A T U L A T I O N S . . .

```

这段代码是一个文本游戏中的游戏结束信息。它表明玩家在游戏中赢得了比赛，表示祝贺。但同时也表明，尽管玩家在游戏中大部分时间里都表现优秀，但有时候运气也会不佳，因此输赢都是家常便饭。


```
YOU HAVE WON--VERY WELL PLAYED.
IT LOOKS LIKE I HAVE MET MY MATCH.
THANKS FOR PLAYING---I CAN'T WIN ALL THE TIME.

"""

LOSE_MSG: Final[
    str
] = """
NICE TRY, BUT IT LOOKS LIKE I HAVE WON.
THANKS FOR PLAYING.

"""


```

这段代码定义了一个名为 `loc_to_num` 的函数，用于将给定的位置转换为空间编号。函数接受两个参数 `location` 和 `fix_align`，其中 `location` 是二元组形式，表示位置的行和列，`fix_align` 是一个布尔值，表示是否固定对齐输出。

函数的实现采用以下步骤：

1. 将 `location` 中给定的行和列转换为对应的列和行数，然后将这两个数相加，得到新的行数。
2. 如果 `fix_align` 为 `True`，则直接使用 `out_str` 返回，其中 `out_str` 是将行数加 8 后的字符串。
3. 如果 `fix_align` 为 `False` 且 `out_str` 的长度小于或等于 3，则返回 `out_str` 本身。
4. 如果 `fix_align` 为 `False` 且 `out_str` 的长度大于 3，则需要在输出字符串中添加空间数，即 `{row + 8 - col}`。

函数的实现实现了将给定的位置转换为空间编号的功能。在 `GAME_BOARD` 变量中，使用了将每个位置的行列号拼接为一个字符串，并将该字符串连接到游戏板的最后一行，实现了输出整个游戏板的功能。


```
def loc_to_num(location: Tuple[int, int], fix_align: bool = False) -> str:
    """Convert a position given by row, column into a space number."""
    row, col = location
    out_str: str = f"{row + 8 - col}{row + 1}"
    if not fix_align or len(out_str) == 3:
        return out_str
    else:
        return out_str + " "


GAME_BOARD: Final[str] = (
    "\n"
    + "\n\n\n".join(
        "".join(f" {loc_to_num((row, col), ALIGNED_BOARD)} " for col in range(8))
        for row in range(8)
    )
    + "\n\n\n"
)


```

这段代码定义了一个名为 `num_to_loc` 的函数，用于将一个整数 `num` 转换为一个位置，该位置由行和列指定。

具体来说，代码中定义了一个名为 `WIN_LOC` 的变量，其初始值为 (7, 0)，表示该计算机的初始位置是在第 7 行，第 0 列。

定义了一个名为 `COMPUTER_SAFE_SPOTS` 的变量，其初始值为 frozenset( [(2, 3), (4, 5), (5, 1), (6, 2)] )。这个集合用于存储计算机可以尝试移动到的所有位置，除了初始位置和目标位置(即 `WIN_LOC`)之外。

最后，代码中没有做太多其他事情，所以这些位置都是计算机可以尝试移动到的位置。


```
def num_to_loc(num: int) -> Tuple[int, int]:
    """Convert a space number into a position given by row, column."""
    row: int = num % 10 - 1
    col: int = row + 8 - (num - row - 1) // 10
    return row, col


# The win location
WIN_LOC: Final[Tuple[int, int]] = (7, 0)

# These are the places (other than the win condition) that the computer will always
# try to move into.
COMPUTER_SAFE_SPOTS: Final[FrozenSet[Tuple[int, int]]] = frozenset(
    [
        (2, 3),
        (4, 5),
        (5, 1),
        (6, 2),
    ]
)

```

这段代码定义了一个游戏中的地图，其中蓝色方块表示可移动的位置，灰色方块表示不可移动的位置，白色方块表示win位置。

计算机可以选择任意一个空位置进行移动，但每次只能选择一个可移动的位置，因此地图上的所有灰色方块表示的位置计算机都不得不用其他方式移动这些位置。

SAFE_SPOTS是一个预定义的列表，表示计算机可以安全移动的位置，包括所有可移动的蓝色方块和win位置。计算机选择移动位置时，必须从SAFE_SPOTS中选择一个位置，如果没有选择则尝试从地图的其他位置中选择一个位置。


```
# These are the places that the computer will always try to move into.
COMPUTER_PREF_MOVES: Final[
    FrozenSet[Tuple[int, int]]
] = COMPUTER_SAFE_SPOTS | frozenset([WIN_LOC])

# These are the locations (not including the win location) from which either player can
# force a win (but the computer will always choose one of the COMPUTER_PREF_MOVES).
SAFE_SPOTS: Final[FrozenSet[Tuple[int, int]]] = COMPUTER_SAFE_SPOTS | frozenset(
    [
        (0, 4),
        (3, 7),
    ]
)


```

This appears to be a program written in Python that is intended to play the board game known as "Tic Tac Toe". It appears to be a command-line version of the game, where the player can place a piece on the board by specifying a row and a column, rather than using生命危险的方法。

The program starts by asking the player what their first move will be, and then it tries to validate the move by checking whether the piece is valid in the current state. If the move is valid, the program updates the game board and returns the new position of the piece. If not, it prompts the player to try again.

It appears that there is a bug in the program that is causing the board to be reset to its original state upon validating a move. It is also possible that there is a maximum number of moves that can be played, as the program is crashing when the player tries to place more than one piece in a single move.


```
def intro() -> None:
    """Print the intro and print instructions if desired."""
    print(" " * 33 + "Queen")
    print(" " * 15 + "Creative Computing  Morristown, New Jersey")
    print("\n" * 2)
    if ask("DO YOU WANT INSTRUCTIONS"):
        print(INSTR_TXT)


def get_move(current_loc: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    """Get the next move from the player."""
    prompt: str
    player_resp: str
    move_raw: int
    new_row: int
    new_col: int
    if current_loc is None:  # It's the first turn
        prompt = "WHERE WOULD YOU LIKE TO START? "
    else:
        prompt = "WHAT IS YOUR MOVE? "
        row, col = current_loc
    while True:
        player_resp = input(prompt).strip()
        try:
            move_raw = int(player_resp)
            if move_raw == 0:  # Forfeit
                return 8, 8
            new_row, new_col = num_to_loc(move_raw)
            if current_loc is None:
                if (new_row == 0 or new_col == 7) and (
                    not FIX_BOARD_BUG or (new_col >= 0 and new_row < 8)
                ):
                    return new_row, new_col
                else:
                    prompt = (
                        "PLEASE READ THE DIRECTIONS AGAIN.\n"
                        "YOU HAVE BEGUN ILLEGALLY.\n\n"
                        "WHERE WOULD YOU LIKE TO START? "
                    )
            else:
                if (
                    (new_row == row and new_col < col)  # move left
                    or (new_col == col and new_row > row)  # move down
                    or (new_row - row == col - new_col)  # move diag left and down
                ) and (not FIX_BOARD_BUG or (new_col >= 0 and new_row < 8)):
                    return new_row, new_col
                else:
                    prompt = "Y O U   C H E A T . . .  TRY AGAIN? "

        except ValueError:
            prompt = "!NUMBER EXPECTED - RETRY INPUT LINE\n? "


```

The programming functionality described in the question is the implementation of the game of Connect-the-Dotts. The game is played by two players, with the objective of one player connecting a series of dots on a grid to their home dot. The home dot is chosen by the player who clicked on the game, and the other player is given a series of random moves by the computer.

The `computer_move()` function takes a location as an input and returns a tuple of the new location. This function uses a combination of seven rules to generate the computer's move. The rules are as follows:

1. If the new location is within one of the seven safe spots, the function returns the new location without any movement.
2. If the new location is on the left or right of the current location, the function returns the new location by moving one space in the diagonal.
3. If the new location is on the bottom or above the current location, the function returns the new location by moving one space up or down.
4. If the new location is on the left or right of the current location and there is a safe spot on the left or right, the function returns the new location without any movement.
5. If the new location is on the left or right of the current location and there is a safe spot on the bottom or above, the function returns the new location without any movement.
6. If the new location is on the left or right of the current location and there is a safe spot on the left or right and it is not on the bottom or above, the function returns the new location without any movement.
7. If all the rules have been checked and the new location cannot be found in any of the above rules, the function returns a random move.

The `random_computer_move()` function is used to generate a random move for the computer. This function chooses a new location by generating a random number between 0 and 1 and choosing a location based on the chart of rules above.

Overall, the Connect-the-Dotts game is a simple game played by two players. The computer moves in response to the player's moves, generating a random move if it cannot find an available safe spot.


```
def random_computer_move(location: Tuple[int, int]) -> Tuple[int, int]:
    """Make a random move."""
    row, col = location
    if (z := random()) > 0.6:
        # Move down one space
        return row + 1, col
    elif z > 0.3:
        # Move diagonaly (left and down) one space
        return row + 1, col - 1
    else:
        # Move left one space
        return row, col - 1


def computer_move(location: Tuple[int, int]) -> Tuple[int, int]:
    """Get the computer's move."""
    # If the player has made an optimal move, then choose a random move
    if location in SAFE_SPOTS:
        return random_computer_move(location)
    # We don't need to implmement the logic of checking for the player's win,
    # because that is checked before this function is called.
    row, col = location
    for k in range(7, 0, -1):
        # If the computer can move left k spaces and end in up in a safe spot or win,
        # do it.
        if (new_loc := (row, col - k)) in COMPUTER_PREF_MOVES:
            return new_loc
        # If the computer can move down k spaces and end up in a safe spot or win, do it.
        if (new_loc := (row + k, col)) in COMPUTER_PREF_MOVES:
            return new_loc
        # If the computer can move diagonally k spaces and end up in a safe spot or win,
        # do it.
        if (new_loc := (row + k, col - k)) in COMPUTER_PREF_MOVES:
            return new_loc
        # As a fallback, do a random move. (NOTE: This shouldn't actally happen--it
        # should always be possible to make an optimal move if the player doesn't play
        # in a location in SAFE_SPOTS.
    return random_computer_move(location)


```

这段代码是一个Python游戏中的主要函数，名为`main_game()`。它是一个while循环，只要游戏没有结束，就会一直循环执行。

游戏开始时，初始化了一些变量，包括`game_over`为False，`location`是一个包含两个整数的可选变量，表示这是游戏的起始位置。然后，进入while循环，只要游戏没有结束，就会执行循环体内的内容。

循环体内的第一个语句是`print("IT LOOKS LIKE I HAVE WON BY FORFEIT.")`，表示游戏已经结束，玩家获胜，但实际游戏中，胜利条件是`location == (8, 8)`或者`location == WIN_LOC`，所以这个条件并不会触发游戏结束的判断。

循环体内的第二个语句是`print(WIN_MSG)`，表示在游戏胜利时输出一个获胜的消息，比如"恭喜你赢得了游戏"。

循环体内的第三个语句是`print("COMPUTER MOVES TO SQUARE {}".format(loc_to_num(location)))`，表示计算机在哪个位置移动了，用`loc_to_num`函数将其转换为数字表示。

循环体内的第四个语句是`if location == (8, 8):`，表示判断起始位置是否为(8, 8)，如果是，则执行`print("你在顶部胜利了！")`，表示玩家在游戏顶部获胜，但实际上这个条件并不会触发游戏结束的判断。

循环体内的第五个语句是`elif location == WIN_LOC:`，表示判断起始位置是否为计算机胜利的位置，如果是，则执行`print("你失败了！")`，表示玩家失败。

循环体内的第六个语句是`else:`，表示如果前五个条件都不成立，那么执行循环体内的最后一个语句，即`location = computer_move(location)`，表示计算机移动位置，`loc_to_num`函数将其转换为数字表示。

循环体内的第七个语句是`print(f"{location[0]} {location[1]}")`，表示输出起始位置的列和行。

第八个语句是`if SHOW_BOARD_ALWAYS`，表示判断是否一直显示游戏板。如果是，那么执行游戏板显示所有的行，否则跳过显示。

最后，如果游戏没有结束，则一直循环执行`print(GAME_BOARD)`，表示输出游戏板。


```
def main_game() -> None:
    """Execute the main game."""
    game_over: bool = False
    location: Optional[Tuple[int, int]] = None  # Indicate it is the first turn
    while not game_over:
        location = get_move(location)
        if location == (8, 8):  # (8, 8) is returned when the player enters 0
            print("\nIT LOOKS LIKE I HAVE WON BY FORFEIT.\n")
            game_over = True
        elif location == WIN_LOC:  # Player wins (in lower left corner)
            print(WIN_MSG)
            game_over = True
        else:
            location = computer_move(location)
            print(f"COMPUTER MOVES TO SQUARE {loc_to_num(location)}")
            if location == WIN_LOC:  # Computer wins (in lower left corner)
                print(LOSE_MSG)
                game_over = True
        # The default behavior is not to show the board each turn, but
        # this can be modified by changing a flag at the start of the file.
        if not game_over and SHOW_BOARD_ALWAYS:
            print(GAME_BOARD)


```

这段代码定义了一个名为 `ask` 的函数，用于询问用户一个“是”或“否”的问题，直到用户给出一个可理解的回答。函数的参数 `prompt` 是一个字符串，用于在函数 prompt 中显示 asking 的消息，函数会不断地 prompt 用户，直到得到一个有效的回答。

函数的功能是通过 while 循环，反复调用 `input` 函数，获得用户输入的问题，并将其转换为大写和不带空格的高帽子字符串，调用大写或小写字母后，会将字符串转换成字符型并返回，然后在循环中，将显示“请回答“YES”或“NO”，最后在循环之外，返回一个布尔值 `False`。

该函数的实现非常简单，但可以有效地质问用户一个“是”或“否”的问题，有助于在某些需要用户回答确认信息的情况下，确认用户是否获得了正确的信息。


```
def ask(prompt: str) -> bool:
    """Ask a yes/no question until user gives an understandable response."""
    inpt: str
    while True:
        # Normalize input to uppercase, no whitespace, then get first character
        inpt = input(prompt + "? ").upper().strip()[0]
        print()
        if inpt == "Y":
            return True
        elif inpt == "N":
            return False
        print("PLEASE ANSWER 'YES' OR 'NO'.")
    return False


```

这段代码是一个Python脚本，它有以下几个主要部分：

1. `if __name__ == "__main__":` 是一个条件语句，它会判断当前脚本是否作为主程序运行。如果是，那么执行下面的代码。否则，跳过这个部分，执行程序的其他部分。
2. `intro()` 是一个函数，它会在程序开始时执行一次。在本例中，它可能是初始化游戏板或者输出一些介绍信息。但是，在本题中，它没有执行任何操作。
3. `still_playing: bool = True` 是一个变量，它的值是一个布尔类型，初始值为`True`。它用于判断游戏是否仍在继续。
4. `while still_playing:` 是一个无限循环，它会在游戏中循环执行。这个循环的条件是 `still_playing` 的值为 `True`。
5. `print(GAME_BOARD)` 在每次循环开始时，它会输出游戏板的内容。这里的 `GAME_BOARD` 应该是从外部传入的，可能是包含游戏元素的二维列表或字符串。
6. `main_game()` 是一个函数，它可能是包含游戏逻辑的函数。但是，在本题中，它没有执行任何操作。
7. `still_playing = ask("ANYONE ELSE CARE TO TRY")` 是一个函数，它可能是从用户那里获取玩家意愿的函数。它使用了 `ask` 函数从用户那里获取一个字符串，然后将其转换为布尔类型。如果用户选择了"是"，那么游戏将继续；如果用户选择了"否"，游戏就结束。
8. `print("\nOK --- THANKS AGAIN.")` 在游戏结束时，它会输出一段感谢信息。


```
if __name__ == "__main__":
    intro()
    still_playing: bool = True
    while still_playing:
        print(GAME_BOARD)
        main_game()
        still_playing = ask("ANYONE ELSE CARE TO TRY")
    print("\nOK --- THANKS AGAIN.")

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/) by Christopher Phan.
Supports Python version 3.8 or later.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Reverse

The game of REVERSE requires you to arrange a list of numbers in numerical order from left to right. To move, you tell the computer how many numbers (counting from the left) to reverse. For example, if the current list is:
```
    2 3 4 5 1 6 7 8 9
```

and you reverse 4, the result will be:
```
    5 4 3 2 1 6 7 8 9
```
Now if you reverse 5, you win!

There are many ways to beat the game, but approaches tend to be either algorithmic or heuristic. The game thus offers the player a chance to play with these concepts in a practical (rather than theoretical) context.

An algorithmic approach guarantees a solution in a predictable number of moves, given the number of items in the list. For example, one method guarantees a solution in 2N - 3 moves when teh list contains N numbers. The essence of an algorithmic approach is that you know in advance what your next move will be. Once could easily program a computer to do this.

A heuristic approach takes advantage of “partial orderings” in the list at any moment. Using this type of approach, your next move is dependent on the way the list currently appears. This way of solving the problem does not guarantee a solution in a predictable number of moves, but if you are lucky and clever, you may come out ahead of the algorithmic solutions. One could not so easily program this method.

In practice, many players adopt a “mixed” strategy, with both algorithmic and heuristic features. Is this better than either “pure” strategy?

The program was created by Peter Sessions of People’s Computer Company and the notes above adapted from his original write-up.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=135)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=150)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `73_Reverse/csharp/Reverse/Program.cs`



This code looks like it is a simple game of skill where the objective is to arrange a list of numbers in numerical order from left to right and then reverse them. The game will start with a list of numbers and the player will be prompted to enter a number to add to the list. The game will then reverse the numbers in the list and display the result. The player will win the game if they manage to reverse all the numbers in the list. If the player wants to quit the game, they can reverse 0 (zero).

It\'s important to note that this code is not complete and may have bugs or issues. For example, it doesn\'t handle the case where the list is empty or if the number entered by the player is not a valid number.


```
﻿using System;

namespace Reverse
{
    class Program
    {
        private static int arrayLength = 9;
        static void Main(string[] args)
        {
            PrintTitle();
            Console.Write("DO YOU WANT THE RULES? ");
            var needRulesInput = Console.ReadLine();
            Console.WriteLine();
            if (string.Equals(needRulesInput, "YES", StringComparison.OrdinalIgnoreCase))
            {
                DisplayRules();
            }

            var tryAgain = string.Empty;
            while (!string.Equals(tryAgain, "NO", StringComparison.OrdinalIgnoreCase))
            {
                var reverser = new Reverser(arrayLength);

                Console.WriteLine("HERE WE GO ... THE LIST IS:");
                PrintList(reverser.GetArrayString());
                var arrayIsInAscendingOrder = false;
                var numberOfMoves = 0;
                while (arrayIsInAscendingOrder == false)
                {
                    int index = ReadNextInput();

                    if (index == 0)
                    {
                        break;
                    }

                    reverser.Reverse(index);
                    PrintList(reverser.GetArrayString());
                    arrayIsInAscendingOrder = reverser.IsArrayInAscendingOrder();
                    numberOfMoves++;
                }

                if (arrayIsInAscendingOrder)
                {
                    Console.WriteLine($"YOU WON IT IN {numberOfMoves} MOVES!!!");

                }

                Console.WriteLine();
                Console.WriteLine();
                Console.Write("TRY AGAIN (YES OR NO) ");
                tryAgain = Console.ReadLine();
            }

            Console.WriteLine();
            Console.WriteLine("OK HOPE YOU HAD FUN!!");
        }

        private static int ReadNextInput()
        {
            Console.Write("HOW MANY SHALL I REVERSE? ");
            var input = ReadIntegerInput();
            while (input > 9 || input < 0)
            {
                if (input > 9)
                {
                    Console.WriteLine($"OOPS! TOO MANY! I CAN REVERSE AT MOST {arrayLength}");
                }

                if (input < 0)
                {
                    Console.WriteLine($"OOPS! TOO FEW! I CAN REVERSE BETWEEN 1 AND {arrayLength}");
                }
                Console.Write("HOW MANY SHALL I REVERSE? ");
                input = ReadIntegerInput();
            }

            return input;
        }

        private static int ReadIntegerInput()
        {
            var input = Console.ReadLine();
            int.TryParse(input, out var index);
            return index;
        }

        private static void PrintList(string list)
        {
            Console.WriteLine();
            Console.WriteLine(list);
            Console.WriteLine();
        }

        private static void PrintTitle()
        {
            Console.WriteLine("\t\t   REVERSE");
            Console.WriteLine("  CREATIVE COMPUTING  MORRISTON, NEW JERSEY");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("REVERSE -- A GAME OF SKILL");
            Console.WriteLine();
        }

        private static void DisplayRules()
        {
            Console.WriteLine();
            Console.WriteLine("THIS IS THE GAME OF 'REVERSE'. TO WIN, ALL YOU HAVE");
            Console.WriteLine("TO DO IS ARRANGE A LIST OF NUMBERS (1 THOUGH 9 )");
            Console.WriteLine("IN NUMERICAL ORDER FROM LEFT TO RIGHT. TO MOVE, YOU");
            Console.WriteLine("TELL ME HOW MANY NUMBERS (COUNTING FROM THE LEFT) TO");
            Console.WriteLine("REVERSE. FOR EXAMPLE, IF THE CURRENT LIST IS:");
            Console.WriteLine();
            Console.WriteLine("2 3 4 5 1 6 7 8 9");
            Console.WriteLine();
            Console.WriteLine("AND YOU REVERSE 4, THE RESULT WILL BE:");
            Console.WriteLine();
            Console.WriteLine("5 4 3 2 1 6 7 8 9");
            Console.WriteLine();
            Console.WriteLine("NOW IF YOU REVERSE 5, YOU WIN!");
            Console.WriteLine();
            Console.WriteLine("1 2 3 4 5 6 7 8 9");
            Console.WriteLine();
            Console.WriteLine("NO DOUBT YOU WILL LIKE THIS GAME, BUT ");
            Console.WriteLine("IF YOU WANT TO QUIT, REVERSE 0 (ZERO)");
            Console.WriteLine();
            Console.WriteLine();
        }
    }
}

```

# `73_Reverse/csharp/Reverse/Reverser.cs`

This is a sample implementation of a reversing class that uses a random array of integers. The `Reverser` class has a constructor that takes an array size as an argument and creates a random array of integers with that size. The `Reverse` method takes an index of the element to reverse and reverses the elements from that index to the end of the array.

The class also has a method `IsArrayInAscendingOrder` that checks if the given array is in ascending order. The `CreateRandomArray` method creates a random array of integers with the specified size.

Note that this implementation is for educational purposes only and may not be suitable for critical applications.


```
﻿using System;
using System.Text;

namespace Reverse
{
    public class Reverser
    {
        protected int[] _array;

        public Reverser(int arraySize)
        {
            _array = CreateRandomArray(arraySize);
        }

        public void Reverse(int index)
        {
            if (index > _array.Length)
            {
                return;
            }

            for (int i = 0; i < index / 2; i++)
            {
                int temp = _array[i];
                int upperIndex = index - 1 - i;
                _array[i] = _array[upperIndex];
                _array[upperIndex] = temp;
            }
        }

        public bool IsArrayInAscendingOrder()
        {
            for (int i = 1; i < _array.Length; i++)
            {
                if (_array[i] < _array[i - 1])
                {
                    return false;
                }
            }

            return true;
        }

        private int[] CreateRandomArray(int size)
        {
            if (size < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(size), "Array size must be a positive integer");
            }

            var array = new int[size];
            for (int i = 1; i <= size; i++)
            {
                array[i - 1] = i;
            }

            var rnd = new Random();

            for (int i = size; i > 1;)
            {
                int k = rnd.Next(i);
                --i;
                int temp = array[i];
                array[i] = array[k];
                array[k] = temp;
            }
            return array;
        }

        public string GetArrayString()
        {
            var sb = new StringBuilder();

            foreach (int i in _array)
            {
                sb.Append(" " + i + " ");
            }

            return sb.ToString();
        }
    }
}

```

# `73_Reverse/csharp/Reverse.Tests/ReverserTests.cs`

This is a unit test that uses a `TestReverser` class to reverse an array of integers. The `TestReverser` class has a `SetArray` method that takes an array of integers as input and returns a reversed array. The `Reverse` method takes an optional integer `index` and a negative integer `reverseCount` and reverses the array by the specified count. The `IsArrayInAscendingOrder` method tests whether an array is in ascending order. It has two tests: one that takes an array of integers in ascending order and another that takes an array of integers in descending order. The `GetArrayString` method tests whether the reversed array can be formatted as a string.

The `[Theory]` attribute indicates that this class is part of a series of tests, and the `[InlineData]` attributes are used to provide examples of the expected behavior. The `[Fact]` and `[FactOf]` attributes are used to mark the methods as a test. The `[Theory]` attribute is used to indicate that this class is a part of a series of tests, and the `[InlineData]` attribute is used to provide examples of the expected behavior.


```
using FsCheck.Xunit;
using Reverse.Tests.Generators;
using System;
using System.Linq;
using Xunit;

namespace Reverse.Tests
{
    public class ReverserTests
    {
        [Fact]
        public void Constructor_CannotAcceptNumberLessThanZero()
        {
            Action action = () => new Reverser(0);

            Assert.Throws<ArgumentOutOfRangeException>(action);
        }

        [Property(Arbitrary = new[] { typeof(PositiveIntegerGenerator) })]
        public void Constructor_CreatesRandomArrayOfSpecifiedLength(int size)
        {
            var sut = new TestReverser(size);

            Assert.Equal(size, sut.GetArray().Length);
        }

        [Property(Arbitrary = new[] { typeof(PositiveIntegerGenerator) })]
        public void ConstructorArray_MaxElementValueIsEqualToSize(int size)
        {
            var sut = new TestReverser(size);

            Assert.Equal(size, sut.GetArray().Max());
        }

        [Property(Arbitrary = new[] { typeof(PositiveIntegerGenerator) })]
        public void ConstructorArray_ReturnsRandomArrayWithDistinctElements(int size)
        {
            var sut = new TestReverser(size);
            var array = sut.GetArray();
            var arrayGroup = array.GroupBy(x => x);
            var duplicateFound = arrayGroup.Any(x => x.Count() > 1);

            Assert.False(duplicateFound);
        }

        [Theory]
        [InlineData(new int[] { 1 }, new int[] { 1 })]
        [InlineData(new int[] { 1, 2 }, new int[] { 2, 1 })]
        [InlineData(new int[] { 1, 2, 3 }, new int[] { 3, 2, 1 })]
        public void Reverse_WillReverseEntireArray(int[] input, int[] output)
        {
            var sut = new TestReverser(1);
            sut.SetArray(input);

            sut.Reverse(input.Length);

            Assert.True(sut.GetArray().SequenceEqual(output));
        }

        [Fact]
        public void Reverse_WithSpecifiedIndex_ReversesItemsUpToThatIndex()
        {
            var input = new int[] { 1, 2, 3, 4 };
            var output = new int[] { 2, 1, 3, 4 };
            var sut = new TestReverser(1);
            sut.SetArray(input);

            sut.Reverse(2);

            Assert.True(sut.GetArray().SequenceEqual(output));
        }

        [Fact]
        public void Reverse_WithIndexOne_DoesNothing()
        {
            var input = new int[] { 1, 2 };
            var output = new int[] { 1, 2 };
            var sut = new TestReverser(1);
            sut.SetArray(input);

            sut.Reverse(1);

            Assert.True(sut.GetArray().SequenceEqual(output));
        }

        [Fact]
        public void Reverse_WithIndexGreaterThanArrayLength_DoesNothing()
        {
            var input = new int[] { 1, 2 };
            var output = new int[] { 1, 2 };
            var sut = new TestReverser(1);
            sut.SetArray(input);

            sut.Reverse(sut.GetArray().Length + 1);

            Assert.True(sut.GetArray().SequenceEqual(output));
        }

        [Fact]
        public void Reverse_WithIndexLessThanZero_DoesNothing()
        {
            var input = new int[] { 1, 2 };
            var output = new int[] { 1, 2 };
            var sut = new TestReverser(1);
            sut.SetArray(input);

            sut.Reverse(-1);

            Assert.True(sut.GetArray().SequenceEqual(output));
        }

        [Theory]
        [InlineData(new int[] { 1 })]
        [InlineData(new int[] { 1, 2 })]
        [InlineData(new int[] { 1, 1 })]
        public void IsArrayInAscendingOrder_WhenArrayElementsAreInNumericAscendingOrder_ReturnsTrue(int[] input)
        {
            var sut = new TestReverser(1);
            sut.SetArray(input);

            var result = sut.IsArrayInAscendingOrder();

            Assert.True(result);
        }

        [Fact]
        public void IsArrayInOrder_WhenArrayElementsAreNotInNumericAscendingOrder_ReturnsFalse()
        {
            var sut = new TestReverser(1);
            sut.SetArray(new int[] { 2, 1 });

            var result = sut.IsArrayInAscendingOrder();

            Assert.False(result);
        }

        [Fact]
        public void GetArrayString_ReturnsSpaceSeparatedElementsOfArrayInStringFormat()
        {
            var sut = new TestReverser(1);
            sut.SetArray(new int[] { 1, 2 });

            var result = sut.GetArrayString();

            Assert.Equal(" 1  2 ", result);
        }
    }
}

```

# `73_Reverse/csharp/Reverse.Tests/TestReverser.cs`

这段代码定义了一个名为"TestReverser"的内部类，它继承自名为"Reverser"的公共类。

在"TestReverser"类中，构造函数接收一个整数参数"arraySize"，用于设置测试反向数组的尺寸。

"GetArray"方法返回测试反向数组，以便在测试过程中使用。

"SetArray"方法接受一个整数数组，将其设置为测试反向数组的值。这个值可以在测试过程中修改，从而测试反向数组的改变。


```
﻿namespace Reverse.Tests
{
    internal class TestReverser : Reverser
    {
        public TestReverser(int arraySize) : base(arraySize) { }

        public int[] GetArray()
        {
            return _array;
        }

        public void SetArray(int[] array)
        {
            _array = array;
        }
    }
}

```

# `73_Reverse/csharp/Reverse.Tests/Generators/PositiveIntegerGenerator.cs`

该代码是一个测试代码，它定义了一个名为 PositiveIntegerGenerator 的类，该类中包含一个名为 Generate 的方法。

方法的作用是返回一个名为 Arbitrary 的接口类型，该接口类型代表任意整数。该方法返回的 Arbitrary 对象通过 Int32 过滤器（ filter()）来生成任意正整数。在这个例子中，生成的数字都大于 0。

生成的数字使用 FsCheck 库进行过滤，该库确保生成的数字只有在其定义的正整数范围内。生成的数字也被返回，因此可以将其打印出来或用于其他应用程序。


```
﻿using FsCheck;

namespace Reverse.Tests.Generators
{
    public static class PositiveIntegerGenerator
    {
        public static Arbitrary<int> Generate() =>
            Arb.Default.Int32().Filter(x => x > 0);
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `73_Reverse/java/Reverse.java`

这段代码是一个名为 "Game of Reverse" 的游戏，它基于 1970 年的一款类似名叫 "BASIC" 的游戏。这款游戏的设计思想是让玩家通过控制计算机的输出，来 "反转" 输入的数字或字符串。

具体来说，这段代码的作用如下：

1. 导入 Scanner 类，用于输入玩家输入的数据；
2. 导入 Math 类，用于数学计算；
3. 定义一个名为 "Game of Reverse" 的类，其中包含如下方法：
	* public static void main(String[] args) {
		Scanner scanner = new Scanner(System.in); // 获取玩家输入的输入流
		while (true) {
			System.out.print("Enter a number or a string: "); // 输出一个输入框，让玩家输入数字或字符串
			int num = scanner.nextInt(); // 获取玩家输入的整数
			String input = scanner.nextLine(); // 获取玩家输入的字符串
			int result = reverse(num); // 计算结果
			System.out.println(num + " reversed to " + result); // 输出结果
		}
	}
	* public static int reverse(int num) {
		int reversed = 0;
		while (num > 0) {
			reversed = reversed * 10 + num % 10; // 将整数转换成字符串，然后将字符串的末尾添加数字
			num = num / 10; // 将整数除以 10
		}
		return reversed;
	}
}

在这段代码中，我们首先导入了 Scanner 和 Math 类，用于输入和处理玩家输入的数据。

然后在 main 方法中，我们创建了一个 Scanner 对象，用于获取玩家输入的输入流。在循环中，我们首先让玩家输入一个数字或字符串，然后使用 reversed 变量将输入转换成字符串，并使用 reverse 方法将整数反转。最后，我们将结果输出到屏幕上。

注意，在这段代码中，我们没有添加任何新的功能，例如错误检查等。


```
import java.util.Scanner;
import java.lang.Math;

/**
 * Game of Reverse
 * <p>
 * Based on the BASIC game of Reverse here
 * https://github.com/coding-horror/basic-computer-games/blob/main/73%20Reverse/reverse.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

```

The `Reverse` class is a implementation of the board game of Reverse. The game is played by arranging a list of numbers in a specific order (from left to right) and then reversing the numbers. The goal is to reverse the numbers so that they form a valid number in the Reverse order.

The `printBoard` method prints the board of numbers in the game. The `printRules` method prints the rules of the game.

The `play` method has the board of numbers and calls the `printBoard` or `printRules` method to start the game.

If the game is won, it will print the message "You Win!" to the console. If the game is lost, it will print the message "You Lose!" to the console.

The `main` method is the entry point of the application. It creates an instance of the `Reverse` class and calls the `play` method to start the game.


```
public class Reverse {

  private final int NUMBER_COUNT = 9;

  private final Scanner scan;  // For user input

  private enum Step {
    INITIALIZE, PERFORM_REVERSE, TRY_AGAIN, END_GAME
  }

  public Reverse() {

    scan = new Scanner(System.in);

  }  // End of constructor Reverse

  public void play() {

    showIntro();
    startGame();

  }  // End of method play

  private static void showIntro() {

    System.out.println(" ".repeat(31) + "REVERSE");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");
    System.out.println("REVERSE -- A GAME OF SKILL");
    System.out.println("");

  }  // End of method showIntro

  private void startGame() {

    int index = 0;
    int numMoves = 0;
    int numReverse = 0;
    int tempVal = 0;
    int[] numList = new int[NUMBER_COUNT + 1];

    Step nextStep = Step.INITIALIZE;

    String userResponse = "";

    System.out.print("DO YOU WANT THE RULES? ");
    userResponse = scan.nextLine();

    if (!userResponse.toUpperCase().equals("NO")) {

      this.printRules();
    }

    // Begin outer while loop
    while (true) {

    // Begin outer switch
    switch (nextStep) {

      case INITIALIZE:

        // Make a random list of numbers
       numList[1] = (int)((NUMBER_COUNT - 1) * Math.random() + 2);

         for (index = 2; index <= NUMBER_COUNT; index++) {

          // Keep generating lists if there are duplicates
          while (true) {

            numList[index] = (int)(NUMBER_COUNT * Math.random() + 1);

            // Search for duplicates
            if (!this.findDuplicates(numList, index)) {
              break;
            }
          }
        }

        System.out.println("");
        System.out.println("HERE WE GO ... THE LIST IS:");

        numMoves = 0;

        this.printBoard(numList);

        nextStep = Step.PERFORM_REVERSE;
        break;

      case PERFORM_REVERSE:

        System.out.print("HOW MANY SHALL I REVERSE? ");
        numReverse = Integer.parseInt(scan.nextLine());

        if (numReverse == 0) {

          nextStep = Step.TRY_AGAIN;

        } else if (numReverse > NUMBER_COUNT) {

          System.out.println("OOPS! TOO MANY! I CAN REVERSE AT MOST " + NUMBER_COUNT);
          nextStep = Step.PERFORM_REVERSE;

        } else {

          numMoves++;

          for (index = 1; index <= (int)(numReverse / 2.0); index++) {

            tempVal = numList[index];
            numList[index] = numList[numReverse - index + 1];
            numList[numReverse - index + 1] = tempVal;
          }

          this.printBoard(numList);

          nextStep = Step.TRY_AGAIN;

          // Check for a win
          for (index = 1; index <= NUMBER_COUNT; index++) {

            if (numList[index] != index) {
              nextStep = Step.PERFORM_REVERSE;
            }
          }

          if (nextStep == Step.TRY_AGAIN) {
            System.out.println("YOU WON IT IN " + numMoves + " MOVES!!!");
            System.out.println("");
          }
        }
        break;

      case TRY_AGAIN:

        System.out.println("");
        System.out.print("TRY AGAIN (YES OR NO)? ");
        userResponse = scan.nextLine();

        if (userResponse.toUpperCase().equals("YES")) {
          nextStep = Step.INITIALIZE;
        } else {
          nextStep = Step.END_GAME;
        }
        break;

      case END_GAME:

        System.out.println("");
        System.out.println("O.K. HOPE YOU HAD FUN!!");
        return;

      default:

        System.out.println("INVALID STEP");
        break;

      }  // End outer switch

    }  // End outer while loop

  }  // End of method startGame

  public boolean findDuplicates(int[] board, int length) {

    int index = 0;

    for (index = 1; index <= length - 1; index++) {

      // Identify duplicates
      if (board[length] == board[index]) {

        return true;  // Found a duplicate
      }
    }

    return false;  // No duplicates found

  }  // End of method findDuplicates

  public void printBoard(int[] board) {

    int index = 0;

    System.out.println("");

    for (index = 1; index <= NUMBER_COUNT; index++) {

      System.out.format("%2d", board[index]);
    }

    System.out.println("\n");

  }  // End of method printBoard

  public void printRules() {

    System.out.println("");
    System.out.println("THIS IS THE GAME OF 'REVERSE'.  TO WIN, ALL YOU HAVE");
    System.out.println("TO DO IS ARRANGE A LIST OF NUMBERS (1 THROUGH " + NUMBER_COUNT + ")");
    System.out.println("IN NUMERICAL ORDER FROM LEFT TO RIGHT.  TO MOVE, YOU");
    System.out.println("TELL ME HOW MANY NUMBERS (COUNTING FROM THE LEFT) TO");
    System.out.println("REVERSE.  FOR EXAMPLE, IF THE CURRENT LIST IS:");
    System.out.println("");
    System.out.println("2 3 4 5 1 6 7 8 9");
    System.out.println("");
    System.out.println("AND YOU REVERSE 4, THE RESULT WILL BE:");
    System.out.println("");
    System.out.println("5 4 3 2 1 6 7 8 9");
    System.out.println("");
    System.out.println("NOW IF YOU REVERSE 5, YOU WIN!");
    System.out.println("");
    System.out.println("1 2 3 4 5 6 7 8 9");
    System.out.println("");
    System.out.println("NO DOUBT YOU WILL LIKE THIS GAME, BUT");
    System.out.println("IF YOU WANT TO QUIT, REVERSE 0 (ZERO).");
    System.out.println("");

  }  // End of method printRules

  public static void main(String[] args) {

    Reverse game = new Reverse();
    game.play();

  }  // End of method main

}  // End of class Reverse

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


# `73_Reverse/javascript/reverse.js`

这段代码的作用是创建了一个输入框和一个用于显示输出信息的文本框。通过创建一个 promise 函数，当用户点击输入框时，它将挂载一个事件监听器，以便在用户按下键时捕获并处理事件。

具体来说，当用户点击输入框时，将创建一个包含输入框和输出框的 DOM 元素，并将输入框的值为“?”。用户将能够在输入框中键入文本，并将在输入框中按任意键时，将捕获到一个新的 Promise 函数。

当用户按任意键时，将捕获到一个新的 input() 函数，该函数将使用上面创建的 Promise 函数来获取用户输入的值。一旦获取了用户的输入值，将使用 print() 函数将其添加到输出框中。然后，将使用 document.getElementById() 获取到输出框的的唯一 ID，并将其添加到输出框中。

最后，当输出框中的内容达到缓冲区上限时，将使用 print() 函数将其中的所有内容添加到输出框中，并打印出新的缓冲区。


```
// REVERSE
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       input_element = document.createElement("INPUT");

                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      print(input_str);
                                                      print("\n");
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

```



这段代码定义了一个名为 "tab" 的函数，它接受一个参数 "space"，它的作用是打印出指定的字符串。

该函数中包含一个 while 循环和一个 space 变量，其中 space 变量在循环中被使用作为条件。在循环内部，函数的字符串被追加到 str 变量中，并在每次循环结束后打印出整个字符串。

该函数还定义了一个名为 "print_rules" 的函数，该函数的作用是打印出游戏规则。

在 print_rules 函数中，使用 print 函数输出了游戏规则，包括打印出要打印的行、列和数字的数量，以及在反向移动中需要考虑的因素。然后，它向用户询问要打印多少行，列和数字，然后告诉他们如何按逆序移动来得到数字的逆序版本。

最后，该段代码还定义了一个名为 "reverse" 的变量，它的作用是打印出指定数字列表的逆序，如果用户想要退出游戏，则可以通过调用 print_rules 函数中的 "1 6 7 8 9" 来告诉游戏规则，这样游戏就会反向移动，从而使得用户赢得游戏。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var a = [];
var n;

// Subroutine to print the rules
function print_rules()
{
    print("\n");
    print("THIS IS THE GAME OF 'REVERSE'.  TO WIN, ALL YOU HAVE\n");
    print("TO DO IS ARRANGE A LIST OF NUMBERS (1 THROUGH " + n + ")\n");
    print("IN NUMERICAL ORDER FROM LEFT TO RIGHT.  TO MOVE, YOU\n");
    print("TELL ME HOW MANY NUMBERS (COUNTING FROM THE LEFT) TO\n");
    print("REVERSE.  FOR EXAMPLE, IF THE CURRENT LIST IS:\n");
    print("\n");
    print("2 3 4 5 1 6 7 8 9\n");
    print("\n");
    print("AND YOU REVERSE 4, THE RESULT WILL BE:\n");
    print("\n");
    print("5 4 3 2 1 6 7 8 9\n");
    print("\n");
    print("NOW IF YOU REVERSE 5, YOU WIN!\n");
    print("\n");
    print("1 2 3 4 5 6 7 8 9\n");
    print("\n");
    print("NO DOUBT YOU WILL LIKE THIS GAME, BUT\n");
    print("IF YOU WANT TO QUIT, REVERSE 0 (ZERO).\n");
    print("\n");
}

```

This appears to be a Python solution to a game of guess the number. The game starts with a list of numbers, and the player is prompted to guess a number. After each guess, the game checks if the猜测 is equal to any of the numbers in the list and, if it is, the game prints out that it is a win and breaks the game. The game then loops through the numbers again and breaks the game if the player knows the number. The game also has a feature where the player can reverse the game, but it is only possible to reverse the last number that was revealed.



```
// Subroutine to print list
function print_list()
{
    print("\n");
    for (k = 1; k <= n; k++)
        print(" " + a[k] + " ");
    print("\n");
    print("\n");
}

// Main program
async function main()
{
    print(tab(32) + "REVERSE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("REVERSE -- A GAME OF SKILL\n");
    print("\n");
    for (i = 0; i <= 20; i++)
        a[i] = 0;
    // *** N=NUMBER OF NUMBER
    n = 9;
    print("DO YOU WANT THE RULES? (YES OR NO)");
    str = await input();
    if (str.toUpperCase() === "YES" || str.toUpperCase() === "Y")
        print_rules();
    while (1) {
        // *** Make a random list a(1) to a(n)
        a[1] = Math.floor((n - 1) * Math.random() + 2);
        for (k = 2; k <= n; k++) {
            do {
                a[k] = Math.floor(n * Math.random() + 1);
                for (j = 1; j <= k - 1; j++) {
                    if (a[k] == a[j])
                        break;
                }
            } while (j <= k - 1) ;
        }
        // *** Print original list and start game
        print("\n");
        print("HERE WE GO ... THE LIST IS:\n");
        t = 0;
        print_list();
        while (1) {
            while (1) {
                print("HOW MANY SHALL I REVERSE");
                r = parseInt(await input());
                if (r == 0)
                    break;
                if (r <= n)
                    break;
                print("OOPS! WRONG! I CAN REVERSE AT MOST " + n + "\n");
            }
            if (r == 0)
                break;
            t++;
            // *** Reverse r numbers and print new list
            for (k = 1; k <= Math.floor(r / 2); k++) {
                z = a[k];
                a[k] = a[r - k + 1];
                a[r - k + 1] = z;
            }
            print_list();
            // *** Check for a win
            for (k = 1; k <= n; k++) {
                if (a[k] != k)
                    break;
            }
            if (k > n) {
                print("YOU WON IT IN " + t + " MOVES!!!\n");
                print("\n");
                break;
            }
        }
        print("\n");
        print("TRY AGAIN? (YES OR NO)");
        str = await input();
        if (str.toUpperCase() === "NO" || str.toUpperCase() === "N")
            break;
    }
    print("\n");
    print("O.K. HOPE YOU HAD FUN!!\n");
}

```

这道题目是一个不完整的程序，缺少了程序的具体内容。请提供更多信息，我会尽力解释该程序的作用。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)
