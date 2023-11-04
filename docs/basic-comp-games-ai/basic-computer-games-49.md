# BasicComputerGames源码解析 49

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `46_Hexapawn/python/hexapawn.py`

这段代码是一个机器学习游戏的描述，它借鉴了马丁·加德纳在《八十二分之一记得》一书中提到的HEXAPAWN游戏。这是一个时间共享系统，由R.A. Kaapke于1976年5月5日创建。该游戏的玩法是通过让学生在有限的时间内学习来评分。

此代码还提到了该游戏最初是在H-P时间复制的系统上实现的，并且是由Jeff Dalton命名的。在2000年，该游戏被转换为 MITS Basic 格式。最后，该游戏是通过Dave LeCompte编写的Python脚本实现的。


```
"""
HEXAPAWN

A machine learning game, an interpretation of HEXAPAWN game as
presented in Martin Gardner's "The Unexpected Hanging and Other
Mathematical Diversions", Chapter Eight: A Matchbox Game-Learning
Machine.

Original version for H-P timeshare system by R.A. Kaapke 5/5/76
Instructions by Jeff Dalton
Conversion to MITS BASIC by Steve North


Port to Python by Dave LeCompte
"""

```

这段代码是一个文本注释，它描述了一个名为 "PORTING NOTES" 的文件。这个文件似乎包含了游戏中的基本代码，以及一些注释。

作者似乎在鼓励其他开发者进行 hard-copy hacking，这种做法可能会有不同的思路。他似乎在强调一个 spoiler，即游戏的目标没有明确的说明，这是为了给玩家带来挑战。

然而，这个文件似乎没有提供任何有用的信息，除了告诉我们游戏的基本结构和一些注释。要真正理解这个游戏，我们需要看到实际的代码或进行更深入的修改。


```
# PORTING NOTES:
#
# I printed out the BASIC code and hand-annotated what each little block
# of code did, which feels amazingly retro.
#
# I encourage other porters that have a complex knot of GOTOs and
# semi-nested subroutines to do hard-copy hacking, it might be a
# different perspective that helps.
#
# A spoiler - the objective of the game is not documented, ostensibly to
# give the human player a challenge. If a player (human or computer)
# advances a pawn across the board to the far row, that player wins. If
# a player has no legal moves (either by being blocked, or all their
# pieces having been captured), that player loses.
#
```

这段代码定义了一个名为BoardLayout的Python类，用于表示BASIC游戏中的游戏板布局。游戏板布局类包含一个2二维列表，每个列表项都是一个由两个数字组成的元组，表示游戏板上的一个位置。游戏板布局类还实现了两个方法，CheckBoardLayout方法用于检查两个游戏板布局是否相等，而removeLosingMoves方法用于移除已经移动的损失游戏。

具体来说，该代码将原始BASIC程序中的22二维表格数据编码成了19种不同的游戏板布局，每个布局都有对应的2维表格。这些布局包括16个大小为4x4的布局以及3个大小为3x3的布局。每个布局的2维表格中，每个位置都被转换成了两个数字，即行和列的ASCII码值。在游戏的进行过程中，AI会不断地覆盖这些损失游戏的位置，将其值设置为0。

为了将这个2维表格数据转换为易于阅读的Python类，BoardLayout类的逻辑是在CheckBoardLayout方法中检查行是否等于0，在removeLosingMoves方法中移除已经移动的损失游戏的位置。


```
# The original BASIC had 2 2-dimensional tables stored in DATA at the
# end of the program. This encoded all 19 different board configurations
# (Hexapawn is a small game), with reflections in one table, and then in
# a parallel table, for each of the 19 rows, a list of legal moves was
# encoded by turning them into 2-digit decimal numbers. As gameplay
# continued, the AI would overwrite losing moves with 0 in the second
# array.
#
# My port takes this "parallel array" structure and turns that
# information into a small Python class, BoardLayout. BoardLayout stores
# the board description and legal moves, but stores the moves as (row,
# column) 2-tuples, which is easier to read. The logic for checking if a
# BoardLayout matches the current board, as well as removing losing move
# have been moved into methods of this class.

```

这段代码定义了一个名为 `ComputerMove` 的类，表示计算机在棋盘上的移动。

它从 `typing.NamedTuple` 类中创建了一个包含棋盘坐标和平移动中的棋子的类。

该类有两个方法：

- `board_index`: 棋盘的行数，从0开始。
- `move_index`: 棋盘的列数，从0开始。
- `m1`: 棋盘上该行该列的电脑控制标记，为-1时表示该位置没有棋子，为0表示该位置的棋子是人类控制的，为 `COMPUTER_PIECE` 表示该位置的棋子是计算机控制的。
- `m2`: 棋盘上该行该列的电脑控制标记，为-1时表示该位置没有棋子，为0表示该位置的棋子是人类控制的，为 `COMPUTER_PIECE` 表示该位置的棋子是计算机控制的。

该类继承自 `NamedTuple` 类，提供了棋盘坐标和平移动中的棋子类型属性。

该类的方法允许在游戏过程中创建 `ComputerMove` 对象，并使用它们来移动棋子。例如，计算机可以从一个位置移动到另一个位置，或者在特定的位置放置棋子。


```
import random
from typing import Iterator, List, NamedTuple, Optional, Tuple

PAGE_WIDTH = 64

HUMAN_PIECE = 1
EMPTY_SPACE = 0
COMPUTER_PIECE = -1


class ComputerMove(NamedTuple):
    board_index: int
    move_index: int
    m1: int
    m2: int


```

这段代码定义了两个函数，分别是`print_centered`和`print_header`。这两个函数的功能是打印一个字符串，并将其居中显示。

具体来说，`print_centered`函数接收一个字符串参数`msg`，然后计算出字符串中`PAGE_WIDTH`与`len(msg)`之间有多少个空白字符，然后使用这些空白字符将字符串和居中显示。这个函数的作用是将一个字符串居中显示。

`print_header`函数接收一个字符串参数`title`，并使用`print_centered`函数打印出这个标题。这个函数的作用是打印一个header，将字符串`title`居中显示。

两个函数的实现都很简单，主要使用了`print_centered`函数来实现字符串居中显示。


```
wins = 0
losses = 0


def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)


def print_header(title: str) -> None:
    print_centered(title)
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")


def print_instructions() -> None:
    print(
        """
```

这段代码是一个用于运行Hexapawn游戏的程序。Hexapawn是一个使用 chess pawns 在3x3棋盘上进行对局的游戏。这些pawns会按照 chess的规则向前移动一个空间，或者向后走一个空间，并尝试捕捉相反方向上的man。在棋盘上，pawns的值为'O'，computer的pawns的值为'X'，而空格的值为'.'。

进入移动棋子时，需要输入该棋子从哪个位置移动到哪个位置，两个数字之间用逗号隔开。计算机程序会随机移动。

计算机程序会在初始时没有任何策略。然而，随着每个游戏的进行，它将学习并随机移动，学习从每个游戏中。因此，在最终的棋局中，计算机程序可能会动态地移动，以最大化它的胜率。


```
THIS PROGRAM PLAYS THE GAME OF HEXAPAWN.
HEXAPAWN IS PLAYED WITH CHESS PAWNS ON A 3 BY 3 BOARD.
THE PAWNS ARE MOVED AS IN CHESS - ONE SPACE FORWARD TO
AN EMPTY SPACE OR ONE SPACE FORWARD AND DIAGONALLY TO
CAPTURE AN OPPOSING MAN.  ON THE BOARD, YOUR PAWNS
ARE 'O', THE COMPUTER'S PAWNS ARE 'X', AND EMPTY
SQUARES ARE '.'.  TO ENTER A MOVE, TYPE THE NUMBER OF
THE SQUARE YOU ARE MOVING FROM, FOLLOWED BY THE NUMBER
OF THE SQUARE YOU WILL MOVE TO.  THE NUMBERS MUST BE
SEPERATED BY A COMMA.

THE COMPUTER STARTS A SERIES OF GAMES KNOWING ONLY WHEN
THE GAME IS WON (A DRAW IS IMPOSSIBLE) AND HOW TO MOVE.
IT HAS NO STRATEGY AT FIRST AND JUST MOVES RANDOMLY.
HOWEVER, IT LEARNS FROM EACH GAME.  THUS, WINNING BECOMES
```

这段代码是一个文本，包含了两个主要的作用：

1. 解释棋规，告诉玩家如何下棋，包括说明棋子的数量、棋子的移动规则以及如何计算胜负等。

2. 帮助玩家计算如何Offset(抵消)他们的初始优势，以便在游戏中更有效地下棋。具体来说，这段代码告诉玩家他们在每局游戏中的初始优势是-3，并且让他们了解如何通过移动来抵消这个初始优势。这使得玩家在游戏中的得分更容易，因为他们可以更容易地获得分数超越对手。


```
MORE AND MORE DIFFICULT.  ALSO, TO HELP OFFSET YOUR
INITIAL ADVANTAGE, YOU WILL NOT BE TOLD HOW TO WIN THE
GAME BUT MUST LEARN THIS BY PLAYING.

THE NUMBERING OF THE BOARD IS AS FOLLOWS:
          123
          456
          789

FOR EXAMPLE, TO MOVE YOUR RIGHTMOST PAWN FORWARD,
YOU WOULD TYPE 9,6 IN RESPONSE TO THE QUESTION
'YOUR MOVE ?'.  SINCE I'M A GOOD SPORT, YOU'LL ALWAYS
GO FIRST.

"""
    )


```

这两段代码是在 Python 中编写的。

第一段代码是一个函数 prompt_yes_no，它 prompts the user to enter "yes" or "no" in a message. 它通过 while True 循环重复执行这个操作，直到用户输入正确的字符串。如果用户输入 "Y"，函数将返回 True，否则将返回 False。

第二段代码是一个函数 reverse_space_name，它返回一个数字，将给定的空格名字符串翻转（逆序）过来。例如，如果输入字符串是 "CPU"，函数将返回数字 4，因为 "CPU" 翻转过来是 "ominv"。

这两段代码的具体实现可能还有优化空间，比如可以使用更加简洁的语法或者重复检查以避免不必要的 while 循环。


```
def prompt_yes_no(msg: str) -> bool:
    while True:
        print(msg)
        response = input().upper()
        if response[0] == "Y":
            return True
        elif response[0] == "N":
            return False


def reverse_space_name(space_name: int) -> int:
    # reverse a space name in the range 1-9 left to right
    assert 1 <= space_name <= 9

    reflections = {1: 3, 2: 2, 3: 1, 4: 6, 5: 5, 6: 4, 7: 9, 8: 8, 9: 7}
    return reflections[space_name]


```

这段代码定义了一个名为 `BoardLayout` 的类，用于模拟棋盘的布局和移动。

```python
def is_space_in_center_column(space_name: int) -> bool:
   return reverse_space_name(space_name) == space_name
```

这个函数判断一个给定的空间名称是否在棋盘中心列。

```python
class BoardLayout:
   def __init__(self, cells: List[int], move_list: List[Tuple[int, int]]) -> None:
       self.cells = cells
       self.moves = move_list
```

这个类定义了一个 `BoardLayout` 类的初始化方法。

```python
def _check_match_no_mirror(cell_list: List[int]) -> bool:
   return all(
       board_contents == cell_list[space_index]
       for space_index, board_contents in enumerate(self.cells)
   )

def _check_match_with_mirror(self, cell_list: List[int]) -> bool:
   for space_index, board_contents in enumerate(self.cells):
       reversed_space_index = reverse_space_name(space_index + 1) - 1
       if board_contents != cell_list[reversed_space_index]:
           return False
   return True

def check_match(self, cell_list: List[int]) -> Tuple[bool, Optional[bool]]:
   if self._check_match_with_mirror(cell_list):
       return True, True
   elif self._check_match_no_mirror(cell_list):
       return True, False
   return False, None

def get_random_move(
   self, reverse_board: Optional[bool]
) -> Optional[Tuple[int, int, int]]:
   if not self.moves:
       return None
   move_index = random.randrange(len(self.moves))

   m1, m2 = self.moves[move_index]
   if reverse_board:
       m1 = reverse_space_name(m1)
       m2 = reverse_space_name(m2)
   return move_index, m1, m2
```

这个类有两个方法，用于检查匹配和获取随机移动。

`check_match` 方法接收一个包含棋盘元素的列表，然后判断给定的元素是否在中心位置，如果是，则返回两个条件都为 `True`，否则返回 `False` 和一个空选项 `None`。

`get_random_move` 方法接收一个可选的 `reverse_board` 参数，如果没有提供，则返回一个包含原位置的棋子位置的列表。如果提供了 `reverse_board` 参数，则随机选择一个位置进行移动。


```
def is_space_in_center_column(space_name: int) -> bool:
    return reverse_space_name(space_name) == space_name


class BoardLayout:
    def __init__(self, cells: List[int], move_list: List[Tuple[int, int]]) -> None:
        self.cells = cells
        self.moves = move_list

    def _check_match_no_mirror(self, cell_list: List[int]) -> bool:
        return all(
            board_contents == cell_list[space_index]
            for space_index, board_contents in enumerate(self.cells)
        )

    def _check_match_with_mirror(self, cell_list: List[int]) -> bool:
        for space_index, board_contents in enumerate(self.cells):
            reversed_space_index = reverse_space_name(space_index + 1) - 1
            if board_contents != cell_list[reversed_space_index]:
                return False
        return True

    def check_match(self, cell_list: List[int]) -> Tuple[bool, Optional[bool]]:
        if self._check_match_with_mirror(cell_list):
            return True, True
        elif self._check_match_no_mirror(cell_list):
            return True, False
        return False, None

    def get_random_move(
        self, reverse_board: Optional[bool]
    ) -> Optional[Tuple[int, int, int]]:
        if not self.moves:
            return None
        move_index = random.randrange(len(self.moves))

        m1, m2 = self.moves[move_index]
        if reverse_board:
            m1 = reverse_space_name(m1)
            m2 = reverse_space_name(m2)

        return move_index, m1, m2


```

以下是 Python 代码实现，对应的 UI 图如下：

```
import numpy as np
import matplotlib.pyplot as plt

class Board:
   def __init__(self, dimensions, pieces):
       self.dimensions = dimensions
       self.pieces = pieces
       
   def layout(self, positions, sizes):
       self.positions = positions
       self.sizes = sizes
       
   def display(self):
       return self.pieces
       
   def get_size(self, piece):
       return self.sizes[piece]
       
   def get_position(self, piece):
       return self.positions[piece]
       
   def add_piece(self, piece, position):
       self.pieces[piece] = position
       self.positions[piece] = position
       self.sizes[piece] = 1
       
   def remove_piece(self, piece, position):
       self.pieces[piece] = position
       self.positions[piece] = position
       self.sizes[piece] = 0
       
   def update(self, events):
       for event in events:
           if event.type == 'gaining':
               self.add_piece(event. piece, event.position)
           elif event.type == 'losing':
               self.remove_piece(event.piece, event.position)
           elif event.type == 'changing_position':
               self.display()
           elif event.type == 'shaking':
               self.get_size(event.piece)
           elif event.type == '旋转':
               self.rotate()
           elif event.type == 'remove_piece_for_rotation':
               self.remove_piece(event.piece, event.position)
           elif event.type == 'swap':
               self.swap(event.piece1, event.piece2)
           elif event.type == 'nothing':
               pass
           else:
               break
       
   def rotate(self):
       self.rotation_number = int(input('Enter rotation number: '))
       self.rotation = np.rotation(self.pieces, self.rotation_number)
       
   def swap(self, piece1, piece2):
       self.pieces[piece1], self.pieces[piece2] = self.pieces[piece2], self.pieces[piece1]
       
   def get_number_of_pieces(self):
       return len(self.pieces)
       
   def get_rotation_positions(self):
       return np.array([self.positions], dtype=int)
       
   def get_rotation_size(self):
       return [self.sizes], dtype=int
       
   def add_rotation(self, piece, angle):
       self.rotation_number = angle
       self.rotation = np.rotation(self.pieces, angle)
       self.update()
       
   def display_rotation(self, position):
       return self.display()
       
   def get_rotation_event(self, index):
       return np.where(self.rotation_number == index)[0]
       
   def get_number_of_rotations(self):
       return len(self)
       
   def is_empty(self):
       return len(self.pieces) == 0
       
   def display(self):
       return self.display_rotation(self.position)
       
   def update_display(self):
       self.display()
       
   def add_event(self, event):
       self.update_display()
       
   def remove_event(self, event):
       self.update_display()
       
   def check_for_rotation(self):
       return self.is_empty()
       
   def check_for_event(self, index, event):
       if event.type == 'gaining':
           self.add_event(index, event)
       elif event.type == 'losing':
           self.remove_event(index, event)
           self.remove_event(index, event)
       elif event.type == 'changing_position':
           self.display_rotation(event.position)
       elif event.type == 'swap':
           self.swap(event.piece1, event.piece2)
       elif event.type == 'nothing':
           pass
       else:
           break
   
   def add_构件(self, piece, position):
       self.add_event(piece, {'type': 'gaining', 'position': position})
   
   def remove_构件(self, piece, position):
       self.remove_event(piece, {'type': 'losing', 'position': position})
   
   def remove_构件_for_rotation(self, piece, position):
       self.remove_event(piece, {'type': 'changing_position', 'position': position})
       self.update()
   
   def get_event(self, index):
       return np.where(self.get_number_of_rotations() == index)[0]
   
   def get_构件_event(self, piece):
       return self.get_event(self.get_number_of_rotations(piece))
   
   def get_size(self):
       return self.sizes
   
   def set_size(self, size):
       self.sizes = size
       
   def get_number_of_layers(self):
       return np.layers(self.sizes, dtype=int)[0]
   
   def set_layers(self, layers, size):
       self.sizes = size
       self.layers = layers
       
   def is_clockwise(self):
       clockwise = [0]
       for angle in self.rotation]
       for event in self.get_rotation_event(len(self.rotation)):
           if event == 0:
               break
           elif event == 1:
               clockwise.append(angle)
               break
           else:
               clockwise.append(2 * angle - event)
               break
       return clockwise
   
   def is_counterclockwise(self):
       counterclockwise = [0]
       for angle in self.get_rotation_event(len(self.rotation))
       for event in self.get_rotation_event(len(self.rotation)-1):
           if event == 0:
               break
           


```
boards = [
    BoardLayout([-1, -1, -1, 1, 0, 0, 0, 1, 1], [(2, 4), (2, 5), (3, 6)]),
    BoardLayout([-1, -1, -1, 0, 1, 0, 1, 0, 1], [(1, 4), (1, 5), (3, 6)]),
    BoardLayout([-1, 0, -1, -1, 1, 0, 0, 0, 1], [(1, 5), (3, 5), (3, 6), (4, 7)]),
    BoardLayout([0, -1, -1, 1, -1, 0, 0, 0, 1], [(3, 6), (5, 8), (5, 9)]),
    BoardLayout([-1, 0, -1, 1, 1, 0, 0, 1, 0], [(1, 5), (3, 5), (3, 6)]),
    BoardLayout([-1, -1, 0, 1, 0, 1, 0, 0, 1], [(2, 4), (2, 5), (2, 6)]),
    BoardLayout([0, -1, -1, 0, -1, 1, 1, 0, 0], [(2, 6), (5, 7), (5, 8)]),
    BoardLayout([0, -1, -1, -1, 1, 1, 1, 0, 0], [(2, 6), (3, 5)]),
    BoardLayout([-1, 0, -1, -1, 0, 1, 0, 1, 0], [(4, 7), (4, 8)]),
    BoardLayout([0, -1, -1, 0, 1, 0, 0, 0, 1], [(3, 5), (3, 6)]),
    BoardLayout([0, -1, -1, 0, 1, 0, 1, 0, 0], [(3, 5), (3, 6)]),
    BoardLayout([-1, 0, -1, 1, 0, 0, 0, 0, 1], [(3, 6)]),
    BoardLayout([0, 0, -1, -1, -1, 1, 0, 0, 0], [(4, 7), (5, 8)]),
    BoardLayout([-1, 0, 0, 1, 1, 1, 0, 0, 0], [(1, 5)]),
    BoardLayout([0, -1, 0, -1, 1, 1, 0, 0, 0], [(2, 6), (4, 7)]),
    BoardLayout([-1, 0, 0, -1, -1, 1, 0, 0, 0], [(4, 7), (5, 8)]),
    BoardLayout([0, 0, -1, -1, 1, 0, 0, 0, 0], [(3, 5), (3, 6), (4, 7)]),
    BoardLayout([0, -1, 0, 1, -1, 0, 0, 0, 0], [(2, 8), (5, 8)]),
    BoardLayout([-1, 0, 0, -1, 1, 0, 0, 0, 0], [(1, 5), (4, 7)]),
]


```

这段代码定义了两个函数，get_move() 和 remove_move()，它们的功能是帮助玩家在游戏棋盘上移动棋子。

get_move()函数接收两个参数，一个是棋盘的索引，另一个是移动棋子的索引。函数首先检查传入的索引是否合法，然后返回棋盘上指定移动棋子的位置。

remove_move()函数也接收两个参数，一个是棋盘的索引，另一个是移动棋子的索引。函数首先检查传入的索引是否合法，然后从棋盘上删除指定移动棋子的位置。


```
def get_move(board_index: int, move_index: int) -> Tuple[int, int]:
    assert board_index >= 0 and board_index < len(boards)
    board = boards[board_index]

    assert move_index >= 0 and move_index < len(board.moves)

    return board.moves[move_index]


def remove_move(board_index: int, move_index: int) -> None:
    assert board_index >= 0 and board_index < len(boards)
    board = boards[board_index]

    assert move_index >= 0 and move_index < len(board.moves)

    del board.moves[move_index]


```

这两段代码是在一个名为`Board`的类中定义的。`init_board()`函数用于初始化游戏棋盘，它返回一个包含3个"电脑棋子"和3个"空格"的列表，即游戏开始时所有位置的棋子情况。`print_board()`函数用于打印棋盘情况，它接收一个包含3个整数的列表（即游戏棋盘上的所有位置），然后根据这个列表制作了下面这个棋盘的英文拼写，最后输出打印出来。


```
def init_board() -> List[int]:
    return [COMPUTER_PIECE] * 3 + [EMPTY_SPACE] * 3 + [HUMAN_PIECE] * 3


def print_board(board: List[int]) -> None:
    piece_dict = {COMPUTER_PIECE: "X", EMPTY_SPACE: ".", HUMAN_PIECE: "O"}

    space = " " * 10
    print()
    for row in range(3):
        line = ""
        for column in range(3):
            line += space
            space_number = row * 3 + column
            space_contents = board[space_number]
            line += piece_dict[space_contents]
        print(line)
    print()


```

这段代码定义了一个名为 `get_coordinates` 的函数，它接受一个元组类型的参数 `Tuple[int, int]`。函数内部使用一个无限循环来不断尝试从用户那里获取坐标信息。

在循环的每次尝试中，函数首先会尝试从用户输入中提取一个坐标。如果用户输入的格式不正确(比如没有空格或多余的逗号)，函数会通过 `print_illegal` 函数来显示一个错误消息并终止循环。否则，函数会将提取出的坐标存储在一个元组中，并返回该元组。

函数首先会从左到右遍历提取出的坐标，然后返回该元组。


```
def get_coordinates() -> Tuple[int, int]:
    while True:
        try:
            print("YOUR MOVE?")
            response = input()
            m1, m2 = (int(c) for c in response.split(","))
            return m1, m2
        except ValueError:
            print_illegal()


def print_illegal() -> None:
    print("ILLEGAL MOVE.")


```

这段代码定义了两个函数，一个是 `board_contents`，另一个是 `is_legal_human_move`。这两个函数都用于在棋盘游戏中检查玩家移动是否合法。

`board_contents` 函数接收一个棋盘和玩家的空间数量作为参数，并返回该棋盘的下一个空位置。这个位置可能是玩家的空间，也可能是电脑的空间。

`is_legal_human_move` 函数接收一个棋盘、玩家的两个空间数量以及玩家的下一步移动作为参数。这个函数会检查玩家移动是否合法，并返回一个布尔值。具体规则如下：

1. 如果玩家的空间不是 HUMAN_PIECE，则不会影响游戏，返回 False。
2. 如果玩家的空间包含 HUMAN_PIECE，则确保玩家可以在这里下棋，不影响游戏，返回 False。
3. 如果玩家已经在目标位置放置了 HUMAN_PIECE，则确保玩家可以在这里下棋，不影响游戏，返回 False。
4. 如果玩家还没有移动，则确保玩家可以移动，不影响游戏，返回 False。
5. 如果玩家移动不合法（例如尝试移动到另一个玩家已经占有的位置），则确保玩家可以在这里下棋，不影响游戏，返回 False。
6. 如果玩家在尝试移动时移动到了棋盘的角部（例如尝试向下或向右移动，但在棋盘角部移动），则确保玩家可以在这里下棋，不影响游戏，返回 False。

最后，如果玩家移动合法，则确保玩家可以在这里下棋，不影响游戏，返回 True。


```
def board_contents(board: List[int], space_number: int) -> int:
    return board[space_number - 1]


def set_board(board: List[int], space_number: int, new_value: int) -> None:
    board[space_number - 1] = new_value


def is_legal_human_move(board: List[int], m1: int, m2: int) -> bool:
    if board_contents(board, m1) != HUMAN_PIECE:
        # Start space doesn't contain player's piece
        return False
    if board_contents(board, m2) == HUMAN_PIECE:
        # Destination space contains player's piece (can't capture your own piece)
        return False

    is_capture = m2 - m1 != -3
    if is_capture and board_contents(board, m2) != COMPUTER_PIECE:
        # Destination does not contain computer piece
        return False

    if m2 > m1:
        # can't move backwards
        return False

    if (not is_capture) and board_contents(board, m2) != EMPTY_SPACE:
        # Destination is not open
        return False

    if m2 - m1 < -4:
        # too far
        return False

    if m1 == 7 and m2 == 3:
        # can't jump corner to corner (wrapping around the board)
        return False
    return True


```

这段代码定义了两个函数，player_piece_on_back_row 和 computer_piece_on_front_row，它们都接受一个棋盘（board）列表作为输入参数，并返回一个布尔值。

player_piece_on_back_row 的作用是检查棋盘上是否有人类棋子。具体来说，它通过检查棋盘中每个位置是否包含人类棋子（即 HUMAN_PIECE）来实现的。这个函数使用了列表遍历，遍历范围是从 1 到 3（不包括 3）。

computer_piece_on_front_row 的作用是检查棋盘上是否有人工智能（compulsive AI）棋子。具体来说，它通过检查棋盘中每个位置是否包含人工智能棋子（即 COMPUTER_PIECE）来实现的。这个函数使用了列表遍历，遍历范围是从 7 到 9（不包括 9）。

all_human_pieces_captured 的作用是检查是否所有人类棋子都被捕获了。具体来说，它返回的是一个布尔值，如果棋盘上所有位置都包含人类棋子，那么这个值就是 True，否则就是 False。

all_computer_pieces_captured 的作用是检查是否所有人工智能棋子都被捕获了。具体来说，它返回的是一个布尔值，如果棋盘上所有位置都包含人工智能棋子，那么这个值就是 True，否则就是 False。


```
def player_piece_on_back_row(board: List[int]) -> bool:
    return any(board_contents(board, space) == HUMAN_PIECE for space in range(1, 4))


def computer_piece_on_front_row(board: List[int]) -> bool:
    return any(board_contents(board, space) == COMPUTER_PIECE for space in range(7, 10))


def all_human_pieces_captured(board: List[int]) -> bool:
    return len(list(get_human_spaces(board))) == 0


def all_computer_pieces_captured(board: List[int]) -> bool:
    return len(list(get_computer_spaces(board))) == 0


```

这段代码定义了两个函数：human_win 和 computer_win，它们用于实现人类和计算机之间的游戏胜利和失败判断。

human_win函数接收一个last_computer_move参数，这个参数表示上一轮游戏中计算机所做的移动。函数首先打印出“YOU WIN”并删除last_computer_move参数，然后将losses变量加1，这个losses变量用于统计人类赢得的局数。

computer_win函数接收一个has_moves参数，这个参数表示这一轮游戏中计算机是否有移动棋子的能力。如果has_moves为False，那么函数会输出“YOU CAN'T MOVE， SO ”，表示计算机无法移动，从而无法获胜。如果has_moves为True，那么函数会输出“I WIN”，表示计算机获得了胜利。

这两个函数的作用是辅助计算机和人类玩家进行游戏判断，当计算机获胜时，会输出“I WIN”，并把wins变量加1，当人类获胜时，会输出“YOU WIN”，并把losses变量加1。


```
def human_win(last_computer_move: ComputerMove) -> None:
    print("YOU WIN")
    remove_move(last_computer_move.board_index, last_computer_move.move_index)
    global losses
    losses += 1


def computer_win(has_moves: bool) -> None:
    if not has_moves:
        msg = "YOU CAN'T MOVE, SO "
    else:
        msg = ""
    msg += "I WIN"
    print(msg)
    global wins
    wins += 1


```

这段代码定义了两个函数，show_scores()是一个函数，而human_has_move()是另一个函数。

show_scores()函数的作用是打印出玩家获胜的游戏数量和输掉的游戏数量，然后返回None。

human_has_move()函数的作用是判断游戏是否可以被人类玩家进行操作。具体来说，该函数会遍历所有的空位置，如果当前位置是空位置，则玩家可以移动该位置的 piece，否则无法进行移动。判断空位置的方法是，如果当前位置在电脑控制的一圈内（以 computer_piece 为中心），则允许移动。如果当前位置是电脑控制的一个 piece，则允许移动。如果当前位置是位于数字 7 或 9 的位置，则允许移动该位置的 piece。如果以上条件都不满足，则不允许移动。

注意，这两个函数内部都是调用的是同一个外部函数 print_scores()，这个函数只是打印出了分数，并不会对 board 参数产生影响。


```
def show_scores() -> None:
    print(f"I HAVE WON {wins} AND YOU {losses} OUT OF {wins + losses} GAMES.\n")


def human_has_move(board: List[int]) -> bool:
    for i in get_human_spaces(board):
        if board_contents(board, i - 3) == EMPTY_SPACE:
            # can move piece forward
            return True
        elif is_space_in_center_column(i):
            if (board_contents(board, i - 2) == COMPUTER_PIECE) or (
                board_contents(board, i - 4) == COMPUTER_PIECE
            ):
                # can capture from center
                return True
            else:
                continue
        elif i < 7:
            assert (i == 4) or (i == 6)
            if board_contents(board, 2) == COMPUTER_PIECE:
                # can capture computer piece at 2
                return True
            else:
                continue
        elif board_contents(board, 5) == COMPUTER_PIECE:
            assert (i == 7) or (i == 9)
            # can capture computer piece at 5
            return True
        else:
            continue
    return False


```

这段代码定义了三个函数，用于生成游戏棋盘上的空间、包含指定类型数值的空间以及包含指定类型数值的人的空间。

get_board_spaces()函数生成1到9的整数，并返回一个迭代器。这个迭代器每次遍历一个数字，从而生成棋盘上的空间名称。

get_board_spaces_with()函数接收一个棋盘数字列表和一个指定类型数值，并返回一个迭代器。这个迭代器遍历所有的空间，如果棋盘上的某个位置包含指定的数值，那么就返回这个位置在数字列表中的索引。

get_human_spaces()函数接收一个棋盘数字列表，并返回一个迭代器。这个迭代器包含所有在棋盘上的人类玩家的空间名称，即1到9的数字。


```
def get_board_spaces() -> Iterator[int]:
    """generates the space names (1-9)"""
    yield from range(1, 10)


def get_board_spaces_with(board: List[int], val: int) -> Iterator[int]:
    """generates spaces containing pieces of type val"""
    for i in get_board_spaces():
        if board_contents(board, i) == val:
            yield i


def get_human_spaces(board: List[int]) -> Iterator[int]:
    yield from get_board_spaces_with(board, HUMAN_PIECE)


```

该代码定义了三个函数，分别是 `get_empty_spaces` 和 `get_computer_spaces`，它们接受一个 `board` 列表作为参数，并返回一个迭代器 `Iterator[int]`。这两个函数的作用是获取与 `board` 列表相邻的空位置和计算机可以占领的位置。

接下来定义了一个名为 `has_computer_move` 的函数，它接受一个 `board` 列表作为参数，并返回一个布尔值，表示给定的 `board` 是否包含可以允许计算机移动的位置。

`get_empty_spaces` 和 `get_computer_spaces` 函数的作用相似，都是获取与 `board` 列表相邻的空位置和计算机可以占领的位置，具体实现是通过调用一个名为 `get_board_spaces_with` 的函数，这个函数接受一个 `board` 列表和一个空格 `EMPTY_SPACE` 或 `COMPUTER_PIECE`，并返回一个包含与 `board` 列表相邻位置的列表。这里 `EMPTY_SPACE` 代表空格，`COMPUTER_PIECE` 代表计算机可以占领的位置。

`has_computer_move` 函数的作用是检查给定的 `board` 是否包含可以允许计算机移动的位置。具体实现是通过遍历 `get_computer_spaces_with` 函数返回的列表，并检查每个位置是否等于 `EMPTY_SPACE` 或 `COMPUTER_PIECE`。如果是，则表示可以允许计算机移动，返回 `True`；否则，表示不可以允许计算机移动，返回 `False`。


```
def get_empty_spaces(board: List[int]) -> Iterator[int]:
    yield from get_board_spaces_with(board, EMPTY_SPACE)


def get_computer_spaces(board: List[int]) -> Iterator[int]:
    yield from get_board_spaces_with(board, COMPUTER_PIECE)


def has_computer_move(board: List[int]) -> bool:
    for i in get_computer_spaces(board):
        if board_contents(board, i + 3) == EMPTY_SPACE:
            # can move forward (down)
            return True

        if is_space_in_center_column(i):
            # i is in the middle column
            if (board_contents(board, i + 2) == HUMAN_PIECE) or (
                board_contents(board, i + 4) == HUMAN_PIECE
            ):
                return True
        else:
            if i > 3:
                # beyond the first row
                if board_contents(board, 8) == HUMAN_PIECE:
                    # can capture on 8
                    return True
                else:
                    continue
            else:
                if board_contents(board, 5) == HUMAN_PIECE:
                    # can capture on 5
                    return True
                else:
                    continue
    return False


```

这两段代码定义了一个名为 `find_board_index_that_matches_board` 的函数，它接收一个包含数字的列表 `board`，并返回匹配该列表的棋盘的索引和布尔值。

函数的核心部分是一个循环，它遍历所有可能的棋盘布局，并使用每个布局的 `check_match` 方法来检查当前的 `board` 是否与布局匹配。如果匹配成功，函数将返回匹配的棋盘索引和布尔值。在循环的结尾，如果仍然没有找到匹配的棋盘，函数将引发 `RuntimeError`。

另一个函数 `pick_computer_move` 接收一个包含数字的列表 `board`，并返回一个随机选择的位置，如果没有选择，则返回 `None`。

函数的核心部分是一个条件语句，它检查 `board` 是否包含计算机可以移动的位置。如果是，函数使用 `get_random_move` 方法从所有位置中选择一个随机位置。如果选择的位置不存在，函数打印 "I RESIGN" 并返回 `None`。


```
def find_board_index_that_matches_board(board: List[int]) -> Tuple[int, Optional[bool]]:
    for board_index, board_layout in enumerate(boards):
        matches, is_reversed = board_layout.check_match(board)
        if matches:
            return board_index, is_reversed

    # This point should never be reached
    # In future, mypy might be able to check exhaustiveness via assert_never
    raise RuntimeError("ILLEGAL BOARD PATTERN.")


def pick_computer_move(board: List[int]) -> Optional[ComputerMove]:
    if not has_computer_move(board):
        return None

    board_index, reverse_board = find_board_index_that_matches_board(board)

    m = boards[board_index].get_random_move(reverse_board)

    if m is None:
        print("I RESIGN")
        return None

    move_index, m1, m2 = m

    return ComputerMove(board_index, move_index, m1, m2)


```

这段代码定义了两个函数，`get_human_move` 和 `apply_move`。这两个函数一起协作，实现了人工智能程序中的一个子游戏。现在，我会详细解释这两个函数的作用。

1. `get_human_move` 函数：

这个函数的主要作用是判断给定的棋盘是否合法人类走法。如果是合法的人类走法，函数会返回两行下标，表示新的棋盘状态。如果不是合法的人类走法，函数会输出一个错误消息。

具体来说，函数首先会调用一个名为 `get_coordinates` 的辅助函数，这个函数的作用是获取一个合法的人类走法。然后，函数会判断输入的 `board` 是否是一个合法的棋盘。如果是合法的棋盘，函数会调用一个名为 `is_legal_human_move` 的辅助函数来判断输入的走法是否合法。如果是合法的人类走法，函数会返回两行下标，表示新的棋盘状态。否则，函数会输出一个错误消息。

2. `apply_move` 函数：

这个函数的作用是将指定的走法应用到给定的棋盘上，并更新棋盘状态。

具体来说，函数首先会获取一个合法的棋盘 `board`，然后获取指定走法的行 `m1` 和列 `m2`。接下来，函数会将指定的走法 `piece_value` 应用到棋盘上，并更新棋盘状态。


```
def get_human_move(board: List[int]) -> Tuple[int, int]:
    while True:
        m1, m2 = get_coordinates()

        if not is_legal_human_move(board, m1, m2):
            print_illegal()
        else:
            return m1, m2


def apply_move(board: List[int], m1: int, m2: int, piece_value: int) -> None:
    set_board(board, m1, EMPTY_SPACE)
    set_board(board, m2, piece_value)


```

这段代码是一个 Python 函数，名为 `play_game`，它用于解释 AI 如何在棋盘上玩棋游戏。AI 初始化了一个棋盘，然后一直循环执行以下操作：

1. 打印棋盘并获取玩家的输入，通常是“HUMAN”或“COMPUTER”。
2. 如果所有计算机 piece 都被捕获了，或者玩家的位置是一个空位置，那么 AI 将认为玩家获胜并返回。
3. 如果所有玩家 move 都已经被应用了，并且棋盘上仍然有一个 computer piece 在前排，那么 AI 将认为计算机获胜并返回。
4. 如果棋盘上仍然有一个 computer piece 在前排，而计算机还没有赢，那么 AI 将移动棋子，通常是玩家的 move。
5. 重复执行步骤 2-4，直到所有的 AI move 都被应用并且棋盘上没有一个 computer piece 在前排。




```
def play_game() -> None:
    last_computer_move = None

    board = init_board()

    while True:
        print_board(board)

        m1, m2 = get_human_move(board)

        apply_move(board, m1, m2, HUMAN_PIECE)

        print_board(board)

        if player_piece_on_back_row(board) or all_computer_pieces_captured(board):
            assert last_computer_move is not None
            human_win(last_computer_move)
            return

        computer_move = pick_computer_move(board)
        if computer_move is None:
            assert last_computer_move is not None
            human_win(last_computer_move)
            return

        last_computer_move = computer_move

        m1, m2 = last_computer_move.m1, last_computer_move.m2

        print(f"I MOVE FROM {m1} TO {m2}")
        apply_move(board, m1, m2, COMPUTER_PIECE)

        print_board(board)

        if computer_piece_on_front_row(board):
            computer_win(True)
            return
        elif (not human_has_move(board)) or (all_human_pieces_captured(board)):
            computer_win(False)
            return


```

这段代码是一个Python程序，名为“main”。程序的主要目的是让用户在两个选项中做出选择（Y或N），然后根据用户的选择，程序会调用不同的函数来显示游戏结果。

具体来说，这段代码执行以下操作：

1. 首先定义了一个名为“main”的函数，该函数包含一个空括号“()”以及一个“None”类型的参数“None”。

2. 在函数内，程序输出一个头部的字符串“HEXAPAWN”，这是用来表示游戏开始的信息。

3. 然后，程序调用一个名为“prompt_yes_no”的函数，该函数会提示用户输入一个字符（Y或N）。

4. 如果用户输入的答案是“Y”，那么程序会调用一个名为“print_instructions”的函数。否则，程序跳过这个函数。

5. 接下来，程序创建了一个名为“wins”的变量，并将其值设置为0。还创建了一个名为“losses”的变量，并将其值设置为0。

6. 程序进入了一个无限循环，该循环将无限重复执行步骤2至4。

7. 在循环中，程序会调用一个名为“play_game”的函数。然而，由于没有提供该函数的具体实现，因此无法知道程序会执行什么操作。

8. 程序会调用一个名为“show_scores”的函数。同样，由于没有提供该函数的具体实现，因此无法知道程序会执行什么操作。

9. 最后，程序跳出了无限循环，并在“if __name__ == "__main__":”这个语句处输出了“main”函数。


```
def main() -> None:
    print_header("HEXAPAWN")
    if prompt_yes_no("INSTRUCTIONS (Y-N)?"):
        print_instructions()

    global wins, losses
    wins = 0
    losses = 0

    while True:
        play_game()
        show_scores()


if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Hi-Lo

This game is an adaptation of the game GUESS; however, instead of just guessing a number between 1 and 100, in this game you win dollars when you guess the number. The directions, in the words of the author, are as follows:
1. There is an amount of money, between one and one hundred dollars, in the “HI-LO” jackpot.
2. You will have six chances in which to guess the amount of money in the jackpot.
3. After each guess, the computer will tell whether the guess was too high or too low.
4. If the correct amount of money is not guessed after six chances, the computer will print the amount in the jackpot.
5. If the correct amount of money is guessed within the six chance limit, the computer will register this amount.
6. After each sequence of guesses, you have the choice of playing again or ending the program. If a new game is played, a new amount of money will constitute the jackpot.
7. If youwin more than once, then your earnings are totalled.

The author is Dean ALtman of Fort Worth, Texas.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=85)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=100)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `47_Hi-Lo/csharp/Program.cs`

这段代码是一个C#程序，它将输出一个游戏“Hi Lo”的规则和提示。这个游戏的规则是在一个有限的时间内，玩家需要猜测一个金额，每次猜测后，程序会告诉玩家猜测的金额是高还是低，直到玩家猜对金额，或者在规定的6次尝试内猜对金额并赢得全部奖金为止。如果玩家在6次尝试内没有猜对金额，那么游戏结束。

程序中，首先定义了一个字符串变量“AMOUNT”，但没有给它赋值。接下来，程序使用Console.WriteLine()方法输出游戏规则和提示，然后再次定义了一个字符串变量“THIS IS THE GAME OF HI LO”，并在这行输出游戏中给出的信息，包括游戏规则和提示。

接下来，程序使用for循环语句，遍历6次，每次循环输出一条游戏规则和提示，然后再次输出一个空行，以便下一行游戏规则和提示能够更清晰地显示在屏幕上。在每次循环中，程序使用Tab()方法来输出指定的字符，以便将游戏规则和提示与输出内容分开显示。

最后，程序再次使用for循环语句，遍历6次，但如果玩家在6次尝试内没有猜对金额，那么程序将结束游戏并输出一条消息。


```
﻿using System;

Console.WriteLine(Tab(34) +                 "HI LO");
Console.WriteLine(Tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
Console.WriteLine();
Console.WriteLine();
Console.WriteLine();
Console.WriteLine("THIS IS THE GAME OF HI LO.");
Console.WriteLine();
Console.WriteLine("YOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE");
Console.WriteLine("HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU");
Console.WriteLine("GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!");
Console.WriteLine("THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,");
Console.WriteLine("IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS.");
Console.WriteLine();

```

这段代码是一个简单的游戏，让用户猜测一个1到100之间的随机数，猜测次数为6次。每次猜测后，程序会告诉用户猜测结果，如果猜中了，程序会输出奖金，否则会提示猜测过大或过小。如果6次都没猜中，程序会结束游戏并输出总奖金。如果用户在猜测时按YEAS表示继续，按NO表示结束。

随机数生成器的作用是生成一个1到100之间的随机整数。在本游戏中，随机数的作用是生成一个1到100之间的随机整数，用于生成用户的猜测范围。


```
// rnd is our random number generator
Random rnd = new();

bool playAgain = false;
int totalWinnings = 0;

do // Our game loop
{
    int jackpot = rnd.Next(100) + 1; // [0..99] + 1 -> [1..100]
    int guess = 1;

    while (true) // Our guessing loop
    {
        Console.WriteLine();
        int amount = ReadInt("YOUR GUESS ");

        if (amount == jackpot)
        {
            Console.WriteLine($"GOT IT!!!!!!!!!!   YOU WIN {jackpot} DOLLARS.");
            totalWinnings += jackpot;
            Console.WriteLine($"YOUR TOTAL WINNINGS ARE NOW {totalWinnings} DOLLARS.");
            break;
        }
        else if (amount > jackpot)
        {
            Console.WriteLine("YOUR GUESS IS TOO HIGH.");
        }
        else
        {
            Console.WriteLine("YOUR GUESS IS TOO LOW.");
        }

        guess++;
        if (guess > 6)
        {
            Console.WriteLine($"YOU BLEW IT...TOO BAD...THE NUMBER WAS {jackpot}");
            break;
        }
    }

    Console.WriteLine();
    Console.Write("PLAY AGAIN (YES OR NO) ");
    playAgain = Console.ReadLine().ToUpper().StartsWith("Y");

} while (playAgain);

```

这段代码是一个 C# 程序，运行后会输出两行文本。

第一行输出的是 "Console.WriteLine()"，表示在程序运行时会先输出这一行文本。

第二行输出的是 "SO LONG.  HOPE YOU ENJOYED YOURSELF!!!"，这是在输出 "SO LONG" 和 "HOPE YOU ENJOYED YOURSELF!" 这两行文本，它们之间有9个字符的空白。

第三行定义了一个名为 Tab的静态函数，该函数接收一个整数参数 n，并返回一个字符串，该字符串中有 n 个空格。

第四行定义了一个名为 ReadInt的静态函数，该函数接收一个字符串参数 question，并等待用户输入一个数字。该函数在循环中一直输出问题，直到用户输入一个数字。如果用户输入的数字可以转换成整数，函数将返回用户输入的数字，否则会输出 "!Invalid Number Entered。"


```
Console.WriteLine();
Console.WriteLine("SO LONG.  HOPE YOU ENJOYED YOURSELF!!!");

// Tab(n) returns n spaces
static string Tab(int n) => new String(' ', n);

// ReadInt asks the user to enter a number
static int ReadInt(string question)
{
    while (true)
    {
        Console.Write(question);
        var input = Console.ReadLine().Trim();
        if (int.TryParse(input, out int value))
        {
            return value;
        }
        Console.WriteLine("!Invalid Number Entered.");
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `47_Hi-Lo/java/src/HiLo.java`

This is a Java class that represents a computer player. It has several methods for interacting with the player, such as displaying a message and asking the player to guess a number. It also has a method for generating a random number.

The class has several instance variables, such as a String for the displayTextAndGetInput method, a StringArray for a variable number of values, and an int for the randomNumber method. It also has a constructor for the ComputerPlayer class.

The class has several methods for getting input from the keyboard, such as playerGuess, which accepts a integer guess from the player. This method first displays a message asking the player to enter a guess and then returns the player's guess as an integer. It also has methods for checking whether the player entered "Y" or "YES" to a question, yesEntered, which takes a string as input and returns true if the string is in the range of "Y" or "YES", and stringIsAnyValue, which takes a string and a variable number of values and returns true if the string is equal to any of the values, it just compare with the variable number of strings passed.

It also has methods for generating random number.


```
import java.util.Scanner;

/**
 * Game of HiLo
 *
 * Based on the Basic game of Hi-Lo here
 * https://github.com/coding-horror/basic-computer-games/blob/main/47%20Hi-Lo/hi-lo.bas
 *
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 *        new features - no additional text, error checking, etc has been added.
 */
public class HiLo {

    public static final int LOW_NUMBER_RANGE = 1;
    public static final int HIGH_NUMBER_RANGE = 100;
    public static final int MAX_GUESSES = 6;

    private enum GAME_STATE {
        STARTING,
        START_GAME,
        GUESSING,
        PLAY_AGAIN,
        GAME_OVER
    }

    // Used for keyboard input
    private final Scanner kbScanner;

    // Current game state
    private GAME_STATE gameState;

    // Players Winnings
    private int playerAmountWon;

    // Players guess count;
    private int playersGuesses;

    // Computers random number
    private int computersNumber;

    public HiLo() {

        gameState = GAME_STATE.STARTING;
        playerAmountWon = 0;

        // Initialise kb scanner
        kbScanner = new Scanner(System.in);
    }

    /**
     * Main game loop
     *
     */
    public void play() {

        do {
            switch (gameState) {

                // Show an introduction the first time the game is played.
                case STARTING:
                    intro();
                    gameState = GAME_STATE.START_GAME;
                    break;

                // Generate computers number for player to guess, etc.
                case START_GAME:
                    init();
                    System.out.println("O.K.  I HAVE A NUMBER IN MIND.");
                    gameState = GAME_STATE.GUESSING;
                    break;

                // Player guesses the number until they get it or run out of guesses
                case GUESSING:
                    int guess = playerGuess();

                    // Check if the player guessed the number
                    if(validateGuess(guess)) {
                        System.out.println("GOT IT!!!!!!!!!!   YOU WIN " + computersNumber
                                + " DOLLARS.");
                        playerAmountWon += computersNumber;
                        System.out.println("YOUR TOTAL WINNINGS ARE NOW "
                                + playerAmountWon + " DOLLARS.");
                        gameState = GAME_STATE.PLAY_AGAIN;
                    } else {
                        // incorrect guess
                        playersGuesses++;
                        // Ran out of guesses?
                        if (playersGuesses == MAX_GUESSES) {
                            System.out.println("YOU BLEW IT...TOO BAD...THE NUMBER WAS "
                                    + computersNumber);
                            playerAmountWon = 0;
                            gameState = GAME_STATE.PLAY_AGAIN;
                        }
                    }
                    break;

                // Play again, or exit game?
                case PLAY_AGAIN:
                    System.out.println();
                    if(yesEntered(displayTextAndGetInput("PLAY AGAIN (YES OR NO) "))) {
                        gameState = GAME_STATE.START_GAME;
                    } else {
                        // Chose not to play again
                        System.out.println("SO LONG.  HOPE YOU ENJOYED YOURSELF!!!");
                        gameState = GAME_STATE.GAME_OVER;
                    }
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    /**
     * Checks the players guess against the computers randomly generated number
     *
     * @param theGuess the players guess
     * @return true if the player guessed correctly, false otherwise
     */
    private boolean validateGuess(int theGuess) {

        // Correct guess?
        if(theGuess == computersNumber) {
            return true;
        }

        if(theGuess > computersNumber) {
            System.out.println("YOUR GUESS IS TOO HIGH.");
        } else {
            System.out.println("YOUR GUESS IS TOO LOW.");
        }

        return false;
    }

    private void init() {
        playersGuesses = 0;
        computersNumber = randomNumber();
    }

    public void intro() {
        System.out.println("HI LO");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println();
        System.out.println("IS THE GAME OF HI LO.");
        System.out.println();
        System.out.println("YOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE");
        System.out.println("HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU");
        System.out.println("GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!");
        System.out.println("THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,");
        System.out.println("IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS.");
    }

    /**
     * Get players guess from kb
     *
     * @return players guess as an int
     */
    private int playerGuess() {
        return Integer.parseInt((displayTextAndGetInput("YOUR GUESS? ")));
    }

    /**
     * Checks whether player entered Y or YES to a question.
     *
     * @param text  player string from kb
     * @return true of Y or YES was entered, otherwise false
     */
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");
    }

    /**
     * Check whether a string equals one of a variable number of values
     * Useful to check for Y or YES for example
     * Comparison is case insensitive.
     *
     * @param text source string
     * @param values a range of values to compare against the source string
     * @return true if a comparison was found in one of the variable number of strings passed
     */
    private boolean stringIsAnyValue(String text, String... values) {

        // Cycle through the variable number of values and test each
        for(String val:values) {
            if(text.equalsIgnoreCase(val)) {
                return true;
            }
        }

        // no matches
        return false;
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }

    /**
     * Generate random number
     * Used as a single digit of the computer player
     *
     * @return random number
     */
    private int randomNumber() {
        return (int) (Math.random()
                * (HIGH_NUMBER_RANGE - LOW_NUMBER_RANGE + 1) + LOW_NUMBER_RANGE);
    }
}

```

# `47_Hi-Lo/java/src/HiLoGame.java`

这段代码定义了一个名为HiLoGame的公共类，其中包含一个名为main的静态方法，该方法接受一个字符串数组args作为参数。在这个方法中，我们创建了一个名为hiLo的整数对象，并将其赋值为一个新的整数对象。然后，我们调用了一个名为play的静态方法来让hiLo玩HiLo游戏。


```
public class HiLoGame {

    public static void main(String[] args) {

        HiLo hiLo = new HiLo();
        hiLo.play();
    }
}

```

# `47_Hi-Lo/javascript/hi-lo.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是将一个字符串 `str` 打印到页面上，并在页面上创建了一个新的文本节点，将 `str` 作为其内容。这个字符串是通过 `BASIC` 语言到 `JAVASCRIPT` 的转换而得来的，因此 `print` 函数使用的是 `document.getElementById` 方法获取页面上的一个 `DOM` 元素，这个元素将包含文本内容。

`input` 函数的作用是从用户接收输入值，并在输入框中聚焦。它将等待用户输入字符，然后将其存储在变量 `input_str` 中。当用户按下 `RETURN` 键或 `SPACE` 键时，将获取输入框中的值并将其存储在 `input_str` 变量中。

`input` 函数的核心部分是使用 `document.getElementById` 方法获取用户输入的输入框，并将其聚焦。聚焦后，用户将输入字符，函数将接收这个字符并将它存储在 `input_str` 变量中。当用户按下 `RETURN` 键或 `SPACE` 键时，将获取输入框中的值并将其存储在 `input_str` 变量中。


```
// HI-LO
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

```
#include <stdio.h>
#include <math.h>

int main() {
   int a, b, r, y, i;
   int猜测[6];
   
   while (1) {
       b = 0;
       y = 100 * Math.random();
       
       for (i = 1; i <= 6; i++) {
           print("YOUR GUESS");
           猜测[i] = Math.floor(100 * Math.random());
           
           if (猜测[i] < y) {
               print("YOUR GUESS IS TOO LOW.\n");
           } else if (猜测[i] > y) {
               print("YOUR GUESS IS TOO HIGH.\n");
           } else {
               break;
           }
           
           print("\n");
       }
       
       if (b > 6) {
           print("YOU BLEW IT...TOO BAD...THE NUMBER WAS " + y + ".\n");
           r = 0;
       } else {
           print("GOT IT!!!!!!!!!!   YOU WIN " + y + " DOLLARS.\n");
           r += y;
           print("YOUR TOTAL WINNINGS ARE NOW " + r + " DOLLARS.\n");
       }
       
       print("\n");
       print("PLAY AGAIN (YES OR NO)");
       str = await input();
       str = str.toUpperCase();
       if (str != "YES")
           break;
   }
   
   print("\n");
   print("SO LONG.  HOPE YOU ENJOYED YOURSELF!!!");
   
   return 0;
}
```



```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// Main program
async function main()
{
    print(tab(34) + "HI LO\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("THIS IS THE GAME OF HI LO.\n");
    print("\n");
    print("YOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE\n");
    print("HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU\n");
    print("GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!\n");
    print("THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,\n");
    print("IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS.\n");
    print("\n");
    r = 0;
    while (1) {
        b = 0;
        print("\n");
        y = Math.floor(100 * Math.random());
        for (b = 1; b <= 6; b++) {
            print("YOUR GUESS");
            a = parseInt(await input());
            if (a < y) {
                print("YOUR GUESS IS TOO LOW.\n");
            } else if (a > y) {
                print("YOUR GUESS IS TOO HIGH.\n");
            } else {
                break;
            }
            print("\n");
        }
        if (b > 6) {
            print("YOU BLEW IT...TOO BAD...THE NUMBER WAS " + y + "\n");
            r = 0;
        } else {
            print("GOT IT!!!!!!!!!!   YOU WIN " + y + " DOLLARS.\n");
            r += y;
            print("YOUR TOTAL WINNINGS ARE NOW " + r + " DOLLARS.\n");
        }
        print("\n");
        print("PLAY AGAIN (YES OR NO)");
        str = await input();
        str = str.toUpperCase();
        if (str != "YES")
            break;
    }
    print("\n");
    print("SO LONG.  HOPE YOU ENJOYED YOURSELF!!!\n");
}

```

这道题目要求解释以下代码的作用，但是我不清楚你指的是哪段代码，因为你没有提供。通常来说，在编程中，`main()` 函数是一个程序的入口点，也是程序的控制中心。在这个函数中，程序会执行一系列操作来完成它的任务。如果你能提供更多的上下文信息，我会尽力解释代码的作用。


```
main();

```