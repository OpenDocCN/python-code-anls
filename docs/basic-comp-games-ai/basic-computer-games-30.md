# BasicComputerGames源码解析 30

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `23_Checkers/javascript/checkers.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是在文档中创建一个输出文本框，将用户输入的字符串作为参数将其添加到该文本框中。

`input` 函数的作用是从用户那里获取一个字符串，并在输入框中聚焦该字符串。该函数通过添加一个 `INPUT` 元素到文档中，设置其 `type` 属性为 `text`，设置其 `length` 属性为 `50`。然后将用户输入的字符串存储在 `input_str` 变量中，并使用 `addEventListener` 函数监听输入框的 `keydown` 事件。在该事件中，如果用户按下了 `13` 键，函数会将 `input_str` 变量中的字符串打印到屏幕上，并使用 `print` 函数将该字符串存储到 `output` 元素中。此外，函数还会在每次事件处理后解雇 `input_element` 元素，以便在之后的 `keydown` 事件中再次使用它。


```
// CHECKERS
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

这两段代码定义了一个名为 tab 的函数和一个名为 try_computer 的函数。

tab 函数的作用是在给定一个空格数后，返回该空间中所有格子的字符串。具体实现是通过一个 while 循环，在每次循环中给空格数减一，并在循环变量 str 中添加一个空格。当循环变量空格数为 0 时，返回空字符串。

try_computer 函数的作用是在给定一个初始位置的四个方向(u,v,x,y)和一个限制条件后，计算出下一步移动的方向和距离。具体实现是通过一个 while 循环，在每次循环中计算出当前位置与初始位置之间的距离 u 和 v，并检查它们是否越过了限制条件(即是否在网格中)。如果是，通过调用一个名为 eval_move 的函数来返回下一步移动的方向，并返回。如果越不过限制条件，就返回一个空字符串。

eval_move 函数的作用是执行从当前位置垂直向下移动 u 步，然后从当前位置水平向右移动 a 步。移动结束后，将返回一个空字符串。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// x,y = origin square
// a,b = movement direction
function try_computer()
{
    u = x + a;
    v = y + b;
    if (u < 0 || u > 7 || v < 0 || v > 7)
        return;
    if (s[u][v] == 0) {
        eval_move();
        return;
    }
    if (s[u][v] < 0)	// Cannot jump over own pieces
        return;
    u += a;
    u += b;
    if (u < 0 || u > 7 || v < 0 || v > 7)
        return;
    if (s[u][v] == 0)
        eval_move();
}

```

这段代码是一个人工智能程序，可以解释为游戏中的智能算法，用于评估每一步的最佳移动。

它通过以下步骤来评估每一天的最佳移动：

1. 评估当前棋盘状态以及目标棋盘状态。
2. 如果目标棋盘状态与当前棋盘状态相同，移动8步。
3. 如果目标棋盘状态与当前棋盘状态不同，根据当前移动计算得分。
4. 如果当前移动得分超过当前分数，将得分更新为当前移动得分，并将当前移动标记为最优移动。
5. 如果没有更多的移动可做，移动为0。

最终结果被保存到数组 `r` 中，以便记录最优移动。


```
// x,y = origin square
// u,v = target square
function eval_move()
{
    if (v == 0 && s[x][y] == -1)
        q += 2;
    if (Math.abs(y - v) == 2)
        q += 5;
    if (y == 7)
        q -= 2;
    if (u == 0 || u == 7)
        q++;
    for (c = -1; c <= 1; c += 2) {
        if (u + c < 0 || u + c > 7 || v + g < 0)
            continue;
        if (s[u + c][v + g] < 0) {	// Computer piece
            q++;
            continue;
        }
        if (u - c < 0 || u - c > 7 || v - g > 7)
            continue;
        if (s[u + c][v + g] > 0 && (s[u - c][v - g] == 0 || (u - c == x && v - g == y)))
            q -= 2;
    }
    if (q > r[0]) {	// Best movement so far?
        r[0] = q;	// Take note of score
        r[1] = x;	// Origin square
        r[2] = y;
        r[3] = u;	// Target square
        r[4] = v;
    }
    q = 0;
}

```

此代码定义了一个名为 more_captures 的函数，其作用是增加捕获次数。

首先，函数将变量 u 和 v 的值分别相加，并检查它们是否小于 0 或大于 7。如果是，函数返回。

然后，函数检查数组中第 u 行和第 v 列的元素是否为 0，以及区域内的所有元素是否都大于 0。如果是，函数执行 eval_move 函数。

最后，函数会遍历数组中的所有元素，对于每个元素，如果它的行列坐标是 0，或者它的值为正数，函数将添加一行到数组中。


```
function more_captures() {
    u = x + a;
    v = y + b;
    if (u < 0 || u > 7 || v < 0 || v > 7)
        return;
    if (s[u][v] == 0 && s[x + a / 2][y + b / 2] > 0)
        eval_move();
}

var r = [-99, 0, 0, 0, 0];
var s = [];

for (x = 0; x <= 7; x++)
    s[x] = [];

```

In this text-based adventure game, the player is presented with a series of choices that they can make to interact with the game world. Each choice affects the game world and the outcome is determined by the value of the choice made.

The game has different layers of difficulty and the difficulty level is determined by the value of the difficulty level番队的 choices made by the player. The player is provided with different options to choose from, each of which has a unique impact on the game world and the outcome.

The game also has different modes of play, such as explore mode and the option to play as a king. Each mode of play has its own set of choices and outcomes.

The game also has a feature for saving and loading the game at any point in the game. This allows the player to easily return to a previous save or to continue the game at a later time.


```
var g = -1;
var data = [1, 0, 1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, -1, 0, -1, 15];
var p = 0;
var q = 0;

// Main program
async function main()
{
    print(tab(32) + "CHECKERS\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("THIS IS THE GAME OF CHECKERS.  THE COMPUTER IS X,\n");
    print("AND YOU ARE O.  THE COMPUTER WILL MOVE FIRST.\n");
    print("SQUARES ARE REFERRED TO BY A COORDINATE SYSTEM.\n");
    print("(0,0) IS THE LOWER LEFT CORNER\n");
    print("(0,7) IS THE UPPER LEFT CORNER\n");
    print("(7,0) IS THE LOWER RIGHT CORNER\n");
    print("(7,7) IS THE UPPER RIGHT CORNER\n");
    print("THE COMPUTER WILL TYPE '+TO' WHEN YOU HAVE ANOTHER\n");
    print("JUMP.  TYPE TWO NEGATIVE NUMBERS IF YOU CANNOT JUMP.\n");
    print("\n");
    print("\n");
    print("\n");
    for (x = 0; x <= 7; x++) {
        for (y = 0; y <= 7; y++) {
            if (data[p] == 15)
                p = 0;
            s[x][y] = data[p];
            p++;
        }
    }
    while (1) {

        // Search the board for the best movement
        for (x = 0; x <= 7; x++) {
            for (y = 0; y <= 7; y++) {
                if (s[x][y] > -1)
                    continue;
                if (s[x][y] == -1) {	// Piece
                    for (a = -1; a <= 1; a += 2) {
                        b = g;	// Only advances
                        try_computer();
                    }
                } else if (s[x][y] == -2) {	// King
                    for (a = -1; a <= 1; a += 2) {
                        for (b = -1; b <= 1; b += 2) {
                            try_computer();
                        }
                    }
                }
            }
        }
        if (r[0] == -99) {
            print("\n");
            print("YOU WIN.\n");
            break;
        }
        print("FROM " + r[1] + "," + r[2] + " TO " + r[3] + "," + r[4]);
        r[0] = -99;
        while (1) {
            if (r[4] == 0) {	// Computer reaches the bottom
                s[r[3]][r[4]] = -2;	// King
                break;
            }
            s[r[3]][r[4]] = s[r[1]][r[2]];	// Move
            s[r[1]][r[2]] = 0;
            if (Math.abs(r[1] - r[3]) == 2) {
                s[(r[1] + r[3]) / 2][(r[2] + r[4]) / 2] = 0;	// Capture
                x = r[3];
                y = r[4];
                if (s[x][y] == -1) {
                    b = -2;
                    for (a = -2; a <= 2; a += 4) {
                        more_captures();
                    }
                } else if (s[x][y] == -2) {
                    for (a = -2; a <= 2; a += 4) {
                        for (b = -2; b <= 2; b += 4) {
                            more_captures();
                        }
                    }
                }
                if (r[0] != -99) {
                    print(" TO " + r[3] + "," + r[4]);
                    r[0] = -99;
                    continue;
                }
            }
            break;
        }
        print("\n");
        print("\n");
        print("\n");
        for (y = 7; y >= 0; y--) {
            str = "";
            for (x = 0; x <= 7; x++) {
                if (s[x][y] == 0)
                    str += ".";
                if (s[x][y] == 1)
                    str += "O";
                if (s[x][y] == -1)
                    str += "X";
                if (s[x][y] == -2)
                    str += "X*";
                if (s[x][y] == 2)
                    str += "O*";
                while (str.length % 5)
                    str += " ";
            }
            print(str + "\n");
            print("\n");
        }
        print("\n");
        z = 0;
        t = 0;
        for (l = 0; l <= 7; l++) {
            for (m = 0; m <= 7; m++) {
                if (s[l][m] == 1 || s[l][m] == 2)
                    z = 1;
                if (s[l][m] == -1 || s[l][m] == -2)
                    t = 1;
            }
        }
        if (z != 1) {
            print("\n");
            print("I WIN.\n");
            break;
        }
        if (t != 1) {
            print("\n");
            print("YOU WIN.\n");
            break;
        }
        do {
            print("FROM");
            e = await input();
            h = parseInt(e.substr(e.indexOf(",") + 1));
            e = parseInt(e);
            x = e;
            y = h;
        } while (s[x][y] <= 0) ;
        do {
            print("TO");
            a = await input();
            b = parseInt(a.substr(a.indexOf(",") + 1));
            a = parseInt(a);
            x = a;
            y = b;
            if (s[x][y] == 0 && Math.abs(a - e) <= 2 && Math.abs(a - e) == Math.abs(b - h))
                break;
            print("WHAT?\n");
        } while (1) ;
        i = 46;
        do {
            s[a][b] = s[e][h]
            s[e][h] = 0;
            if (Math.abs(e - a) != 2)
                break;
            s[(e + a) / 2][(h + b) / 2] = 0;
            while (1) {
                print("+TO");
                a1 = await input();
                b1 = parseInt(a1.substr(a1.indexOf(",") + 1));
                a1 = parseInt(a1);
                if (a1 < 0)
                    break;
                if (s[a1][b1] == 0 && Math.abs(a1 - a) == 2 && Math.abs(b1 - b) == 2)
                    break;
            }
            if (a1 < 0)
                break;
            e = a;
            h = b;
            a = a1;
            b = b1;
            i += 15;
        } while (1);
        if (b == 7)	// Player reaches top
            s[a][b] = 2;	// Convert to king
    }
}

```

这是经典的 "Hello, World!" 程序，用于在控制台输出 "Hello, World!" 消息。

```
main() 是一个函数名，表示程序的入口点。当程序运行时，首先会执行 main() 函数中的代码，然后逐步执行下去。

在这个程序中，只有一个 main() 函数，它没有参数，也没有返回值，只是一个简单的函数入口点。因此，程序运行后，不会产生任何输出结果。

```


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)

Note: This version has lines and columns numbers to help you with choosing the cell
to move from and to, so you don't have to continually count. It also puts a "." only for
blank cells you can move to, which I think makes for a more pleasing look and makes
it easier to play. If you want the original behavior, start the program with an arg
of "-o" for the original behavior.


# `23_Checkers/python/checkers.py`

这是一段使用 Python 语言编写的字符串，表示一个简单的文本游戏界面，让玩家选择下棋。

整个字符串定义了一个名为 "CHECKERS" 的字符串，然后紧跟一个字符串 "How about a nice game of checkers?"。接下来是两行文本，分别解释了游戏规则和目的。

接下来是两个变量定义，分别定义了棋盘的页宽和人类玩家的编号。然后定义了一个名为 "HUMAN_PLAYER" 的变量，表示人类玩家的编号，并定义了一个名为 "COMPUTER_PLAYER" 的变量，表示计算机玩家的编号。

接着定义了一个名为 "HUMAN_PIECE" 的变量，表示人类棋子的编号。

最后，在字符串的末尾，用 "CHECKERS" 和 "Dave LeCompte" 表示了游戏的作者信息。


```
"""
CHECKERS

How about a nice game of checkers?

Ported by Dave LeCompte
"""

from typing import Iterator, NamedTuple, Optional, Tuple

PAGE_WIDTH = 64

HUMAN_PLAYER = 1
COMPUTER_PLAYER = -1
HUMAN_PIECE = 1
```

这段代码定义了一个名为 `MoveRecord` 的类，该类包含以下字段：

- `quality`: 棋的品质，本游戏中为 0 或 1。
- `start_x`: 棋在水平方向上的起始位置，本游戏中原初位置为 0。
- `start_y`: 棋在垂直方向上的起始位置，本游戏中原初位置为 0。
- `dest_x`: 棋在水平方向上的目标位置，本游戏中原初位置为 0。
- `dest_y`: 棋在垂直方向上的目标位置，本游戏中原初位置为 0。

该类还定义了一个 `__new__` 方法，用于创建新棋实例。该方法返回一个名为 `MoveRecord` 的类对象，该对象将初始化与原初位置相同的棋的品质、位置和目标位置。

该文件中定义的类 `MoveRecord` 以及相关的字段和方法，是一个基于 `NamedTuple` 的数据类。该数据类允许在 Python 3 中使用类型注释来简化类型定义。通过使用 `NamedTuple` 类型注释，可以提供一种易于阅读和理解的方法来定义命名参数的类，使得从函数参数中读取和理解参数类型时更加容易。


```
HUMAN_KING = 2
COMPUTER_PIECE = -1
COMPUTER_KING = -2
EMPTY_SPACE = 0

TOP_ROW = 7
BOTTOM_ROW = 0


class MoveRecord(NamedTuple):
    quality: int
    start_x: int
    start_y: int
    dest_x: int
    dest_y: int


```



该代码定义了一个名为 `print_centered` 的函数，用于在页面宽度为 `PAGE_WIDTH` 的情况下，对传入的 `msg` 参数进行居中输出，并在输出的起始位置和结束位置分别添加定制的字符空间。

另一个名为 `print_header` 的函数，用于在标题为 `title` 的情况下，输出类似于 `console.log()` 的消息，并将其居中。

该代码的最后一个函数 `get_coordinates` 接受一个名为 `prompt` 的字符串参数，用于在向用户询问其坐标之前获取其坐标，并将其返回。该函数首先向用户输出一个错误消息，如果用户在输出的过程中没有提供有效的坐标，然后尝试从用户的输入中解析出一个整数对。如果解析出整数对成功，该函数返回该坐标，否则将输出一个错误消息并继续尝试获取输入。


```
def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)


def print_header(title: str) -> None:
    print_centered(title)
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")


def get_coordinates(prompt: str) -> Tuple[int, int]:
    err_msg = "ENTER COORDINATES in X,Y FORMAT"
    while True:
        print(prompt)
        response = input()
        if "," not in response:
            print(err_msg)
            continue

        try:
            x, y = (int(c) for c in response.split(","))
        except ValueError:
            print(err_msg)
            continue

        return x, y


```



This is a class that represents a game of chess. It has methods for moving pieces, checking the board state, and playing human or computer moves.

The `get_coordinates` method takes a start position (`start_x` and `start_y`) and returns a tuple of the coordinates of the destination position (`dest_x` and `dest_y`). This is useful for printing the board and reporting invalid moves.

The `check_pieces` method checks if the computer has any pieces left or if the board is complete. This is a condition that must be met before playing a move.

The `play_human_move` method takes a start position (`start_x`, `start_y`, and `dest_x`, and `dest_y`) and removes the pieces from the board, if the move is valid. It then updates the board state and returns nothing.

The `play_computer_move` method takes a start position (`start_x`, `start_y`, and `dest_x`, and `dest_y`) and removes the pieces from the board, if the move is valid. It then updates the board state and returns nothing.

The `spaces_with_computer_pieces` method returns a list of the squares on the board that the computer has control over. This is useful for reporting the current state of the game.

The `abs` method returns the absolute value of a given value.

The `print_human_won` method prints a message to the console when the human wins the game.

The `print_computer_won` method prints a message to the console when the computer wins the game.


```
def is_legal_board_coordinate(x: int, y: int) -> bool:
    return (0 <= x <= 7) and (0 <= y <= 7)


class Board:
    def __init__(self) -> None:
        self.spaces = [[0 for y in range(8)] for x in range(8)]
        for x in range(8):
            if (x % 2) == 0:
                self.spaces[x][6] = COMPUTER_PIECE
                self.spaces[x][2] = HUMAN_PIECE
                self.spaces[x][0] = HUMAN_PIECE
            else:
                self.spaces[x][7] = COMPUTER_PIECE
                self.spaces[x][5] = COMPUTER_PIECE
                self.spaces[x][1] = HUMAN_PIECE

    def __str__(self) -> str:
        pieces = {
            EMPTY_SPACE: ".",
            HUMAN_PIECE: "O",
            HUMAN_KING: "O*",
            COMPUTER_PIECE: "X",
            COMPUTER_KING: "X*",
        }

        s = "\n\n\n"
        for y in range(7, -1, -1):
            for x in range(0, 8):
                piece_str = pieces[self.spaces[x][y]]
                piece_str += " " * (5 - len(piece_str))
                s += piece_str
            s += "\n"
        s += "\n\n"

        return s

    def get_spaces(self) -> Iterator[Tuple[int, int]]:
        for x in range(0, 8):
            for y in range(0, 8):
                yield x, y

    def get_spaces_with_computer_pieces(self) -> Iterator[Tuple[int, int]]:
        for x, y in self.get_spaces():
            contents = self.spaces[x][y]
            if contents < 0:
                yield x, y

    def get_spaces_with_human_pieces(self) -> Iterator[Tuple[int, int]]:
        for x, y in self.get_spaces():
            contents = self.spaces[x][y]
            if contents > 0:
                yield x, y

    def get_legal_deltas_for_space(self, x: int, y: int) -> Iterator[Tuple[int, int]]:
        contents = self.spaces[x][y]
        if contents == COMPUTER_PIECE:
            for delta_x in (-1, 1):
                yield (delta_x, -1)
        else:
            for delta_x in (-1, 1):
                for delta_y in (-1, 1):
                    yield (delta_x, delta_y)

    def get_legal_moves(self, x: int, y: int) -> Iterator[MoveRecord]:
        for delta_x, delta_y in self.get_legal_deltas_for_space(x, y):
            new_move_record = self.check_move(x, y, delta_x, delta_y)

            if new_move_record is not None:
                yield new_move_record

    def pick_computer_move(self) -> Optional[MoveRecord]:
        move_record = None

        for start_x, start_y in self.get_spaces_with_computer_pieces():
            for delta_x, delta_y in self.get_legal_deltas_for_space(start_x, start_y):
                new_move_record = self.check_move(start_x, start_y, delta_x, delta_y)

                if new_move_record is None:
                    continue

                if (move_record is None) or (
                    new_move_record.quality > move_record.quality
                ):
                    move_record = new_move_record

        return move_record

    def check_move(
        self, start_x: int, start_y: int, delta_x: int, delta_y: int
    ) -> Optional[MoveRecord]:
        new_x = start_x + delta_x
        new_y = start_y + delta_y
        if not is_legal_board_coordinate(new_x, new_y):
            return None

        contents = self.spaces[new_x][new_y]
        if contents == EMPTY_SPACE:
            return self.evaluate_move(start_x, start_y, new_x, new_y)
        if contents < 0:
            return None

        # check jump landing space, which is an additional dx, dy from new_x, newy
        landing_x = new_x + delta_x
        landing_y = new_y + delta_y

        if not is_legal_board_coordinate(landing_x, landing_y):
            return None
        if self.spaces[landing_x][landing_y] == EMPTY_SPACE:
            return self.evaluate_move(start_x, start_y, landing_x, landing_y)
        return None

    def evaluate_move(
        self, start_x: int, start_y: int, dest_x: int, dest_y: int
    ) -> MoveRecord:
        quality = 0
        if dest_y == 0 and self.spaces[start_x][start_y] == COMPUTER_PIECE:
            # promoting is good
            quality += 2
        if abs(dest_y - start_y) == 2:
            # jumps are good
            quality += 5
        if start_y == 7:
            # prefer to defend back row
            quality -= 2
        if dest_x in (0, 7):
            # moving to edge column
            quality += 1
        for delta_x in (-1, 1):
            if not is_legal_board_coordinate(dest_x + delta_x, dest_y - 1):
                continue

            if self.spaces[dest_x + delta_x][dest_y - 1] < 0:
                # moving into "shadow" of another computer piece
                quality += 1

            if not is_legal_board_coordinate(dest_x - delta_x, dest_y + 1):
                continue

            if (
                (self.spaces[dest_x + delta_x][dest_y - 1] > 0)
                and (self.spaces[dest_x - delta_x][dest_y + 1] == EMPTY_SPACE)
                or ((dest_x - delta_x == start_x) and (dest_y + 1 == start_y))
            ):
                # we are moving up to a human checker that could jump us
                quality -= 2
        return MoveRecord(quality, start_x, start_y, dest_x, dest_y)

    def remove_r_pieces(self, move_record: MoveRecord) -> None:
        self.remove_pieces(
            move_record.start_x,
            move_record.start_y,
            move_record.dest_x,
            move_record.dest_y,
        )

    def remove_pieces(
        self, start_x: int, start_y: int, dest_x: int, dest_y: int
    ) -> None:
        self.spaces[dest_x][dest_y] = self.spaces[start_x][start_y]
        self.spaces[start_x][start_y] = EMPTY_SPACE

        if abs(dest_x - start_x) == 2:
            mid_x = (start_x + dest_x) // 2
            mid_y = (start_y + dest_y) // 2
            self.spaces[mid_x][mid_y] = EMPTY_SPACE

    def play_computer_move(self, move_record: MoveRecord) -> None:
        print(
            f"FROM {move_record.start_x} {move_record.start_y} TO {move_record.dest_x} {move_record.dest_y}"
        )

        while True:
            if move_record.dest_y == BOTTOM_ROW:
                # KING ME
                self.remove_r_pieces(move_record)
                self.spaces[move_record.dest_x][move_record.dest_y] = COMPUTER_KING
                return
            else:
                self.spaces[move_record.dest_x][move_record.dest_y] = self.spaces[
                    move_record.start_x
                ][move_record.start_y]
                self.remove_r_pieces(move_record)

                if abs(move_record.dest_x - move_record.start_x) != 2:
                    return

                landing_x = move_record.dest_x
                landing_y = move_record.dest_y

                best_move = None
                if self.spaces[landing_x][landing_y] == COMPUTER_PIECE:
                    for delta_x in (-2, 2):
                        test_record = self.try_extend(landing_x, landing_y, delta_x, -2)
                        if (move_record is not None) and (
                            (best_move is None)
                            or (move_record.quality > best_move.quality)
                        ):
                            best_move = test_record
                else:
                    assert self.spaces[landing_x][landing_y] == COMPUTER_KING
                    for delta_x in (-2, 2):
                        for delta_y in (-2, 2):
                            test_record = self.try_extend(
                                landing_x, landing_y, delta_x, delta_y
                            )
                            if (move_record is not None) and (
                                (best_move is None)
                                or (move_record.quality > best_move.quality)
                            ):
                                best_move = test_record

                if best_move is None:
                    return
                else:
                    print(f"TO {best_move.dest_x} {best_move.dest_y}")
                    move_record = best_move

    def try_extend(
        self, start_x: int, start_y: int, delta_x: int, delta_y: int
    ) -> Optional[MoveRecord]:
        new_x = start_x + delta_x
        new_y = start_y + delta_y

        if not is_legal_board_coordinate(new_x, new_y):
            return None

        jumped_x = start_x + delta_x // 2
        jumped_y = start_y + delta_y // 2

        if (self.spaces[new_x][new_y] == EMPTY_SPACE) and (
            self.spaces[jumped_x][jumped_y] > 0
        ):
            return self.evaluate_move(start_x, start_y, new_x, new_y)
        return None

    def get_human_move(self) -> Tuple[int, int, int, int]:
        is_king = False

        while True:
            start_x, start_y = get_coordinates("FROM?")

            legal_moves = list(self.get_legal_moves(start_x, start_y))
            if not legal_moves:
                print(f"({start_x}, {start_y}) has no legal moves. Choose again.")
                continue
            if self.spaces[start_x][start_y] > 0:
                break

        is_king = self.spaces[start_x][start_y] == HUMAN_KING

        while True:
            dest_x, dest_y = get_coordinates("TO?")

            if (not is_king) and (dest_y < start_y):
                # CHEATER! Trying to move non-king backwards
                continue
            is_free = self.spaces[dest_x][dest_y] == 0
            within_reach = abs(dest_x - start_x) <= 2
            is_diagonal_move = abs(dest_x - start_x) == abs(dest_y - start_y)
            if is_free and within_reach and is_diagonal_move:
                break
        return start_x, start_y, dest_x, dest_y

    def get_human_extension(
        self, start_x: int, start_y: int
    ) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        is_king = self.spaces[start_x][start_y] == HUMAN_KING

        while True:
            dest_x, dest_y = get_coordinates("+TO?")

            if dest_x < 0:
                return False, None
            if (not is_king) and (dest_y < start_y):
                # CHEATER! Trying to move non-king backwards
                continue
            if (
                (self.spaces[dest_x][dest_y] == EMPTY_SPACE)
                and (abs(dest_x - start_x) == 2)
                and (abs(dest_y - start_y) == 2)
            ):
                return True, (start_x, start_y, dest_x, dest_y)

    def play_human_move(
        self, start_x: int, start_y: int, dest_x: int, dest_y: int
    ) -> None:
        self.remove_pieces(start_x, start_y, dest_x, dest_y)

        if dest_y == TOP_ROW:
            # KING ME
            self.spaces[dest_x][dest_y] = HUMAN_KING

    def check_pieces(self) -> bool:
        if len(list(self.get_spaces_with_computer_pieces())) == 0:
            print_human_won()
            return False
        if len(list(self.get_spaces_with_computer_pieces())) == 0:
            print_computer_won()
            return False
        return True


```

这两段代码定义了两个函数，分别是print_instructions()和print_human_won()。

print_instructions()函数的作用是输出游戏的规则，包括字符串"THIS IS THE GAME OF CHECKERS. THE COMPUTER IS X,"、"AND YOU ARE O. THE COMPUTER WILL MOVE FIRST."、"SQUARES ARE REFERRED TO BY A COORDINATE SYSTEM."、"(0,0) IS THE LOWER LEFT CORNER)"、"(0,7) IS THE UPPER LEFT CORNER)"、"(7,0) IS THE LOWER RIGHT CORNER)"和"(7,7) IS THE UPPER RIGHT CORNER)"，以及一个提示框，告知玩家计算机在输出的数字后，会提示玩家进行操作。最后，函数返回None，表示没有返回值。

print_human_won()函数的作用是输出"YOU WIN。"，表示游戏已经结束，并且你赢了。


```
def print_instructions() -> None:
    print("THIS IS THE GAME OF CHECKERS.  THE COMPUTER IS X,")
    print("AND YOU ARE O.  THE COMPUTER WILL MOVE FIRST.")
    print("SQUARES ARE REFERRED TO BY A COORDINATE SYSTEM.")
    print("(0,0) IS THE LOWER LEFT CORNER")
    print("(0,7) IS THE UPPER LEFT CORNER")
    print("(7,0) IS THE LOWER RIGHT CORNER")
    print("(7,7) IS THE UPPER RIGHT CORNER")
    print("THE COMPUTER WILL TYPE '+TO' WHEN YOU HAVE ANOTHER")
    print("JUMP.  TYPE TWO NEGATIVE NUMBERS IF YOU CANNOT JUMP.\n\n\n")


def print_human_won() -> None:
    print("\nYOU WIN.")


```

这两段代码定义了一个计算机会游戏，玩家可以与计算机轮流进行决策，计算机的决策基于最近2次操作人类玩家的结果。在游戏过程中，计算机会检测自己的棋子是否被对手抓住了，如果没有被抓住，计算机会继续尝试攻击人类的棋子。如果计算机的攻击与人类的攻击相同，则游戏继续进行，直到玩家之一赢得了游戏。


```
def print_computer_won() -> None:
    print("\nI WIN.")


def play_game() -> None:
    board = Board()

    while True:
        move_record = board.pick_computer_move()
        if move_record is None:
            print_human_won()
            return
        board.play_computer_move(move_record)

        print(board)

        if not board.check_pieces():
            return

        start_x, start_y, dest_x, dest_y = board.get_human_move()
        board.play_human_move(start_x, start_y, dest_x, dest_y)
        if abs(dest_x - start_x) == 2:
            while True:
                extend, move = board.get_human_extension(dest_x, dest_y)
                assert move is not None
                if not extend:
                    break
                start_x, start_y, dest_x, dest_y = move
                board.play_human_move(start_x, start_y, dest_x, dest_y)


```

这段代码是一个Python程序，名为“main”。程序的主要作用是让用户输入两个字符串，然后输出游戏结果。

具体来说，这段代码包含以下几个部分：

1. `def main() -> None:`是一个函数定义，表示程序的入口点。函数体内部没有定义任何参数和变量，因此这个函数可以被视为一个“无参函数”。

2. `print_header("CHECKERS")`是一个函数调用，会在程序开始时打印出“CHECKERS”这个字符串。这个函数的作用可能是输出一个消息，告诉用户正在使用的是哪个游戏。

3. `print_instructions()`是一个函数调用，也会在程序开始时打印出一些游戏规则的说明。这个函数的作用可能是输出一些游戏规则，告诉用户如何玩游戏。

4. `play_game()`是一个外部函数，可能是game.py文件中的函数，这个函数负责运行游戏。在这个函数中，程序会运行game.py文件中的代码，来玩这个游戏。

5. `if __name__ == "__main__":`是一个判断语句，用于检查当前正在运行的程序是否是一个独立的Python脚本。如果当前运行的程序是一个独立的Python脚本，那么就会执行main()函数中的代码。

6. `main()`是一个函数定义，表示当前程序的入口点。这个函数会调用`print_header()`、`print_instructions()`和`play_game()`这三个函数，然后等待用户输入两个字符串，最后调用`play_game()`函数来运行游戏。


```
def main() -> None:
    print_header("CHECKERS")
    print_instructions()

    play_game()


if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)

This version preserves the underlying algorithms and functionality of
the original while using more modern programming constructs
(functions, classes, symbols) and providing much more detailed
comments.  It also fixes some (but not all) of the bugs.


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Chemist

The fictitious chemical, kryptocyanic acid, can only be diluted by the ratio of 7 parts water to 3 parts acid. Any other ratio causes an unstable compound which soon explodes. Given an amount of acid, you must determine how much water to add to the dilution. If you’re more than 5% off, you lose one of your nine lives. The program continues to play until you lose all nine lives or until it is interrupted.

It was originally written by Wayne Teeter of Ridgecrest, California.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=42)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=57)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- There is a typo in the original Basic, "...DECIDE **WHO** MUCH WATER..." should be "DECIDE **HOW** MUCH WATER"

#### Porting Notes

(please note any difficulties or challenges in porting here)

#### External Links
 - C: https://github.com/ericfischer/basic-computer-games/blob/main/24%20Chemist/c/chemist.c


# `24_Chemist/csharp/Program.cs`

这段代码的主要目的是输出一段文字，其中包含了关于化学家、创意计算和摩尔斯信息的内容。然后，它将展示一个程序，这个程序将会向用户提问，询问是否想要改变酸的浓度。如果用户想要改变酸的浓度，它将提示用户必须按照什么比例加水，否则酸就会变得不稳定并爆炸。


```
﻿using System;
const int maxLives = 9;

WriteCentred("Chemist");
WriteCentred("Creative Computing, Morristown, New Jersey");
Console.WriteLine(@"


The fictitious chemical kryptocyanic acid can only be
diluted by the ratio of 7 parts water to 3 parts acid.
If any other ratio is attempted, the acid becomes unstable
and soon explodes.  Given the amount of acid, you must
decide who much water to add for dilution.  If you miss
you face the consequences.
");

```

这段代码使用了Java随机数生成器来生成一个0到49之间的随机整数，表示这箱子里的药水。然后，它计算出这些药水对人体的危害程度，并让用户输入这个危害程度。如果用户输入的值在2到7.5之间，那么它会告诉用户可以呼吸了，否则它会给出一个滑稽的警告，告诉用户它缺氧了。

在每次循环中，它会让用户输入一个药水危害程度，并根据这个值决定如何处理。如果用户输入的值在2到7.5之间，那么它会告诉用户可以呼吸了，否则它会给出一个滑稽的警告，告诉用户它缺氧了。如果用户已经尝试过呼吸，那么它会再次警告用户缺氧的危害，并将livesUsed加1，这可能有助于用户在之后的循环中更好地理解这个程序。


```
var random = new Random();
int livesUsed = 0;
while (livesUsed < maxLives)
{
    int krypto = random.Next(1, 50);
    double water = krypto * 7.0 / 3.0;

    Console.WriteLine($"{krypto} Liters of kryptocyanic acid.  How much water?");
    double answer = double.Parse(Console.ReadLine());

    double diff = Math.Abs(answer - water);
    if (diff <= water / 20)
    {
        Console.WriteLine("Good job! You may breathe now, but don't inhale the fumes"!);
        Console.WriteLine();
    }
    else
    {
        Console.WriteLine("Sizzle!  You have just been desalinated into a blob\nof quivering protoplasm!");
        Console.WriteLine();
        livesUsed++;

        if (livesUsed < maxLives)
            Console.WriteLine("However, you may try again with another life.");
    }
}
```

这段代码定义了一个名为 WriteCentered 的静态函数，其作用是输出一段文本，并将其置于字符串的中心位置。

具体来说，函数接收一个字符串参数 text，然后计算出该字符串在控制台窗口宽度加上该字符串长度后除以 2 的整数倍得到的字符数 indent，使用这个整数倍作为格式化字符串的宽度，并在字符串左侧填充字符并将其置于其中心位置。最后，函数使用 Console.WriteLine 方法将计算得到的字符串输出到控制台。

例如，如果调用 WriteCentered 函数并传入字符串 "Your"，则函数将输出 "Your  lives are used, but you will be long remembered for your contributions to the field of comic book chemistry。"，其中心位置为字符串的第一和第二个字符。


```
Console.WriteLine($"Your {maxLives} lives are used, but you will be long remembered for\nyour contributions to the field of comic book chemistry.");

static void WriteCentred(string text)
{
    int indent = (Console.WindowWidth + text.Length) / 2;
    Console.WriteLine($"{{0,{indent}}}", text);
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `24_Chemist/java/src/Chemist.java`

This is a Java program that simulates the appearance of a tab character in a string of text. The program accepts a user input of the number of spaces required for the tab character.

The program first defines a simulateTabs() method that accepts an integer representing the number of spaces and returns a new string with those spaces appended at the specified number of spaces.

The program then defines a displayTextAndGetNumber() method that accepts a string representing the text to be displayed on the screen. This method first displays the message on the screen and then accepts the input from the Keyboard. The method converts the input to an Integer and returns it.

The program also defines a displayTextAndGetInput() method that is similar to the displayTextAndGetNumber() method but accepts the text to be displayed on the screen and returns the input as a String.

Finally, the program uses a library called kbScanner to handle the Keyboard input.

Overall, this program simulates the appearance of a tab character in a string of text based on the number of spaces entered by the user.


```
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Chemist
 * <p>
 * Based on the Basic game of Chemist here
 * https://github.com/coding-horror/basic-computer-games/blob/main/24%20Chemist/chemist.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Chemist {

    public static final int MAX_LIVES = 9;

    // Used for keyboard input
    private final Scanner kbScanner;

    private enum GAME_STATE {
        START_GAME,
        INPUT,
        BLOWN_UP,
        SURVIVED,
        GAME_OVER
    }

    // Current game state
    private GAME_STATE gameState;

    private int timesBlownUp;

    public Chemist() {
        kbScanner = new Scanner(System.in);

        gameState = GAME_STATE.START_GAME;
    }

    /**
     * Main game loop
     */
    public void play() {

        do {
            switch (gameState) {

                case START_GAME:
                    intro();
                    timesBlownUp = 0;
                    gameState = GAME_STATE.INPUT;
                    break;

                case INPUT:

                    int amountOfAcid = (int) (Math.random() * 50);
                    int correctAmountOfWater = (7 * amountOfAcid) / 3;
                    int water = displayTextAndGetNumber(amountOfAcid + " LITERS OF KRYPTOCYANIC ACID.  HOW MUCH WATER? ");

                    // Calculate if the player mixed enough water
                    int result = Math.abs(correctAmountOfWater - water);

                    // Ratio of water wrong?
                    if (result > (correctAmountOfWater / 20)) {
                        gameState = GAME_STATE.BLOWN_UP;
                    } else {
                        // Got the ratio correct
                        gameState = GAME_STATE.SURVIVED;
                    }
                    break;

                case BLOWN_UP:
                    System.out.println(" SIZZLE!  YOU HAVE JUST BEEN DESALINATED INTO A BLOB");
                    System.out.println(" OF QUIVERING PROTOPLASM!");

                    timesBlownUp++;

                    if (timesBlownUp < MAX_LIVES) {
                        System.out.println(" HOWEVER, YOU MAY TRY AGAIN WITH ANOTHER LIFE.");
                        gameState = GAME_STATE.INPUT;
                    } else {
                        System.out.println(" YOUR " + MAX_LIVES + " LIVES ARE USED, BUT YOU WILL BE LONG REMEMBERED FOR");
                        System.out.println(" YOUR CONTRIBUTIONS TO THE FIELD OF COMIC BOOK CHEMISTRY.");
                        gameState = GAME_STATE.GAME_OVER;
                    }

                    break;

                case SURVIVED:
                    System.out.println(" GOOD JOB! YOU MAY BREATHE NOW, BUT DON'T INHALE THE FUMES!");
                    System.out.println();
                    gameState = GAME_STATE.INPUT;
                    break;

            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    private void intro() {
        System.out.println(simulateTabs(33) + "CHEMIST");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("THE FICTITIOUS CHEMICAL KRYPTOCYANIC ACID CAN ONLY BE");
        System.out.println("DILUTED BY THE RATIO OF 7 PARTS WATER TO 3 PARTS ACID.");
        System.out.println("IF ANY OTHER RATIO IS ATTEMPTED, THE ACID BECOMES UNSTABLE");
        System.out.println("AND SOON EXPLODES.  GIVEN THE AMOUNT OF ACID, YOU MUST");
        System.out.println("DECIDE WHO MUCH WATER TO ADD FOR DILUTION.  IF YOU MISS");
        System.out.println("YOU FACE THE CONSEQUENCES.");
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     * Converts input to an Integer
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private int displayTextAndGetNumber(String text) {
        return Integer.parseInt(displayTextAndGetInput(text));
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
     * Simulate the old basic tab(xx) command which indented text by xx spaces.
     *
     * @param spaces number of spaces required
     * @return String with number of spaces
     */
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }
}

```

# `24_Chemist/java/src/ChemistGame.java`

这段代码定义了一个名为ChemistGame的public类，其中包含一个名为main的静态方法，其参数为字符串数组args，表示程序启动时传递的命令行参数。在main方法中，创建了一个名为chemate的Chemist对象，然后调用chemate的play()方法。

ChemistGame是一个模拟化学试验的游戏，play()方法可能是用于让玩家进行一些操作，例如添加试剂、观察反应等等。但由于没有提供具体的实现，无法确定这段代码的实际作用。


```
public class ChemistGame {
    public static void main(String[] args) {
        Chemist chemist = new Chemist();
        chemist.play();
    }
}

```

# `24_Chemist/javascript/chemist.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是打印字符串，将字符串附加到文档中的一个元素上。这个字符串以 "CHEMIST" 为前缀，以换行符为分隔符。

`input` 函数的作用是从用户接收输入字符串，并将其存储在变量 `input_str` 中。该函数通过创建一个带有 `type="text"` 属性的 `INPUT` 元素，将输入的字符串附加到文档中的一个元素上，并设置该元素的 `length` 属性为 "50"。然后将该元素聚焦，并添加事件监听器 `keyup` 类型，以便在用户按下键时接收输入的字符串，并将其存储在 `input_str` 变量中。

最后，函数通过调用 `print` 函数将接收到的字符串输出，并在调用结束时返回它。


```
// CHEMIST
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

It looks like this is a program that simulates the chemical reaction of kryptocyanic acid (a type of acid) and water. The program uses a random number generator to simulate the reaction, and outputs the resulting acid concentration in liters. The program also includes a section on the risks associated with the acid, as well as a message inviting the user to add more water to the mixture if they find it to be too strong.



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
    print(tab(33) + "CHEMIST\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("THE FICTITIOUS CHECMICAL KRYPTOCYANIC ACID CAN ONLY BE\n");
    print("DILUTED BY THE RATIO OF 7 PARTS WATER TO 3 PARTS ACID.\n");
    print("IF ANY OTHER RATIO IS ATTEMPTED, THE ACID BECOMES UNSTABLE\n");
    print("AND SOON EXPLODES.  GIVEN THE AMOUNT OF ACID, YOU MUST\n");
    print("DECIDE WHO MUCH WATER TO ADD FOR DILUTION.  IF YOU MISS\n");
    print("YOU FACE THE CONSEQUENCES.\n");
    t = 0;
    while (1) {
        a = Math.floor(Math.random() * 50);
        w = 7 * a / 3;
        print(a + " LITERS OF KRYPTOCYANIC ACID.  HOW MUCH WATER");
        r = parseFloat(await input());
        d = Math.abs(w - r);
        if (d > w / 20) {
            print(" SIZZLE!  YOU HAVE JUST BEEN DESALINATED INTO A BLOB\n");
            print(" OF QUIVERING PROTOPLASM!\n");
            t++;
            if (t == 9)
                break;
            print(" HOWEVER, YOU MAY TRY AGAIN WITH ANOTHER LIFE.\n");
        } else {
            print(" GOOD JOB! YOU MAY BREATHE NOW, BUT DON'T INHALE THE FUMES!\n");
            print("\n");
        }
    }
    print(" YOUR 9 LIVES ARE USED, BUT YOU WILL BE LONG REMEMBERED FOR\n");
    print(" YOUR CONTRIBUTIONS TO THE FIELD OF COMIC BOOK CHEMISTRY.\n");
}

```

这是C++中的一个标准的main函数，其作用是程序的入口点。在main函数中，程序会开始执行，从命令行开始读取输入，并逐行执行程序体中的代码。

对于这段代码而言，它只是一个简单的程序，它包含了一个main函数，但是没有具体的函数体。这意味着程序不会输出任何结果，因为它没有进行任何计算或操作。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `24_Chemist/python/chemist.py`

这段代码是一个简单的 Python 程序，它演示了一个有趣的数学游戏。在这个游戏中，玩家需要在一个化学实验室中合成一种名为“Kryptocyanic Acid”的化学品。为了完成这个任务，玩家需要使用一种名为“acid”的化学品和一种名为“water”的化学品。

在游戏的开始，程序会生成一种名为“Kryptocyanic Acid”的化学品，并且需要玩家提供一个足够的水来配制这种化学品。然后，程序会问玩家想要生成多少“Kryptocyanic Acid”化学品，并告知玩家需要多少水来配制这种化学品。

接下来，程序会随机生成一个数字，表示玩家需要使用多少“acid”化学品来配制“Kryptocyanic Acid”。然后，程序会提示玩家输入想要生成多少“Kryptocyanic Acid”化学品，以及需要多少水来配制这种化学品。

如果玩家输入的数量不正确，或者生成的“Kryptocyanic Acid”化学品数量不符合要求，那么程序会显示一个错误消息，并提示玩家重新尝试。如果玩家正确地生成了一种“Kryptocyanic Acid”化学品，并且使用了足够的水来配制这种化学品，那么程序会显示一个成功消息，并提示玩家可以继续进行下一步游戏。


```
"""
CHEMIST

A math game posing as a chemistry word problem.

Ported by Dave LeCompte
"""

import random

MAX_LIVES = 9


def play_scenario() -> bool:
    acid_amount = random.randint(1, 50)

    water_amount = 7 * acid_amount / 3

    print(f"{acid_amount} LITERS OF KRYPTOCYANIC ACID.  HOW MUCH WATER?")

    response = float(input())

    difference = abs(water_amount - response)

    acceptable_difference = water_amount / 20

    if difference > acceptable_difference:
        show_failure()

        return False
    else:
        show_success()

        return True


```

这段代码定义了三个函数，分别是show_failure、show_success和show_ending。这三个函数的作用是打印一些信息。

show_failure函数打印出了一个警告信息，说明创建者已经被剥离并且给予了过多关注。

show_success函数打印出了一个成功信息，告诉读者他们现在可以呼吸了，但是不要吸入烟雾。

show_ending函数打印出了一个信息，告诉读者他们的寿命都已经被用完了，但是他们为化学领域做出的贡献将会被铭记在历史中。


```
def show_failure() -> None:
    print(" SIZZLE!  YOU HAVE JUST BEEN DESALINATED INTO A BLOB")
    print(" OF QUIVERING PROTOPLASM!")


def show_success() -> None:
    print(" GOOD JOB! YOU MAY BREATHE NOW, BUT DON'T INHALE THE FUMES!\n")


def show_ending() -> None:
    print(f" YOUR {MAX_LIVES} LIVES ARE USED, BUT YOU WILL BE LONG REMEMBERED FOR")
    print(" YOUR CONTRIBUTIONS TO THE FIELD OF COMIC BOOK CHEMISTRY.")


def main() -> None:
    print(" " * 33 + "CHEMIST")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

    print("THE FICTITIOUS CHEMICAL KRYPTOCYANIC ACID CAN ONLY BE")
    print("DILUTED BY THE RATIO OF 7 PARTS WATER TO 3 PARTS ACID.")
    print("IF ANY OTHER RATIO IS ATTEMPTED, THE ACID BECOMES UNSTABLE")
    print("AND SOON EXPLODES.  GIVEN THE AMOUNT OF ACID, YOU MUST")
    print("DECIDE WHO MUCH WATER TO ADD FOR DILUTION.  IF YOU MISS")
    print("YOU FACE THE CONSEQUENCES.")

    lives_used = 0

    while True:
        success = play_scenario()

        if not success:
            lives_used += 1

            if lives_used == MAX_LIVES:
                show_ending()
                return


```

这段代码是一个Python程序中的一个if语句，其作用是：

如果当前脚本被作为主程序(即，脚本被运行时被视为命令行脚本而不是交互式脚本)运行，那么程序将跳转到该if语句的末尾执行main函数。

具体来说，当脚本被运行时，Python解释器检查是否执行了`__main__`目录下的脚本。如果是，则执行该目录下的`main`函数，如果是其他目录的脚本，则跳转到当前目录并执行`main`函数。

因此，该代码的作用是用于确保脚本在作为主程序运行时能够正常工作，即使它没有被直接命名为`__main__`目录下的脚本。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)
