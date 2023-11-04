# BasicComputerGames源码解析 55

# `55_Life/javascript/life.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

1. `print` 函数的作用是在页面上输出一个字符串，当用户点击页面上的元素时，这个函数将接收该元素的价格并将其打印到页面上。

2. `input` 函数的作用是从用户那里获取一个字符串，它会提示用户输入字符串，然后将该字符串打印到页面上。它还监听用户按键，当用户按下键盘上的 13 时，它将捕获到用户输入的字符串，打印到页面上并将其从元素中删除。


```
// LIFE
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

This appears to be a Python implementation of a simple text editor. It is written in a mix of Python and some other language that it uses for certain parts of the code, such as the f-strings. The program appears to be parsing a text file and replacing certain text with different text based on the rules specified in the f-strings. It also appears to be implementing a CSS class system, with different styles being applied based on the class name and value.

The main functionality of the program is reading a text file and replacing certain text with other text based on the rules specified in the f-strings.


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var bs = [];
var a = [];

// Main program
async function main()
{
    print(tab(34) + "LIFE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("ENTER YOUR PATTERN:\n");
    x1 = 1;
    y1 = 1;
    x2 = 24;
    y2 = 70;
    for (c = 1; c <= 24; c++) {
        bs[c] = "";
        a[c] = [];
        for (d = 1; d <= 70; d++)
            a[c][d] = 0;
    }
    c = 1;
    while (1) {
        bs[c] = await input();
        if (bs[c] == "DONE") {
            bs[c] = "";
            break;
        }
        if (bs[c].substr(0, 1) == ".")
            bs[c] = " " + bs[c].substr(1);
        c++;
    }
    c--;
    l = 0;
    for (x = 1; x <= c - 1; x++) {
        if (bs[x].length > l)
            l = bs[x].length;
    }
    x1 = 11 - (c >> 1);
    y1 = 33 - (l >> 1);
    p = 0;
    for (x = 1; x <= c; x++) {
        for (y = 1; y <= bs[x].length; y++) {
            if (bs[x][y - 1] != " ") {
                a[x1 + x][y1 + y] = 1;
                p++;
            }
        }
    }
    print("\n");
    print("\n");
    print("\n");
    i9 = false;
    g = 0;
    while (g < 100) {
        print("GENERATION: " + g + " POPULATION: " + p + " ");
        if (i9)
            print("INVALID!");
        x3 = 24;
        y3 = 70;
        x4 = 1;
        y4 = 1;
        p = 0;
        g++;
        for (x = 1; x <= x1 - 1; x++)
            print("\n");
        for (x = x1; x <= x2; x++) {
            print("\n");
            str = "";
            for (y = y1; y <= y2; y++) {
                if (a[x][y] == 2) {
                    a[x][y] = 0;
                    continue;
                } else if (a[x][y] == 3) {
                    a[x][y] = 1;
                } else if (a[x][y] != 1) {
                    continue;
                }
                while (str.length < y)
                    str += " ";
                str += "*";
                if (x < x3)
                    x3 = x;
                if (x > x4)
                    x4 = x;
                if (y < y3)
                    y3 = y;
                if (y > y4)
                    y4 = y;
            }
            print(str);
        }
        for (x = x2 + 1; x <= 24; x++)
            print("\n");
        x1 = x3;
        x2 = x4;
        y1 = y3;
        y2 = y4;
        if (x1 < 3) {
            x1 = 3;
            i9 = true;
        }
        if (x2 > 22) {
            x2 = 22;
            i9 = true;
        }
        if (y1 < 3) {
            y1 = 3;
            i9 = true;
        }
        if (y2 > 68) {
            y2 = 68;
            i9 = true;
        }
        p = 0;
        for (x = x1 - 1; x <= x2 + 1; x++) {
            for (y = y1 - 1; y <= y2 + 1; y++) {
                c = 0;
                for (i = x - 1; i <= x + 1; i++) {
                    for (j = y - 1; j <= y + 1; j++) {
                        if (a[i][j] == 1 || a[i][j] == 2)
                            c++;
                    }
                }
                if (a[x][y] == 0) {
                    if (c == 3) {
                        a[x][y] = 3;
                        p++;
                    }
                } else {
                    if (c < 3 || c > 4) {
                        a[x][y] = 2;
                    } else {
                        p++;
                    }
                }
            }
        }
        x1--;
        y1--;
        x2++;
        y2++;
    }
}

```

这道题目缺少上下文，无法给出具体的解释。通常来说，在编程中，`main()` 函数是程序的入口点，程序从此处开始执行。它的作用是启动程序，告诉操作系统程序需要开始执行哪些代码。`main()` 函数可以是程序中的任何部分，但通常用来包含程序的主要函数。


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


# `55_Life/python/life.py`

这段代码是一个名为"LIFE"的Python类，它实现了著名的康威生命游戏(也称为"Conway's Game of Life")。这个游戏的特点是，在一个网格中，每个细胞在每一代中根据其周围细胞的状态决定自己的状态，状态只有两种，即生或死。

具体来说，这段代码定义了一个名为"Cell"的类，每个实例的属性包括一个表示细胞状态的二维数组(也就是网格)，以及一个表示该网格的宽度和高度的变量。

接着，代码定义了一个名为"PAGE_WIDTH"和"MAX_WIDTH"的变量，分别表示页面在横向和纵向的最大宽度。然后，代码定义了一个名为"MAX_HEIGHT"的变量，表示纵向的最大高度。

最后，代码通过以下几行代码创建了一个包含256个网格的二维列表，这些网格的初始状态都是已知的。具体来说，这些网格按照行和列的顺序初始化为"alive"，也就是生状态。

通过调用"Life.create_grid()"方法，可以创建一个和上面描述相同的网格。而通过调用"Life.run()"方法，则可以模拟游戏20代的运行过程，并输出每个网格的最终状态。


```
"""
LIFE

An implementation of John Conway's popular cellular automaton

Ported by Dave LeCompte
"""

from typing import Dict

PAGE_WIDTH = 64

MAX_WIDTH = 70
MAX_HEIGHT = 24


```

这段代码定义了两个函数，分别是 `print_centered` 和 `print_header`。这两个函数的功能是打印特定的字符串，并将其置于文本中心。

`print_centered` 函数接受一个字符串参数 `msg`，并在字符串两侧填充字符 ` " "`，使其总宽度填满 `PAGE_WIDTH` 除以 2 的整数倍。最后，函数先打印字符 ` " "`，然后将其与 `msg` 混合，打印出来。

`print_header` 函数接受一个字符串参数 `title`，并打印一个包含头部的字符串。它通过调用 `print_centered` 函数来确保在打印字符串时，字符串两侧填充字符 ` " "`，并将其置于文本中心。然后，函数在打印结果中添加了一些额外的字符，以及换行符 `\n`，使得输出更易于阅读。

这两个函数的实现主要依赖于 `get_pattern` 函数，这个函数接受一个整数类型参数，并要求用户输入模式。它将输入的字符串存储在一个字典中，并返回该字典。函数在每一次调用时，它会提示用户输入模式，并将其存储在字典中。之后，在 `print_centered` 和 `print_header` 函数中，我们可以通过调用 `get_pattern` 函数来获取用户输入的模式，并将其存储在字典中，以便在需要时动态地应用模式。


```
def print_centered(msg) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)


def print_header(title) -> None:
    print_centered(title)
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print()
    print()
    print()


def get_pattern() -> Dict[int, str]:
    print("ENTER YOUR PATTERN:")
    c = 0

    pattern: Dict[int, str] = {}
    while True:
        line = input()
        if line == "DONE":
            return pattern

        # BASIC input would strip of leading whitespace.
        # Python input does not. The following allows you to start a
        # line with a dot to disable the whitespace stripping. This is
        # unnecessary for Python, but for historical accuracy, it's
        # staying in.

        if line[0] == ".":
            line = " " + line[1:]
        pattern[c] = line
        c += 1


```

It seems like you're trying to implement a solution for a problem that involves updating a 2D list `a` based on certain rules, and then printing the resulting list. However, the problem description and the solution you're trying to implement are not matching.

The problem description you provided is actually a description of a different problem, which involves finding the next largest or smallest value in a list of values. The solution you've provided involves updating the value of variable `next_max_x` and `next_min_y` based on the min and max values of the list, which is not the same as finding the next largest or smallest value.

If you're still having trouble understanding the problem, please let me know, and I'll do my best to help you.


```
def main() -> None:
    print_header("LIFE")

    pattern = get_pattern()

    pattern_height = len(pattern)
    pattern_width = 0
    for _line_num, line in pattern.items():
        pattern_width = max(pattern_width, len(line))

    min_x = 11 - pattern_height // 2
    min_y = 33 - pattern_width // 2
    max_x = MAX_HEIGHT - 1
    max_y = MAX_WIDTH - 1

    a = [[0 for y in range(MAX_WIDTH)] for x in range(MAX_HEIGHT)]
    p = 0
    g = 0
    invalid = False

    # line 140
    # transcribe the input pattern into the active array
    for x in range(0, pattern_height):
        for y in range(0, len(pattern[x])):
            if pattern[x][y] != " ":
                a[min_x + x][min_y + y] = 1
                p += 1

    print()
    print()
    print()
    while True:
        if invalid:
            inv_str = "INVALID!"
        else:
            inv_str = ""

        print(f"GENERATION: {g}\tPOPULATION: {p} {inv_str}")

        next_min_x = MAX_HEIGHT - 1
        next_min_y = MAX_WIDTH - 1
        next_max_x = 0
        next_max_y = 0

        p = 0
        g += 1
        for _ in range(min_x):
            print()

        for x in range(min_x, max_x + 1):
            print()
            line_list = [" "] * MAX_WIDTH
            for y in range(min_y, max_y + 1):
                if a[x][y] == 2:
                    a[x][y] = 0
                    continue
                elif a[x][y] == 3:
                    a[x][y] = 1
                elif a[x][y] != 1:
                    continue

                line_list[y] = "*"

                next_min_x = min(x, next_min_x)
                next_max_x = max(x, next_max_x)
                next_min_y = min(y, next_min_y)
                next_max_y = max(y, next_max_y)

            print("".join(line_list))

        # line 295
        for _ in range(max_x + 1, MAX_HEIGHT):
            print()

        print()

        min_x = next_min_x
        max_x = next_max_x
        min_y = next_min_y
        max_y = next_max_y

        if min_x < 3:
            min_x = 3
            invalid = True
        if max_x > 22:
            max_x = 22
            invalid = True
        if min_y < 3:
            min_y = 3
            invalid = True
        if max_y > 68:
            max_y = 68
            invalid = True

        # line 309
        p = 0

        for x in range(min_x - 1, max_x + 2):
            for y in range(min_y - 1, max_y + 2):
                count = 0
                for i in range(x - 1, x + 2):
                    for j in range(y - 1, y + 2):
                        if a[i][j] == 1 or a[i][j] == 2:
                            count += 1
                if a[x][y] == 0:
                    if count == 3:
                        a[x][y] = 3
                        p += 1
                elif (count < 3) or (count > 4):
                    a[x][y] = 2
                else:
                    p += 1

        # line 635
        min_x = min_x - 1
        min_y = min_y - 1
        max_x = max_x + 1
        max_y = max_y + 1


```

这段代码是一个Python程序中的一个if语句。if语句可以用来进行条件判断，判断的条件是在程序运行时是否遇到了当前文件名。如果程序运行时遇到了当前文件名，则执行if语句内部的代码，否则跳过if语句，继续执行if语句外面的代码。

在这个例子中，if __name__ == "__main__": 是一个特殊的if语句，它会在程序运行时判断当前文件是否为程序的主文件(main file)，如果是，则执行if语句内部的代码，否则跳过if语句，继续执行if语句外面的代码。在这个例子中，if __name__ == "__main__": 的判断条件是当前文件是否为程序的主文件，如果不是，则程序不会执行if语句内部的代码，直接跳过。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


# Conway's Life

Original from David Ahl's _Basic Computer Games_, downloaded from http://www.vintage-basic.net/games.html.

Ported to Rust by Jon Fetter-Degges

Developed and tested on Rust 1.64.0

## How to Run

Install Rust using the instructions at [rust-lang.org](https://www.rust-lang.org/tools/install).

At a command or shell prompt in the `rust` subdirectory, enter `cargo run`.

## Differences from Original Behavior

* The simulation stops if all cells die.
* `.` at the beginning of an input line is supported but optional.
* Input of more than 66 columns is rejected. Input will automatically terminate after 20 rows. Beyond these bounds, the original
implementation would have marked the board as invalid, and beyond 68 cols/24 rows it would have had an out of bounds array access.
* The check for the string "DONE" at the end of input is case-independent.
* The program pauses for half a second between each generation.


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Life for Two

LIFE-2 is based on Conway’s game of Life. You must be familiar with the rules of LIFE before attempting to play LIFE-2.

There are two players; the game is played on a 5x5 board and each player has a symbol to represent his own pieces of ‘life.’ Live cells belonging to player 1 are represented by `*` and live cells belonging to player 2 are represented by the symbol `#`.

The # and * are regarded as the same except when deciding whether to generate a live cell. An empty cell having two `#` and one `*` for neighbors will generate a `#`, i.e. the live cell generated belongs to the player who has the majority of the 3 live cells surrounding the empty cell where life is to be generated, for example:

```
|   | 1 | 2 | 3 | 4 | 5 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 |   |   |   |   |   |
| 2 |   |   | * |   |   |
| 3 |   |   |   | # |   |
| 4 |   |   | # |   |   |
| 5 |   |   |   |   |   |
```

A new cell will be generated at (3,3) which will be a `#` since there are two `#` and one `*` surrounding. The board will then become:
```
|   | 1 | 2 | 3 | 4 | 5 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 |   |   |   |   |   |
| 2 |   |   |   |   |   |
| 3 |   |   | # | # |   |
| 4 |   |   |   |   |   |
| 5 |   |   |   |   |   |
```
On the first move each player positions 3 pieces of life on the board by typing in the co-ordinates of the pieces. (In the event of the same cell being chosen by both players that cell is left empty.)

The board is then adjusted to the next generation and printed out.

On each subsequent turn each player places one piece on the board, the object being to annihilate his opponent’s pieces. The board is adjusted for the next generation and printed out after both players have entered their new piece.

The game continues until one player has no more live pieces. The computer will then print out the board and declare the winner.

The idea for this game, the game itself, and the above write-up were written by Brian Wyvill of Bradford University in Yorkshire, England.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=102)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=117)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html


#### Porting Notes

(please note any difficulties or challenges in porting here)

Note: The original program has a bug. The instructions say that if both players
enter the same cell that the cell is set to 0 or empty. However, the original
Basic program tells the player "ILLEGAL COORDINATES" and makes another cell be entered,
giving a slightly unfair advantage to the 2nd player.

The Perl verson of the program fixes the bug and follows the instructions.

Note: The original code had "GOTO 800" but label 800 didn't exist; it should have gone to label 999.
The Basic program has been fixed.

Note: The Basic program is written to assume it's being played on a Teletype, i.e. output is printed
on paper. To play on a terminal the input must not be echoed, which can be a challenge to do portably
and without tying the solution to a specific OS. Some versions may tell you how to do this, others might not.


# `56_Life_for_Two/csharp/Board.cs`

This is a class that represents a 6x6 game board. It has a property `_cells` that is a 2D array, representing the cells of the board, and a property `_cellCounts` that is a count of how many cells of each type there are in the `_cells` array. It has methods `ClearCell`, `AddPlayer1Piece`, `AddPlayer2Piece`, `GetCellDisplay`, `GetEnumerator`, `IsEmptyAt`, `ClearCell`, `AddPlayer1Piece`, `AddPlayer2Piece`, `ToString`, `GetCellDisplay`, `IsEmptyAt`, `ClearCell`, `AddPlayer1Piece`, `AddPlayer2Piece`, `GetEnumerator`, `IsEmptyAt`, `ClearCell`, `AddPlayer1Piece`, `AddPlayer2Piece`, `ToString`, `GetCellDisplay`, `IsEmptyAt`, `ClearCell`, `AddPlayer1Piece`, `AddPlayer2Piece`, `GetEnumerator`, `IsEmptyAt`, `ClearCell`, `AddPlayer1Piece`, `AddPlayer2Piece`, `GetEnumerator`, `IsEmptyAt`, `ClearCell`, `AddPlayer1Piece`, `AddPlayer2Piece`, `GetEnumerator`, `IsEmptyAt`, `ClearCell`, `AddPlayer1Piece`, `AddPlayer2Piece`, `GetEnumerator`, `IsEmptyAt`, `ClearCell`, `AddPlayer1Piece`, `AddPlayer2Piece`, `GetEnumerator`, `IsEmptyAt`, `ClearCell`, `AddPlayer1Piece`, `AddPlayer2Piece`, `GetEnumerator`, `IsEmptyAt`, `ClearCell`, `GetCellDisplay`, `GetEnumerator`, `IsEmptyAt`, `ClearCell`, `GetCellDisplay`, `GetEnumerator`, `IsEmptyAt`, `ClearCell`, `GetCellDisplay`, `GetEnumerator`, `IsEmptyAt`, `ClearCell`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt`, `GetCellDisplay`, `IsEmptyAt


```
using System.Collections;
using System.Text;

namespace LifeforTwo;

internal class Board : IEnumerable<Coordinates>
{
    private readonly Piece[,] _cells = new Piece[7, 7];
    private readonly Dictionary<int, int> _cellCounts = 
        new() { [Piece.None] = 0, [Piece.Player1] = 0, [Piece.Player2] = 0 };

    public Piece this[Coordinates coordinates]
    {
        get => this[coordinates.X, coordinates.Y];
        set => this[coordinates.X, coordinates.Y] = value;
    }

    private Piece this[int x, int y]
    {
        get => _cells[x, y];
        set
        {
            if (!_cells[x, y].IsEmpty) { _cellCounts[_cells[x, y]] -= 1; }
            _cells[x, y] = value;
            _cellCounts[value] += 1;
        }
    }

    public int Player1Count => _cellCounts[Piece.Player1];
    public int Player2Count => _cellCounts[Piece.Player2];

    internal bool IsEmptyAt(Coordinates coordinates) => this[coordinates].IsEmpty;

    internal void ClearCell(Coordinates coordinates) => this[coordinates] = Piece.NewNone();
    internal void AddPlayer1Piece(Coordinates coordinates) => this[coordinates] = Piece.NewPlayer1();
    internal void AddPlayer2Piece(Coordinates coordinates) => this[coordinates] = Piece.NewPlayer2();

    public override string ToString()
    {
        var builder = new StringBuilder();

        for (var y = 0; y <= 6; y++)
        {
            builder.AppendLine();
            for (var x = 0; x <= 6; x++)
            {
                builder.Append(GetCellDisplay(x, y));
            }
        }

        return builder.ToString();
    }

    private string GetCellDisplay(int x, int y) =>
        (x, y) switch
        {
            (0 or 6, _) => $" {y % 6} ",
            (_, 0 or 6) => $" {x % 6} ",
            _ => $" {this[x, y]} "
        };

    public IEnumerator<Coordinates> GetEnumerator()
    {
        for (var x = 1; x <= 5; x++)
        {
            for (var y = 1; y <= 5; y++)
            {
                yield return new(x, y);
            }
        }
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}

```

# `56_Life_for_Two/csharp/Coordinates.cs`

这段代码定义了一个名为LifeforTwo的命名空间，其中包含一个名为Coordinates的内部记录类型，该类型记录了两个整数类型的值，以及一个名为operator+的算术重载类型，用于将两个 Coordinates 对象相加。

函数TryCreate尝试创建一个名为(X, Y)的 floating-point 数对，如果它们小于零或大于五，则返回false，否则创建一个名为(0, 0)的 Coordinates 对象并返回true。

函数operator+重载了两个 Coordinates 对象，将它们相加并返回一个新的 Coordinates 对象。

函数GetNeighbors返回一个异步迭代器，它尝试创建一个邻居的 Coordinates 对象，并在创建失败时返回。


```
namespace LifeforTwo;

internal record Coordinates (int X, int Y)
{
    public static bool TryCreate((float X, float Y) values, out Coordinates coordinates)
    {
        if (values.X <= 0 || values.X > 5 || values.Y <= 0 || values.Y > 5)
        {
            coordinates = new(0, 0);
            return false;
        }

        coordinates = new((int)values.X, (int)values.Y);
        return true;
    }

    public static Coordinates operator +(Coordinates coordinates, int value) =>
        new (coordinates.X + value, coordinates.Y + value);

    public IEnumerable<Coordinates> GetNeighbors()
    {
        yield return new(X - 1, Y);
        yield return new(X + 1, Y);
        yield return new(X, Y - 1);
        yield return new(X, Y + 1);
        yield return new(X - 1, Y - 1);
        yield return new(X + 1, Y - 1);
        yield return new(X - 1, Y + 1);
        yield return new(X + 1, Y + 1);
    }
}

```

# `56_Life_for_Two/csharp/Game.cs`



这是一段使用Java编写的代码，定义了一个名为`Game`的内部类，包含一个私有字段`IReadWrite`和一个构造函数。构造函数接收一个`IReadWrite`参数，并将其赋值给`_io`字段。

该类有一个名为`Play`的公共方法，该方法首先输出游戏的标题，然后创建一个名为`Life`的类，该类包含一个私有字段`FirstGeneration`，以及一个循环，该循环输出每个`Life`对象的`FirstGeneration`值。在循环内部，游戏还输出每个`Life`对象。最后，游戏输出`Life`对象的`Result`值（如果有的话），否则输出"No result"。

总之，该代码创建了一个`Game`类，该类具有一个`Play`方法，用于在游戏开始时输出标题，然后输出`Life`对象的`FirstGeneration`值，并在循环中输出每个`Life`对象的值。


```
internal class Game
{
    private readonly IReadWrite _io;

    public Game(IReadWrite io)
    {
        _io = io;
    }

    public void Play()
    {
        _io.Write(Streams.Title);

        var life = new Life(_io);

        _io.Write(life.FirstGeneration);

        foreach (var generation in life)
        {
            _io.WriteLine();
            _io.Write(generation);
        }

        _io.WriteLine(life.Result ?? "No result");
    }
}

```

# `56_Life_for_Two/csharp/Generation.cs`

This is a implementation of the board game in C#. It uses the `Board` class to store the game board, `Piece` class to store each piece on the board, `io` class to read and write to the console, `Streams` class to read and write to the console, `Coordinates` class to store the coordinates of each piece.

The `SetInitialPieces` method sets the initial pieces on the game board.

The `CalculateNextGeneration` method generates the next generation of the game board and returns it.

The `AddPieces` method adds pieces to the game board. It reads coordinates from the console, adds the piece to its coordinates, and writes the coordinate back to the console.

The `CountNeighbours` method counts the number of neighbors for each piece on the game board.

The `ToString` method returns the string representation of the game board.

It should be noted that this implementation uses a hardcoded board size and does not include any sort of randomness or AI which could make it more interesting.


```
internal class Generation
{
    private readonly Board _board;

    public Generation(Board board)
    {
        _board = board;
        CountNeighbours();
    }

    public Board Board => _board;

    public int Player1Count => _board.Player1Count;
    public int Player2Count => _board.Player2Count;

    public string? Result => 
        (Player1Count, Player2Count) switch
        {
            (0, 0) => Strings.Draw,
            (_, 0) => string.Format(Formats.Winner, 1),
            (0, _) => string.Format(Formats.Winner, 2),
            _ => null
        };

    public static Generation Create(IReadWrite io)
    {
        var board = new Board();

        SetInitialPieces(1, coord => board.AddPlayer1Piece(coord));
        SetInitialPieces(2, coord => board.AddPlayer2Piece(coord));

        return new Generation(board);

        void SetInitialPieces(int player, Action<Coordinates> setPiece)
        {
            io.WriteLine(Formats.InitialPieces, player);
            for (var i = 1; i <= 3; i++)
            {
                setPiece(io.ReadCoordinates(board));
            }
        }
    }

    public Generation CalculateNextGeneration()
    {
        var board = new Board();

        foreach (var coordinates in _board)
        {
            board[coordinates] = _board[coordinates].GetNext();
        }

        return new(board);
    }
    
    public void AddPieces(IReadWrite io)
    {
        var player1Coordinate = io.ReadCoordinates(1, _board);
        var player2Coordinate = io.ReadCoordinates(2, _board);

        if (player1Coordinate == player2Coordinate)
        {
            io.Write(Streams.SameCoords);
            // This is a bug existing in the original code. The line should be _board[_coordinates[_player]] = 0;
            _board.ClearCell(player1Coordinate + 1);
        }
        else
        {
            _board.AddPlayer1Piece(player1Coordinate);
            _board.AddPlayer2Piece(player2Coordinate);
        }
    }

    private void CountNeighbours()
    {
        foreach (var coordinates in _board)
        {
            var piece = _board[coordinates];
            if (piece.IsEmpty) { continue; }

            foreach (var neighbour in coordinates.GetNeighbors())
            {
                _board[neighbour] = _board[neighbour].AddNeighbour(piece);
            }
        }
    }

    public override string ToString() => _board.ToString();
}
```

# `56_Life_for_Two/csharp/IOExtensions.cs`

这段代码是一个名为 IOExtensions 的内部类，其作用是扩展了 IO 扩展函数的支持，以方便在游戏中读取和操作坐标。

ReadCoordinates 函数接受一个 IReadWrite 类型的 io 参数，以及一个整数类型的参数 board，并返回一个 Coordinates 类型的变量。这个函数的作用是读取玩家在游戏中的位置，并返回该位置的坐标。在函数中，首先将玩家信息输出到屏幕上，然后从 io 中读取一个坐标值，接着判断该坐标值是否已存在于坐标中，如果已存在，则返回该坐标值，否则继续尝试。

另一个名为 ReadCoordinates 的函数，同样接受一个 IReadWrite 类型的 io 参数和一个整数类型的参数 board，但返回类型是 Coordinates 类型。这个函数的作用是读取 board 上一个空白位置的坐标，并在尝试获取该位置坐标时，如果当前位置已经被占有，则返回该位置的坐标。


```
internal static class IOExtensions
{
    internal static Coordinates ReadCoordinates(this IReadWrite io, int player, Board board)
    {
        io.Write(Formats.Player, player);
        return io.ReadCoordinates(board);
    }

    internal static Coordinates ReadCoordinates(this IReadWrite io, Board board)
    {
        while (true)
        {
            io.WriteLine("X,Y");
            var values = io.Read2Numbers("&&&&&&\r");
            if (Coordinates.TryCreate(values, out var coordinates) && board.IsEmptyAt(coordinates))
            {
                return coordinates;
            }
            io.Write(Streams.IllegalCoords);
        }
    }
}
```

# `56_Life_for_Two/csharp/Life.cs`

这段代码定义了一个名为Life的内部类，该类实现了为名为Generation的接口添加内容的功能。Life内部包含一个IReadWrite类型的成员变量_io，一个名为FirstGeneration的内部类成员变量，一个名为Result的私有成员变量，以及一个名为GetEnumerator的内部方法。

FirstGeneration是一个内部类，该类实现了IGeneration的接口，该接口定义了一个Generation类型的成员函数CalculateNextGeneration，以及一个名为AddPieces的内部函数，该函数将指定数量和新的一块内容添加到指定位置。

在构造函数中，使用FirstGeneration的CalculateNextGeneration函数从文件中读取数据，并将其存储在_io中，然后设置FirstGeneration为当前生命周期。

在GetEnumerator方法中，首先创建并返回FirstGeneration实例，然后使用该实例的CalculateNextGeneration函数获取下一个生命周期，并将结果添加到当前生命周期中。然后，继续递归执行该过程，直到当前生命周期中的结果为null。

此外，在FirstGeneration中，还实现了IGeneration的接口，该接口定义了一个名为AddPieces的函数，该函数将指定数量和新的一块内容添加到指定位置。


```
using System.Collections;

internal class Life : IEnumerable<Generation>
{
    private readonly IReadWrite _io;

    public Life(IReadWrite io)
    {
        _io = io;
        FirstGeneration = Generation.Create(io);
    }

    public Generation FirstGeneration { get; }
    public string? Result { get; private set; }
    
    public IEnumerator<Generation> GetEnumerator()
    {
        var current = FirstGeneration;
        while (current.Result is null)
        {
            current = current.CalculateNextGeneration();
            yield return current;

            if (current.Result is null) { current.AddPieces(_io); }
        }

        Result = current.Result;
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator(); 
}
```

# `56_Life_for_Two/csharp/Piece.cs`

This is a Rust implementation of a simple Rockstar game. It creates a `Piece` struct to represent each piece on the game board, with an initial value of zero and a `ToImmutableHashSet` method to convert the piece to a hardware-accelerated set of integers.

The `Piece` struct has an `IsEmpty` method to check if the piece is empty, and a `Value` method to get the current value of the piece.

The `Piece` class has methods for adding neighbors and getting the next piece based on which player's piece it is on. It also has a `ToString` method to return a string representation of the piece.

The `ToImmutableHashSet` method is defined in the `ImmutableHashSet` trait and is used to convert the piece to a hardware-accelerated set of integers. This allows the game to use the Rockstar game's built-in caching to store the set of integers, which can improve performance.

The `Player` struct is defined to represent the two players in the game. It has a `Index` field to indicate which player the piece belongs to, and a `PieceMask` field to indicate the mask for that player's piece.

The `Piece` struct is defined to take an integer value and convert it to a `Piece` struct with a value of zero and a `ToImmutableHashSet` method to convert the piece to a hardware-accelerated set of integers.


```
using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;

namespace LifeforTwo;

public struct Piece
{
    public const int None = 0x0000;
    public const int Player1 = 0x0100;
    public const int Player2 = 0x1000;
    private const int PieceMask = Player1 | Player2;
    private const int NeighbourValueOffset = 8;

    private static readonly ImmutableHashSet<int> _willBePlayer1 = 
        new[] { 0x0003, 0x0102, 0x0103, 0x0120, 0x0130, 0x0121, 0x0112, 0x0111, 0x0012 }.ToImmutableHashSet();
    private static readonly ImmutableHashSet<int> _willBePlayer2 = 
        new[] { 0x0021, 0x0030, 0x1020, 0x1030, 0x1011, 0x1021, 0x1003, 0x1002, 0x1012 }.ToImmutableHashSet();

    private int _value;

    private Piece(int value) => _value = value;

    public int Value => _value & PieceMask;
    public bool IsEmpty => (_value & PieceMask) == None;

    public static Piece NewNone() => new(None);
    public static Piece NewPlayer1() => new(Player1);
    public static Piece NewPlayer2() => new(Player2);

    public Piece AddNeighbour(Piece neighbour)
    {
        _value += neighbour.Value >> NeighbourValueOffset;
        return this;
    }

    public Piece GetNext() => new(
        _value switch
        {
            _ when _willBePlayer1.Contains(_value) => Player1,
            _ when _willBePlayer2.Contains(_value) => Player2,
            _ => None
        });

    public override string ToString() =>
        (_value & PieceMask) switch
        {
            Player1 => "*",
            Player2 => "#",
            _ => " "
        };

    public static implicit operator Piece(int value) => new(value);
    public static implicit operator int(Piece piece) => piece.Value;
}
```

# `56_Life_for_Two/csharp/Program.cs`

这段代码使用了三个命名空间（namespaces）：Games.Common.IO，LifeforTwo.Resources，以及LifeforTwo。它还使用了两个全局变量：new Game（一个ConsoleIO类实例）和new ConsoleIO（一个IO类实例）。

全局变量是让整个程序都可以访问的变量，而不必在使用命名空间时每次都实例化。这种情况下，new Game和new ConsoleIO将直接在当前作用域中访问，而不是在Games.Common.IO和LifeforTwo中。

最后，该代码调用了名为new Game（应该是一个Game类实例）的Play()方法，这个方法可能实现了从Games.Common.IO命名空间中加载资源并对其进行操作的功能。但具体的作用取决于所使用的Game类如何实现了这些功能。


```
global using Games.Common.IO;
global using static LifeforTwo.Resources.Resource;
global using LifeforTwo;

new Game(new ConsoleIO()).Play();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `56_Life_for_Two/csharp/Resources/Resource.cs`



这段代码是一个自定义的 `Resource` 类，它包含了多个 `Stream`、`Format` 和 `String` 类。

`Streams` 类包含了三个构造函数，分别用于获取三个不同文档的标题流。

`Formats` 类包含了三个构造函数，分别用于将字符串初始化为不同的格式。

`Strings` 类包含了一个构造函数，用于获取一个特定的字符串。

`GetStream` 方法使用 `Assembly.GetExecutingAssembly()` 获取执行的程序集，并获取其程序集名称为 `{typeof(Resource).Namespace}` 和要获取的资源类型名称为 `name` 的资源文件对应的引用。如果资源文件找不到，则会抛出异常。

`GetString` 方法使用 `Assembly.GetExecutingAssembly()` 获取执行的程序集，并获取其程序集名称为 `{typeof(Resource).Namespace}` 和要获取的资源类型名称为 `name` 的资源文件对应的字符串。


```
using System.Reflection;
using System.Runtime.CompilerServices;

namespace LifeforTwo.Resources;

internal static class Resource
{
    internal static class Streams
    {
        public static Stream Title => GetStream();
        public static Stream IllegalCoords => GetStream();
        public static Stream SameCoords => GetStream();
    }

    internal static class Formats
    {
        public static string InitialPieces => GetString();
        public static string Player => GetString();
        public static string Winner => GetString();
    }

    internal static class Strings
    {
        public static string Draw => GetString();
    }

    private static string GetString([CallerMemberName] string? name = null)
    {
        using var stream = GetStream(name);
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    private static Stream GetStream([CallerMemberName] string? name = null) =>
        Assembly.GetExecutingAssembly().GetManifestResourceStream($"{typeof(Resource).Namespace}.{name}.txt")
            ?? throw new Exception($"Could not find embedded resource stream '{name}'.");
}
```

# `56_Life_for_Two/java/LifeForTwo.java`

This is a Java class that simulates a game of U.B. Life. The game has two players, player 1 and player 2, and each player has a score that is updated based on the number of live cells that are surrounding a cell. The score is determined by the number of live cells in a cell, so if a cell is surrounded by a thick layer of live cells, it is considered to be 'alive' and will not be declared as 'dead'.

The class also has a Coordinate class that represents a cell, and a Scores class that represents the scores of the game.

The program has a main method that first prints the instructions, then reads the input from the user, and finally updates the scores of the players. It also has a try-catch block that is used to catch any exceptions that may occur, and a finally block that is used to perform any cleanup tasks.

Overall, this class appears to be a key part of a larger game program that allows players to play the game by inputting their moves.


```
import java.util.*;
import java.util.stream.IntStream;

/**
 * Life for Two
 * <p>
 * The original BASIC program uses a grid with an extras border of cells all around,
 * probably to simplify calculations and manipulations. This java program has the exact
 * grid size and instead uses boundary check conditions in the logic.
 * <p>
 * Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
 */
public class LifeForTwo {

    final static int GRID_SIZE = 5;

    //Pair of offset which when added to the current cell's coordinates,
    // give the coordinates of the neighbours
    final static int[] neighbourCellOffsets = {
            -1, 0,
            1, 0,
            0, -1,
            0, 1,
            -1, -1,
            1, -1,
            -1, 1,
            1, 1
    };

    //The best term that I could come with to describe these numbers was 'masks'
    //They act like indicators to decide which player won the cell. The value is the score of the cell after all the
    // generation calculations.
    final static List<Integer> maskPlayer1 = List.of(3, 102, 103, 120, 130, 121, 112, 111, 12);
    final static List<Integer> maskPlayer2 = List.of(21, 30, 1020, 1030, 1011, 1021, 1003, 1002, 1012);

    public static void main(String[] args) {
        printIntro();
        Scanner scan = new Scanner(System.in);
        scan.useDelimiter("\\D");

        int[][] grid = new int[GRID_SIZE][GRID_SIZE];

        initializeGrid(grid);

        //Read the initial 3 moves for each player
        for (int b = 1; b <= 2; b++) {
            System.out.printf("\nPLAYER %d - 3 LIVE PIECES.%n", b);
            for (int k1 = 1; k1 <= 3; k1++) {
                var player1Coordinates = readUntilValidCoordinates(scan, grid);
                grid[player1Coordinates.x - 1][player1Coordinates.y - 1] = (b == 1 ? 3 : 30);
            }
        }

        printGrid(grid);

        calculatePlayersScore(grid); //Convert 3, 30 to 100, 1000

        resetGridForNextGen(grid);
        computeCellScoresForOneGen(grid);

        var playerScores = calculatePlayersScore(grid);
        resetGridForNextGen(grid);

        boolean gameOver = false;
        while (!gameOver) {
            printGrid(grid);
            if (playerScores.getPlayer1Score() == 0 && playerScores.getPlayer2Score() == 0) {
                System.out.println("\nA DRAW");
                gameOver = true;
            } else if (playerScores.getPlayer2Score() == 0) {
                System.out.println("\nPLAYER 1 IS THE WINNER");
                gameOver = true;
            } else if (playerScores.getPlayer1Score() == 0) {
                System.out.println("\nPLAYER 2 IS THE WINNER");
                gameOver = true;
            } else {
                System.out.print("PLAYER 1 ");
                Coordinate player1Move = readCoordinate(scan);
                System.out.print("PLAYER 2 ");
                Coordinate player2Move = readCoordinate(scan);
                if (!player1Move.equals(player2Move)) {
                    grid[player1Move.x - 1][player1Move.y - 1] = 100;
                    grid[player2Move.x - 1][player2Move.y - 1] = 1000;
                }
                //In the original, B is assigned 99 when both players choose the same cell
                //and that is used to control the flow
                computeCellScoresForOneGen(grid);
                playerScores = calculatePlayersScore(grid);
                resetGridForNextGen(grid);
            }
        }

    }

    private static void initializeGrid(int[][] grid) {
        for (int[] row : grid) {
            Arrays.fill(row, 0);
        }
    }

    private static void computeCellScoresForOneGen(int[][] grid) {
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                if (grid[i][j] >= 100) {
                    calculateScoreForOccupiedCell(grid, i, j);
                }
            }
        }
    }

    private static Scores calculatePlayersScore(int[][] grid) {
        int m2 = 0;
        int m3 = 0;
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                if (grid[i][j] < 3) {
                    grid[i][j] = 0;
                } else {
                    if (maskPlayer1.contains(grid[i][j])) {
                        m2++;
                    } else if (maskPlayer2.contains(grid[i][j])) {
                        m3++;
                    }
                }
            }
        }
        return new Scores(m2, m3);
    }

    private static void resetGridForNextGen(int[][] grid) {
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                if (grid[i][j] < 3) {
                    grid[i][j] = 0;
                } else {
                    if (maskPlayer1.contains(grid[i][j])) {
                        grid[i][j] = 100;
                    } else if (maskPlayer2.contains(grid[i][j])) {
                        grid[i][j] = 1000;
                    } else {
                        grid[i][j] = 0;
                    }
                }
            }
        }
    }

    private static void calculateScoreForOccupiedCell(int[][] grid, int i, int j) {
        var b = 1;
        if (grid[i][j] > 999) {
            b = 10;
        }
        for (int k = 0; k < 15; k += 2) {
            //check bounds
            var neighbourX = i + neighbourCellOffsets[k];
            var neighbourY = j + neighbourCellOffsets[k + 1];
            if (neighbourX >= 0 && neighbourX < GRID_SIZE &&
                    neighbourY >= 0 && neighbourY < GRID_SIZE) {
                grid[neighbourX][neighbourY] = grid[neighbourX][neighbourY] + b;
            }

        }
    }

    private static void printGrid(int[][] grid) {
        System.out.println();
        printRowEdge();
        System.out.println();
        for (int i = 0; i < grid.length; i++) {
            System.out.printf("%d ", i + 1);
            for (int j = 0; j < grid[i].length; j++) {
                System.out.printf(" %c ", mapChar(grid[i][j]));
            }
            System.out.printf(" %d", i + 1);
            System.out.println();
        }
        printRowEdge();
        System.out.println();
    }

    private static void printRowEdge() {
        System.out.print("0 ");
        IntStream.range(1, GRID_SIZE + 1).forEach(i -> System.out.printf(" %s ", i));
        System.out.print(" 0");
    }

    private static char mapChar(int i) {
        if (i == 3 || i == 100) {
            return '*';
        }
        if (i == 30 || i == 1000) {
            return '#';
        }
        return ' ';
    }

    private static Coordinate readUntilValidCoordinates(Scanner scanner, int[][] grid) {
        boolean coordinateInRange = false;
        Coordinate coordinate = null;
        while (!coordinateInRange) {
            coordinate = readCoordinate(scanner);
            if (coordinate.x <= 0 || coordinate.x > GRID_SIZE
                    || coordinate.y <= 0 || coordinate.y > GRID_SIZE
                    || grid[coordinate.x - 1][coordinate.y - 1] != 0) {
                System.out.println("ILLEGAL COORDS. RETYPE");
            } else {
                coordinateInRange = true;
            }
        }
        return coordinate;
    }

    private static Coordinate readCoordinate(Scanner scanner) {
        Coordinate coordinate = null;
        int x, y;
        boolean valid = false;

        System.out.println("X,Y");
        System.out.print("XXXXXX\r");
        System.out.print("$$$$$$\r");
        System.out.print("&&&&&&\r");

        while (!valid) {
            try {
                System.out.print("? ");
                y = scanner.nextInt();
                x = scanner.nextInt();
                valid = true;
                coordinate = new Coordinate(x, y);
            } catch (InputMismatchException e) {
                System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE");
                valid = false;
            } finally {
                scanner.nextLine();
            }
        }
        return coordinate;
    }

    private static void printIntro() {
        System.out.println("                                LIFE2");
        System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println("\n\n");

        System.out.println("\tU.B. LIFE GAME");
    }

    private static class Coordinate {
        private final int x, y;

        public Coordinate(int x, int y) {
            this.x = x;
            this.y = y;
        }

        public int getX() {
            return x;
        }

        public int getY() {
            return y;
        }

        @Override
        public String toString() {
            return "Coordinate{" +
                    "x=" + x +
                    ", y=" + y +
                    '}';
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Coordinate that = (Coordinate) o;
            return x == that.x && y == that.y;
        }

        @Override
        public int hashCode() {
            return Objects.hash(x, y);
        }
    }

    private static class Scores {
        private final int player1Score;
        private final int player2Score;

        public Scores(int player1Score, int player2Score) {
            this.player1Score = player1Score;
            this.player2Score = player2Score;
        }

        public int getPlayer1Score() {
            return player1Score;
        }

        public int getPlayer2Score() {
            return player2Score;
        }
    }


}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `56_Life_for_Two/javascript/lifefortwo.js`

这段代码定义了两个函数：`print()` 和 `input()`。

`print()` 函数的作用是在文档的 `id` 为 `output` 的元素中添加一个新的文本节点，并将其内容设置为传入的参数 `str`。

`input()` 函数的作用是从用户那里获取输入值，并在获取到用户输入后将其存储在变量 `input_str` 中。该函数通过添加一个 `INPUT` 元素元素到文档中，然后使用 `focus()` 方法将元素的焦点转移该元素，这样用户输入时可以获得输入验证和自动完成帮助。该函数还监听 `keydown` 事件，当用户按下键盘上的 `13` 键时，它会在输入框中设置新的输入值，并将其存储在 `input_str` 变量中。然后，函数使用 `print()` 函数将新的输入值打印到文档中，并使用 `print()` 函数的 `removeChild()` 方法将 `INPUT` 元素从文档中删除。


```
// LIFE FOR TWO
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

这段代码定义了一个名为“tab”的函数，它接受一个参数“space”，用于指定在空格中插入的字符个数。

函数内部首先创建一个空字符串“str”，并使用一个变量“space”来跟踪在空格中插入的字符数量。每次，“space”减1，如果“space”为0，循环结束。在循环中，使用“+”将一个或多个空格插入到“str”字符串中。

然后，将生成的字符串“str”返回给调用者。

在代码的最后部分，定义了三个变量：na、ka和aa，分别存储了一个包含数字的数组。

另外，定义了两个变量：xa和ya，以及一个变量j。变量xa和ya似乎没有在函数内部使用，而变量j则没有在代码中定义或说明其作用。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var na = [];
var ka = [, 3,102,103,120,130,121,112,111,12,
          21,30,1020,1030,1011,1021,1003,1002,1012];
var aa = [,-1,0,1,0,0,-1,0,1,-1,-1,1,-1,-1,1,1,1];
var xa = [];
var ya = [];
var j;
```

这段代码定义了一个名为 show_data 的函数，它输出了一些数字和字符串，用于显示数据。函数内部定义了三个整数变量 k、m2 和 m3，用于跟踪输出的数字数量。

函数体中，通过一个 for 循环来遍历一个包含六个数字的数组。在循环内部，使用两个 if 语句来判断当前数字是否等于数轴上对应的偏移量。如果是，则执行特定的操作，并将对应的数字计数加一。

如果当前数字既不在数轴上对应的偏移量中，也不是零，那么将该数字输出，并将其对应的计数加一。在输出数字时，使用不同的空格和圆括号来分隔不同的数字。


```
var k;
var m2;
var m3;

function show_data()
{
    k = 0;
    m2 = 0;
    m3 = 0;
    for (j = 0; j <= 6; j++) {
        print("\n");
        for (k = 0; k <= 6; k++) {
            if (j == 0 || j == 6) {
                if (k == 6)
                    print(" 0 ");
                else
                    print(" " + k + " ");
            } else if (k == 0 || k == 6) {
                if (j == 6)
                    print(" 0\n");
                else
                    print(" " + j + " ");
            } else {
                if (na[j][k] >= 3) {
                    for (o1 = 1; o1 <= 18; o1++) {
                        if (na[j][k] == ka[o1])
                            break;
                    }
                    if (o1 <= 18) {
                        if (o1 <= 9) {
                            na[j][k] = 100;
                            m2++;
                            print(" * ");
                        } else {
                            na[j][k] = 1000;
                            m3++;
                            print(" # ");
                        }
                    } else {
                        na[j][k] = 0;
                        print("   ");
                    }
                } else {
                    na[j][k] = 0;
                    print("   ");
                }
            }
        }
    }
}

```

这是一段 JavaScript 代码，名为 `process_board()`。这段代码的主要目的是处理一个棋盘（具体是一个 5x5 的棋盘）中的数据，并在处理完成后将数据输出。

具体来说，这段代码将以下内容：

1. 遍历棋盘的每个位置（从 1 到 5，即行和列）。
2. 对于每个位置，按照行和列的顺序遍历（从 1 到 5，即从左上角到右下角）。
3. 如果当前位置（例如，第 3 行第 2 列的位置）的数值大于 99，则执行以下操作：
  a. 如果该位置是数字 9 或 10，则将 `b` 赋值为 1，并将 `aa[o1]` 和 `aa[o1 + 1]` 的值加 1。
  b. 对于每个 `aa[o1]`，将其值（即数字 9 或 10）加上 `b`。
4. 处理完所有位置后，将数据输出。

由于 `na` 数组未在代码中定义，所以这段代码无法确定数组中的元素具体是什么。


```
function process_board()
{
    for (j = 1; j <= 5; j++) {
        for (k = 1; k <= 5; k++) {
            if (na[j][k] > 99) {
                b = 1;
                if (na[j][k] > 999)
                    b = 10;
                for (o1 = 1; o1 <= 15; o1 += 2) {
                    na[j + aa[o1]][k + aa[o1 + 1]] = na[j + aa[o1]][k + aa[o1 + 1]] + b;
                }
            }
        }
    }
    show_data();
}

```

This is a program written in the JavaScript programming language. It appears to simulate a game of Connect-the-Dot, where players connect two 5x5 boards to each other, with the goal of connecting a diagonal line from one player's team's dots to the other player's team's dots. The code implements the game logic, as well as the界面 and some basic graphics.

The game initializes the board and each player's starting position, and then enters a loop that continues until one of the players has won or the game is over.

In each iteration of the loop, the code checks for the players' turn, and if it's the player's turn, displays a prompt asking them to either draw a line or select a square to connect. After the user has made a choice, the code checks for the connectivity of the selected square, and if the connection is valid (i.e. the player has connected all four dots in a line), the code updates the board accordingly.

If the player has not made a valid choice or the game is over (e.g. one of the players has won), the code displays a graphic of a connected board (using the default Connect-the-Dot graphics).

Note that the code also includes some basic graphics to display the board, as well as some styling to make the output more readable.


```
// Main program
async function main()
{
    print(tab(33) + "LIFE2\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print(tab(10) + "U.B. LIFE GAME\n");
    m2 = 0;
    m3 = 0;
    for (j = 0; j <= 6; j++) {
        na[j] = [];
        for (k = 0; k <= 6; k++)
            na[j][k] = 0;
    }
    for (b = 1; b <= 2; b++) {
        p1 = (b == 2) ? 30 : 3;
        print("\n");
        print("PLAYER " + b + " - 3 LIVE PIECES.\n");
        for (k1 = 1; k1 <= 3; k1++) {
            while (1) {
                print("X,Y\n");
                str = await input();
                ya[b] = parseInt(str);
                xa[b] = parseInt(str.substr(str.indexOf(",") + 1));
                if (xa[b] > 0 && xa[b] < 6 && ya[b] > 0 && ya[b] < 5 && na[xa[b]][ya[b]] == 0)
                    break;
                print("ILLEGAL COORDS. RETYPE\n");
            }
            if (b != 1) {
                if (xa[1] == xa[2] && ya[1] == ya[2]) {
                    print("SAME COORD.  SET TO 0\n");
                    na[xa[b] + 1][ya[b] + 1] = 0;
                    b = 99;
                }
            }
            na[xa[b]][ya[b]] = p1;
        }
    }
    show_data();
    while (1) {
        print("\n");
        process_board();
        if (m2 == 0 && m3 == 0) {
            print("\n");
            print("A DRAW\n");
            break;
        }
        if (m3 == 0) {
            print("\n");
            print("PLAYER 1 IS THE WINNER\n");
            break;
        }
        if (m2 == 0) {
            print("\n");
            print("PLAYER 2 IS THE WINNER\n");
            break;
        }
        for (b = 1; b <= 2; b++) {
            print("\n");
            print("\n");
            print("PLAYER " + b + " ");
            while (1) {
                print("X,Y\n");
                str = await input();
                ya[b] = parseInt(str);
                xa[b] = parseInt(str.substr(str.indexOf(",") + 1));
                if (xa[b] > 0 && xa[b] < 6 && ya[b] > 0 && ya[b] < 5 && na[xa[b]][ya[b]] == 0)
                    break;
                print("ILLEGAL COORDS. RETYPE\n");
            }
            if (b != 1) {
                if (xa[1] == xa[2] && ya[1] == ya[2]) {
                    print("SAME COORD.  SET TO 0\n");
                    na[xa[b] + 1][ya[b] + 1] = 0;
                    b = 99;
                }
            }
            if (b == 99)
                break;
        }
        if (b <= 2) {
            na[x[1]][y[1]] = 100;
            na[x[2]][y[2]] = 1000;
        }
    }
}

```

这是 C 语言中的一个标准函数，名为 `main()`。这个函数是程序的入口点，程序从此处开始执行。

在 `main()` 函数中，程序员可以编写程序执行所需的代码。通常，程序会在 `main()` 函数内开始执行，并尽可能地完成一些必要的工作，如初始化计算机硬件、加载操作系统和相关设置、启动应用程序等。

请注意，即使 `main()` 函数内没有编写任何实际的代码，程序仍然会在启动时执行。所以，在 `main()` 函数内，程序员应该始终编写完整的、可以执行的代码，以确保程序能够正常运行。


```
main();

```