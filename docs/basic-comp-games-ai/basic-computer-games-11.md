# BasicComputerGames源码解析 11

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `02_Amazing/python/amazing.py`

This is a Python implementation of a maze, where each "cell" has a list of integers that represent the color of the cell (0 for wall, 1 for enterable area). The maze has a width and a length, and there is an entrance at row 0, column 0. The `Maze` class has a `__init__` method that creates the maze's walls and uses it to store the cells' color values, and a `display` method that prints the maze as a string. The `Maze` class also has a `Maze` proper类 that inherits from `dataclasses`, and it has a `width` and `length` attribute, and a `enter_col` attribute that stores the index of the entrance.


```
import enum
import random
from dataclasses import dataclass
from typing import List, Tuple

# Python translation by Frank Palazzolo - 2/2021


class Maze:
    def __init__(self, width: int, length: int) -> None:
        assert width >= 2 and length >= 2
        used: List[List[int]] = []
        walls: List[List[int]] = []
        for _ in range(length):
            used.append([0] * width)
            walls.append([0] * width)

        # Pick a random entrance, mark as used
        enter_col = random.randint(0, width - 1)
        used[0][enter_col] = 1

        self.used = used
        self.walls = walls
        self.enter_col = enter_col
        self.width = width
        self.length = length

    def add_exit(self) -> None:
        """Modifies 'walls' to add an exit to the maze."""
        col = random.randint(0, self.width - 1)
        row = self.length - 1
        self.walls[row][col] = self.walls[row][col] + 1

    def display(self) -> None:
        for col in range(self.width):
            if col == self.enter_col:
                print(".  ", end="")
            else:
                print(".--", end="")
        print(".")
        for row in range(self.length):
            print("I", end="")
            for col in range(self.width):
                if self.walls[row][col] < 2:
                    print("  I", end="")
                else:
                    print("   ", end="")
            print()
            for col in range(self.width):
                if self.walls[row][col] == 0 or self.walls[row][col] == 2:
                    print(":--", end="")
                else:
                    print(":  ", end="")
            print(".")


```

这段代码定义了一个名为 Direction 的枚举类型，它有四个成员变量，分别表示四个方向，分别为 LEFT、UP、RIGHT 和 DOWN。

定义了一个名为 Position 的类，它包含一个 col 和一个 row 的成员变量，用于表示当前的位置。

定义了一个名为 EXIT_DOWN 的类常量，它表示向下走的方向。

最后，通过 EXIT_DOWN 常量，给 Position 类添加了一个名为 "EXIT_DOWN" 的方法，用于设置给定位置的朝向为向下。


```
class Direction(enum.Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


@dataclass
class Position:
    row: int
    col: int


# Give Exit directions nice names
EXIT_DOWN = 1
```

这段代码是一个Python程序，主要作用是输出一个AMAZING PROGRAM，并引导用户输入一个Maze的维度。

具体来说，它首先通过调用`get_maze_dimensions`函数获取一个指定大小的二维迷宫的宽度和长度，然后调用`build_maze`函数构建了一个迷宫，并将它显示出来。

需要注意的是，这个程序并没有对迷宫进行任何处理，只是一个简单的示例，可以让用户自己创建一个迷宫并输出它的结构。


```
EXIT_RIGHT = 2


def main() -> None:
    print_intro()
    width, length = get_maze_dimensions()
    maze = build_maze(width, length)
    maze.display()


def print_intro() -> None:
    print(" " * 28 + "AMAZING PROGRAM")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")


```

这段代码定义了一个名为 `build_maze` 的函数，它接受两个整数参数 `width` 和 `length`，并返回一个名为 `Maze` 的类实例。

函数内部首先设置两个二维数组 `maze` 和 `position`，分别从 0 开始。然后设置一个计数器 `count`，用于统计当前已处理到的细胞数。

接下来，函数内部使用 `get_possible_directions` 函数获取所有可能的移动方向，并存储在一个列表中。如果列表非空，则说明可以移动，函数将移动到 `position` 所在的下一个空位置，并将 `count` 自增 1。否则，函数将尝试所有可能的移动方向，直到找到一个可以移动的位置或者到达了所有的空位置。

在移动过程中，如果当前位置处理到了一个被堵住的细胞，函数将记录下这个位置，并继续尝试下一个可能的移动方向。如果当前位置还没有被堵住，则说明已经到达了一个空位置，函数将处理到这个位置并尝试将其设为 1，以便统计已处理到的细胞数。

最后，函数会将所有被处理到的细胞数加 1，并调用 `Maze` 类的一个名为 `add_exit` 的方法，将所有出口的位置添加到 `maze` 中。函数返回一个名为 `Maze` 的类实例，以确保所有的输入参数和函数内部的数据结构都被正确初始化和处理。


```
def build_maze(width: int, length: int) -> Maze:
    """Build two 2D arrays."""
    #
    # used:
    #   Initially set to zero, unprocessed cells
    #   Filled in with consecutive non-zero numbers as cells are processed
    #
    # walls:
    #   Initially set to zero, (all paths blocked)
    #   Remains 0 if there is no exit down or right
    #   Set to 1 if there is an exit down
    #   Set to 2 if there is an exit right
    #   Set to 3 if there are exits down and right
    assert width >= 2 and length >= 2

    maze = Maze(width, length)
    position = Position(row=0, col=maze.enter_col)
    count = 2

    while count != width * length + 1:
        possible_dirs = get_possible_directions(maze, position)

        # If we can move in a direction, move and make opening
        if len(possible_dirs) != 0:
            position, count = make_opening(maze, possible_dirs, position, count)
        # otherwise, move to the next used cell, and try again
        else:
            while True:
                if position.col != width - 1:
                    position.col += 1
                elif position.row != length - 1:
                    position.row, position.col = position.row + 1, 0
                else:
                    position.row, position.col = 0, 0
                if maze.used[position.row][position.col] != 0:
                    break

    maze.add_exit()
    return maze


```

这段代码定义了一个名为 `make_opening` 的函数，它接受一个迷宫 `maze`、一个或多个方向 `possible_dirs` 和一个位置 `pos`，并返回一个元组 `(pos, count)`。

函数的主要作用是改变迷宫中的状态，以便在需要时能够方便地收集坚果或点击墙。具体来说，当需要点击某个位置时，函数会根据 `possible_dirs` 中随机选择一个方向，然后根据所选方向对位置进行移动，并且更新相应的墙的状态。当玩家收集到 2 颗坚果时，函数会增加点击墙的次数并更新状态。

可以认为，这个函数是一个辅助函数，用于在游戏中帮助玩家更方便地收集坚果或点击墙。


```
def make_opening(
    maze: Maze,
    possible_dirs: List[Direction],
    pos: Position,
    count: int,
) -> Tuple[Position, int]:
    """
    Attention! This modifies 'used' and 'walls'
    """
    direction = random.choice(possible_dirs)
    if direction == Direction.LEFT:
        pos.col = pos.col - 1
        maze.walls[pos.row][pos.col] = EXIT_RIGHT
    elif direction == Direction.UP:
        pos.row = pos.row - 1
        maze.walls[pos.row][pos.col] = EXIT_DOWN
    elif direction == Direction.RIGHT:
        maze.walls[pos.row][pos.col] = maze.walls[pos.row][pos.col] + EXIT_RIGHT
        pos.col = pos.col + 1
    elif direction == Direction.DOWN:
        maze.walls[pos.row][pos.col] = maze.walls[pos.row][pos.col] + EXIT_DOWN
        pos.row = pos.row + 1
    maze.used[pos.row][pos.col] = count
    count = count + 1
    return pos, count


```

该函数 `get_possible_directions` 接收一个迷宫 `maze` 和一个位置 `pos` 作为参数，并返回一个包含所有未受到阻挡的方向的列表。

函数首先检查 `pos` 所在的行是否为 0，如果是，则将向左、上、右和下游的箭头类型排除在外。然后，它检查 `pos` 所在的列是否为迷宫的宽度 -1，如果是，则将向上和向下的箭头类型排除在外。接下来，它检查 `pos` 所在的行是否为迷宫的长度 -1，如果是，则将向右和向左的箭头类型排除在外。最后，它检查 `pos` 所在的列是否为 0，如果是，则将其添加到返回的列表中。

函数返回的列表包含了所有未受到阻挡的箭头类型。


```
def get_possible_directions(maze: Maze, pos: Position) -> List[Direction]:
    """
    Get a list of all directions that are not blocked.

    Also ignore hit cells that we have already processed
    """
    possible_dirs = list(Direction)
    if pos.col == 0 or maze.used[pos.row][pos.col - 1] != 0:
        possible_dirs.remove(Direction.LEFT)
    if pos.row == 0 or maze.used[pos.row - 1][pos.col] != 0:
        possible_dirs.remove(Direction.UP)
    if pos.col == maze.width - 1 or maze.used[pos.row][pos.col + 1] != 0:
        possible_dirs.remove(Direction.RIGHT)
    if pos.row == maze.length - 1 or maze.used[pos.row + 1][pos.col] != 0:
        possible_dirs.remove(Direction.DOWN)
    return possible_dirs


```

这段代码定义了一个名为 `get_maze_dimensions` 的函数，用于获取一个棋盘(maze)的宽度和高度。函数使用了无限循环来读取用户输入的棋盘宽度和高，并返回一个元组表示宽度和高度。

在函数内部，首先进行了一个 while 循环，该循环将一直运行，直到用户输入的含义不明确(即没有给出棋盘的宽度和高度)。当用户输入的含义不明确时，函数将输出 "Meaningless dimensions. Try again." 并要求用户重新输入。

一旦用户提供了棋盘的宽度和高度，函数将使用这些值来检查是否可以创建一个有效的棋盘。如果宽度大于 1 或高度大于 1，则函数将退出无限循环，并返回无效的棋盘大小。否则，函数将继续运行，并尝试创建一个有效的棋盘。

如果用户成功创建了有效的棋盘，函数将返回其宽度和高度，并结束程序。否则，函数将继续运行，并再次尝试创建一个有效的棋盘。


```
def get_maze_dimensions() -> Tuple[int, int]:
    while True:
        input_str = input("What are your width and length?")
        if input_str.count(",") == 1:
            width_str, length_str = input_str.split(",")
            width = int(width_str)
            length = int(length_str)
            if width > 1 and length > 1:
                break
        print("Meaningless dimensions. Try again.")
    return width, length


if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)

Converted to Ruby (with tons of inspiration from the Python version) by @marcheiligers

Run `ruby amazing.rb`.

Run `DEBUG=1 ruby amazing.ruby` to see how it works (requires at least Ruby 2.7).


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by Anthony Rubick [AnthonyMichaelTDM](https://github.com/AnthonyMichaelTDM)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Animal

Unlike other computer games in which the computer picks a number or letter and you must guess what it is, in this game _you_ think of an animal and the _computer_ asks you questions and tries to guess the name of your animal. If the computer guesses incorrectly, it will ask you for a question that differentiates the animal you were thinking of. In this way the computer “learns” new animals. Questions to differentiate new animals should be input without a question mark.

This version of the game does not have a SAVE feature. If your system allows, you may modify the program to save and reload the array when you want to play the game again. This way you can save what the computer learns over a series of games.

At any time if you reply “LIST” to the question “ARE YOU THINKING OF AN ANIMAL,” the computer will tell you all the animals it knows so far.

The program starts originally by knowing only FISH and BIRD. As you build up a file of animals you should use broad, general questions first and then narrow down to more specific ones with later animals. For example, if an elephant was to be your first animal, the computer would ask for a question to distinguish an elephant from a bird. Naturally, there are hundreds of possibilities, however, if you plan to build a large file of animals a good question would be “IS IT A MAMMAL.”

This program can be easily modified to deal with categories of things other than animals by simply modifying the initial data and the dialogue references to animals. In an educational environment, this would be a valuable program to teach the distinguishing characteristics of many classes of objects — rock formations, geography, marine life, cell structures, etc.

Originally developed by Arthur Luehrmann at Dartmouth College, Animal was subsequently shortened and modified by Nathan Teichholtz at DEC and Steve North at Creative Computing.

---

As published in Basic Computer Games (1978)
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=4)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=19)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `03_Animal/csharp/Branch.cs`

这段代码定义了一个名为 "Branch" 的类，用于表示分支。该类包含两个私有字段 "Text" 和 "IsEnd"，以及两个公有字段 "Yes" 和 "No"。

"IsEnd" 字段是一个布尔类型变量，其值由两个分支的值决定，如果两个分支都为假(即 "")，则 "IsEnd" 的值为真，否则为假。

"Yes" 和 "No" 字段也是布尔类型变量，分别表示当前分支是否为真(即分支的值为真)和假(即分支的值为假)。

该类还重写了 "ToString" 方法，用于将分支对象的字符串表示为 "Text: IsEnd true" 或 "Text: IsEnd false"。

总的来说，该代码定义了一个表示分支的类，通过对 "Text" 和 "IsEnd" 字段的设置，可以控制分支的真假，通过 "Yes" 和 "No" 字段的设置，可以判断当前分支的情况。而 "ToString" 方法则可以用来获取分支的字符串表示。


```
﻿namespace Animal
{
    public class Branch
    {
        public string Text { get; set; }

        public bool IsEnd => Yes == null && No == null;

        public Branch Yes { get; set; }

        public Branch No { get; set; }

        public override string ToString()
        {
            return $"{Text} : IsEnd {IsEnd}";
        }
    }
}

```

# `03_Animal/csharp/Program.cs`

这段代码是一个使用Animal类（可能是自定义的，我无法确定）的示例。首先，它创建了一个包含字符串'ANIMAL'和'CREATIVE COMPUTING'的列表，然后将它们连接在一起并输出。接着，它输出了一行字符串'PLAY 'GUESS THE ANIMAL'"。最后，它要求用户猜测一个动物，然后计算机猜测并提供提示以帮助用户猜测。


```
﻿using System;
using System.Collections.Generic;
using System.Linq;

using Animal;

Console.WriteLine(new string(' ', 32) + "ANIMAL");
Console.WriteLine(new string(' ', 15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
Console.WriteLine();
Console.WriteLine();
Console.WriteLine();
Console.WriteLine("PLAY 'GUESS THE ANIMAL'");
Console.WriteLine();
Console.WriteLine("THINK OF AN ANIMAL AND THE COMPUTER WILL TRY TO GUESS IT.");
Console.WriteLine();

```

这段代码创建了一个根问题树，其中包含三个分支：一个名为“DOES IT SWIM”的分支，它有两个子分支，“FISH”和“BIRD”。这两个子分支分别代表问题的真假两个方面。另外，还有一个名为“TRUE”的分支，它代表问题的正确答案。

接着，定义了一个数组TRUE_INPUTS，其中包含四个字符串元素，分别是个“Y”、“YES”、“T”和“TRUE”。另一个数组FALSE_INPUTS，其中包含四个字符串元素，分别是个“N”、“NO”、“F”和“FALSE”。

在while循环中，不断调用MainGameLoop函数，这个函数可能是用于显示问题树的用户界面部分。注意，问题树的具体结构没有给出，我们无法进一步了解这个代码的实际作用。


```
// Root of the question and answer tree
Branch rootBranch = new Branch
{
    Text = "DOES IT SWIM",
    Yes = new Branch { Text = "FISH" },
    No = new Branch { Text = "BIRD" }
};

string[] TRUE_INPUTS = { "Y", "YES", "T", "TRUE" };
string[] FALSE_INPUTS = { "N", "NO", "F", "FALSE" };


while (true)
{
    MainGameLoop();
}

```

This appears to be a program written in C# that is designed to simulate a simple interview system where the user is asked to choose between two possible answers for a given question about an animal. The user's response is then used to determine the correct branch of the tree and continue the interview process until the user has provided an answer that is accepted or the interview is over.

The program also includes some comments at the beginning explaining what it is doing, but it doesn't provide any additional context or explanations for the functions and variables used in the program. It is possible that there are other parts of the program that are not included in this code snippet, such as the code for the interview itself.


```
void MainGameLoop()
{
    // Wait fora YES or LIST command
    string input = null;
    while (true)
    {
        input = GetInput("ARE YOU THINKING OF AN ANIMAL");
        if (IsInputListCommand(input))
        {
            ListKnownAnimals(rootBranch);
        }
        else if (IsInputYes(input))
        {
            break;
        }
    }

    // Walk through the tree following the YES and NO
    // branches based on user input.
    Branch currentBranch = rootBranch;
    while (!currentBranch.IsEnd)
    {
        while (true)
        {
            input = GetInput(currentBranch.Text);
            if (IsInputYes(input))
            {
                currentBranch = currentBranch.Yes;
                break;
            }
            else if (IsInputNo(input))
            {
                currentBranch = currentBranch.No;
                break;
            }
        }
    }

    // Was the answer correct?
    input = GetInput($"IS IT A {currentBranch.Text}");
    if (IsInputYes(input))
    {
        Console.WriteLine("WHY NOT TRY ANOTHER ANIMAL?");
        return;
    }

    // Interview the user to add a new question and answer
    // branch to the tree
    string newAnimal = GetInput("THE ANIMAL YOU WERE THINKING OF WAS A");
    string newQuestion = GetInput($"PLEASE TYPE IN A QUESTION THAT WOULD DISTINGUISH A {newAnimal} FROM A {currentBranch.Text}");
    string newAnswer = null;
    while (true)
    {
        newAnswer = GetInput($"FOR A {newAnimal} THE ANSWER WOULD BE");
        if (IsInputNo(newAnswer))
        {
            currentBranch.No = new Branch { Text = newAnimal };
            currentBranch.Yes = new Branch { Text = currentBranch.Text };
            currentBranch.Text = newQuestion;
            break;
        }
        else if (IsInputYes(newAnswer))
        {
            currentBranch.Yes = new Branch { Text = newAnimal };
            currentBranch.No = new Branch { Text = currentBranch.Text };
            currentBranch.Text = newQuestion;
            break;
        }
    }
}

```

这段代码是一个 C# 类，的作用是获取用户输入并将其转换为大写形式。

GetInput 方法使用了一个字符串 prompt，用来在控制台输出一个带问号的提示，然后等待用户输入并将其存储在 result 变量中。如果输入是空字符串或只包含一个换行符，那么该方法会重新调用自身，继续提示用户输入。

IsInputYes 和 IsInputNo 方法用于判断用户输入是否为“是”。在代码中，这两个方法都是基于输入字符串的 ToUpperInvariant() 方法来获取输入的字符串，并将其存储在输入变量中。如果输入字符串中只包含小写字母，那么这些方法会将其转换为大写字符串并返回 false，否则会返回 true。

最后，需要注意，这段代码并没有对输入进行验证，以确保输入不会是 null、空或只包含换行符。


```
string GetInput(string prompt)
{
    Console.Write($"{prompt}? ");
    string result = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(result))
    {
        return GetInput(prompt);
    }

    return result.Trim().ToUpper();
}

bool IsInputYes(string input) => TRUE_INPUTS.Contains(input.ToUpperInvariant().Trim());

bool IsInputNo(string input) => FALSE_INPUTS.Contains(input.ToUpperInvariant().Trim());

```

这段代码有两个函数，第一个函数是`bool IsInputListCommand(string input) =>` ，它返回一个布尔值，表示输入是否为列表命令行界面(ListCommand)。函数的作用是检查输入字符串是否以大括号"LIST"开头，如果是，则返回真，否则返回假。

第二个函数是`string[] GetKnownAnimals(Branch branch)`，它返回一个字符串数组，包含了所有已知动物的名称。函数的作用是从根分支开始递归搜索所有已知动物的名称，并将这些名称存储到一个字符串数组中。如果当前分支是终点(IsEnd)，则返回该分支的名称，否则递归调用`GetKnownAnimals`函数获取子分支的名称，并将它们添加到结果字符串数组中。最后，返回结果字符串数组。


```
bool IsInputListCommand(string input) => input.ToUpperInvariant().Trim() == "LIST";

string[] GetKnownAnimals(Branch branch)
{
    List<string> result = new List<string>();
    if (branch.IsEnd)
    {
        return new[] { branch.Text };
    }
    else
    {
        result.AddRange(GetKnownAnimals(branch.Yes));
        result.AddRange(GetKnownAnimals(branch.No));
        return result.ToArray();
    }
}

```



这段代码是一个名为 "ListKnownAnimals" 的函数，其作用是打印已知动物的列表。函数接受一个 "Branch" 类型的参数，但并未定义该参数的具体类型和含义。

函数内部首先定义了一个名为 "animals" 的字符串数组，使用 "GetKnownAnimals" 函数获取了某个分支上所有已知动物的名称。

接着，函数遍历 "animals" 数组中的所有元素，并输出到控制台。在循环中，首先判断当前循环位置的列是否为 0，如果是 0，则输出一个空格，如果不是 0，则输出该列上的动物名称和对齐方式(0 列对齐、15 列对齐等)。然后，在循环输出动物名称的同时，根据输出对齐方式调整字符串长度，使其只显示一行的字符。最后，输出一个换行符。

由于循环中的分支处理程序并未定义具体分支类型，因此 ListKnownAnimals 函数的行为无法确定，可能会对不同的分支类型产生不同的结果。


```
void ListKnownAnimals(Branch branch)
{
    string[] animals = GetKnownAnimals(branch);
    for (int x = 0; x < animals.Length; x++)
    {
        int column = (x % 4);
        if (column == 0)
        {
            Console.WriteLine();
        }

        Console.Write(new string(' ', column == 0 ? 0 : 15) + animals[x]);
    }
    Console.WriteLine();
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `03_Animal/java/src/Animal.java`

这段代码定义了一个名为`Animal`的类，从`java.util.List`接口中继承了`ArrayList`类。这个类包含了一个`Scanner`对象，用于从用户输入中读取游戏中的问题。

首先，该类中定义了一个` Locale`对象，这可能是一个用来处理不同语言问题的工具类。然后，通过构造函数，用从用户输入中获得的` Locale`对象初始化了这个`Locale`对象。

接着，该类中定义了一个` List<String>`对象，这可能是一个`ArrayList`对象中包含了一些字符串类型的元素。然后，用这个` List<String>`对象实现了` Java 8 `中提供的一个`Collectors`接口的`mapToPrettyString`方法。这个方法的第一个参数是一个`Map<String, Integer>`对象，其中键是游戏中的问题，值是每道题的分数。第二个参数是一个`PrettyStringBuilder`对象，用于将Map中的键和值转换成相应的字符串，使得这些字符串可以被打印出来。

接下来，该类中定义了一个`Map<String, Integer>`对象，用于存储游戏中的问题及其得分。然后，通过实现`java.util.Map`接口，将上述的`List<String>`对象和`Map<String, Integer>`对象映射到同一个键`String`上。

接着，该类中定义了一个`Scanner`对象，用于从用户输入中读取游戏中的问题。然后，使用` Locale`对象中的`setLanguage(locale)`方法，将当前的`Locale`对象中的语言设置为指定的语言。

接下来，使用`Scanner`对象中的` askQuestion()`方法，向用户询问问题。如果用户提供了回答，则使用从用户输入中获得的` Locale`对象中的语言来决定要问哪个问题。如果用户没有回答或者提供了无效的回答，则继续问下一个问题。

最后，使用`Scanner`对象中的`printTree()`方法，将游戏决策数据打印为树形结构，以便于调试和可视化游戏状态。


```
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;
import java.util.stream.Collectors;

/**
 * ANIMAL
 * <p>
 * Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
 * The original BASIC program uses an array to maintain the questions and answers and to decide which question to
 * ask next. Updated this Java implementation to use a tree instead of the earlier faulty one based on a list (thanks @patimen).
 *
 * Bonus option: TREE --> prints the game decision data as a tree to visualize/debug the state of the game
 */
```

This appears to be a JavaScript implementation of a simple question tree, where each node is a QuestionNode, AnimalNode, or other custom node. The QuestionNode has a question, trueAnswer, and falseAnswer, while the AnimalNode has an animal.

The toString() method is used for each node to print it in a more human-readable format.

It appears that the buffer is being constructed by appending the toString() method of each node to it.

There is also a function called print which is using the println() function to print the node's information.

It's worth noting that this appears to be a simple example, and it does not handle some edge cases, such as when the tree is empty or the node is a root node without any children.


```
public class Animal {

    public static void main(String[] args) {
        printIntro();
        Scanner scan = new Scanner(System.in);

        Node root = new QuestionNode("DOES IT SWIM",
                new AnimalNode("FISH"), new AnimalNode("BIRD"));

        boolean stopGame = false;
        while (!stopGame) {
            String choice = readMainChoice(scan);
            switch (choice) {
                case "TREE":
                    printTree(root);
                    break;
                case "LIST":
                    printKnownAnimals(root);
                    break;
                case "Q":
                case "QUIT":
                    stopGame = true;
                    break;
                default:
                    if (choice.toUpperCase(Locale.ROOT).startsWith("Y")) {
                        Node current = root; //where we are in the question tree
                        Node previous; //keep track of parent of current in order to place new questions later on.

                        while (current instanceof QuestionNode) {
                            var currentQuestion = (QuestionNode) current;
                            var reply = askQuestionAndGetReply(currentQuestion, scan);

                            previous = current;
                            current = reply ? currentQuestion.getTrueAnswer() : currentQuestion.getFalseAnswer();
                            if (current instanceof AnimalNode) {
                                //We have reached a animal node, so offer it as the guess
                                var currentAnimal = (AnimalNode) current;
                                System.out.printf("IS IT A %s ? ", currentAnimal.getAnimal());
                                var animalGuessResponse = readYesOrNo(scan);
                                if (animalGuessResponse) {
                                    //we guessed right! end this round
                                    System.out.println("WHY NOT TRY ANOTHER ANIMAL?");
                                } else {
                                    //we guessed wrong :(, ask for feedback
                                    //cast previous to QuestionNode since we know at this point that it is not a leaf node
                                    askForInformationAndSave(scan, currentAnimal, (QuestionNode) previous, reply);
                                }
                            }
                        }
                    }
            }
        }
    }

    /**
     * Prompt for information about the animal we got wrong
     * @param current The animal that we guessed wrong
     * @param previous The root of current
     * @param previousToCurrentDecisionChoice Whether it was a Y or N answer that got us here. true = Y, false = N
     */
    private static void askForInformationAndSave(Scanner scan, AnimalNode current, QuestionNode previous, boolean previousToCurrentDecisionChoice) {
        //Failed to get it right and ran out of questions
        //Let's ask the user for the new information
        System.out.print("THE ANIMAL YOU WERE THINKING OF WAS A ? ");
        String animal = scan.nextLine();
        System.out.printf("PLEASE TYPE IN A QUESTION THAT WOULD DISTINGUISH A %s FROM A %s ? ", animal, current.getAnimal());
        String newQuestion = scan.nextLine();
        System.out.printf("FOR A %s THE ANSWER WOULD BE ? ", animal);
        boolean newAnswer = readYesOrNo(scan);
        //Add it to our question store
        addNewAnimal(current, previous, animal, newQuestion, newAnswer, previousToCurrentDecisionChoice);
    }

    private static void addNewAnimal(Node current,
                                     QuestionNode previous,
                                     String animal,
                                     String newQuestion,
                                     boolean newAnswer,
                                     boolean previousToCurrentDecisionChoice) {
        var animalNode = new AnimalNode(animal);
        var questionNode = new QuestionNode(newQuestion,
                newAnswer ? animalNode : current,
                !newAnswer ? animalNode : current);

        if (previous != null) {
            if (previousToCurrentDecisionChoice) {
                previous.setTrueAnswer(questionNode);
            } else {
                previous.setFalseAnswer(questionNode);
            }
        }
    }

    private static boolean askQuestionAndGetReply(QuestionNode questionNode, Scanner scanner) {
        System.out.printf("%s ? ", questionNode.question);
        return readYesOrNo(scanner);
    }

    private static boolean readYesOrNo(Scanner scanner) {
        boolean validAnswer = false;
        Boolean choseAnswer = null;
        while (!validAnswer) {
            String answer = scanner.nextLine();
            if (answer.toUpperCase(Locale.ROOT).startsWith("Y")) {
                validAnswer = true;
                choseAnswer = true;
            } else if (answer.toUpperCase(Locale.ROOT).startsWith("N")) {
                validAnswer = true;
                choseAnswer = false;
            }
        }
        return choseAnswer;
    }

    private static void printKnownAnimals(Node root) {
        System.out.println("\nANIMALS I ALREADY KNOW ARE:");

        List<AnimalNode> leafNodes = collectLeafNodes(root);
        String allAnimalsString = leafNodes.stream().map(AnimalNode::getAnimal).collect(Collectors.joining("\t\t"));

        System.out.println(allAnimalsString);
    }

    //Traverse the tree and collect all the leaf nodes, which basically have all the animals.
    private static List<AnimalNode> collectLeafNodes(Node root) {
        List<AnimalNode> collectedNodes = new ArrayList<>();
        if (root instanceof AnimalNode) {
            collectedNodes.add((AnimalNode) root);
        } else {
            var q = (QuestionNode) root;
            collectedNodes.addAll(collectLeafNodes(q.getTrueAnswer()));
            collectedNodes.addAll(collectLeafNodes(q.getFalseAnswer()));
        }
        return collectedNodes;
    }

    private static String readMainChoice(Scanner scan) {
        System.out.print("ARE YOU THINKING OF AN ANIMAL ? ");
        return scan.nextLine();
    }

    private static void printIntro() {
        System.out.println("                                ANIMAL");
        System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println("\n\n");
        System.out.println("PLAY 'GUESS THE ANIMAL'");
        System.out.println("\n");
        System.out.println("THINK OF AN ANIMAL AND THE COMPUTER WILL TRY TO GUESS IT.");
    }

    //Based on https://stackoverflow.com/a/8948691/74057
    private static void printTree(Node root) {
        StringBuilder buffer = new StringBuilder(50);
        print(root, buffer, "", "");
        System.out.println(buffer);
    }

    private static void print(Node root, StringBuilder buffer, String prefix, String childrenPrefix) {
        buffer.append(prefix);
        buffer.append(root.toString());
        buffer.append('\n');

        if (root instanceof QuestionNode) {
            var questionNode = (QuestionNode) root;
            print(questionNode.getTrueAnswer(), buffer, childrenPrefix + "├─Y─ ", childrenPrefix + "│   ");
            print(questionNode.getFalseAnswer(), buffer, childrenPrefix + "└─N─ ", childrenPrefix + "    ");
        }
    }


    /**
     * Base interface for all nodes in our question tree
     */
    private interface Node {
    }

    private static class QuestionNode implements Node {
        private final String question;
        private Node trueAnswer;
        private Node falseAnswer;

        public QuestionNode(String question, Node trueAnswer, Node falseAnswer) {
            this.question = question;
            this.trueAnswer = trueAnswer;
            this.falseAnswer = falseAnswer;
        }

        public String getQuestion() {
            return question;
        }

        public Node getTrueAnswer() {
            return trueAnswer;
        }

        public void setTrueAnswer(Node trueAnswer) {
            this.trueAnswer = trueAnswer;
        }

        public Node getFalseAnswer() {
            return falseAnswer;
        }

        public void setFalseAnswer(Node falseAnswer) {
            this.falseAnswer = falseAnswer;
        }

        @Override
        public String toString() {
            return "Question{'" + question + "'}";
        }
    }

    private static class AnimalNode implements Node {
        private final String animal;

        public AnimalNode(String animal) {
            this.animal = animal;
        }

        public String getAnimal() {
            return animal;
        }

        @Override
        public String toString() {
            return "Animal{'" + animal + "'}";
        }
    }

}

```

# `03_Animal/javascript/animal.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是接收一个字符串参数，将其添加到页面上一个叫做 "output" 的元素中。这个元素是一个 `<textarea>` 元素，可以用来输入和输出文本内容。

`input` 函数的作用是接收一个字符串参数，并返回一个Promise对象。它通过创建一个 `<input>` 元素来获取用户的输入，然后将其添加到页面上一个叫做 "output" 的元素中，并将其设置为可见状态。接着，它将监听 `keydown` 事件，当用户按下了键盘上的13时，它会获取用户输入的字符串，并将其添加到 "output" 元素中，并输出该字符串。


```
// ANIMAL
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

这段代码定义了一个名为 `tab` 的函数，它会接收一个参数 `space`，该参数代表在空格中要填写的字符数。函数内部创建了一个空字符串 `str`，然后使用 while 循环从 `space` 开始数，每次填写一个空格，直到 `space` 自减为 0。这样，在循环结束后，字符串 `str` 中包含了从左往右数从 0 到 `space` 之间的所有空白字符。

接下来，函数返回字符串 `str`。

在函数外部，分别调用 `tab(32)`、`tab(15)` 和 `tab()` 三个不同的函数，并将其返回值赋给变量 `str`. 然后，在同一行的末尾添加了四个空行。

接着，在下一行输出了一行字符串 `"PLAY 'GUESS THE ANIMAL'"`。

再次在同一行的末尾输出了一行字符串 `"ANIMAL"`，以及另外一行字符串 `"CREATIVE COMPUTING"`，以及另外一行字符串 `"MORRISTOWN, NEW JERSEY"`。

最后，在同一行的末尾添加了四个空行。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

print(tab(32) + "ANIMAL\n");
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
print("\n");
print("\n");
print("\n");
print("PLAY 'GUESS THE ANIMAL'\n");
print("\n");
```

这段代码是一个简单的 JavaScript 程序，它的主要目的是输出一段文本，并通过用户输入猜测一个动物的名字，然后输出该动物名称在计算机中的猜测结果。

具体来说，代码首先定义了 7 个变量：k、n、str、q、z、c、t，然后定义了一个名为 animals 的字符串数组，包含了一些动物名称。接着，代码通过 var 关键字定义了 k、n、str、q、z、c、t 7 个变量，并分别给它们赋值为菜肴名称。

接下来，代码使用 var 关键字定义了一个名为 c 的变量，并给它赋值为变量 t 的值，即 c = t。这样，代码就可以通过调用 t.guess() 来获取 c 的值了。

最后，代码通过 print 函数输出了两行文本，第一行是公司常见的问候语，第二行是告诉用户输入一个动物名称，然后猜测该动物在计算机中的名字。用户输入动物名称后，程序会尝试猜测并输出该动物名称在计算机中的猜测结果。


```
print("THINK OF AN ANIMAL AND THE COMPUTER WILL TRY TO GUESS IT.\n");
print("\n");

var k;
var n;
var str;
var q;
var z;
var c;
var t;

var animals = [
               "\\QDOES IT SWIM\\Y1\\N2\\",
               "\\AFISH",
               "\\ABIRD",
               ];

```

这段代码定义了一个名为 show_animals 的函数，其作用是输出所有已知动物的名称。函数内部首先定义了一个变量 x，用于打印字符串中的行数。接着，通过循环遍历 animals 数组中的每个元素，判断其前两个元素是否为 "\A" 标识，如果是，则说明这个动物已经在控制台上输出过，进入了一个循环，其中用 " " 字符填充空格，直到该元素后边的所有字符串长度之和等于 15 与当前循环行数 $i$ 的乘积，然后将当前循环行数 $i$ 加 1，并将字符串中的所有元素复制到 str 变量中。循环结束后，如果当前循环行数 $i$ 的字符串长度已经达到了 4，则将循环结束并输出 str，然后将 str 字符串清空。最后，如果 str 变量中仍然有字符串，则输出 str，否则不再输出。


```
n = animals.length;

function show_animals() {
    var x;

    print("\n");
    print("ANIMALS I ALREADY KNOW ARE:\n");
    str = "";
    x = 0;
    for (var i = 0; i < n; i++) {
        if (animals[i].substr(0, 2) == "\\A") {
            while (str.length < 15 * x)
                str += " ";
            for (var z = 2; z < animals[i].length; z++) {
                if (animals[i][z] == "\\")
                    break;
                str += animals[i][z];
            }
            x++;
            if (x == 4) {
                x = 0;
                print(str + "\n");
                str = "";
            }
        }
    }
    if (str != "")
        print(str + "\n");
}

```

This is a Python program that allows the user to choose from a list of animals and


```
// Main control section
async function main()
{
    while (1) {
        while (1) {
            print("ARE YOU THINKING OF AN ANIMAL");
            str = await input();
            if (str == "LIST")
                show_animals();
            if (str[0] == "Y")
                break;
        }

        k = 0;
        do {
            // Subroutine to print questions
            q = animals[k];
            while (1) {
                str = "";
                for (z = 2; z < q.length; z++) {
                    if (q[z] == "\\")
                        break;
                    str += q[z];
                }
                print(str);
                c = await input();
                if (c[0] == "Y" || c[0] == "N")
                    break;
            }
            t = "\\" + c[0];
            x = q.indexOf(t);
            k = parseInt(q.substr(x + 2));
        } while (animals[k].substr(0,2) == "\\Q") ;

        print("IS IT A " + animals[k].substr(2));
        a = await input();
        if (a[0] == "Y") {
            print("WHY NOT TRY ANOTHER ANIMAL?\n");
            continue;
        }
        print("THE ANIMAL YOU WERE THINKING OF WAS A ");
        v = await input();
        print("PLEASE TYPE IN A QUESTION THAT WOULD DISTINGUISH A\n");
        print(v + " FROM A " + animals[k].substr(2) + "\n");
        x = await input();
        while (1) {
            print("FOR A " + v + " THE ANSWER WOULD BE ");
            a = await input();
            a = a.substr(0, 1);
            if (a == "Y" || a == "N")
                break;
        }
        if (a == "Y")
            b = "N";
        if (a == "N")
            b = "Y";
        z1 = animals.length;
        animals[z1] = animals[k];
        animals[z1 + 1] = "\\A" + v;
        animals[k] = "\\Q" + x + "\\" + a + (z1 + 1) + "\\" + b + z1 + "\\";
    }
}

```

这道题目要求解释以下代码的作用，不要输出源代码。从代码中可以看出，只有一个名为main的函数，而main函数内部没有任何代码，因此整个程序的作用就是没有其他代码的话，程序是不会运行的。


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


# `03_Animal/python/animal.py`

It sounds like you're playing a game where you answer questions about an animal and the computer tries to guess the name of the animal. You can also ask the computer questions to help the computer learn new animals. The computer will ask you questions to differentiate the animal it guessed from the one you were thinking of.

If you're not sure what the computer is asking, you can tell it. The computer will tell you all the animals it knows so far if you reply with "LIST".

The program starts originally by knowing only FISH and BIRD. As you build up a file of animals, you should use broad, general questions first and then narrow down to more specific ones with later animals. For example, if an elephant was to be your first animal, the computer would ask for a question to distinguish an elephant from a bird.


```
"""
Animal

From: Basic computer Games(1978)

   Unlike other computer games in which the computer
  picks a number or letter and you must guess what it is,
  in this game you think of an animal and the computer asks
  you questions and tries to guess the name of your animal.
  If the computer guesses incorrectly, it will ask you for a
  question that differentiates the animal it guessed
  from the one you were thinking of. In this way the
  computer "learns" new animals. Questions to differentiate
  new animals should be input without a question mark.
   This version of the game does not have a SAVE feature.
  If your sistem allows, you may modify the program to
  save array A$, then reload the array  when you want
  to play the game again. This way you can save what the
  computer learns over a series of games.
   At any time if you reply 'LIST' to the question "ARE YOU
  THINKING OF AN ANIMAL", the computer will tell you all the
  animals it knows so far.
   The program starts originally by knowing only FISH and BIRD.
  As you build up a file of animals you should use broad,
  general questions first and then narrow down to more specific
  ones with later animals. For example, If an elephant was to be
  your first animal, the computer would ask for a question to distinguish
  an elephant from a bird. Naturally there are hundreds of possibilities,
  however, if you plan to build a large file of animals a good question
  would be "IS IT A MAMAL".
   This program can be easily modified to deal with categories of
  things other than animals by simply modifying the initial data
  in Line 530 and the dialogue references to animal in Lines 10,
  40, 50, 130, 230, 240 and 600. In an educational environment, this
  would be a valuable program to teach the distinguishing chacteristics
  of many classes of objects -- rock formations, geography, marine life,
  cell structures, etc.
   Originally developed by Arthur Luehrmann at Dartmouth College,
  Animal was subsequently shortened and modified by Nathan Teichholtz at
  DEC and Steve North at Creative Computing
```



这段代码定义了一个名为 `Node` 的类，代表二叉树中的一个节点。每个节点包含一个文本字符串 `text`，以及一个指向 `Node` 对象的引用 `yes_node` 和一个指向 `Node` 对象的引用 `no_node`。

在 `__init__` 方法中，我们创建了一个新的 `Node` 对象，将 `text`、`yes_node` 和 `no_node` 都设置为 `None`。然后，我们实现了 `update_node` 方法，用于更新节点的文本和答案。

`update_node` 方法接收一个新问题 `new_question`、新的答案 `answer_new_ques` 和新的动物 `new_animal`。我们根据 `new_question` 的值更新 `text`、如果是 "y"，则将 `self.yes_node` 设置为新的动物，同时将 `self.no_node` 设置为新的动物；如果是 "n"，则将 `self.yes_node` 设置为新的动物，同时将 `self.no_node` 设置为新的动物。这样，当 `self.text` 和 `self.yes_node` 或 `self.no_node` 有一个发生了变化，我们就会通知 `Node` 对象来更新它。

`is_leaf` 方法用于检查一个节点是否是叶子节点。如果 `self.yes_node` 和 `self.no_node` 都是 `None`，则该节点是叶子节点，否则不是。


```
"""

from typing import Optional


class Node:
    """
    Node of the binary tree of questions.
    """

    def __init__(
        self, text: str, yes_node: Optional["Node"], no_node: Optional["Node"]
    ):
        # the nodes that are leafs have as text the animal's name, otherwise
        # a yes/no question
        self.text = text
        self.yes_node = yes_node
        self.no_node = no_node

    def update_node(
        self, new_question: str, answer_new_ques: str, new_animal: str
    ) -> None:
        # update the leaf with a question
        old_animal = self.text
        # we replace the animal with a new question
        self.text = new_question

        if answer_new_ques == "y":
            self.yes_node = Node(new_animal, None, None)
            self.no_node = Node(old_animal, None, None)
        else:
            self.yes_node = Node(old_animal, None, None)
            self.no_node = Node(new_animal, None, None)

    # the leafs have as children None
    def is_leaf(self) -> bool:
        return self.yes_node is None and self.no_node is None


```



这段代码定义了一个名为 `list_known_animals` 的函数，它递归地遍历一个二叉树中的节点。函数接收一个名为 `root_node` 的选项对象作为参数，表示要遍历的树根节点。函数返回 None，表示在遍历过程中没有发现任何节点。

函数首先检查 `root_node` 是否为空节点，如果是，函数返回并结束。如果不是，函数递归地遍历 `root_node` 的所有子节点，直到到达叶子节点。在叶子节点处，函数打印该节点的文本并返回。

如果 `root_node` 是 `yes_node` 节点，函数递归地遍历 `root_node.yes_node` 子节点，而不是 `root_node` 本身。递归结束后，函数返回。

如果 `root_node` 是 `no_node` 节点，函数递归地遍历 `root_node.no_node` 子节点，而不是 `root_node` 本身。递归结束后，函数返回。

这段代码的主要目的是打印出二叉树中的所有已知动物(如果有的话)，并输出它们。


```
def list_known_animals(root_node: Optional[Node]) -> None:
    """Traversing the tree by recursion until we reach the leafs."""
    if root_node is None:
        return

    if root_node.is_leaf():
        print(root_node.text, end=" " * 11)
        return

    if root_node.yes_node:
        list_known_animals(root_node.yes_node)

    if root_node.no_node:
        list_known_animals(root_node.no_node)


```

这段代码定义了一个名为 `parse_input` 的函数，用于解析用户输入的可读字符串并返回其含义。该函数只接受 "是" 或 "否" 类型的输入，并支持列表操作。

函数首先定义了一个名为 `token` 的字符串变量，用于存储用户输入的可读字符串。然后使用一个 while 循环来读取用户输入的字符串，并将其存储在 `token` 变量中。

接下来，函数使用 `if` 语句检查用户输入是否包含 "是" 或 "否"，如果是，则调用一个名为 `list_known_animals` 的函数，该函数接收一个 `Node` 类型的参数 `root_node`，用于存储已知的动物列表。调用 `list_known_animals` 函数时，如果它接收到了一个 "是"，则使用打印语句输出已知的动物列表，并在最后加上一个换行符。

如果输入不是空字符串，则函数将清理 `token` 并重新读取用户输入，以准备下一个循环。如果循环完成了而 `token` 仍然为空，则函数返回一个空字符串。

最后，函数返回 `token`，即上面存储的可读字符串，用于表示用户输入的动物列表。


```
def parse_input(message: str, check_list: bool, root_node: Optional[Node]) -> str:
    """only accepts yes or no inputs and recognizes list operation"""
    token = ""
    while token not in ["y", "n"]:
        inp = input(message)

        if check_list and inp.lower() == "list":
            print("Animals I already know are:")
            list_known_animals(root_node)
            print("\n")

        if len(inp) > 0:
            token = inp[0].lower()
        else:
            token = ""

    return token


```

This is a simple text-based game where the user has to guess an animal by asking yes or no questions.

The animal is either a fish or a plant depending on whether it can swim or not.

The `Node` class is used to store the text and children of a node in the tree.

The `parse_input` function is used to parse the user's input, return a float or a boolean, and update the node in the tree accordingly.

The game has a main loop that prints the introduction of the game, and then keeps asking the user until they decide to stop.

In the end, the root node of the tree contains the final answer to the user's question.


```
def avoid_void_input(message: str) -> str:
    answer = ""
    while answer == "":
        answer = input(message)
    return answer


def print_intro() -> None:
    print(" " * 32 + "Animal")
    print(" " * 15 + "Creative Computing Morristown, New Jersey\n")
    print("Play ´Guess the Animal´")
    print("Think of an animal and the computer will try to guess it.\n")


def main() -> None:
    # Initial tree
    yes_child = Node("Fish", None, None)
    no_child = Node("Bird", None, None)
    root = Node("Does it swim?", yes_child, no_child)

    # Main loop of game
    print_intro()
    keep_playing = parse_input("Are you thinking of an animal? ", True, root) == "y"
    while keep_playing:
        keep_asking = True
        # Start traversing the tree by the root
        actual_node: Node = root

        while keep_asking:

            if not actual_node.is_leaf():

                # we have to keep asking i.e. traversing nodes
                answer = parse_input(actual_node.text, False, None)

                # As this is an inner node, both children are not None
                if answer == "y":
                    assert actual_node.yes_node is not None
                    actual_node = actual_node.yes_node
                else:
                    assert actual_node.no_node is not None
                    actual_node = actual_node.no_node
            else:
                # we have reached a possible answer
                answer = parse_input(f"Is it a {actual_node.text}? ", False, None)
                if answer == "n":
                    # add the new animal to the tree
                    new_animal = avoid_void_input(
                        "The animal you were thinking of was a ? "
                    )
                    new_question = avoid_void_input(
                        "Please type in a question that would distinguish a "
                        f"{new_animal} from a {actual_node.text}: "
                    )
                    answer_new_question = parse_input(
                        f"for a {new_animal} the answer would be: ", False, None
                    )

                    actual_node.update_node(
                        new_question + "?", answer_new_question, new_animal
                    )

                else:
                    print("Why not try another animal?")

                keep_asking = False

        keep_playing = parse_input("Are you thinking of an animal? ", True, root) == "y"


```

这段代码是一个用于存储问题和动物的数据结构，是一个二叉树。每个非根节点（即一个非叶子节点）存储一个问题，而叶子节点（即一个叶子节点）存储一个动物。

该程序的主要目的是提供一个数据库来存储问题，可以记录问题的历史。它还可以通过修改初始数据来修改问题，还可以根据用户的提问来做出猜测，通过调整树的结构来提高猜测的准确性。


```
########################################################
# Porting Notes
#
#   The data structure used for storing questions and
#   animals is a binary tree where each non-leaf node
#   has a question, while the leafs store the animals.
#
#   As the original program, this program doesn't store
#   old questions and animals. A good modification would
#   be to add a database to store the tree.
#    Also as the original program, this one can be easily
#   modified to not only make guesses about animals, by
#   modyfing the initial data of the tree, the questions
#   that are asked to the user and the initial message
#   function  (Lines 120 to 130, 135, 158, 160, 168, 173)

```

这段代码是一个 Python 程序，主要作用是定义一个函数 main()，并在程序运行时始终运行这个函数。

具体来说，这段代码包含以下几个部分：

1. `#` 开头的 non-printing抬头，表示这是一个被打印但不包含输出内容的笔记。

2. `if __name__ == "__main__":` 是一个判定语句，用于检查当前程序是否作为主程序运行。如果这个条件为真，程序将执行下面的代码。

3. `main()` 是一个函数定义，表示程序的入口点。这个函数可以在程序的其他部分中被调用，也可以被其他程序或代码块导入。

4. 上述代码下面的三个部分是函数体，其中 `main()` 是函数名，占位符 `()` 内的内容被替换成了函数的实际参数和体。这部分代码定义了一个函数，可以被程序调用，也可以在其他程序或代码块中使用。


```
########################################################

if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by [Anton Kaiukov](https://github.com/batk0)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)

This takes some inspiration from the [C# port of Animal](https://github.com/zspitz/basic-computer-games/tree/main/03_Animal/csharp).

The `Game` class takes a console abstraction (`ConsoleAdapterBase`), which could also be used for different UIs, such as WinForms or a web page.
This solution also has an xUnit tests project.
Responses can be entered in any capitalization, but animals and the distinguishing question will be converted to uppercase.


### Awari

Awari is an ancient African game played with seven sticks and thirty-six stones or beans laid out as shown above. The board is divided into six compartments or pits on each side. In addition, there are two special home pits at the ends.

A move is made by taking all the beans from any (non-empty) pit on your own side. Starting from the pit to the right of this one, these beans are ‘sown’ one in each pit working around the board anticlockwise.

A turn consists of one or two moves. If the last bean of your move is sown in your own home you may take a second move.

If the last bean sown in a move lands in an empty pit, provided that the opposite pit is not empty, all the beans in the opposite pit, together with the last bean sown are ‘captured’ and moved to the player’s home.

When either side is empty, the game is finished. The player with the most beans in his home has won.

In the computer version, the board is printed as 14 numbers representing the 14 pits.

```
    3   3   3   3   3   3
0                           0
    3   3   3   3   3   3
```

The pits on your (lower) side are numbered 1-6 from left to right. The pits on my (the computer’s) side are numbered from my left (your right).

To make a move you type in the number of a pit. If the last bean lands in your home, the computer types ‘AGAIN?’ and then you type in your second move.

The computer’s move is typed, followed by a diagram of the board in its new state. The computer always offers you the first move. This is considered to be a slight advantage.

There is a learning mechanism in the program that causes the play of the computer to improve as it playes more games.

The original version of Awari is adopted from one originally written by Geoff Wyvill of Bradford, Yorkshire, England.

---

As published in Basic Computer Games (1978)
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=6)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=21)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)



# `04_Awari/csharp/Game.cs`

This is a Java class that represents a game of Tic Tac Toe where the player has to make moves to land on one of the cells, and the computer has to land on the other cells. The game has a score that is updated based on the outcome of each move. The move counting is also updated.

The `Score` class is used to keep track of the score. It calculates the score by taking the sum of the number of beans the player has landed on each move, taking away the number of beans the player landed on previously.

The `ScoreBestNextPlayerMove` class is used to get the best move for the computer. It takes the hypothetical beans array as input and returns the best move that the computer could make.

The `ScoreNextPlayerMove` class is used to calculate the score of the move. It takes the hypothetical beans array and the move number as input and returns the score based on the outcome of the move.

The `Beans` class is used to store the beans that have landed on each player's moves so far.

The `Notes` class is used to store the not won game moves. It is initialized with a list of four elements, each representing a move number.

The `Game` class is the main class that starts the game and calls the appropriate methods to update the game score and move counting.


```
namespace Awari;

public class Game
{
    public int[] PlayerPits => _beans[0..6];
    public int[] ComputerPits => _beans[7..13];
    public int PlayerHome => _beans[_playerHome];
    public int ComputerHome => _beans[_computerHome];

    private bool IsDone =>
        PlayerPits.All(b => b == 0) // if all the player's pits are empty
     || ComputerPits.All(b => b == 0); // or if all the computer's pits are empty

    public GameState State { get; private set; }

    public void Reset()
    {
        State = GameState.PlayerMove;

        Array.Fill(_beans, _initialPitValue);
        _beans[_playerHome] = 0;
        _beans[_computerHome] = 0;

        _moveCount = 0;
        _notWonGameMoves[^1] = 0;
    }

    public bool IsLegalPlayerMove(int move) =>
        move is > 0 and < 7
     && _beans[move - 1] > 0; // arrays are zero-based, but moves are one-based

    public void PlayerMove(int move) => MoveAndRegister(move - 1, _playerHome);

    public List<int> ComputerTurn()
    {
        // keep a list of moves made by the computer in a single turn (1 or 2)
        List<int> moves = new();

        moves.Add(ComputerMove()); // ComputerMove() returns the move made

        // only if a second move is possible, do it
        if (State == GameState.ComputerSecondMove)
            moves.Add(ComputerMove());

        return moves;
    }

    public GameOutcome GetOutcome()
    {
        if (State != GameState.Done)
            throw new InvalidOperationException("Game is not yet done.");

        int difference = _beans[_playerHome] - _beans[_computerHome];
        var winner = difference switch
        {
            < 0 => GameWinner.Computer,
            0 => GameWinner.Draw,
            > 0 => GameWinner.Player,
        };

        return new GameOutcome(winner, Math.Abs(difference));
    }

    private void MoveAndRegister(int pit, int homePosition)
    {
        int lastMovedBean = Move(_beans, pit, homePosition);

        // encode moves by player and computer into a 'base 6' number
        // e.g. if the player moves 5, the computer moves 2, and the player moves 4,
        // that would be encoded as ((5 * 6) * 6) + (2 * 6) + 4 = 196
        if (pit > 6) pit -= 7;
        _moveCount++;
        if (_moveCount < 9)
            _notWonGameMoves[^1] = _notWonGameMoves[^1] * 6 + pit;

        // determine next state based on current state, whether the game's done, and whether the last moved bean moved
        // into the player's home position
        State = (State, IsDone, lastMovedBean == homePosition) switch
        {
            (_, true, _) => GameState.Done,
            (GameState.PlayerMove, _, true) => GameState.PlayerSecondMove,
            (GameState.PlayerMove, _, false) => GameState.ComputerMove,
            (GameState.PlayerSecondMove, _, _) => GameState.ComputerMove,
            (GameState.ComputerMove, _, true) => GameState.ComputerSecondMove,
            (GameState.ComputerMove, _, false) => GameState.PlayerMove,
            (GameState.ComputerSecondMove, _, _) => GameState.PlayerMove,
            _ => throw new InvalidOperationException("Unexpected game state"),
        };

        // do some bookkeeping if the game is done, but not won by the computer
        if (State == GameState.Done
         && _beans[_playerHome] >= _beans[_computerHome])
            // add an entry for the next game
            _notWonGameMoves.Add(0);
    }

    private static int Move(int[] beans, int pit, int homePosition)
    {
        int beansToMove = beans[pit];
        beans[pit] = 0;

        // add the beans that were in the pit to other pits, moving clockwise around the board
        for (; beansToMove >= 1; beansToMove--)
        {
            // wrap around if pit exceeds 13
            pit = (pit + 1) % 14;

            beans[pit]++;
        }

        if (beans[pit] == 1 // if the last bean was sown in an empty pit
         && pit is not _playerHome and not _computerHome // which is not either player's home
         && beans[12 - pit] != 0) // and the pit opposite is not empty
        {
            // move the last pit sown and the _beans in the pit opposite to the player's home
            beans[homePosition] = beans[homePosition] + beans[12 - pit] + 1;
            beans[pit] = 0;
            beans[12 - pit] = 0;
        }

        return pit;
    }

    private int ComputerMove()
    {
        int move = DetermineComputerMove();
        MoveAndRegister(move, homePosition: _computerHome);

        // the result is only used to return it to the application, so translate it from an array index (between 7 and
        // 12) to a pit number (between 1 and 6)
        return move - 6;
    }

    private int DetermineComputerMove()
    {
        int bestScore = -99;
        int move = 0;

        // for each of the computer's possible moves, simulate them to calculate a score and pick the best one
        for (int j = 7; j < 13; j++)
        {
            if (_beans[j] <= 0)
                continue;

            int score = SimulateMove(j);

            if (score >= bestScore)
            {
                move = j;
                bestScore = score;
            }
        }

        return move;
    }

    private int SimulateMove(int move)
    {
        // make a copy of the current state, so we can safely mess with it
        var hypotheticalBeans = new int[14];
        _beans.CopyTo(hypotheticalBeans, 0);

        // simulate the move in our copy
        Move(hypotheticalBeans, move, homePosition: _computerHome);

        // determine the 'best' move the player could make after this (best for them, not for the computer)
        int score = ScoreBestNextPlayerMove(hypotheticalBeans);

        // score this move by calculating how far ahead we would be after the move, and subtracting the player's next
        // move score
        score = hypotheticalBeans[_computerHome] - hypotheticalBeans[_playerHome] - score;

        // have we seen the current set of moves before in a drawn/lost game? after 8 moves it's unlikely we'll find any
        // matches, since games will have diverged. also we don't have space to store that many moves.
        if (_moveCount < 8)
        {
            int translatedMove = move - 7;  // translate from 7 through 12 to 0 through 5

            // if the first two moves in this game were 1 and 2, and this hypothetical third move would be a 3,
            // movesSoFar would be (1 * 36) + (2 * 6) + 3 = 51
            int movesSoFar = _notWonGameMoves[^1] * 6 + translatedMove;

            // since we store moves as a 'base 6' number, we need to divide stored moves by a power of 6
            // let's say we've a stored lost game where the moves were, in succession, 1 through 8, the value stored
            // would be:
            // 8 + (7 * 6) + (6 * 36) + (5 * 216) + (4 * 1296) + (3 * 7776) + (2 * 46656) + (1 * 279936) = 403106
            // to figure out the first three moves, we'd need to divide by 7776, resulting in 51.839...
            double divisor = Math.Pow(6.0, 7 - _moveCount);

            foreach (int previousGameMoves in _notWonGameMoves)
                // if this combination of moves so far ultimately resulted in a draw/loss, give it a lower score
                // note that this can happen multiple times
                if (movesSoFar == (int) (previousGameMoves / divisor + 0.1))
                    score -= 2;
        }

        return score;
    }

    private static int ScoreBestNextPlayerMove(int[] hypotheticalBeans)
    {
        int bestScore = 0;

        for (int i = 0; i < 6; i++)
        {
            if (hypotheticalBeans[i] <= 0)
                continue;

            int score = ScoreNextPlayerMove(hypotheticalBeans, i);

            if (score > bestScore)
                bestScore = score;
        }

        return bestScore;
    }

    private static int ScoreNextPlayerMove(int[] hypotheticalBeans, int move)
    {
        // figure out where the last bean will land
        int target = hypotheticalBeans[move] + move;
        int score = 0;

        // if it wraps around, that means the player is adding to his own pits, which is good
        if (target > 13)
        {
            // prevent overrunning the number of pits we have
            target %= 14;
            score = 1;
        }

        // if the player's move ends up in an empty pit, add the value of the pit on the opposite side to the score
        if (hypotheticalBeans[target] == 0 && target is not _playerHome and not _computerHome)
            score += hypotheticalBeans[12 - target];

        return score;
    }

    private const int _playerHome = 6;
    private const int _computerHome = 13;
    private const int _initialPitValue = 3;

    private readonly int[] _beans = new int[14];
    private readonly List<int> _notWonGameMoves = new() { 0 };    // not won means draw or lose
    private int _moveCount;
}

```



这段代码定义了两个枚举类型GameState和GameWinner，用于表示游戏的当前状态以及胜利情况。

GameState枚举类型定义了五种状态，分别为PlayerMove、PlayerSecondMove、ComputerMove、ComputerSecondMove和Done，分别表示游戏中的玩家移动、玩家移动两次、电脑移动、电脑移动两次以及游戏结束。

GameWinner枚举类型定义了三种胜利情况，分别为Player、Computer和Draw，表示游戏中的获胜方。

此代码并没有定义任何函数或方法，所以它只是一个枚举类型的定义，不会输出任何内容。


```
public enum GameState
{
    PlayerMove,
    PlayerSecondMove,
    ComputerMove,
    ComputerSecondMove,
    Done,
}

public enum GameWinner
{
    Player,
    Computer,
    Draw,
}

```

这段代码定义了一个名为`GameOutcome`的结构体类型，用于表示游戏的结果。该类型有两个成员变量，一个是`GameWinner`类型的变量`Winner`，另一个是整型变量`Difference`。

`public record struct GameOutcome`表示该结构体类型是公共的，可以在程序中任何地方使用。

`GameOutcome`类型的实例化方式如下：

``` 
GameOutcome gameResult = GameOutcome(
   GameWinner.GetMaxValue(0),
   0
);
```

这段代码创建了一个名为`gameResult`的`GameOutcome`实例，游戏的胜利者是`GetMaxValue(0)`函数的返回值，也就是0。`Difference`变量没有初始化，因此其值将为0。

需要注意的是，`GameOutcome`类型中的`GameWinner`和`Difference`成员变量是标记为`public`的，这意味着它们是可访问的，可以在任何地方被访问。


```
public record struct GameOutcome(GameWinner Winner, int Difference);

```