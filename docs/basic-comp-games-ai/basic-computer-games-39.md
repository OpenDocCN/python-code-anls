# BasicComputerGames源码解析 39

# `35_Even_Wins/javascript/gameofevenwins.js`

该代码是一个网页游戏中的BASIC代码，通过Oscar Toledo G.的转换后，以JavaScript形式呈现在网页上。

具体来说，该游戏的目的是让玩家猜一个8位二进制数的输出结果。玩家每输入一个数字，游戏会提示玩家输入的字符，直到输入正确的字符。当玩家输入数字13时，游戏会自动计算并显示该数字的输出结果，并在计算完成后将输入结果输出到页面上。

代码中定义了两个函数：`print()` 和 `input()`。`print()`函数的作用是打印字符串，将指定的字符串输出到页面上指定的元素中。`input()`函数的作用是获取用户输入的字符串，将其转换为8位二进制数，并返回该二进制数的输出结果。

整个游戏的逻辑可以在代码中找到，包括输入验证、计算输出结果等步骤。


```
// GAME OF EVEN WINS
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

This appears to be a program that plays the game of Connect Four. The program takes turns between the player and the computer, with the player having the initial choice to make a move. The computer's move is determined by the number 1 being inputted by the player, and the computer taking one chip from the board. If the computer's move is 1, it will print a message and take one chip from the board, otherwise it will print a message and take the number of chips specified by the player. The computer will continue to take one chip from the board until the game is over, either when the player wins or when the computer wins.



```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var r = [[], []];

// Main program
async function main()
{
    print(tab(28) + "GAME OF EVEN WINS\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("DO YOU WANT INSTRUCTIONS (YES OR NO)");
    str = await input();
    print("\n");
    if (str != "NO") {
        print("THE GAME IS PLAYED AS FOLLOWS:\n");
        print("\n");
        print("AT THE BEGINNING OF THE GAME, A RANDOM NUMBER OF CHIPS ARE\n");
        print("PLACED ON THE BOARD.  THE NUMBER OF CHIPS ALWAYS STARTS\n");
        print("AS AN ODD NUMBER.  ON EACH TURN, A PLAYER MUST TAKE ONE,\n");
        print("TWO, THREE, OR FOUR CHIPS.  THE WINNER IS THE PLAYER WHO\n");
        print("FINISHES WITH A TOTAL NUMBER OF CHIPS THAT IS EVEN.\n");
        print("THE COMPUTER STARTS OUT KNOWING ONLY THE RULES OF THE\n");
        print("GAME.  IT GRADUALLY LEARNS TO PLAY WELL.  IT SHOULD BE\n");
        print("DIFFICULT TO BEAT THE COMPUTER AFTER TWENTY GAMES IN A ROW.\n");
        print("TRY IT!!!!\n");
        print("\n");
        print("TO QUIT AT ANY TIME, TYPE A '0' AS YOUR MOVE.\n");
        print("\n");
    }
    l = 0;
    b = 0;
    for (i = 0; i <= 5; i++) {
        r[1][i] = 4;
        r[0][i] = 4;
    }
    while (1) {
        a = 0;
        b = 0;
        e = 0;
        l = 0;
        p = Math.floor((13 * Math.random() + 9) / 2) * 2 + 1;
        while (1) {
            if (p == 1) {
                print("THERE IS 1 CHIP ON THE BOARD.\n");
            } else {
                print("THERE ARE " + p + " CHIPS ON THE BOARD.\n");
            }
            e1 = e;
            l1 = l;
            e = a % 2;
            l = p % 6;
            if (r[e][l] < p) {
                m = r[e][l];
                if (m <= 0) {
                    m = 1;
                    b = 1;
                    break;
                }
                p -= m;
                if (m == 1)
                    print("COMPUTER TAKES 1 CHIP LEAVING " + p + "... YOUR MOVE");
                else
                    print("COMPUTER TAKES " + m + " CHIPS LEAVING " + p + "... YOUR MOVE");
                b += m;
                while (1) {
                    m = parseInt(await input());
                    if (m == 0)
                        break;
                    if (m < 1 || m > 4 || m > p) {
                        print(m + " IS AN ILLEGAL MOVE ... YOUR MOVE");
                    } else {
                        break;
                    }
                }
                if (m == 0)
                    break;
                if (m == p)
                    break;
                p -= m;
                a += m;
            } else {
                if (p == 1) {
                    print("COMPUTER TAKES 1 CHIP.\n");
                } else {
                    print("COMPUTER TAKES " + p + " CHIPS.\n");
                }
                r[e][l] = p;
                b += p;
                break;
            }
        }
        if (m == 0)
            break;
        if (b % 2 != 0) {
            print("GAME OVER ... YOU WIN!!!\n");
            print("\n");
            if (r[e][l] != 1) {
                r[e][l]--;
            } else if (r[e1][l1] != 1) {
                r[e1][l1]--;
            }
        } else {
            print("GAME OVER ... I WIN!!!\n");
            print("\n");
        }
    }
}

```

这道题目要求解释以下代码的作用，但是我不清楚你想要我解释哪个代码。如果你可以提供更多上下文或者更多的代码，我会尽力帮助你的。


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


# `35_Even_Wins/python/evenwins.py`

这段代码的作用是一个基于随机游戏的决定码，它不遵循原始 source，计算机从随机位置选择方块。

为了简化，使用了全局变量来存储游戏状态。

然而，这个代码不是很短，但它非常容易理解和修改。

在无限循环中使用“while True?”来简化代码，在某些地方使用“continue”来跳回循环顶部，使用“return”来退出函数。

虽然这种风格通常被认为是不好的，但在这种情况下，它简化了代码并使其更容易阅读。


```
"""
This version of evenwins.bas based on game decscription and does *not*
follow the source. The computer chooses marbles at random.

For simplicity, global variables are used to store the game state.
A good exercise would be to replace this with a class.
The code is not short, but hopefully it is easy for beginners to understand
and modify.

Infinite loops of the style "while True:" are used to simplify some of the
code. The "continue" keyword is used in a few places to jump back to the top
of the loop. The "return" keyword is also used to break out of functions.
This is generally considered poor style, but in this case it simplifies the
code and makes it easier to read (at least in my opinion). A good exercise
would be to remove these infinite loops, and uses of continue, to follow a
```

这段代码定义了一个名为 `MarbleCounts` 的类，使用了 `dataclasses` 装饰，可以是一个可读的 JSON 或 YAML 数据类。

在这个类中，定义了一个 `PlayerType` 类元组变量，包含两个字符串类型的成员变量，`human` 和 `computer`，它们都是枚举类型，定义了玩家的两种状态。

接着定义了一个 `MarbleCounts` 类，使用了 `@dataclass` 装饰，说明这个类也使用了 `dataclasses` 的装饰。在类中，定义了一个 `middle` 成员变量，类型为整数类型。

然后，在类中定义了一个 `human` 和 `computer` 成员变量，类型也都是整数类型。

最后，在类中添加了一个 `__init__` 方法，用于初始化这些成员变量的值，但是并没有传入任何参数。


```
more structured style.
"""


from dataclasses import dataclass
from typing import Literal, Tuple

PlayerType = Literal["human", "computer"]


@dataclass
class MarbleCounts:
    middle: int
    human: int
    computer: int


```

这段代码是一个函数，名为 `print_intro()`，它不会输出任何函数调用本身。该函数通过 `return None` 返回一个 `None` 对象，这意味着它不会执行任何具体的操作。

函数体内部包含以下几行代码：

```
print("Welcome to Even Wins!")
print("Based on evenwins.bas from Creative Computing")
print()
print("Even Wins is a two-person game. You start with")
print("27 marbles in the middle of the table.")
print()
print("Players alternate taking marbles from the middle.")
print("A player can take 1 to 4 marbles on their turn, and")
print("turns cannot be skipped. The game ends when there are")
print("no marbles left, and the winner is the one with an even")
print("number of marbles.")
print()
```

这些代码在函数内部输出了一些字符串和信息，包括游戏规则，玩家轮流从中间抽取一颗球，每个玩家每次只能抽取1到4颗球，总共抽取5颗球， game ends 的时候 所有的球都拿完了， 并且数字里面 是奇数。


```
def print_intro() -> None:
    print("Welcome to Even Wins!")
    print("Based on evenwins.bas from Creative Computing")
    print()
    print("Even Wins is a two-person game. You start with")
    print("27 marbles in the middle of the table.")
    print()
    print("Players alternate taking marbles from the middle.")
    print("A player can take 1 to 4 marbles on their turn, and")
    print("turns cannot be skipped. The game ends when there are")
    print("no marbles left, and the winner is the one with an even")
    print("number of marbles.")
    print()


```

这段代码定义了一个名为"marbles_str"的函数和一个名为"choose_first_player"的函数。

"marbles_str"函数接收一个整数参数"n"，并返回一个字符串。如果"n"等于1，函数返回字符串"1 marble"，否则返回字符串"{n} marbles"。函数的作用是输出不同棋子的数量。

"choose_first_player"函数是一个无限循环的函数，它接收一个字符串参数，用于表示玩家是选择自己还是选择电脑。函数首先提示玩家输入"y"来选择自己，然后再次提示玩家输入"y"或"n"来选择自己或电脑。如果玩家输入"n"，函数将发送一个字符串"Please enter 'y' if you want to play first, or 'n' if you want to play second."来重新询问玩家。然后函数将再次提示玩家输入"y"或"n"。这个函数的作用是让玩家输入选择自己的方式。


```
def marbles_str(n: int) -> str:
    if n == 1:
        return "1 marble"
    return f"{n} marbles"


def choose_first_player() -> PlayerType:
    while True:
        ans = input("Do you want to play first? (y/n) --> ")
        if ans == "y":
            return "human"
        elif ans == "n":
            return "computer"
        else:
            print()
            print('Please enter "y" if you want to play first,')
            print('or "n" if you want to play second.')
            print()


```

这段代码定义了两个函数：toggle_player 和 to_int。

toggle_player 函数接受一个参数 wh霸是字符串类型(PlayerType)，并返回一个字符串类型(PlayerType)。这个函数的作用是判断轮到谁来决定游戏，如果是人类玩家，则返回 "computer"，如果是计算机玩家，则返回 "human"。

to_int 函数接受一个参数 s，也是字符串类型，并返回一个元组类型(bool, int)。这个函数的作用是将字符串 s 转换成 if 为 True,int 类型的值，如果转换成功，则返回 (True, n)，否则返回 (False, 0)。

总的来说，这两个函数并没有做更多的的事情，只是定义了如何根据 wh 轮到谁来决定游戏，以及如何将字符串转换为 if 类型或 int 类型。


```
def toggle_player(whose_turn: PlayerType) -> PlayerType:
    if whose_turn == "human":
        return "computer"
    else:
        return "human"


def to_int(s: str) -> Tuple[bool, int]:
    """Convert a string s to an int, if possible."""
    try:
        n = int(s)
        return True, n
    except Exception:
        return False, 0


```

这是一个用Python编写的游戏策略。它定义了两个函数：print_board 和 human_turn。这两个函数用于打印游戏板和决定由玩家还是电脑选择要移动的棋子。

print_board函数接收一个名为marbles的MarbleCounts对象。这个函数的作用是打印游戏板的当前状态，并显示棋子数量和位置。它通过打印 "marbles in the middle: {marbles.middle}" 和 "    # marbles you have: {marbles.human}" 来显示游戏板的当前状态。然后分别显示电脑拥有的棋子和玩家拥有的棋子。最后，它通过打印 "It's your turn!" 来提醒玩家轮到他们移动棋子了。

human_turn函数接收一个名为marbles的MarbleCounts对象。这个函数用于决定由玩家还是电脑选择要移动的棋子。它首先要求玩家输入选择数量，然后检查输入是否为整数。如果为整数，那么它将打印 "It's your turn!"，然后提示玩家输入选择。如果输入不是整数，那么它将提示玩家输入一个整数。如果玩家输入的不是数字1-4，那么它将提示玩家输入正确的数字。然后，它将更新游戏板的状态，将选择数量从中间位置的棋子数中减去，并将选择数量添加到玩家的的人类棋子数中。最后，它将返回更新后的游戏板状态。


```
def print_board(marbles: MarbleCounts) -> None:
    print()
    print(f" marbles in the middle: {marbles.middle} " + marbles.middle * "*")
    print(f"    # marbles you have: {marbles.human}")
    print(f"# marbles computer has: {marbles.computer}")
    print()


def human_turn(marbles: MarbleCounts) -> None:
    """get number in range 1 to min(4, marbles.middle)"""
    max_choice = min(4, marbles.middle)
    print("It's your turn!")
    while True:
        s = input(f"Marbles to take? (1 - {max_choice}) --> ")
        ok, n = to_int(s)
        if not ok:
            print(f"\n  Please enter a whole number from 1 to {max_choice}\n")
            continue
        if n < 1:
            print("\n  You must take at least 1 marble!\n")
            continue
        if n > max_choice:
            print(f"\n  You can take at most {marbles_str(max_choice)}\n")
            continue
        print(f"\nOkay, taking {marbles_str(n)} ...")
        marbles.middle -= n
        marbles.human += n
        return


```

这段代码定义了两个函数：game_over 和 computer_turn。game_over 函数用于在玩家赢得游戏时输出一些信息，并显示一个游戏板。computer_turn 函数则用于计算机出牌，它会根据玩家出牌的个数和游戏板的大小来决定是否出牌，并在出牌时输出相关信息。

game_over 函数的作用是在玩家赢得游戏时输出一些信息，并显示一个游戏板。这里使用了 Python 的 print 函数来输出信息，print 函数会输出指定的字符和信息。这里通过 print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")来输出 "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"。print("!! All the marbles are taken: Game Over!") 则是在输出游戏板后，让玩家意识到游戏已经结束，并通知玩家已经赢得了游戏。print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") 是 print 函数用来输出这些信息的下一个输出，print_board(marbles) 用来输出游戏板。通过调用 game_over 函数，在游戏结束时输出一些信息并显示游戏板，最后让玩家知道已经赢得了游戏，或者通知玩家游戏已经结束。

computer_turn 函数则用于计算机出牌。它根据游戏板的大小和玩家出牌的个数来决定是否出牌，并在出牌时输出相关信息。这里使用了 Python 的 print 函数来输出信息，print 函数会输出指定的字符和信息。这里通过 print("It's the computer's turn ...")来通知玩家这是计算机的回合，然后 print_board(marbles) 来输出游戏板。然后通过 if 语句判断玩家出牌的个数是否为 2，如果是，就计算出牌数，然后输出相关信息。这里通过调用 computer_turn 函数，在出牌时根据玩家的出牌数计算出牌数，并通知玩家出牌了。


```
def game_over(marbles: MarbleCounts) -> None:
    print()
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!! All the marbles are taken: Game Over!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print()
    print_board(marbles)
    if marbles.human % 2 == 0:
        print("You are the winner! Congratulations!")
    else:
        print("The computer wins: all hail mighty silicon!")
    print()


def computer_turn(marbles: MarbleCounts) -> None:
    marbles_to_take = 0

    print("It's the computer's turn ...")
    r = marbles.middle - 6 * int(marbles.middle / 6)

    if int(marbles.human / 2) == marbles.human / 2:
        if r < 1.5 or r > 5.3:
            marbles_to_take = 1
        else:
            marbles_to_take = r - 1

    elif marbles.middle < 4.2:
        marbles_to_take = marbles.middle
    elif r > 3.4:
        if r < 4.7 or r > 3.5:
            marbles_to_take = 4
    else:
        marbles_to_take = r + 1

    print(f"Computer takes {marbles_str(marbles_to_take)} ...")
    marbles.middle -= marbles_to_take
    marbles.computer += marbles_to_take


```

这段代码是一个用Python实现的玩家游戏，其目的是让两个玩家轮流投掷骰子，然后根据骰子的点数来判断游戏是否结束。

首先，在游戏开始时，游戏设置了一个3x3的棋盘，其中人类玩家的两个球记为HUMAN，电脑玩家的两个球记为COMPUTER。然后，函数将打印出这个棋盘。

在游戏循环中，首先检查是否轮到人类玩家。如果是，函数将调用人类玩家的函数，该函数将转动棋盘并打印出当前局面。然后，将当前轮到的人类玩家的球指向另一个玩家，并将游戏状态更改为电脑玩家。

如果是电脑玩家，函数将调用电脑玩家的函数，该函数将转动棋盘并打印出当前局面。然后，将当前轮到的人类玩家的球指向另一个玩家，并将游戏状态更改为人类玩家。

如果尝试调用一个不存在的玩家，函数将 raise一个异常并退出游戏。


```
def play_game(whose_turn: PlayerType) -> None:
    marbles = MarbleCounts(middle=27, human=0, computer=0)
    print_board(marbles)

    while True:
        if marbles.middle == 0:
            game_over(marbles)
            return
        elif whose_turn == "human":
            human_turn(marbles)
            print_board(marbles)
            whose_turn = toggle_player(whose_turn)
        elif whose_turn == "computer":
            computer_turn(marbles)
            print_board(marbles)
            whose_turn = toggle_player(whose_turn)
        else:
            raise Exception(f"whose_turn={whose_turn} is not 'human' or 'computer'")


```

这段代码是一个 Python 语言的函数，名为 `main()`。它定义了一个函数体，其中包含一个空格，这表明该函数没有返回任何值。

以下是 `main()` 函数的作用：

1. 首先，函数调用一个名为 `print_intro()` 的函数，但没有传入参数。
2. 接着，函数进入一个无限循环，即 `while True:`。
3. 在循环的每次迭代中，函数调用一个名为 `choose_first_player()` 的函数，但这个函数没有传入参数。
4. 调用 `play_game()` 函数，这个函数没有接收任何参数，因此不会执行任何操作。
5. 接着，函数输出一个 `print()` 函数，并在其内部输出了一条消息。
6. 然后，函数要求用户再次输入是否要再次玩。如果用户输入 "y"，那么函数会再次输出消息并返回。
7. 如果用户输入 "n"，那么函数会结束，不再进一步执行。
8. 最后，函数返回一个空值，表示它没有返回任何值。


```
def main() -> None:
    print_intro()

    while True:
        whose_turn = choose_first_player()
        play_game(whose_turn)

        print()
        again = input("Would you like to play again? (y/n) --> ").lower()
        if again == "y":
            print("\nOk, let's play again ...\n")
        else:
            print("\nOk, thanks for playing ... goodbye!\n")
            return


```

这段代码是一个条件判断语句，它的作用是判断当前程序是否作为主程序运行。如果当前程序作为主程序运行，那么程序会执行if语句块内的内容，否则跳过if语句块。

具体来说，这段代码的作用是：如果当前程序作为主程序运行，那么程序会先执行if语句块内的内容，如果条件为真，那么执行if语句块内的代码，否则跳过if语句块。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Flip Flop

The object of this game is to change a row of ten X’s

```
X X X X X X X X X X
```

to a row of ten 0’s

```
0 0 0 0 0 0 0 0 0 0
```

by typing in a number corresponding to the position of an “X” in the line. On some numbers one position will change while on other numbers, two will change. For example, inputting a 3 may reverse the X and 0 in position 3, but it might possibly reverse some of other position too! You ought to be able to change all 10 in 12 or fewer moves. Can you figure out a good winning strategy?

To reset the line to all X’s (same game), type 0 (zero). To start a new game at any point, type 11.

The original author of this game was Micheal Kass of New Hyde Park, New York.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=63)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=78)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `36_Flip_Flop/csharp/FlipFlop.cs`



This is a program that will play the popular puzzle game, Puzzle Box, to a player. It uses a standard implementation of the game, where the objective of the game is to reach 100% game entropy and escape by solving the puzzle. The game starts with a standard 12x12 game board and uses the挡板 rule to separate players into two teams. 

The program has a while loop that runs the game until one of the players wins or the game is not completed within 12 moves. The while loop runs the game by making moves, checking for win conditions, and printing the game board. It also keeps track of the game entropy and number of moves remaining for each player.

The program also has a try again functionality that will reset the game board and start a new game.

The program also has a variable that prints the game result, after the game is over, the game entropy will be printed.


```
﻿// Flip Flop Game

PrintGameInfo();

bool startNewGame = true;

string[] board = new string[] { "X", "X", "X", "X", "X", "X", "X", "X", "X", "X" };

do
{
    int stepsCount = 0;
    int lastMove = -1;
    int moveIndex;
    int gameSum;
    double gameEntropyRate = Rnd();
    bool toPlay = false;
    bool setNewBoard = true;

    Print();
    Print("HERE IS THE STARTING LINE OF X'S.");
    Print();

    do
    {
        bool illegalEntry;
        bool equalToLastMove;

        if (setNewBoard)
        {
            PrintNewBoard();
            board = new string[] { "X", "X", "X", "X", "X", "X", "X", "X", "X", "X" };
            setNewBoard = false;
            toPlay = true;
        }

        stepsCount++;
        gameSum = 0;

        // Read User's move
        do
        {
            Write("INPUT THE NUMBER? ");
            var input = Console.ReadLine();
            illegalEntry = !int.TryParse(input, out moveIndex);

            if (illegalEntry || moveIndex > 11)
            {
                illegalEntry = true;
                Print("ILLEGAL ENTRY--TRY AGAIN.");
            }
        }
        while (illegalEntry);

        if (moveIndex == 11)
        {
            // Run new game, To start a new game at any point
            toPlay = false;
            stepsCount = 12;
            startNewGame = true;
        }


        if (moveIndex == 0)
        {
            // To reset the line to all X, same game
            setNewBoard = true;
            toPlay = false;
        }

        if (toPlay)
        {
            board[moveIndex - 1] = board[moveIndex - 1] == "O" ? "X" : "O";

            if (lastMove == moveIndex)
            {
                equalToLastMove = true;
            }
            else
            {
                equalToLastMove = false;
                lastMove = moveIndex;
            }

            do
            {
                moveIndex = equalToLastMove
                    ? GetMoveIndexWhenEqualeLastMove(moveIndex, gameEntropyRate)
                    : GetMoveIndex(moveIndex, gameEntropyRate);

                board[moveIndex] = board[moveIndex] == "O" ? "X" : "O";
            }
            while (lastMove == moveIndex && board[moveIndex] == "X");

            PrintGameBoard(board);

            foreach (var item in board)
            {
                if (item == "O")
                {
                    gameSum++;
                }
            }
        }
    }
    while (stepsCount < 12 && gameSum < 10);

    if (toPlay)
    {
        PrintGameResult(gameSum, stepsCount);

        Write("DO YOU WANT TO TRY ANOTHER PUZZLE ");

        var toContinue = Console.ReadLine();

        if (!string.IsNullOrEmpty(toContinue) && toContinue?.ToUpper()[0] == 'N')
        {
            startNewGame = false;
        }

        Print();
    }
}
```

这段代码是一个while循环，它将一直运行，直到你手动停止它。

对于每个循环迭代，它将调用一个名为Print的函数。Print函数是一个void类型的函数，它没有返回值，但是它使用了Console.WriteLine来输出字符串类型的参数str。

对于每个循环迭代，它将调用一个名为Write的函数。Write函数是一个void类型的函数，它也没有返回值，但是它使用了Console.Write来输出字符串类型的参数value。

对于每个循环迭代，它将调用一个名为Tab的函数。Tab是一个匿名函数，它返回了一个字符串，使用了New和字符串类型的@参数。

对于每个循环迭代，它将调用一个名为Rnd的函数。Rnd是一个匿名函数，它返回了一个double类型的随机数。

对于每个循环迭代，它将调用一个名为GetMoveIndex的函数。GetMoveIndex函数是一个int类型的函数，它返回了一个int类型的参数moveIndex，使用了Math.Tan和Math.Sin和一些数学计算。

对于每个循环迭代，它将计算一个名为rate的变量，使用了Math.Tan和Math.Sin和一些数学计算。rate变量是一个double类型的变量，它代表了游戏熵增加率。

对于每个循环迭代，它将使用Console.Write来输出rate变量和rate - Math.Floor(rate)的差值，使用了Math.Floor和Math.Tan计算。

最后，它将返回rate - Math.Floor(rate)。


```
while (startNewGame);

void Print(string str = "") => Console.WriteLine(str);

void Write(string value) => Console.Write(value);

string Tab(int pos) => new(' ', pos);

double Rnd() => new Random().NextDouble();

int GetMoveIndex(int moveIndex, double gameEntropyRate)
{
    double rate = Math.Tan(gameEntropyRate + moveIndex / gameEntropyRate - moveIndex) - Math.Sin(gameEntropyRate / moveIndex) + 336 * Math.Sin(8 * moveIndex);
    return Convert.ToInt32(Math.Floor(10 * (rate - Math.Floor(rate))));
}

```

这段代码定义了两个函数，`GetMoveIndexWhenEqualeLastMove()` 和 `PrintNewBoard()`。函数 `GetMoveIndexWhenEqualeLastMove()` 接收两个参数，一个是整数 `moveIndex`，另一个是双精度浮点数 `gameEntropyRate`，它返回一个整数，表示在游戏 entropy 高时移动棋子的位置。函数的核心计算公式是通过数学变换得到一个与棋子移动位置相关的双精度浮点数，然后再将其转换成整数并返回。

函数 `PrintNewBoard()` 是一个纯函数，它输出一个带有 11 个棋子的随机棋盘，用于开发者日后在测试中创建或打印棋盘。函数的主要目的是创建一个空白的棋盘，并将其输出到控制台。

函数 `PrintGameBoard()` 也是一个纯函数，它接收一个字符串数组 `board`，并输出一个带有 11 个棋子的随机棋盘。它遍历 `board` 数组，并输出每个棋子的位置。这个函数可以被用于创建或打印随机棋盘，以及生成其他棋类游戏的棋盘。


```
int GetMoveIndexWhenEqualeLastMove(int moveIndex, double gameEntropyRate)
{
    double rate = 0.592 * (1 / Math.Tan(gameEntropyRate / moveIndex + gameEntropyRate)) / Math.Sin(moveIndex * 2 + gameEntropyRate) - Math.Cos(moveIndex);
    return Convert.ToInt32(Math.Floor(10 * (rate - Math.Floor(rate))));
}

void PrintNewBoard()
{
    Print("1 2 3 4 5 6 7 8 9 10");
    Print("X X X X X X X X X X");
    Print();
}

void PrintGameBoard(string[] board)
{
    Print("1 2 3 4 5 6 7 8 9 10");

    foreach (var item in board)
    {
        Write($"{item} ");
    }

    Print();
    Print();
}

```

这段代码定义了两个函数，PrintGameResult和一个PrintGameInfo。

PrintGameResult函数的作用是打印游戏结果，包括游戏得分和猜数步骤。具体来说，如果猜对了，输出“VERY GOOD. YOU GUESSED IT IN JUST {stepsCount} GUESSES.”如果没有猜对，则输出“TRY HARDER NEXT TIME. IT TOOK YOU {stepsCount} GUESSES.”。

PrintGameInfo函数的作用是打印游戏信息，包括游戏背景和提示信息。具体来说，首先输出游戏的背景信息，然后输出“FLIPFLOP”。接下来，输出“THE OBJECT OF THIS PUZZLE IS TO CHANGE THIS:”信息，然后输出“X X X X X X X X X”。接下来，按照猜数步长变化，输出数字，然后输出“O O O O O O O O O O”。最后，按照每行交替输出数字，包括在猜数成功后输出“ZERO”和重新开始游戏时输出“11 (ELEVEN)”。


```
void PrintGameResult(int gameSum, int stepsCount)
{
    if (gameSum == 10)
    {
        Print($"VERY GOOD.  YOU GUESSED IT IN ONLY {stepsCount} GUESSES.");
    }
    else
    {
        Print($"TRY HARDER NEXT TIME.  IT TOOK YOU {stepsCount} GUESSES.");
    }
}

void PrintGameInfo()
{
    Print(Tab(32) + "FLIPFLOP");
    Print(Tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    Print();
    Print("THE OBJECT OF THIS PUZZLE IS TO CHANGE THIS:");
    Print();

    Print("X X X X X X X X X X");
    Print();
    Print("TO THIS:");
    Print();
    Print("O O O O O O O O O O");
    Print();

    Print("BY TYPING THE NUMBER CORRESPONDING TO THE POSITION OF THE");
    Print("LETTER ON SOME NUMBERS, ONE POSITION WILL CHANGE, ON");
    Print("OTHERS, TWO WILL CHANGE.  TO RESET LINE TO ALL X'S, TYPE 0");
    Print("(ZERO) AND TO START OVER IN THE MIDDLE OF A GAME, TYPE ");
    Print("11 (ELEVEN).");
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `36_Flip_Flop/java/FlipFlop.java`

这段代码是一个名为 "Game of FlipFlop" 的游戏，基于 1970 年初的 BASIC 游戏。该游戏的目标是通过编写 Java 代码，创建一个与 BASIC 游戏相似的游戏，但没有添加新的功能，如文本显示、错误检查等。

具体来说，这段代码实现了以下功能：

1. 定义了一个名为 "Game of FlipFlop" 的类。
2. 导入了 Java 标准库中的 Scanner 类和 Math 类。
3. 创建了一个 Scanner 对象，用于从玩家那里获取输入。
4. 通过 Math.random() 方法生成一个 0 到 1 之间的随机数。
5. 如果随机数是 0 或 1，那么游戏胜利，否则失败。
6. 输出游戏结果，以便玩家了解游戏结果。

这段代码的主要目的是提供一个简单的 FlipFlop 游戏，让玩家在玩游戏的同时，了解计算机编程的基础知识。这个游戏没有添加其他功能，而是重点关注于基本的游戏逻辑和玩法。


```
import java.util.Scanner;
import java.lang.Math;

/**
 * Game of FlipFlop
 * <p>
 * Based on the BASIC game of FlipFlop here
 * https://github.com/coding-horror/basic-computer-games/blob/main/36%20Flip%20Flop/flipflop.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

```

This is a Java program that simulates a board game where the user has to guess a randomly chosen number within 11 attempts. The game has different steps for getting a number, guessing a number, and trying again if the user doesn't get it right. It also has a count of how many guesses the user made.

The program first initializes the game board and then enters a while loop that runs the game until the user chooses to stop or a certain number of attempts are made. The loop gets through each step of the game and updates the game board, displays the board, and makes a request to the user if they want to continue or try again.

The game also has a count of how many guesses the user made, which is incremented each time the user makes a guess. If the user makes more than 11 guesses without getting the number, the game prints a message and goes back to the step where the user tries again.

The program ends with a main method that initializes the game and calls the play method, which displays the game board and makes the game play until the user stops it.


```
public class FlipFlop {

  private final Scanner scan;  // For user input

  private enum Step {
    RANDOMIZE, INIT_BOARD, GET_NUMBER, ILLEGAL_ENTRY, FLIP_POSITION, SET_X_FIRST, SET_X_SECOND,
    GENERATE_R_FIRST, GENERATE_R_SECOND, PRINT_BOARD, QUERY_RETRY
  }

  public FlipFlop() {

    scan = new Scanner(System.in);

  }  // End of constructor FlipFlop

  public void play() {

    showIntro();
    startGame();

  }  // End of method play

  private static void showIntro() {

    System.out.println(" ".repeat(31) + "FLIPFLOP");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("");

  }  // End of method showIntro

  private void startGame() {

    double mathVal = 0;
    double randVal = 0;
    double tmpVal = 0;

    int index = 0;
    int match = 0;
    int numFlip = 0;
    int numGuesses = 0;

    Step nextStep = Step.RANDOMIZE;

    String userResponse = "";

    String[] board = new String[21];

    System.out.println("THE OBJECT OF THIS PUZZLE IS TO CHANGE THIS:");
    System.out.println("");
    System.out.println("X X X X X X X X X X");
    System.out.println("");
    System.out.println("TO THIS:");
    System.out.println("");
    System.out.println("O O O O O O O O O O");
    System.out.println("");
    System.out.println("BY TYPING THE NUMBER CORRESPONDING TO THE POSITION OF THE");
    System.out.println("LETTER ON SOME NUMBERS, ONE POSITION WILL CHANGE, ON");
    System.out.println("OTHERS, TWO WILL CHANGE.  TO RESET LINE TO ALL X'S, TYPE 0");
    System.out.println("(ZERO) AND TO START OVER IN THE MIDDLE OF A GAME, TYPE ");
    System.out.println("11 (ELEVEN).");
    System.out.println("");

    // Begin outer while loop
    while (true) {

      // Begin switch
      switch (nextStep) {

        case RANDOMIZE:

          randVal = Math.random();

          System.out.println("HERE IS THE STARTING LINE OF X'S.");
          System.out.println("");

          numGuesses = 0;
          nextStep = Step.INIT_BOARD;
          break;

        case INIT_BOARD:

          System.out.println("1 2 3 4 5 6 7 8 9 10");
          System.out.println("X X X X X X X X X X");
          System.out.println("");

          // Avoid out of bounds error by starting at zero
          for (index = 0; index <= 10; index++) {
            board[index] = "X";
          }

          nextStep = Step.GET_NUMBER;
          break;

        case GET_NUMBER:

          System.out.print("INPUT THE NUMBER? ");
          userResponse = scan.nextLine();

          try {
            numFlip = Integer.parseInt(userResponse);
          }
          catch (NumberFormatException ex) {
            nextStep = Step.ILLEGAL_ENTRY;
            break;
          }

          // Command to start a new game
          if (numFlip == 11) {
            nextStep = Step.RANDOMIZE;
            break;
          }

          if (numFlip > 11) {
            nextStep = Step.ILLEGAL_ENTRY;
            break;
          }

          // Command to reset the board
          if (numFlip == 0) {
            nextStep = Step.INIT_BOARD;
            break;
          }

          if (match == numFlip) {
            nextStep = Step.FLIP_POSITION;
            break;
          }

          match = numFlip;

          if (board[numFlip].equals("O")) {
            nextStep = Step.SET_X_FIRST;
            break;
          }

          board[numFlip] = "O";
          nextStep = Step.GENERATE_R_FIRST;
          break;

        case ILLEGAL_ENTRY:
          System.out.println("ILLEGAL ENTRY--TRY AGAIN.");
          nextStep = Step.GET_NUMBER;
          break;

        case GENERATE_R_FIRST:

          mathVal = Math.tan(randVal + numFlip / randVal - numFlip) - Math.sin(randVal / numFlip) + 336
                    * Math.sin(8 * numFlip);

          tmpVal = mathVal - (int)Math.floor(mathVal);

          numFlip = (int)(10 * tmpVal);

          if (board[numFlip].equals("O")) {
            nextStep = Step.SET_X_FIRST;
            break;
          }

          board[numFlip] = "O";
          nextStep = Step.PRINT_BOARD;
          break;

        case SET_X_FIRST:
          board[numFlip] = "X";

          if (match == numFlip) {
            nextStep = Step.GENERATE_R_FIRST;
          } else {
            nextStep = Step.PRINT_BOARD;
          }
          break;

        case FLIP_POSITION:

          if (board[numFlip].equals("O")) {
            nextStep = Step.SET_X_SECOND;
            break;
          }

          board[numFlip] = "O";
          nextStep = Step.GENERATE_R_SECOND;
          break;

        case GENERATE_R_SECOND:

          mathVal = 0.592 * (1 / Math.tan(randVal / numFlip + randVal)) / Math.sin(numFlip * 2 + randVal)
                    - Math.cos(numFlip);

          tmpVal = mathVal - (int)mathVal;
          numFlip = (int)(10 * tmpVal);

          if (board[numFlip].equals("O")) {
            nextStep = Step.SET_X_SECOND;
            break;
          }

          board[numFlip] = "O";
          nextStep = Step.PRINT_BOARD;
          break;

        case SET_X_SECOND:

          board[numFlip] = "X";
          if (match == numFlip) {
            nextStep = Step.GENERATE_R_SECOND;
            break;
          }

          nextStep = Step.PRINT_BOARD;
          break;

        case PRINT_BOARD:
          System.out.println("1 2 3 4 5 6 7 8 9 10");

          for (index = 1; index <= 10; index++) {
            System.out.print(board[index] + " ");
          }

          numGuesses++;

          System.out.println("");

          for (index = 1; index <= 10; index++) {
            if (!board[index].equals("O")) {
              nextStep = Step.GET_NUMBER;
              break;
            }
          }

          if (nextStep == Step.GET_NUMBER) {
            break;
          }

          if (numGuesses > 12) {
            System.out.println("TRY HARDER NEXT TIME.  IT TOOK YOU " + numGuesses + " GUESSES.");
          } else {
            System.out.println("VERY GOOD.  YOU GUESSED IT IN ONLY " + numGuesses + " GUESSES.");
          }
          nextStep = Step.QUERY_RETRY;
          break;

        case QUERY_RETRY:

          System.out.print("DO YOU WANT TO TRY ANOTHER PUZZLE? ");
          userResponse = scan.nextLine();

          if (userResponse.toUpperCase().charAt(0) == 'N') {
            return;
          }
          System.out.println("");
          nextStep = Step.RANDOMIZE;
          break;

        default:
          System.out.println("INVALID STEP");
          nextStep = Step.QUERY_RETRY;
          break;

      }  // End of switch

    }  // End outer while loop

  }  // End of method startGame

  public static void main(String[] args) {

    FlipFlop game = new FlipFlop();
    game.play();

  }  // End of method main

}  // End of class FlipFlop

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `36_Flip_Flop/javascript/flipflop.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是接收一个字符串参数，将其追加到页面上一个名为 `output` 的 div 元素中。

`input` 函数的作用是接收用户输入的字符串，将其存储在变量 `input_str` 中，然后将其显示在页面上。为了获取用户输入的字符串，它使用了 `document.getElementById` 函数来获取 input 元素，然后使用 `addEventListener` 函数来监听输入元素中的 `keydown` 事件。在该事件中，当用户按下了 `13` 键时，函数会将 `input_str` 赋值为当前输入元素中的值，并将其显示在页面上。


```
// FLIPFLOP
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

This appears to be a program that is meant to solve a type of puzzle called a "pigmacci sequence." This sequence is a series of numbers where each number is the sum of the two preceding numbers, such as 2, 3, 5, 8, 13, 21, etc. The program looks for a solution to the puzzle by guessing values until it finds a valid one. The program also appears to use a "remaining challenge" system where the number of guesses the program has to solve the puzzle is displayed on the screen.



```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var as = [];

// Main program
async function main()
{
    print(tab(32) + "FLIPFLOP\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    // *** Created by Michael Cass
    print("THE OBJECT OF THIS PUZZLE IS TO CHANGE THIS:\n");
    print("\n");
    print("X X X X X X X X X X\n");
    print("\n");
    print("TO THIS:\n");
    print("\n");
    print("O O O O O O O O O O\n");
    print("\n");
    print("BY TYPING THE NUMBER CORRESPONDING TO THE POSITION OF THE\n");
    print("LETTER ON SOME NUMBERS, ONE POSITION WILL CHANGE, ON\n");
    print("OTHERS, TWO WILL CHANGE.  TO RESET LINE TO ALL X'S, TYPE 0\n");
    print("(ZERO) AND TO START OVER IN THE MIDDLE OF A GAME, TYPE \n");
    print("11 (ELEVEN).\n");
    print("\n");
    while (1) {
        start = 1;
        do {
            z = 1;
            if (start == 1) {
                m = 0;
                q = Math.random();
                print("HERE IS THE STARTING LINE OF X'S.\n");
                print("\n");
                c = 0;
                start = 2;
            }
            if (start == 2) {
                print("1 2 3 4 5 6 7 8 9 10\n");
                print("X X X X X X X X X X\n");
                print("\n");
                for (x = 1; x <= 10; x++)
                    as[x] = "X";
                start = 0;
            }
            print("INPUT THE NUMBER");
            while (1) {
                n = parseInt(await input());
                if (n >= 0 && n <= 11)
                    break;
                print("ILLEGAL ENTRY--TRY AGAIN.\n");
            }
            if (n == 11) {
                start = 1;
                continue;
            }
            if (n == 0) {
                start = 2;
                continue;
            }
            if (m != n) {
                m = n;
                as[n] = (as[n] == "O" ? "X" : "O");
                do {
                    r = Math.tan(q + n / q - n) - Math.sin(q / n) + 336 * Math.sin(8 * n);
                    n = r - Math.floor(r);
                    n = Math.floor(10 * n);
                    as[n] = (as[n] == "O" ? "X" : "O");
                } while (m == n) ;
            } else {
                as[n] = (as[n] == "O" ? "X" : "O");
                do {
                    r = 0.592 * (1 / Math.tan(q / n + q)) / Math.sin(n * 2 + q) - Math.cos(n);
                    n = r - Math.floor(r);
                    n = Math.floor(10 * n);
                    as[n] = (as[n] == "O" ? "X" : "O");
                } while (m == n) ;
            }
            print("1 2 3 4 5 6 7 8 9 10\n");
            for (z = 1; z <= 10; z++)
                print(as[z] + " ");
            c++;
            print("\n");
            for (z = 1; z <= 10; z++) {
                if (as[z] != "O")
                    break;
            }
        } while (z <= 10) ;
        if (c <= 12) {
            print("VERY GOOD.  YOU GUESSED IT IN ONLY " + c + " GUESSES.\n");
        } else {
            print("TRY HARDER NEXT TIME.  IT TOOK YOU " + c + " GUESSES.\n");
        }
        print("DO YOU WANT TO TRY ANOTHER PUZZLE");
        str = await input();
        if (str.substr(0, 1) == "N")
            break;
    }
    print("\n");
}

```

这道题目缺少上下文，无法给出具体的解释。通常来说，在编程中， `main()` 函数是程序的入口点，程序从此处开始执行。它的作用是启动程序，告诉操作系统程序要开始执行哪些代码。所以，`main()` 函数可以被视为程序的“开始”函数。


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


# `36_Flip_Flop/python/flipflop.py`

这段代码是一个简单的游戏，其目的是通过输入数字来控制游戏地图中X和O的布局。具体来说，这段代码定义了一个名为"Flip Flop"的游戏对象，其作用是通过输入数字来改变游戏地图中X和O的布局，使得所有的X都排成一列，而所有的O都排成一列。

地图上的X和O代表游戏中的行，而数字则代表每个位置上的X或O。通过输入数字，用户可以选择一个或多个位置上的X，从而改变地图中O和X的布局。例如，当输入数字3时，它可能会反转位置3的X和O，但也有可能改变其他位置的布局。

在实际应用中，这段代码仅能改变部分位置上的X和O，而不能在所有位置上改变它们的布局。此外，由于 X和O的数量是固定的，因此有些位置可能无法改变它们的布局。


```
# Flip Flop
#
# The object of this game is to change a row of ten X's
# X X X X X X X X X X
# to a row of ten O's:
# O O O O O O O O O O
# by typing in a number corresponding
# to the position of an "X" in the line. On
# some numbers one position will
# change while on other numbers, two
# will change. For example, inputting a 3
# may reverse the X and O in position 3,
# but it might possibly reverse some
# other position too! You ought to be able
# to change all 10 in 12 or fewer
```

这段代码是一个简单的棋类游戏，玩家可以在两个整数之间进行输入来选择行。棋盘初始状态是所有行都是"X"，玩家可以使用0来清空行，使用11来开始新的游戏。

虽然这段代码的实现非常简单，但它仍然具有很多有趣的性质。例如，在游戏开始时，所有行都是"X"，这意味着玩家需要立即选择行。此外，由于游戏中的所有行都是等效的，因此这个游戏没有明确的胜利条件。

如果玩家在游戏中选择0，那么他们会将所有行都重置为"X"，这实际上是在向玩家提出一个条件，即所有行都是相同的，并且没有任何明显的胜利条件。


```
# moves. Can you figure out a good win-
# ning strategy?
# To reset the line to all X's (same
# game), type 0 (zero). To start a new
# game at any point, type 11.
# The original author of this game was
# Michael Kass of New Hyde Park, New
# York.
import math
import random
from typing import Callable, List, Tuple

flip_dict = {"X": "O", "O": "X"}


```

该函数 `flip_bits` 接受一个列表 `row`、一个整数 `m` 和一个整数 `n`，以及一个函数 `r_function`。它返回一个元组 `[row, n]`，其中 `row` 是反向排列的字符，`n` 是旋转的步数。

函数的核心部分是一个 while 循环，该循环会在 `m` 与 `n` 之间遍历。在每次循环中，函数调用一个名为 `r_function` 的函数，它接受一个整数参数 `n`，并返回一个计算为浮点数的值。

在 while 循环内部，函数首先将 `r_function(n)` 的值赋给 `n`，然后将 `n` 的值乘以 10，并将结果除以 2，得到一个新的 `n`。接下来，函数遍历 `row` 中的所有字符。对于每个字符 `c`，函数检查当前字符 `c` 在 `row` 中的位置是否为 `"X"`，如果是，则将 `c` 更改为 `"O"`，否则将 `c` 更改为 `"X"`。这样，经过一次循环后，字符的逆序位置会发生变化，从而实现了反向排列。

函数返回的元组包含 `row` 和 `n`，其中 `row` 是反向排列的字符，`n` 是旋转的步数。


```
def flip_bits(
    row: List[str], m: int, n: int, r_function: Callable[[int], float]
) -> Tuple[List[str], int]:
    """
    Function that flips the positions at the computed steps
    """
    while m == n:
        r = r_function(n)
        n_tmp = r - int(math.floor(r))
        n = int(10 * n_tmp)
        if row[n] == "X":
            row[n] = "O"
            break
        elif row[n] == "O":
            row[n] = "X"
    return row, n


```

这段代码是一个Python函数，名为`print_instructions()`。它包含一系列输出指令，用于在控制台上显示一些信息。

具体来说，这段代码会输出以下内容：

```
       _
      _           _                           
     (_)_)          (_)_)                           
    (             )                             
   /  ---             \                             
  /                      \                             
 /                        \                             
/                          \                             
/______________________________\______________________
```

然后，它会输出一个乘以32的星号，一个带有不同字符的短语，以及一段描述这个谜题对象的文章。文章包括了要改变什么，以及如何改变它，以及如何重新开始，等等。

最后，它会提示用户输入一个数字，用于改变星星的形状。


```
def print_instructions() -> None:
    print(" " * 32 + "FLIPFLOP")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print("\n" * 2)
    print("THE OBJECT OF THIS PUZZLE IS TO CHANGE THIS:\n")
    print("X X X X X X X X X X\n")
    print("TO THIS:\n")
    print("O O O O O O O O O O\n")
    print("BY TYPING TH NUMBER CORRESPONDING TO THE POSITION OF THE")
    print("LETTER ON SOME NUMBERS, ONE POSITION WILL CHANGE, ON")
    print("OTHERS, TWO WILL CHANGE. TO RESET LINE TO ALL X'S, TYPE 0")
    print("(ZERO) AND TO START OVER IN THE MIDDLE OF A GAME, TYPE ")
    print("11 (ELEVEN).\n")


```

It looks like you're trying to implement the Tic-tac-toe game. While I'm an AI language model and can certainly help you with any questions you may have, it's important to keep in mind that this particular implementation may not be the most efficient or user-friendly.

That being said, here are some suggestions for improving your implementation:

1. Use a function to flip the bits of the board. This will make it easier to keep track of which bits have been set to X and allow you to update the game state more efficiently.
2. Use a variable to keep track of the current player's turn. This will help you maintain the game state between each player's turn.
3. Use a variable to store the counter-turns count. This will help you keep track of how many turns have passed without any winner.
4. Add a function to reset the game state. This will be useful if the user accidentally deletes any part of the game state.
5. Use the input() function to prompt the user for their move. This will make it easier for the user to enter their choice.
6. Use a variable to store the legal moves list. This will help you keep track of the valid moves that the game state allows.

Here's an example of how you could implement some of these suggestions:
```python
import random

def tic_tac_toe(board_size, num_colors):
   # Initialize the game board
   board = [[random.randint(0, 9) for _ in range(board_size)] for _ in range(board_size)]
   player_turn = 1
   counter_turns = 0
   valid_moves = []

   while True:
       # Print the board
       print(" ".join(row[1:]) + "\n")

       # Ask the user for their move
       m_str = input("PLACE YOUR Decision DOUBLING/QUIT/TRY AGAIN OR enter the number to play: ")
       m = int(m_str)
       if m > 11 or m < 0:
           print("ILLEGAL ENTRY--TRY AGAIN")
           continue

       # Update the game state
       board[player_turn], player_turn = swap_bits(board[player_turn], m), swap_bits(board[player_turn], m - 1)
       counter_turns += 1
       if counter_turns == 12:
           counter_turns = 0

       # Check if the board is a valid game state
       if all_win(board):
           valid_moves.append(m)
           print(f"Congratulations {player_name}! You win!")
           break
       elif all_ tied:
           valid_moves.append(m)
           print(f"It's a tie, unfortunately! The game is still a tie!")
           break
       else:
           valid_moves.append(m)

   # Handle the counter-turns
   while counter_turns > 0:
       # Print the board
       print()
       print(" ".join(row[1:]) + "\n")

       # Ask the user for their move
       m_str = input("PLACE YOUR Decision DOUBLING/QUIT/TRY AGAIN OR enter the number to play: ")
       m = int(m_str)
       if m > 11 or m < 0:
           print("ILLEGAL ENTRY--TRY AGAIN")
           continue

       # Update the game state
       board[player_turn], player_turn = swap_bits(board[player_turn], m), swap_bits(board[player_turn], m - 1)
       counter_turns -= 1

   # Print the board to show the final game state
   print()
       print(" ".join(row[1:]) + "\n")

   # Print the
```


```
def main() -> None:
    q = random.random()

    print("HERE IS THE STARTING LINE OF X'S.\n")
    # We add an extra 0-th item because this sometimes is set to something
    # but we never check what it is for completion of the puzzle
    row = [""] + ["X"] * 10
    counter_turns = 0
    n = -1
    legal_move = True
    while row[1:] != ["O"] * 10:
        if legal_move:
            print(" ".join([str(i) for i in range(1, 11)]))
            print(" ".join(row[1:]) + "\n")
        m_str = input("INPUT THE NUMBER\n")
        try:
            m = int(m_str)
            if m > 11 or m < 0:
                raise ValueError()
        except ValueError:
            print("ILLEGAL ENTRY--TRY AGAIN")
            legal_move = False
            continue
        legal_move = True
        if m == 11:
            # completely reset the puzzle
            counter_turns = 0
            row = [""] + ["X"] * 10
            q = random.random()
            continue
        elif m == 0:
            # reset the board, but not the counter or the random number
            row = [""] + ["X"] * 10
        elif m == n:
            row[n] = flip_dict[row[n]]
            r_function = lambda n_t: 0.592 * (1 / math.tan(q / n_t + q)) / math.sin(
                n_t * 2 + q
            ) - math.cos(n_t)
            row, n = flip_bits(row, m, n, r_function)
        else:
            n = m
            row[n] = flip_dict[row[n]]
            r_function = lambda n_t: (
                math.tan(q + n_t / q - n_t)
                - math.sin(n_t * 2 + q)
                + 336 * math.sin(8 * n_t)
            )
            row, n = flip_bits(row, m, n, r_function)

        counter_turns += 1
        print()

    if counter_turns <= 12:
        print(f"VERY GOOD. YOU GUESSED IT IN ONLY {counter_turns} GUESSES.")
    else:
        print(f"TRY HARDER NEXT TIME. IT TOOK YOU {counter_turns} GUESSES.")
    return


```

这段代码是一个简单的 Python 程序，它的主要目的是让用户尝试解决一系列谜题。现在让我们逐步分析这段代码的作用。

1. 首先，定义了一个名为 `__main__` 的内置函数。如果程序在运行时直接以 `__main__` 作为文件名，那么程序会执行这个函数体内的内容。

2. 在 `__main__` 函数体内，调用了一个名为 `print_instructions` 的函数。由于这个函数在程序启动时没有被定义为 `__main__` 函数，所以它的作用可能不是很清楚。但从代码中看，似乎这个函数没有做太多事情，只是输出了一些指示符。

3. 接下来，程序创建了一个名为 `another` 的空字符串变量。然后进入了一个名为 `while` 的无限循环。只要另一个字符串（也就是用户输入）不是名为 `"NO"`，循环就会一直持续。

4. 当用户尝试解决一个谜题时，程序调用了一个名为 `main` 的函数。由于这个函数没有定义为 `__main__` 函数，所以它的作用也不是很清楚。但从代码中看，它可能与程序的核心功能有关。

5. 最后，程序接收了一个用户输入，用于判断用户是否想要继续尝试下一个谜题。这个输入会被传递给 `main` 函数，但在这里并没有被使用。


```
if __name__ == "__main__":
    print_instructions()

    another = ""
    while another != "NO":
        main()
        another = input("DO YOU WANT TO TRY ANOTHER PUZZLE\n")

```