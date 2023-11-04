# BasicComputerGames源码解析 16

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `08_Batnum/java/src/BatNum.java`

This is a Java class that simulates a basic tab-based command-line interface for playing a game.

The class includes several methods:

* `displayTextAndGetNumber(String text)`: This method displays a message on the screen and then accepts input from the keyboard. It converts the input to an integer and returns it.
* `displayTextAndGetInput(String text)`: This method displays a message on the screen and then accepts input from the keyboard. It converts the input to a string and returns it.
* `simulateTabs(int spaces)`: This method simulates the old basic tab command by indenting the text by the specified number of spaces. It returns the simulated text.
* `getDelimitedValue(String text, int pos)`: This method accepts a string with values separated by commas. It returns the nth delimited value (starting at count 0).

The class also includes several println statements that print messages to the screen.


```
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of BatNum
 * <p>
 * Based on the Basic game of BatNum here
 * https://github.com/coding-horror/basic-computer-games/blob/main/08%20Batnum/batnum.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class BatNum {

    private enum GAME_STATE {
        STARTING,
        START_GAME,
        CHOOSE_PILE_SIZE,
        SELECT_WIN_OPTION,
        CHOOSE_MIN_AND_MAX,
        SELECT_WHO_STARTS_FIRST,
        PLAYERS_TURN,
        COMPUTERS_TURN,
        ANNOUNCE_WINNER,
        GAME_OVER
    }

    // Used for keyboard input
    private final Scanner kbScanner;

    // Current game state
    private GAME_STATE gameState;

    private int pileSize;

    // How to win the game options
    enum WIN_OPTION {
        TAKE_LAST,
        AVOID_LAST
    }

    // Tracking the winner
    enum WINNER {
        COMPUTER,
        PLAYER
    }

    private WINNER winner;

    private WIN_OPTION winOption;

    private int minSelection;
    private int maxSelection;

    // Used by computer for optimal move
    private int rangeOfRemovals;

    public BatNum() {

        gameState = GAME_STATE.STARTING;

        // Initialise kb scanner
        kbScanner = new Scanner(System.in);
    }

    /**
     * Main game loop
     */
    public void play() {

        do {
            switch (gameState) {

                // Show an introduction and optional instructions the first time the game is played.
                case STARTING:
                    intro();
                    gameState = GAME_STATE.START_GAME;
                    break;

                // Start new game
                case START_GAME:
                    gameState = GAME_STATE.CHOOSE_PILE_SIZE;
                    break;

                case CHOOSE_PILE_SIZE:
                    System.out.println();
                    System.out.println();
                    pileSize = displayTextAndGetNumber("ENTER PILE SIZE ");
                    if (pileSize >= 1) {
                        gameState = GAME_STATE.SELECT_WIN_OPTION;
                    }
                    break;

                case SELECT_WIN_OPTION:
                    int winChoice = displayTextAndGetNumber("ENTER WIN OPTION - 1 TO TAKE LAST, 2 TO AVOID LAST: ");
                    if (winChoice == 1) {
                        winOption = WIN_OPTION.TAKE_LAST;
                        gameState = GAME_STATE.CHOOSE_MIN_AND_MAX;
                    } else if (winChoice == 2) {
                        winOption = WIN_OPTION.AVOID_LAST;
                        gameState = GAME_STATE.CHOOSE_MIN_AND_MAX;
                    }
                    break;

                case CHOOSE_MIN_AND_MAX:
                    String range = displayTextAndGetInput("ENTER MIN AND MAX ");
                    minSelection = getDelimitedValue(range, 0);
                    maxSelection = getDelimitedValue(range, 1);
                    if (maxSelection > minSelection && minSelection >= 1) {
                        gameState = GAME_STATE.SELECT_WHO_STARTS_FIRST;
                    }

                    // Used by computer in its turn
                    rangeOfRemovals = minSelection + maxSelection;
                    break;

                case SELECT_WHO_STARTS_FIRST:
                    int playFirstChoice = displayTextAndGetNumber("ENTER START OPTION - 1 COMPUTER FIRST, 2 YOU FIRST ");
                    if (playFirstChoice == 1) {
                        gameState = GAME_STATE.COMPUTERS_TURN;
                    } else if (playFirstChoice == 2) {
                        gameState = GAME_STATE.PLAYERS_TURN;
                    }
                    break;

                case PLAYERS_TURN:
                    int playersMove = displayTextAndGetNumber("YOUR MOVE ");

                    if (playersMove == 0) {
                        System.out.println("I TOLD YOU NOT TO USE ZERO! COMPUTER WINS BY FORFEIT.");
                        winner = WINNER.COMPUTER;
                        gameState = GAME_STATE.ANNOUNCE_WINNER;
                        break;
                    }

                    if (playersMove == pileSize && winOption == WIN_OPTION.AVOID_LAST) {
                        winner = WINNER.COMPUTER;
                        gameState = GAME_STATE.ANNOUNCE_WINNER;
                        break;
                    }

                    // Check if players move is with the min and max possible
                    if (playersMove >= minSelection && playersMove <= maxSelection) {
                        // Valid so reduce pileSize by amount player entered
                        pileSize -= playersMove;

                        // Did this move result in there being no more objects on pile?
                        if (pileSize == 0) {
                            // Was the game setup so the winner was whoever took the last object
                            if (winOption == WIN_OPTION.TAKE_LAST) {
                                // Player won
                                winner = WINNER.PLAYER;
                            } else {
                                // Computer one
                                winner = WINNER.COMPUTER;
                            }
                            gameState = GAME_STATE.ANNOUNCE_WINNER;
                        } else {
                            // There are still items left.
                            gameState = GAME_STATE.COMPUTERS_TURN;
                        }
                    } else {
                        // Invalid move
                        System.out.println("ILLEGAL MOVE, REENTER IT ");
                    }
                    break;

                case COMPUTERS_TURN:
                    int pileSizeLeft = pileSize;
                    if (winOption == WIN_OPTION.TAKE_LAST) {
                        if (pileSize > maxSelection) {

                            int objectsToRemove = calculateComputersTurn(pileSizeLeft);

                            pileSize -= objectsToRemove;
                            System.out.println("COMPUTER TAKES " + objectsToRemove + " AND LEAVES " + pileSize);
                            gameState = GAME_STATE.PLAYERS_TURN;
                        } else {
                            System.out.println("COMPUTER TAKES " + pileSize + " AND WINS.");
                            winner = WINNER.COMPUTER;
                            gameState = GAME_STATE.ANNOUNCE_WINNER;
                        }
                    } else {
                        pileSizeLeft--;
                        if (pileSize > minSelection) {
                            int objectsToRemove = calculateComputersTurn(pileSizeLeft);
                            pileSize -= objectsToRemove;
                            System.out.println("COMPUTER TAKES " + objectsToRemove + " AND LEAVES " + pileSize);
                            gameState = GAME_STATE.PLAYERS_TURN;
                        } else {
                            System.out.println("COMPUTER TAKES " + pileSize + " AND LOSES.");
                            winner = WINNER.PLAYER;
                            gameState = GAME_STATE.ANNOUNCE_WINNER;
                        }
                    }
                    break;

                case ANNOUNCE_WINNER:
                    switch (winner) {
                        case PLAYER:
                            System.out.println("CONGRATULATIONS, YOU WIN.");
                            break;
                        case COMPUTER:
                            System.out.println("TOUGH LUCK, YOU LOSE.");
                            break;
                    }
                    gameState = GAME_STATE.START_GAME;
                    break;
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    /**
     * Figure out the computers turn - i.e. how many objects to remove
     *
     * @param pileSizeLeft current size
     * @return the number of objects to remove.
     */
    private int calculateComputersTurn(int pileSizeLeft) {
        int computersNumberToRemove = pileSizeLeft - rangeOfRemovals * (pileSizeLeft / rangeOfRemovals);
        if (computersNumberToRemove < minSelection) {
            computersNumberToRemove = minSelection;
        }
        if (computersNumberToRemove > maxSelection) {
            computersNumberToRemove = maxSelection;
        }

        return computersNumberToRemove;
    }

    private void intro() {
        System.out.println(simulateTabs(33) + "BATNUM");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("THIS PROGRAM IS A 'BATTLE OF NUMBERS' GAME, WHERE THE");
        System.out.println("COMPUTER IS YOUR OPPONENT.");
        System.out.println();
        System.out.println("THE GAME STARTS WITH AN ASSUMED PILE OF OBJECTS. YOU");
        System.out.println("AND YOUR OPPONENT ALTERNATELY REMOVE OBJECTS FROM THE PILE.");
        System.out.println("WINNING IS DEFINED IN ADVANCE AS TAKING THE LAST OBJECT OR");
        System.out.println("NOT. YOU CAN ALSO SPECIFY SOME OTHER BEGINNING CONDITIONS.");
        System.out.println("DON'T USE ZERO, HOWEVER, IN PLAYING THE GAME.");
        System.out.println("ENTER A NEGATIVE NUMBER FOR NEW PILE SIZE TO STOP PLAYING.");
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     * Converts input to Integer
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

    /**
     * Accepts a string delimited by comma's and returns the nth delimited
     * value (starting at count 0).
     *
     * @param text - text with values separated by comma's
     * @param pos  - which position to return a value for
     * @return the int representation of the value
     */
    private int getDelimitedValue(String text, int pos) {
        String[] tokens = text.split(",");
        return Integer.parseInt(tokens[pos]);
    }
}

```

# `08_Batnum/java/src/BatNumGame.java`

这段代码定义了一个名为BatNumGame的public类，其中包含一个名为main的静态方法。

在main方法中，使用new关键字创建了一个名为BatNum的实例变量，并将其赋值为一个新的BatNum对象。

接着，使用play()方法对创建的BatNum对象进行操作，但是batNum对象没有定义任何方法，因此这个方法也是没有意义的。


```
public class BatNumGame {

    public static void main(String[] args) {

        BatNum batNum = new BatNum();
        batNum.play();
    }
}

```

# `08_Batnum/javascript/batnum.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是在页面上打印一段字符串，接收一个字符串参数，并将该参数附加给一个元素，然后将该元素添加到页面上。

`input` 函数的作用是从用户接收输入，接收一个字符串参数。该函数会创建一个 `INPUT` 元素，设置其 `type` 属性为 `text`，设置其 `length` 属性为 `50`，然后将该元素添加到页面上。然后，函数会绑定一个 `keydown` 事件到该元素的 `addEventListener` 方法中。当用户按下 `13` 键时，函数会将 `input` 元素中的字符串值赋给 `input_str`，并删除该元素。最后，函数会将 `input_str` 的值输出到页面上，并输出 `"\n"`。


```
// BATNUM
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

This is a program that plays the game of chess. It takes two arguments, the number of moves as a win condition and the number of moves as a loss condition. The program starts by randomly choosing which player will move and then enters a loop that plays the game.

For each move, the program checks if the move was valid and if the player has won or lost. If the move was valid and the player has won, the program prints a message and resets the player's score to 1. If the move was valid and the player has lost, the program prints a different message depending on whether or not they have zero moves left.

If the player has won or lost, the program checks if they have any left moves and if they do, it updates the player's score accordingly. If the player has run out of moves and has no left moves, the program prints a message indicating that the game was too far away for the player to win or lose.

If the player has won, the program prints a different message depending on whether or not they have zero moves left. If the player has run out of moves and has no left moves, the program prints a message indicating that the game was too far away for the player to win or lose.

If the player has lost, the program prints a message and updates the player's score to 1.

The program also has a third argument which is the number of moves as a win condition. If the player has made the move limit number of moves, the program will print a message and the player will lose by default.


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
    print(tab(33) + "BATNUM\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("THIS PROGRAM IS A 'BATTLE OF NUMBERS' GAME, WHERE THE\n");
    print("COMPUTER IS YOUR OPPONENT.\n");
    print("\n");
    print("THE GAME STARTS WITH AN ASSUMED PILE OF OBJECTS. YOU\n");
    print("AND YOUR OPPONENT ALTERNATELY REMOVE OBJECTS FROM THE PILE.\n");
    print("WINNING IS DEFINED IN ADVANCE AS TAKING THE LAST OBJECT OR\n");
    print("NOT. YOU CAN ALSO SPECIFY SOME OTHER BEGINNING CONDITIONS.\n");
    print("DON'T USE ZERO, HOWEVER, IN PLAYING THE GAME.\n");
    print("ENTER A NEGATIVE NUMBER FOR NEW PILE SIZE TO STOP PLAYING.\n");
    print("\n");
    first_time = 1;
    while (1) {
        while (1) {
            if (first_time == 1) {
                first_time = 0;
            } else {
                for (i = 1; i <= 10; i++)
                    print("\n");
            }
            print("ENTER PILE SIZE");
            n = parseInt(await input());
            if (n >= 1)
                break;
        }
        while (1) {
            print("ENTER WIN OPTION - 1 TO TAKE LAST, 2 TO AVOID LAST: ");
            m = parseInt(await input());
            if (m == 1 || m == 2)
                break;
        }
        while (1) {
            print("ENTER MIN AND MAX ");
            str = await input();
            a = parseInt(str);
            b = parseInt(str.substr(str.indexOf(",") + 1));
            if (a <= b && a >= 1)
                break;
        }
        while (1) {
            print("ENTER START OPTION - 1 COMPUTER FIRST, 2 YOU FIRST ");
            s = parseInt(await input());
            print("\n");
            print("\n");
            if (s == 1 || s == 2)
                break;
        }
        w = 0;
        c = a + b;
        while (1) {
            if (s == 1) {
                // Computer's turn
                q = n;
                if (m != 1)
                    q--;
                if (m != 1 && n <= a) {
                    w = 1;
                    print("COMPUTER TAKES " + n + " AND LOSES.\n");
                } else if (m == 1 && n <= b) {
                    w = 1;
                    print("COMPUTER TAKES " + n + " AND WINS.\n");
                } else {
                    p = q - c * Math.floor(q / c);
                    if (p < a)
                        p = a;
                    if (p > b)
                        p = b;
                    n -= p;
                    print("COMPUTER TAKES " + p + " AND LEAVES " + n + "\n");
                    w = 0;
                }
                s = 2;
            }
            if (w)
                break;
            if (s == 2) {
                while (1) {
                    print("\n");
                    print("YOUR MOVE ");
                    p = parseInt(await input());
                    if (p == 0) {
                        print("I TOLD YOU NOT TO USE ZERO! COMPUTER WINS BY FORFEIT.\n");
                        w = 1;
                        break;
                    } else if (p >= a && p <= b && n - p >= 0) {
                        break;
                    }
                }
                if (p != 0) {
                    n -= p;
                    if (n == 0) {
                        if (m != 1) {
                            print("TOUGH LUCK, YOU LOSE.\n");
                        } else {
                            print("CONGRATULATIONS, YOU WIN.\n");
                        }
                        w = 1;
                    } else {
                        w = 0;
                    }
                }
                s = 1;
            }
            if (w)
                break;
        }
    }
}

```

这道题目是一个不完整的C语言代码，缺少了程序的输入输出语句。我们来分析一下这个代码可能会执行哪些操作：

1. `main()` 是C语言中的一个函数，表示程序的入口点。通常情况下，这个函数会负责启动程序的执行。

2. `int main()` 是C语言中的一个函数，定义了程序的返回类型。返回类型表示程序在运行结束时会返回给用户的数据类型。

3. `int main()` 表示程序从 `main()` 函数开始，该函数的实现会根据 `int main()` 的参数进行执行。

4. `int main()` 是程序的入口点，程序从这里开始执行。在程序执行之前，可能会通过 `printf()` 等函数输出一些信息，告诉用户程序的用途。

由于缺少程序的输入输出语句，我们无法得知程序在运行时具体会执行哪些操作。


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

Conversion to [Pascal](https://en.wikipedia.org/wiki/Pascal_(programming_language))


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/) by [Austin White](https://github.com/austinwhite)


# `08_Batnum/python/batnum.py`

这段代码定义了一个名为 "WinOptions" 的类，它是一个 IntEnum 类型的枚举。IntEnum 是一种用于定义具有默认值的枚举类型的库函数，可以通过带参数的 @classmethod 方法进行自定义。

在代码中，首先定义了三个枚举值：Undefined、TakeLast 和 AvoidLast，它们的值分别为 0、1 和 2。然后，定义了一个名为 "WinOptions" 的类，继承自 IntEnum 类。

在 WinOptions 类的 _missing_ 方法中，返回了一个 "WinOptions" 类的实例，该实例的 value 属性使用了参数化的 int 函数来尝试将值解析为整数。如果失策失败，该方法将返回 IntEnum 中定义的 Undefined 值。

该代码的主要目的是定义了一个枚举类型，用于表示 Windows 操作系统的不同选项，以及一个类来描述这些选项。这个类提供了一个简单的接口，让用户可以通过一个可读的名称来选择一个选项，而不是直接从 Windows 命令行界面中使用数字或字符串。


```
from enum import IntEnum
from typing import Any, Tuple


class WinOptions(IntEnum):
    Undefined = 0
    TakeLast = 1
    AvoidLast = 2

    @classmethod
    def _missing_(cls, value: Any) -> "WinOptions":
        try:
            int_value = int(value)
        except Exception:
            return WinOptions.Undefined
        if int_value == 1:
            return WinOptions.TakeLast
        elif int_value == 2:
            return WinOptions.AvoidLast
        else:
            return WinOptions.Undefined


```

这段代码定义了一个名为 "StartOptions" 的类，属于 IntEnum 枚举类型。其中，枚举类型中定义了三个成员变量，分别为 Undefined、ComputerFirst 和 PlayerFirst，成员变量分别为 0、1 和 2。

接着，代码中定义了一个名为 "@classmethod" 的方法，属于 Python 3 中的特殊方法，用于在运行时创建类的实例。

在 "@classmethod" 方法的体内，定义了一个名为 "_missing_" 的方法，该方法接收一个参数 "value"，用于在创建类的实例时，如果 "value" 参数未被正确类型化成 Int 类型，则返回 "StartOptions.Undefined"。

另外，如果 "value" 参数正确类型化成 Int 类型，那么会根据不同的值，返回不同的实例化类别，分别为 "StartOptions.ComputerFirst" 和 "StartOptions.PlayerFirst"。如果尝试将 "value" 转换为 Int 类型时出现异常，则返回 "StartOptions.Undefined"。


```
class StartOptions(IntEnum):
    Undefined = 0
    ComputerFirst = 1
    PlayerFirst = 2

    @classmethod
    def _missing_(cls, value: Any) -> "StartOptions":
        try:
            int_value = int(value)
        except Exception:
            return StartOptions.Undefined
        if int_value == 1:
            return StartOptions.ComputerFirst
        elif int_value == 2:
            return StartOptions.PlayerFirst
        else:
            return StartOptions.Undefined


```

这段代码是一个 Python 函数，名为 `print_intro()`。函数体中包含以下几行输出：

```
BATNUMBER.rjust(33, " ")
"CREATIVE COMPUTING"
"MORRISSTOWN, NEW JERSEY".rjust(15, " ")
"THIS PROGRAM IS A 'BATTLE OF NUMBERS' GAME,"
"WHERE THE COMPUTER IS YOUR OPPONENT."
```

这些输出是为了在游戏开始时向玩家介绍游戏规则和背景。

具体来说，第一行输出了游戏的主要参数，包括游戏名称和版本，第二行输出了游戏的创造者信息，第三行是游戏规则的简单说明。第四行和第五行是游戏的起点规则，第六行到第十行是游戏的详细规则，最后一行是声明，告诉玩家如何编写自己的游戏。


```
def print_intro() -> None:
    """Print out the introduction and rules for the game."""
    print("BATNUM".rjust(33, " "))
    print("CREATIVE COMPUTING  MORRISSTOWN, NEW JERSEY".rjust(15, " "))
    print()
    print()
    print()
    print("THIS PROGRAM IS A 'BATTLE OF NUMBERS' GAME, WHERE THE")
    print("COMPUTER IS YOUR OPPONENT.")
    print()
    print("THE GAME STARTS WITH AN ASSUMED PILE OF OBJECTS. YOU")
    print("AND YOUR OPPONENT ALTERNATELY REMOVE OBJECTS FROM THE PILE.")
    print("WINNING IS DEFINED IN ADVANCE AS TAKING THE LAST OBJECT OR")
    print("NOT. YOU CAN ALSO SPECIFY SOME OTHER BEGINNING CONDITIONS.")
    print("DON'T USE ZERO, HOWEVER, IN PLAYING THE GAME.")
    print("ENTER A NEGATIVE NUMBER FOR NEW PILE SIZE TO STOP PLAYING.")
    print()
    return


```

这段代码定义了一个名为 `get_params` 的函数，它返回了游戏需要必要参数的元组。这个函数通过调用其他函数来获取这五个参数。

函数的第一个参数 `pile_size` 返回了玩家开始时堆栈的大小。如果这个大小小于 0，那么函数将返回一个元组 `(-1, 0, 0, `未定义的选择参数 `, WinOptions.Undefined)`。

函数的第二个参数 `min_select` 返回了每个回合最小需要选择的对象数量。

函数的第三个参数 `max_select` 返回了每个回合最大需要选择的对象数量。

函数的第四个参数 `start_option` 返回了计算机是否在游戏开始时拥有优先选择权，它有两个选项：1 和 2。

函数的第五个参数 `win_option` 返回了游戏的目标。它有两个选项：1 和 2。如果游戏的目标是让玩家不抓最后一个对象，那么函数将返回 2。

这个函数返回了一个元组，它由五个参数组成，这些参数是游戏开始时需要的信息。


```
def get_params() -> Tuple[int, int, int, StartOptions, WinOptions]:
    """This requests the necessary parameters to play the game.

    Returns a set with the five game parameters:
        pile_size - the starting size of the object pile
        min_select - minimum selection that can be made on each turn
        max_select - maximum selection that can be made on each turn
        start_option - 1 if the computer is first
                      or 2 if the player is first
        win_option - 1 if the goal is to take the last object
                    or 2 if the goal is to not take the last object
    """
    pile_size = get_pile_size()
    if pile_size < 0:
        return (-1, 0, 0, StartOptions.Undefined, WinOptions.Undefined)
    win_option = get_win_option()
    min_select, max_select = get_min_max()
    start_option = get_start_option()
    return (pile_size, min_select, max_select, start_option, win_option)


```



This code defines two functions, `get_pile_size()` and `get_win_option()`.

`get_pile_size()` is a function that returns the size of a pile of blocks. It does this by first setting the pile size to 0 and then repeatedly trying to get a number from the user. If the user enters a negative number, the function returns 0.

The function uses a while loop to keep prompting the user for input until they enter a valid number. The function also checks if the input is a valid integer, and if not, it returns 0.

Finally, the function returns the pile size.

`get_win_option()` is a function that returns a `WinOptions` object. It does this by first initializing the `win_option` variable to an instance of the `WinOptions` class.

The function then enters a while loop that prompts the user to enter either "1" to "TAKE LAST" or "2" to "AVOID LAST". Depending on the user's input, the function updates the `win_option` object.

Finally, the function returns the `win_option` object.


```
def get_pile_size() -> int:
    # A negative number will stop the game.
    pile_size = 0
    while pile_size == 0:
        try:
            pile_size = int(input("ENTER PILE SIZE "))
        except ValueError:
            pile_size = 0
    return pile_size


def get_win_option() -> WinOptions:
    win_option: WinOptions = WinOptions.Undefined
    while win_option == WinOptions.Undefined:
        win_option = WinOptions(input("ENTER WIN OPTION - 1 TO TAKE LAST, 2 TO AVOID LAST: "))  # type: ignore
    return win_option


```

这段代码是一个Python函数，名为`get_min_max`，它返回两个整数类型的元组，分别是`min_select`和`max_select`。函数的作用是读取两个整数，来自用户输入，并让用户有一个选择，然后返回这两个整数中的最小值和最大值。

接下来是一个函数`get_start_option`，它返回一个`StartOptions`对象，其中`StartOptions`是一个枚举类型，定义了三种可能的选项：`StartOptions.Undefined`，`StartOptions.ComputerFirst`和`StartOptions.YouFirst`。函数的作用是让用户从三个选项中选择一个，并返回选中的选项。


```
def get_min_max() -> Tuple[int, int]:
    min_select = 0
    max_select = 0
    while min_select < 1 or max_select < 1 or min_select > max_select:
        (min_select, max_select) = (
            int(x) for x in input("ENTER MIN AND MAX ").split(" ")
        )
    return min_select, max_select


def get_start_option() -> StartOptions:
    start_option: StartOptions = StartOptions.Undefined
    while start_option == StartOptions.Undefined:
        start_option = StartOptions(input("ENTER START OPTION - 1 COMPUTER FIRST, 2 YOU FIRST "))  # type: ignore
    return start_option


```

这段代码定义了一个名为 `player_move` 的函数，用于处理玩家的回合。函数接受四个参数：堆大小 `pile_size`、最小选择数量 `min_select`、最大选择数量 `max_select` 和胜利选项 `win_option`。

函数的作用是让玩家进行投票，然后检查游戏是否结束。在投票之前，函数首先检查玩家输入是否为 0，如果不是，函数会提示玩家不要使用零，否则会认为玩家犯了一个大错误，游戏直接结束。如果玩家输入为 0，游戏也会直接结束，所以玩家需要输入一个整数来进行投票。

在投票之后，函数会计算堆的大小，并检查堆是否为空或已满。如果是，函数会根据胜利选项来处理游戏结果。如果堆为空或已满，但玩家成功占领了所有资源，游戏会结束，并打印出胜利者的新堆大小。否则，游戏会结束，并打印出失败者的新堆大小。


```
def player_move(
    pile_size: int, min_select: int, max_select: int, win_option: WinOptions
) -> Tuple[bool, int]:
    """This handles the player's turn - asking the player how many objects
    to take and doing some basic validation around that input.  Then it
    checks for any win conditions.

    Returns a boolean indicating whether the game is over and the new pile_size."""
    player_done = False
    while not player_done:
        player_move = int(input("YOUR MOVE "))
        if player_move == 0:
            print("I TOLD YOU NOT TO USE ZERO!  COMPUTER WINS BY FORFEIT.")
            return (True, pile_size)
        if player_move > max_select or player_move < min_select:
            print("ILLEGAL MOVE, REENTER IT")
            continue
        pile_size = pile_size - player_move
        player_done = True
        if pile_size <= 0:
            if win_option == WinOptions.AvoidLast:
                print("TOUGH LUCK, YOU LOSE.")
            else:
                print("CONGRATULATIONS, YOU WIN.")
            return (True, pile_size)
    return (False, pile_size)


```

这段代码定义了一个名为 `computer_pick` 的函数，用于处理计算机在抽卡时选择张卡的数量。

函数接收四个参数，分别为：

- `pile_size`：堆叠卡牌的大小，是一个整数类型。
- `min_select`：最小选择张数，是一个整数类型。
- `max_select`：最大选择张数，是一个整数类型。
- `win_option`：抽卡胜利选项，可以选择 AvoidLast、Redraw 张卡或两者都不选择。

函数的实现逻辑如下：

1. 如果 `win_option` 为 `WinOptions.AvoidLast`，则计算机将从堆叠卡牌中选择数量等于堆叠卡牌大小减一的张卡。
2. 如果 `win_option` 为 `WinOptions.Redraw`，则计算机将从堆叠卡牌中选择数量等于选择张数加最大张数(包括最大张数)的张卡，并且不会选择上一张抽出的卡牌。
3. 如果 `win_option` 为 `WinOptions.Default`，则计算机将从堆叠卡牌中选择数量等于 `min_select` 张卡。
4. 如果 `computer_pick` 小于 `min_select`，则将 `min_select` 作为计算机选择张数的下限。
5. 如果 `computer_pick` 大于 `max_select`，则将 `max_select` 作为计算机选择张数的上限。
6. 函数返回计算机选择张数的值。

函数可以作为一个实参函数被调用，例如：

```
# 选择 10 张卡
computer_pick(pile_size=10, min_select=3, max_select=5)
```

这样，函数将会选择堆叠卡牌中数量为 10 张卡，其中选择张数为 3 张，最小选择张数为 3。


```
def computer_pick(
    pile_size: int, min_select: int, max_select: int, win_option: WinOptions
) -> int:
    """This handles the logic to determine how many objects the computer
    will select on its turn.
    """
    q = pile_size - 1 if win_option == WinOptions.AvoidLast else pile_size
    c = min_select + max_select
    computer_pick = q - (c * int(q / c))
    if computer_pick < min_select:
        computer_pick = min_select
    if computer_pick > max_select:
        computer_pick = max_select
    return computer_pick


```

这段代码定义了一个名为 `computer_move` 的函数，用于处理电脑在游戏中的移动操作。

函数接收四个参数，分别为：

- `pile_size`：堆叠物的总高度。
- `min_select`：最小选择数量，即当前堆叠物高度达到此值时，电脑必须选择的最小物品数。
- `max_select`：最大选择数量，即当前堆叠物高度达到此值时，电脑必须选择的最大物品数。
- `win_option`：胜利选项，包括以下可能的值：
 - `WinOptions.TakeLast`：电脑将选择堆叠物的最后一个物品。
 - `WinOptions.AvoidLast`：电脑将选择堆叠物中剩余的最小的物品，并丢弃掉最后一个物品。
 - `WinOptions.TakeAll`：电脑将选择堆叠物中的所有物品。

函数首先检查指定的胜利选项，然后根据指定的胜利选项计算电脑的新堆叠物高度，并输出结果。

最后，函数会输出当前堆叠物的剩余数量，以及电脑选择的新堆叠物高度。


```
def computer_move(
    pile_size: int, min_select: int, max_select: int, win_option: WinOptions
) -> Tuple[bool, int]:
    """This handles the computer's turn - first checking for the various
    win/lose conditions and then calculating how many objects
    the computer will take.

    Returns a boolean indicating whether the game is over and the new pile_size."""
    # First, check for win conditions on this move
    # In this case, we win by taking the last object and
    # the remaining pile is less than max select
    # so the computer can grab them all and win
    if win_option == WinOptions.TakeLast and pile_size <= max_select:
        print(f"COMPUTER TAKES {pile_size} AND WINS.")
        return (True, pile_size)
    # In this case, we lose by taking the last object and
    # the remaining pile is less than minsize and the computer
    # has to take all of them.
    if win_option == WinOptions.AvoidLast and pile_size <= min_select:
        print(f"COMPUTER TAKES {min_select} AND LOSES.")
        return (True, pile_size)

    # Otherwise, we determine how many the computer selects
    curr_sel = computer_pick(pile_size, min_select, max_select, win_option)
    pile_size = pile_size - curr_sel
    print(f"COMPUTER TAKES {curr_sel} AND LEAVES {pile_size}")
    return (False, pile_size)


```

这段代码定义了一个名为 `play_game` 的函数，用于控制游戏的进行。函数包含了以下参数：

- `pile_size`：堆叠棋盘的大小。
- `min_select`：选择棋的优先级下限，最低优先级为 `min_select`。
- `max_select`：选择棋的优先级上限，最高优先级为 `max_select`。
- `start_option`：下棋时玩家是否先手，值为 `StartOptions.PlayerFirst` 时玩家先手，值为 `StartOptions.ComputerFirst` 时计算机先手。
- `win_option`：当游戏是否胜利的条件，值为 `WinOptions.Yes` 时游戏胜利，值为 `WinOptions.No` 时游戏不胜利。

函数的作用是让玩家与计算机轮流下棋，直到有一方胜利或游戏平局。


```
def play_game(
    pile_size: int,
    min_select: int,
    max_select: int,
    start_option: StartOptions,
    win_option: WinOptions,
) -> None:
    """This is the main game loop - repeating each turn until one
    of the win/lose conditions is met.
    """
    game_over = False
    # players_turn is a boolean keeping track of whether it's the
    # player's or computer's turn
    players_turn = start_option == StartOptions.PlayerFirst

    while not game_over:
        if players_turn:
            (game_over, pile_size) = player_move(
                pile_size, min_select, max_select, win_option
            )
            players_turn = False
            if game_over:
                return
        if not players_turn:
            (game_over, pile_size) = computer_move(
                pile_size, min_select, max_select, win_option
            )
            players_turn = True


```

这段代码是一个Python程序，名为“main”。它定义了一个函数，名为“main”，返回None。在这个函数中，定义了一个无限循环的段落，该段落将不断执行以下操作：

1. 打印出介绍信息
2. 通过调用一个名为“get_params”的函数，获取堆叠大小、最小选择数、最大选择数和开始选项等参数
3. 检查堆叠大小是否为负数，如果是，那么函数将返回
4. 调用一个名为“play_game”的函数，将获取到的参数传递给该函数，并开始游戏
5. 无限循环将继续执行步骤2至4的操作，直到程序被用户通过控制台发送的“Ctrl + C”停止。


```
def main() -> None:
    while True:
        print_intro()
        (pile_size, min_select, max_select, start_option, win_option) = get_params()

        if pile_size < 0:
            return

        # Just keep playing the game until the user kills it with ctrl-C
        play_game(pile_size, min_select, max_select, start_option, win_option)


if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Battle

BATTLE is based on the popular game Battleship which is primarily played to familiarize people with the location and designation of points on a coordinate plane.

BATTLE first randomly sets up the bad guy’s fleet disposition on a 6 by 6 matrix or grid. The fleet consists of six ships:
- Two destroyers (ships number 1 and 2) which are two units long
- Two cruisers (ships number 3 and 4) which are three units long
- Two aircraft carriers (ships number 5 and 6) which are four units long

The program then prints out this fleet disposition in a coded or disguised format (see the sample computer print-out). You then proceed to sink the various ships by typing in the coordinates (two digits. each from 1 to 6, separated by a comma) of the place where you want to drop a bomb, if you’ll excuse the expression. The computer gives the appropriate response (splash, hit, etc.) which you should record on a 6 by 6 matrix. You are thus building a representation of the actual fleet disposition which you will hopefully use to decode the coded fleet disposition printed out by the computer. Each time a ship is sunk, the computer prints out which ships have been sunk so far and also gives you a “SPLASH/HIT RATIO.”

The first thing you should learn is how to locate and designate positions on the matrix, and specifically the difference between “3,4” and “4,3.” Our method corresponds to the location of points on the coordinate plane rather than the location of numbers in a standard algebraic matrix: the first number gives the column counting from left to right and the second number gives the row counting from bottom to top.

The second thing you should learn about is the splash/hit ratio. “What is a ratio?” A good reply is “It’s a fraction or quotient.” Specifically, the spash/hit ratio is the number of splashes divided by the number of hits. If you had 9 splashes and 15 hits, the ratio would be 9/15 or 3/5, both of which are correct. The computer would give this splash/hit ratio as .6.

The main objective and primary education benefit of BATTLE comes from attempting to decode the bad guys’ fleet disposition code. To do this, you must make a comparison between the coded matrix and the actual matrix which you construct as you play the game.

The original author of both the program and these descriptive notes is Ray Westergard of Lawrence Hall of Science, Berkeley, California.

---

As published in Basic Computer Games (1978)
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=15)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=30)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

- The original game has no way to re-view the fleet disposition code once it scrolls out of view.  Ports should consider allowing the user to enter "?" at the "??" prompt, to reprint the disposition code.  (This is added by the MiniScript port under Alternate Languages, for example.)

# `09_Battle/csharp/Game.cs`

This is a program written in C# that appears to simulate a game of depth-first search (DFS) where the player is trying to sink a ship in a scenario where the ship has been damaged and must repeat the action until it sinks or the player gives up. The program uses a combination of printing, writing, and random number generators to display information about the game state and provide feedback to the player.

The program consists of several classes, including船舶类（Ship）、游戏类（Game）、以及util类（Util）。

The Ship class represents a ship with a unique number called "shipNum"，每个船舶的船体上有4个位置，分别对应酸甜苦辣，用于显示当前船舶的受伤情况。

The Game类模拟了游戏的主要逻辑，包括检查游戏是否结束、是否可以继续游戏、游戏胜利和游戏失败的情况。

util类则包含了一些工具方法，如打印（Print）、写（Write）、格式化输出（Tab）、生成随机数（Rnd）、以及获取输入值（GetInput）。

通过这些类和函数，可以实现模拟游戏的功能，让玩家尝试猜测并尝试把船底弄好，直到成功。


```
﻿using System;
using System.Linq;

namespace Battle
{
    public class Game
    {
        private int[,] field = new int[7, 7];

        private Random random = new Random();

        public void Run()
        {
            DisplayIntro();

            while (true)
            {
                field = new int[7, 7];

                foreach (var shipType in new []{ 1, 2, 3})
                {
                    foreach (var ship in new int[] { 1, 2 })
                    {
                        while (!SetShip(shipType, ship)) { }
                    }
                }

                UserInteraction();
            }
        }

        private bool SetShip(int shipType, int shipNum)
        {
            var shipSize = 4 - shipType;
            int direction;
            int[] A = new int[5];
            int[] B = new int[5];
            int row, col;

            do
            {
                row = Rnd(6) + 1;
                col = Rnd(6) + 1;
                direction = Rnd(4) + 1;
            } while (field[row, col] > 0);

            var M = 0;

            switch (direction)
            {
                case 1:
                    B[1] = col;
                    B[2] = 7;
                    B[3] = 7;

                    for (var K = 1; K <= shipSize; K++)
                    {
                        if (!(M > 1 || B[K] == 6 || field[row, B[K] + 1] > 0))
                        {
                            B[K + 1] = B[K] + 1;
                            continue;
                        }

                        M = 2;
                        var Z = 1;

                        if (B[1] < B[2] && B[1] < B[3]) Z = B[1];
                        if (B[2] < B[1] && B[2] < B[3]) Z = B[2];
                        if (B[3] < B[1] && B[3] < B[2]) Z = B[3];

                        if (Z == 1 || field[row, Z - 1] > 0) return false;

                        B[K + 1] = Z - 1;
                    }

                    field[row, col] = 9 - 2 * shipType - shipNum;

                    for (var K = 1; K <= shipSize; K++)
                    {
                        field[row, B[K + 1]] = field[row, col];
                    }
                    break;

                case 2:
                    A[1] = row;
                    B[1] = col;
                    A[2] = 0;
                    A[3] = 0;
                    B[2] = 0;
                    B[3] = 0;

                    for (var K = 1; K <= shipSize; K++)
                    {
                        if (!(M > 1
                            || A[K] == 1 || B[K] == 1
                            || field[A[K] - 1, B[K] - 1] > 0
                            || (field[A[K] - 1, B[K]] > 0 && field[A[K] - 1, B[K]] == field[A[K], B[K] - 1])))
                        {
                            A[K + 1] = A[K] - 1;
                            B[K + 1] = B[K] - 1;
                            continue;
                        }

                        M = 2;
                        var Z1 = 1;
                        var Z2 = 1;

                        if (A[1] > A[2] && A[1] > A[3]) Z1 = A[1];
                        if (A[2] > A[1] && A[2] > A[3]) Z1 = A[2];
                        if (A[3] > A[1] && A[3] > A[2]) Z1 = A[3];
                        if (B[1] > B[2] && B[1] > B[3]) Z2 = B[1];
                        if (B[2] > B[1] && B[2] > B[3]) Z2 = B[2];
                        if (B[3] > B[1] && B[3] > B[2]) Z2 = B[3];

                        if (Z1 == 6 || Z2 == 6
                            || field[Z1 + 1, Z2 + 1] > 0
                            || (field[Z1, Z2 + 1] > 0 && field[Z1, Z2 + 1] == field[Z1 + 1, Z2])) return false;

                        A[K + 1] = Z1 + 1;
                        B[K + 1] = Z2 + 1;
                    }

                    field[row, col] = 9 - 2 * shipType - shipNum;

                    for (var K = 1; K <= shipSize; K++)
                    {
                        field[A[K + 1], B[K + 1]] = field[row, col];
                    }
                    break;

                case 3:
                    A[1] = row;
                    A[2] = 7;
                    A[3] = 7;

                    for (var K = 1; K <= shipSize; K++)
                    {
                        if (!(M > 1 || A[K] == 6
                            || field[A[K] + 1, col] > 0))
                        {
                            A[K + 1] = A[K] + 1;
                            continue;
                        }

                        M = 2;
                        var Z = 1;

                        if (A[1] < A[2] && A[1] < A[3]) Z = A[1];
                        if (A[2] < A[1] && A[2] < A[3]) Z = A[2];
                        if (A[3] < A[1] && A[3] < A[2]) Z = A[3];

                        if (Z == 1 || field[Z - 1, col] > 0) return false;

                        A[K + 1] = Z - 1;
                    }

                    field[row, col] = 9 - 2 * shipType - shipNum;

                    for (var K = 1; K <= shipSize; K++)
                    {
                        field[A[K + 1], col] = field[row, col];
                    }
                    break;

                case 4:
                default:
                    A[1] = row;
                    B[1] = col;
                    A[2] = 7;
                    A[3] = 7;
                    B[2] = 0;
                    B[3] = 0;

                    for (var K = 1; K <= shipSize; K++)
                    {
                        if (!(M > 1 || A[K] == 6 || B[K] == 1
                            || field[A[K] + 1, B[K] - 1] > 0
                            || (field[A[K] + 1, B[K]] > 0 && field[A[K] + 1, B[K]] == field[A[K], B[K] - 1])))
                        {
                            A[K + 1] = A[K] + 1;
                            B[K + 1] = B[K] - 1;
                            continue;
                        }

                        M = 2;
                        var Z1 = 1;
                        var Z2 = 1;

                        if (A[1] < A[2] && A[1] < A[3]) Z1 = A[1];
                        if (A[2] < A[1] && A[2] < A[3]) Z1 = A[2];
                        if (A[3] < A[1] && A[3] < A[2]) Z1 = A[3];
                        if (B[1] > B[2] && B[1] > B[3]) Z2 = B[1];
                        if (B[2] > B[1] && B[2] > B[3]) Z2 = B[2];
                        if (B[3] > B[1] && B[3] > B[2]) Z2 = B[3];

                        if (Z1 == 1 || Z2 == 6
                            || field[Z1 - 1, Z2 + 1] > 0
                            || (field[Z1, Z2 + 1] > 0 && field[Z1, Z2 + 1] == field[Z1 - 1, Z2])) return false;

                        A[K + 1] = Z1 - 1;
                        B[K + 1] = Z2 + 1;
                    }

                    field[row, col] = 9 - 2 * shipType - shipNum;

                    for (var K = 1; K <= shipSize; K++)
                    {
                        field[A[K + 1], B[K + 1]] = field[row, col];
                    }

                    break;
            }

            return true;
        }

        public void DisplayIntro()
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Print(Tab(33) + "BATTLE");
            Print(Tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            //-- BATTLE WRITTEN BY RAY WESTERGARD  10 / 70
            // COPYRIGHT 1971 BY THE REGENTS OF THE UNIV.OF CALIF.
            // PRODUCED AT THE LAWRENCE HALL OF SCIENCE, BERKELEY
        }

        public void UserInteraction()
        {
            Print();
            Print("THE FOLLOWING CODE OF THE BAD GUYS' FLEET DISPOSITION");
            Print("HAS BEEN CAPTURED BUT NOT DECODED:");
            Print();

            for (var row = 1; row <= 6; row++)
            {
                for (var col = 1; col <= 6; col++)
                {
                    Write(field[col, row].ToString());
                }

                Print();
            }

            Print();
            Print("DE-CODE IT AND USE IT IF YOU CAN");
            Print("BUT KEEP THE DE-CODING METHOD A SECRET.");
            Print();

            var hit = new int[7, 7];
            var lost = new int[4];
            var shipHits = new[] { 0, 2, 2, 1, 1, 0, 0 };
            var splashes = 0;
            var hits = 0;

            Print("START GAME");

            do
            {
                var input = Console.ReadLine().Split(',').Select(x => int.TryParse(x, out var num) ? num : 0).ToArray();

                if (!IsValid(input))
                {
                    Print("INVALID INPUT.  TRY AGAIN.");
                    continue;
                }

                var col = input[0];
                var row = 7 - input[1];
                var shipNum = field[row, col];

                if (shipNum == 0)
                {
                    splashes = splashes + 1;
                    Print("SPLASH!  TRY AGAIN.");
                    continue;
                }

                if (shipHits[shipNum] > 3)
                {
                    Print("THERE USED TO BE A SHIP AT THAT POINT, BUT YOU SUNK IT.");
                    Print("SPLASH!  TRY AGAIN.");
                    splashes = splashes + 1;
                    continue;
                }

                if (hit[row, col] > 0)
                {
                    Print($"YOU ALREADY PUT A HOLE IN SHIP NUMBER {shipNum} AT THAT POINT.");
                    Print("SPLASH!  TRY AGAIN.");
                    splashes = splashes + 1;
                    continue;
                }

                hits = hits + 1;
                hit[row, col] = shipNum;

                Print($"A DIRECT HIT ON SHIP NUMBER {shipNum}");
                shipHits[shipNum] = shipHits[shipNum] + 1;

                if (shipHits[shipNum] < 4)
                {
                    Print("TRY AGAIN.");
                    continue;
                }

                var shipType = (shipNum - 1) / 2 + 1;
                lost[shipType] = lost[shipType] + 1;

                Print("AND YOU SUNK IT.  HURRAH FOR THE GOOD GUYS.");
                Print("SO FAR, THE BAD GUYS HAVE LOST");
                Write($"{lost[1]} DESTROYER(S), {lost[2]} CRUISER(S), AND ");
                Print($"{lost[3]} AIRCRAFT CARRIER(S).");
                Print($"YOUR CURRENT SPLASH/HIT RATIO IS {splashes / hits}");

                if ((lost[1] + lost[2] + lost[3]) < 6) continue;

                Print();
                Print("YOU HAVE TOTALLY WIPED OUT THE BAD GUYS' FLEET");
                Print($"WITH A FINAL SPLASH/HIT RATIO OF {splashes / hits}");

                if ((splashes / hits) == 0)
                {
                    Print("CONGRATULATIONS -- A DIRECT HIT EVERY TIME.");
                }

                Print();
                Print("****************************");
                Print();

                return;

            } while (true);
        }

        public bool IsValid(int[] input) => input.Length == 2 && input.All(Valid);

        public bool Valid(int value) => value > 0 && value < 7;

        public void Print(string str = "") => Console.WriteLine(str);

        public void Write(string value) => Console.Write(value);

        public string Tab(int pos) => new String(' ', pos);

        public int Rnd(int seed) => random.Next(seed);
    }
}

```

# `09_Battle/csharp/Program.cs`

这段代码是一个 C# 程序，定义了一个名为 "Battle" 的namespace，一个名为 "Program" 的类，一个名为 "Main" 的静态函数，以及一个名为 "Game" 的类。

在Main函数中，调用了一个名为Run的静态函数，这个函数没有参数，并返回一个void类型的值。

Main函数的代码中没有其他函数或类定义，说明它只是一个程序的入口点。

由于没有定义任何函数或类，所以Run函数也无法被调用。


```
﻿using System;

namespace Battle
{
    class Program
    {
        static void Main(string[] args)
        {
            new Game().Run();
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `09_Battle/java/Battle.java`

This is a Java program that simulates a game of blockly探索 and conquest. In this game, players take turns reporting hits and sinking ships. The player can also attack the enemy ships with their fleet. The game keeps track of the number of hits, losses, and the ratio of hits to losses for each ship.

Here is the main class of the game:

```java
public class Main {
   private int[][] sizes;
   private int hits;
   private int losses;
   private double ratioHit;
   private double ratioMiss;
   private double countHits;
   private double countMisses;

   public static void main(String[] args) {
       MyIni ini = new MyIni();
       My game = new MyGame(ini.readInt("sizeHits", 3), ini.readInt("sizeLosses", 3));
       game.run();
   }
}
```

The game has several different variables that keep track of the game state.

* `sizes`: A 2-dimensional array that keeps track of the sizes of the ships.
* `hits`: A count of the number of hits for each ship.
* `losses`: A count of the number of losses for each ship.
* `ratioHit`: A double that represents the ratio of hits to losses for each ship.
* `ratioMiss`: A double that represents the ratio of misses to hits for each ship.
* `countHits`: A count of the number of hits for each ship.
* `countMisses`: A count of the number of misses for each ship.

The game also has two double variables that calculate the splash/hit ratio and the direct hit ratio, respectively.

* `splashHitRatio`: A double that represents the ratio of splash/hits to hits.
* `directHitRatio`: A double that represents the ratio of direct hits to hits.

The game has a `run` method that runs the game. This method first reads the game parameters from the command-line using `MyIni` and then initializes the game state.

The `run` method then calls a `run` method that simulates the game. This method has several private variables, including `ship`


```
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;
import java.util.function.Predicate;
import java.text.NumberFormat;


/* This class holds the game state and the game logic */
public class Battle {

    /* parameters of the game */
    private int seaSize;
    private int[] sizes;
    private int[] counts;

    /* The game setup - the ships and the sea */
    private ArrayList<Ship> ships;
    private Sea sea;

    /* game state counts */
    private int[] losses;    // how many of each type of ship have been sunk
    private int hits;        // how many hits the player has made
    private int misses;      // how many misses the player has made

    // Names of ships of each size. The game as written has ships of size 3, 4 and 5 but
    // can easily be modified. It makes no sense to have a ship of size zero though.
    private static String NAMES_BY_SIZE[] = {
        "error",
        "size1",
        "destroyer",
        "cruiser",
        "aircraft carrier",
        "size5" };

    // Entrypoint
    public static void main(String args[]) {
        Battle game = new Battle(6,                        // Sea is 6 x 6 tiles
                                 new int[] { 2, 3, 4 },    // Ships are of sizes 2, 3, and 4
                                 new int[] { 2, 2, 2 });   // there are two ships of each size
        game.play();
    }

    public Battle(int scale, int[] shipSizes, int[] shipCounts) {
        seaSize = scale;
        sizes = shipSizes;
        counts = shipCounts;

        // validate parameters
        if (seaSize < 4) throw new RuntimeException("Sea Size " + seaSize + " invalid, must be at least 4");

        for (int sz : sizes) {
            if ((sz < 1) || (sz > seaSize))
                throw new RuntimeException("Ship has invalid size " + sz);
        }

        if (counts.length != sizes.length) {
            throw new RuntimeException("Ship counts must match");
        }

        // Initialize game state
        sea = new Sea(seaSize);          // holds what ship if any occupies each tile
        ships = new ArrayList<Ship>();   // positions and states of all the ships
        losses = new int[counts.length]; // how many ships of each type have been sunk

        // Build up the list of all the ships
        int shipNumber = 1;
        for (int type = 0; type < counts.length; ++type) {
            for (int i = 0; i < counts[i]; ++i) {
                ships.add(new Ship(shipNumber++, sizes[type]));
            }
        }

        // When we put the ships in the sea, we put the biggest ones in first, or they might
        // not fit
        ArrayList<Ship> largestFirst = new ArrayList<>(ships);
        Collections.sort(largestFirst, Comparator.comparingInt((Ship ship) -> ship.size()).reversed());

        // place each ship into the sea
        for (Ship ship : largestFirst) {
            ship.placeRandom(sea);
        }
    }

    public void play() {
        System.out.println("The following code of the bad guys' fleet disposition\nhas been captured but not decoded:\n");
        System.out.println(sea.encodedDump());
        System.out.println("De-code it and use it if you can\nbut keep the de-coding method a secret.\n");

        int lost = 0;
        System.out.println("Start game");
        Input input = new Input(seaSize);
        try {
            while (lost < ships.size()) {          // the game continues while some ships remain unsunk
                if (! input.readCoordinates()) {   // ... unless there is no more input from the user
                    return;
                }

                // The computer thinks of the sea as a grid of rows, from top to bottom.
                // However, the user will use X and Y coordinates, with Y going bottom to top
                int row = seaSize - input.y();
                int col = input.x() - 1;

                if (sea.isEmpty(col, row)) {
                    ++misses;
                    System.out.println("Splash!  Try again.");
                } else {
                    Ship ship = ships.get(sea.get(col, row) - 1);
                    if (ship.isSunk()) {
                        ++misses;
                        System.out.println("There used to be a ship at that point, but you sunk it.");
                        System.out.println("Splash!  Try again.");
                    } else if (ship.wasHit(col, row)) {
                        ++misses;
                        System.out.println("You already put a hole in ship number " + ship.id());
                        System.out.println("Splash!  Try again.");
                    } else {
                        ship.hit(col, row);
                        ++hits;
                        System.out.println("A direct hit on ship number " + ship.id());

                        // If a ship was hit, we need to know whether it was sunk.
                        // If so, tell the player and update our counts
                        if (ship.isSunk()) {
                            ++lost;
                            System.out.println("And you sunk it.  Hurrah for the good guys.");
                            System.out.print("So far, the bad guys have lost ");
                            ArrayList<String> typeDescription = new ArrayList<>();
                            for (int i = 0 ; i < sizes.length; ++i) {
                                if (sizes[i] == ship.size()) {
                                    ++losses[i];
                                }
                                StringBuilder sb = new StringBuilder();
                                sb.append(losses[i]);
                                sb.append(" ");
                                sb.append(NAMES_BY_SIZE[sizes[i]]);
                                if (losses[i] != 1)
                                    sb.append("s");
                                typeDescription.add(sb.toString());
                            }
                            System.out.println(String.join(", ", typeDescription));
                            double ratioNum = ((double)misses)/hits;
                            String ratio = NumberFormat.getInstance().format(ratioNum);
                            System.out.println("Your current splash/hit ratio is " + ratio);

                            if (lost == ships.size()) {
                                System.out.println("You have totally wiped out the bad guys' fleet");
                                System.out.println("With a final splash/hit ratio of " + ratio);

                                if (misses == 0) {
                                    System.out.println("Congratulations - A direct hit every time.");
                                }

                                System.out.println("\n****************************\n");
                            }
                        }
                    }
                }
            }
        }
        catch (IOException e) {
            // This should not happen running from console, but java requires us to check for it
            System.err.println("System error.\n" + e);
        }
    }
}

```