# BasicComputerGames源码解析 61

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Nicomachus

One of the most ancient forms of arithmetic puzzle is sometimes referred to as a “boomerang.” At some time, everyone has been asked to “think of a number,” and, after going through some process of private calculation, to state the result, after which the questioner promptly tells you the number you originally thought of. There are hundreds of varieties of this puzzle.

The oldest recorded example appears to be that given in _Arithmetica_ of Nicomachus, who died about the year 120. He tells you to think of any whole number between 1 and 100 and divide it successfully by 3, 5, and 7, telling him the remainder in each case. On receiving this information, he promptly discloses the number you thought of.

Can you discover a simple method of mentally performing this feat? If not, you can see how the ancient mathematician did it by looking at this program.

Nicomachus was written by David Ahl.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=117)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=132)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `64_Nicomachus/csharp/program.cs`

This appears to be a class that controls a game of "Nicomachus", a computational problem-solving game. The game involves assigning a number to each of four players, and then asking each player to come up with a number that is divisible by 3, 5, and 7, while keeping the sum of the numbers being submitted must be a multiple of 3. The game then displays the four numbers to the players, and prompts them to choose one of the numbers to submit as their own number. If the player chooses a number that is not divisible by 3, 5, or 7, it will display an error message. If the player chooses a number that is divisible by 3, 5, or 7, the game will display the number and ask the player if they are satisfied with it. If the player chooses "Yes", the game will continue to the next round. If the player chooses "No", the game will end and the four numbers will be displayed again. The game will continue until one of the players abandons ship, at which point the game will end and the four numbers will be displayed again.

The game also has a method called "Play", which will play one round of the game and display the four numbers to the players. The method also has a "DisplayIntro" method, which displays the introduction message to the game.

Note that the game assumes that the players have an integer data type, and that the program has aleviates to the high level of abstraction.


```
using System.Text;
using System.Threading;

namespace Nicomachus
{
    class Nicomachus
    {
        private void DisplayIntro()
        {
            Console.WriteLine();
            Console.WriteLine("NICOMA".PadLeft(23));
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("Boomerang puzzle from Arithmetica of Nicomachus -- A.D. 90!");
       }

        private bool PromptYesNo(string Prompt)
        {
            bool Success = false;

            while (!Success)
            {
                Console.Write(Prompt);
                string LineInput = Console.ReadLine().Trim().ToLower();

                if (LineInput.Equals("yes"))
                    return true;
                else if (LineInput.Equals("no"))
                    return false;
                else
                    Console.WriteLine("Eh?  I don't understand '{0}'  Try 'Yes' or 'No'.", LineInput);
            }

            return false;
        }

        private int PromptForNumber(string Prompt)
        {
            bool InputSuccess = false;
            int ReturnResult = 0;

            while (!InputSuccess)
            {
                Console.Write(Prompt);
                string Input = Console.ReadLine().Trim();
                InputSuccess = int.TryParse(Input, out ReturnResult);
                if (!InputSuccess)
                    Console.WriteLine("*** Please enter a valid number ***");
            }   

            return ReturnResult;
        }

        private void PlayOneRound()
        {
            Random rand = new Random();
            int A_Number = 0;
            int B_Number = 0;
            int C_Number = 0;
            int D_Number = 0;

            Console.WriteLine();
            Console.WriteLine("Please think of a number between 1 and 100.");

            A_Number = PromptForNumber("Your number divided by 3 has a remainder of? ");
            B_Number = PromptForNumber("Your number divided by 5 has a remainder of? ");
            C_Number = PromptForNumber("Your number divided by 7 has a remainder of? ");

            Console.WriteLine();
            Console.WriteLine("Let me think a moment...");

            Thread.Sleep(2000);

            D_Number = 70 * A_Number + 21 * B_Number + 15 * C_Number;

            while (D_Number > 105)
            {
                D_Number -= 105;
            }

            if (PromptYesNo("Your number was " + D_Number.ToString() + ", right? "))
            {
                Console.WriteLine();
                Console.WriteLine("How about that!!");
            }
            else
            {
                Console.WriteLine();
                Console.WriteLine("I feel your arithmetic is in error.");
            }

            Console.WriteLine();

       }

        public void Play()
        {
            bool ContinuePlay = true;

            DisplayIntro();

            do 
            {
                PlayOneRound();

                ContinuePlay = PromptYesNo("Let's try another? ");
            }
            while (ContinuePlay);
        }
    }
    class Program
    {
        static void Main(string[] args)
        {

            new Nicomachus().Play();

        }
    }
}
```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `64_Nicomachus/java/src/Nicomachus.java`

This code appears to be a simple game where the player is prompted to enter a series of choices. The game appears to have a goal of guessing the answer to each question, but it is not clear how the player's answers are being compared to the correct answer.

The `Nicomachus` class appears to be the main class that is responsible for handling the communication between the player and the game. It has methods for displaying the correct answer, prompting the player for their answer, and checking whether the entered answer is correct.

The `displayTextAndGetInput` method displays the correct answer on the screen and returns the player's entered answer.

The `addSpaces` method returns a string with the specified number of spaces filled in.

The `isAnyValue` method checks whether a given string is equal to one of a variable number of values.

The `play` method of the `Nicomachus` class appears to be the main method of entry for the game.

Overall, this code is not providing enough information to determine what the full functionality of the game is, but it appears to be a simple game where the player is prompted to enter a series of choices.


```
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Nichomachus
 * <p>
 * Based on the Basic game of Nichomachus here
 * https://github.com/coding-horror/basic-computer-games/blob/main/64%20Nicomachus/nicomachus.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */

public class Nicomachus {

    public static final long TWO_SECONDS = 2000;

    // Used for keyboard input
    private final Scanner kbScanner;

    private enum GAME_STATE {
        START_GAME,
        GET_INPUTS,
        RESULTS,
        PLAY_AGAIN
    }

    int remainderNumberDividedBy3;
    int remainderNumberDividedBy5;
    int remainderNumberDividedBy7;

    // Current game state
    private GAME_STATE gameState;

    public Nicomachus() {
        kbScanner = new Scanner(System.in);
        gameState = GAME_STATE.START_GAME;
    }

    /**
     * Main game loop
     */
    public void play() throws Exception {

        do {
            switch (gameState) {

                case START_GAME:
                    intro();
                    gameState = GAME_STATE.GET_INPUTS;
                    break;

                case GET_INPUTS:

                    System.out.println("PLEASE THINK OF A NUMBER BETWEEN 1 AND 100.");
                    remainderNumberDividedBy3 = displayTextAndGetNumber("YOUR NUMBER DIVIDED BY 3 HAS A REMAINDER OF? ");
                    remainderNumberDividedBy5 = displayTextAndGetNumber("YOUR NUMBER DIVIDED BY 5 HAS A REMAINDER OF? ");
                    remainderNumberDividedBy7 = displayTextAndGetNumber("YOUR NUMBER DIVIDED BY 7 HAS A REMAINDER OF? ");

                    gameState = GAME_STATE.RESULTS;

                case RESULTS:
                    System.out.println("LET ME THINK A MOMENT...");
                    // Simulate the basic programs for/next loop to delay things.
                    // Here we are sleeping for one second.
                    Thread.sleep(TWO_SECONDS);

                    // Calculate the number the player was thinking of.
                    int answer = (70 * remainderNumberDividedBy3) + (21 * remainderNumberDividedBy5)
                            + (15 * remainderNumberDividedBy7);

                    // Something similar was in the original basic program
                    // (to test if the answer was 105 and deducting 105 until it was <= 105
                    while (answer > 105) {
                        answer -= 105;
                    }

                    do {
                        String input = displayTextAndGetInput("YOUR NUMBER WAS " + answer + ", RIGHT? ");
                        if (yesEntered(input)) {
                            System.out.println("HOW ABOUT THAT!!");
                            break;
                        } else if (noEntered(input)) {
                            System.out.println("I FEEL YOUR ARITHMETIC IS IN ERROR.");
                            break;
                        } else {
                            System.out.println("EH?  I DON'T UNDERSTAND '" + input + "'  TRY 'YES' OR 'NO'.");
                        }
                    } while (true);

                    gameState = GAME_STATE.PLAY_AGAIN;
                    break;

                case PLAY_AGAIN:
                    System.out.println("LET'S TRY ANOTHER");
                    gameState = GAME_STATE.GET_INPUTS;
                    break;
            }

            // Original basic program looped until CTRL-C
        } while (true);
    }

    private void intro() {
        System.out.println(addSpaces(33) + "NICOMA");
        System.out.println(addSpaces(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("BOOMERANG PUZZLE FROM ARITHMETICA OF NICOMACHUS -- A.D. 90!");
        System.out.println();
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
        return kbScanner.nextLine();
    }

    /**
     * Return a string of x spaces
     *
     * @param spaces number of spaces required
     * @return String with number of spaces
     */
    private String addSpaces(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }

    /**
     * Checks whether player entered Y or YES to a question.
     *
     * @param text player string from kb
     * @return true of Y or YES was entered, otherwise false
     */
    private boolean yesEntered(String text) {
        return stringIsAnyValue(text, "Y", "YES");
    }

    /**
     * Checks whether player entered N or NO to a question.
     *
     * @param text player string from kb
     * @return true of N or NO was entered, otherwise false
     */
    private boolean noEntered(String text) {
        return stringIsAnyValue(text, "N", "NO");
    }

    /**
     * Check whether a string equals one of a variable number of values
     * Useful to check for Y or YES for example
     * Comparison is case insensitive.
     *
     * @param text   source string
     * @param values a range of values to compare against the source string
     * @return true if a comparison was found in one of the variable number of strings passed
     */
    private boolean stringIsAnyValue(String text, String... values) {

        return Arrays.stream(values).anyMatch(str -> str.equalsIgnoreCase(text));
    }

    public static void main(String[] args) throws Exception {

        Nicomachus nicomachus = new Nicomachus();
        nicomachus.play();
    }
}

```

# `64_Nicomachus/javascript/nicomachus.js`

这段代码定义了两个函数，分别是`print()`和`input()`。

`print()`函数的作用是在页面上打印一段字符串，将字符串添加到页面上的一行中。

`input()`函数的作用是接收用户输入的字符串，并返回该字符串的ASCII值。该函数通过监听输入元素的`keydown`事件来获取用户输入的字符串，当用户按下回车键时，将获取到的字符串打印到页面上，并从页面上移除输入元素。


```
// NICOMACHUS
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

This is a program that simulates a game of莫神秘事件。在这个游戏中，玩家需要猜测一个三位数，程序会通过不断地除以3、5、7，来猜测这个数，直到猜对为止。如果猜错了，程序会告诉玩家，让玩家重新猜测。




```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var str;
var b;

// Main program
async function main()
{
    print(tab(33) + "NICOMA\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("BOOMERANG PUZZLE FROM ARITHMETICA OF NICOMACHUS -- A.D. 90!\n");
    while (1) {
        print("\n");
        print("PLEASE THINK OF A NUMBER BETWEEN 1 AND 100.\n");
        print("YOUR NUMBER DIVIDED BY 3 HAS A REMAINDER OF");
        a = parseInt(await input());
        print("YOUR NUMBER DIVIDED BY 5 HAS A REMAINDER OF");
        b = parseInt(await input());
        print("YOUR NUMBER DIVIDED BY 7 HAS A REMAINDER OF");
        c = parseInt(await input());
        print("\n");
        print("LET ME THINK A MOMENT...\n");
        print("\n");
        d = 70 * a + 21 * b + 15 * c;
        while (d > 105)
            d -= 105;
        print("YOUR NUMBER WAS " + d + ", RIGHT");
        while (1) {
            str = await input();
            print("\n");
            if (str == "YES") {
                print("HOW ABOUT THAT!!\n");
                break;
            } else if (str == "NO") {
                print("I FEEL YOUR ARITHMETIC IS IN ERROR.\n");
                break;
            } else {
                print("EH?  I DON'T UNDERSTAND '" + str + "'  TRY 'YES' OR 'NO'.\n");
            }
        }
        print("\n");
        print("LET'S TRY ANOTHER.\n");
    }
}

```

这是 C 语言中的一个简单程序，包含一个名为 main 的函数，但没有定义任何变量或参数。

一般来说，这个程序的作用是在主函数中声明函数，但没有函数体，也就是不执行任何操作或返回结果。

具体来说，这个程序的作用是告诉编译器这个程序要做什么，它将编译为一个可执行文件，然后通过调用者的操作来执行程序。如果程序成功运行，它将不会做任何特别的事情，只是一个空载的程序。


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


# `64_Nicomachus/python/nicomachus.py`

这段代码是一个用于数学练习的演示，名叫"NICOMA"。它将打印出以下文本："NICOMA数学练习/演示"。

代码的主要目的是提供一个数学练习，但没有输出任何内容。它使用了Python的time模块来等待用户执行它，因为该代码在练习期间停止运行。


```
"""
NICOMACHUS

Math exercise/demonstration

Ported by Dave LeCompte
"""

# PORTING NOTE
#
# The title, as printed ingame, is "NICOMA", hinting at a time when
# filesystems weren't even 8.3, but could only support 6 character
# filenames.

import time


```

这段代码定义了一个名为 `get_yes_or_no` 的函数，用于获取用户输入的文本，并输出 "是" 或 "否"。

然后，代码定义了一个名为 `play_game` 的函数，用于让用户猜一个1到100之间的随机整数。如果用户猜对，函数会打印结果并退出游戏。如果猜错，函数会提示用户重新猜测，并继续循环尝试。

在 `play_game` 函数中，首先让用户猜一个1到100之间的随机整数，然后计算这个数除以3、5、7的余数，并将结果打印出来。接着，代码会等待一段时间(2.5秒)，然后再让用户重新猜测。如果用户在两次尝试后仍然猜对，函数会打印一个消息并退出游戏。否则，函数会提示用户他们的数学计算出现了错误。然后，函数会再次尝试让用户猜对。


```
def get_yes_or_no() -> bool:
    while True:
        response = input().upper()
        if response == "YES":
            return True
        elif response == "NO":
            return False
        print(f"EH?  I DON'T UNDERSTAND '{response}'  TRY 'YES' OR 'NO'.")


def play_game() -> None:
    print("PLEASE THINK OF A NUMBER BETWEEN 1 AND 100.")
    print("YOUR NUMBER DIVIDED BY 3 HAS A REMAINDER OF")
    a = int(input())
    print("YOUR NUMBER DIVIDED BY 5 HAS A REMAINDER OF")
    b = int(input())
    print("YOUR NUMBER DIVIDED BY 7 HAS A REMAINDER OF")
    c = int(input())
    print()
    print("LET ME THINK A MOMENT...")
    print()

    time.sleep(2.5)

    d = (70 * a + 21 * b + 15 * c) % 105

    print(f"YOUR NUMBER WAS {d}, RIGHT?")

    response = get_yes_or_no()

    if response:
        print("HOW ABOUT THAT!!")
    else:
        print("I FEEL YOUR ARITHMETIC IS IN ERROR.")
    print()
    print("LET'S TRY ANOTHER")


```

这段代码是一个Python程序，名为“main”。程序的主要作用是输出一个有趣的数学谜题，并提示用户输入答案。现在让我们逐步分析程序的功能。

1. 首先，程序定义了一个名为“main”的函数，该函数没有返回值。函数内部使用print函数输出了一个数学谜题的文本，其中包括33个空格和15个换行符。这部分代码的作用是输出一个提示用户需要输入答案的数学谜题。

2. 接下来，程序使用print函数再次输出了另一个数学谜题的文本。谜题包含一个空格和一个换行符，这部分代码的作用与上面部分相同，输出一个提示用户需要输入答案的数学谜题。

3. 在这两部分代码输出完谜题之后，程序使用while True循环，该循环将一直运行，直到程序被手动中断。这部分代码的作用是重复输出谜题并提示用户输入答案。

4. while True循环中的代码调用了一个名为“play_game”的函数，但这是另一个定义在外的函数，不是main函数内部的。play_game函数的作用未知，但可以推测它与谜题的答案有关。

5. 最后，程序通过调用main函数来执行整个程序。main函数内部的代码在整个程序中都是运行的，包括前面定义的函数和循环。


```
def main() -> None:
    print(" " * 33 + "NICOMA")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

    print("BOOMERANG PUZZLE FROM ARITHMETICA OF NICOMACHUS -- A.D. 90!")
    print()
    while True:
        play_game()


if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Nim

NIM is one of the oldest two-person games known to man; it is believed to have originated in ancient China. The name, which was coined by the first mathematician to analyze it, comes from an archaic English verb which means to steal or to take away. Objects are arranged in rows between the two opponents as in the following example:
|         |       |           |
|---------|-------|-----------|
| XXXXXXX | Row 1 | 7 Objects |
| XXXXX   | Row 2 | 5 Objects |
| XXX     | Row 3 | 3 Objects |
| X       | Row 4 | 1 Object  |

Opponents take turns removing objects until there are none left. The one who picks up the last object wins. The moves are made according to the following rules:
1. On any given turn only objects from one row may be removed. There is no restriction on which row or on how many objects you remove. Of course, you cannot remove more than are in the row.
2. You cannot skip a move or remove zero objects.

The winning strategy can be mathematically defined, however, rather than presenting it here, we’d rather let you find it on your own. HINT: Play a few games with the computer and mark down on a piece of paper the number of objects in each stack (in binary!) after each move. Do you see a pattern emerging?

This game of NIM is from Dartmouth College and allows you to specify any starting size for the four piles and also a win option. To play traditional NIM, you would simply specify 7,5,3 and 1, and win option 1.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=118)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=133)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

This can be a real challenge to port because of all the `GOTO`s going out of loops down to code. You may need breaks and continues, or other techniques.

#### Known Bugs

- If, after the player moves, all piles are gone, the code prints "MACHINE LOSES" regardless of the win condition (when line 1550 jumps to line 800).  This should instead jump to line 800 ("machine loses") if W=1, but jump to 820 ("machine wins") if W=2.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `65_Nim/javascript/nim.js`

该代码定义了一个名为 print 的函数，该函数将字符串转换为小写并返回其 JavaScript 代码。

接下来是一个名为 input 的函数，该函数提示用户输入一行字符串，并返回一个 Promise 对象。该函数创建一个带有 Focus 属性的 INPUT 元素，将其设置为文本类型，长度为 50，并将结果添加到页面上。然后，该函数将焦点添加到 INPUT 元素上，并设置一个事件监听器，以便在按下回车键时接收输入并将其作为参数传递给 print 函数。在该事件处理程序中，如果事件参数的键是 13（回车键），则将其设置为输入的字符串，删除 INPUT 元素，并在页面上打印该字符串，并将其保存在 Promise 对象中。


```
// NIM
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

This appears to be a program written in the programming language。 It is a simple game where two players take turns removing tiles from a game board until one of them wins or the game is completed. The game board is divided into rows and columns, with each tile having a unique integer value. The player who removes the last tile from the board wins the game. If the game is completed, or if the player who removes the tile has run out of moves, the game ends and the other player wins. If the player who removes the tile clicks "NO" to start a new game, the game starts again.

The program starts by defining some global variables at the beginning of the game. These variables include the game board, the size of the game board, and the current size of the piles. The game board is then initialized with a loop that goes through each tile on the board, removing any tile that has a value of 0.

The player's first move is then made by the player by calling a function called "print()" which displays the current state of the game board. After that, the player is prompted to enter the number of the tile to be removed. The number is then stored in the "x" variable and passed to the "parseInt()" function to convert the number to an integer.

If the number entered by the player is not a valid integer, the program continues to the next iteration. If the number is valid, the tile is then removed from its position on the game board and the player's score is updated accordingly.

After the game board has been completely updated, the program displays a prompt asking if the player wants to start a new game or if they want to continue with the current game. If the player chooses to start a new game, the program displays another prompt asking if the player wants to continue or if they want to end the game. If the player chooses to continue, the program displays a message indicating that the game has been won by the player who removed the tile, or if the game is still being played. If the player chooses to end the game, the program displays a message indicating that the game has ended due to the player's score.

The program also includes a function called "game_completed()" which is used to check if the game is already won by one of the players. If the game is won, the program displays a message and the game is over. If not, the program displays a message and the player's score is updated accordingly.

The program also includes a function called


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var a = [];
var b = [];
var d = [];

// Main program
async function main()
{
    print(tab(33) + "NIM\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    for (i = 1; i <= 100; i++) {
        a[i] = 0;
        b[i] = [];
        for (j = 0; j <= 10; j++)
            b[i][j] = 0;
    }
    d[0] = 0;
    d[1] = 0;
    d[2] = 0;
    print("DO YOU WANT INSTRUCTIONS");
    while (1) {
        str = await input();
        str = str.toUpperCase();
        if (str == "YES" || str == "NO")
            break;
        print("PLEASE ANSWER YES OR NO\n");
    }
    if (str == "YES") {
        print("THE GAME IS PLAYED WITH A NUMBER OF PILES OF OBJECTS.\n");
        print("ANY NUMBER OF OBJECTS ARE REMOVED FROM ONE PILE BY YOU AND\n");
        print("THE MACHINE ALTERNATELY.  ON YOUR TURN, YOU MAY TAKE\n");
        print("ALL THE OBJECTS THAT REMAIN IN ANY PILE, BUT YOU MUST\n");
        print("TAKE AT LEAST ONE OBJECT, AND YOU MAY TAKE OBJECTS FROM\n");
        print("ONLY ONE PILE ON A SINGLE TURN.  YOU MUST SPECIFY WHETHER\n");
        print("WINNING IS DEFINED AS TAKING OR NOT TAKING THE LAST OBJECT,\n");
        print("THE NUMBER OF PILES IN THE GAME, AND HOW MANY OBJECTS ARE\n");
        print("ORIGINALLY IN EACH PILE.  EACH PILE MAY CONTAIN A\n");
        print("DIFFERENT NUMBER OF OBJECTS.\n");
        print("THE MACHINE WILL SHOW ITS MOVE BY LISTING EACH PILE AND THE\n");
        print("NUMBER OF OBJECTS REMAINING IN THE PILES AFTER  EACH OF ITS\n");
        print("MOVES.\n");
    }
    while (1) {
        print("\n");
        while (1) {
            print("ENTER WIN OPTION - 1 TO TAKE LAST, 2 TO AVOID LAST");
            w = parseInt(await input());
            if (w == 1 || w == 2)
                break;
        }
        while (1) {
            print("ENTER NUMBER OF PILES");
            n = parseInt(await input());
            if (n >= 1 && n <= 100)
                break;
        }
        print("ENTER PILE SIZES\n");
        for (i = 1; i <= n; i++) {
            while (1) {
                print(i + " ");
                a[i] = parseInt(await input());
                if (a[i] >= 1 && a[i] <= 2000)
                    break;
            }
        }
        print("DO YOU WANT TO MOVE FIRST");
        while (1) {
            str = await input();
            str = str.toUpperCase();
            if (str == "YES" || str == "NO")
                break;
            print("PLEASE ANSWER YES OR NO.\n");
        }
        if (str == "YES")
            player_first = true;
        else
            player_first = false;
        while (1) {
            if (!player_first) {
                if (w != 1) {
                    c = 0;
                    for (i = 1; i <= n; i++) {
                        if (a[i] == 0)
                            continue;
                        c++;
                        if (c == 3)
                            break;
                        d[c] = i;
                    }
                    if (i > n) {
                        if (c == 2) {
                            if (a[d[1]] == 1 || a[d[2]] == 1) {
                                print("MACHINE WINS\n");
                                break;
                            }
                        } else {
                            if (a[d[1]] > 1)
                                print("MACHINE WINS\n");
                            else
                                print("MACHINE LOSES\n");
                            break;
                        }

                    } else {
                        c = 0;
                        for (i = 1; i <= n; i++) {
                            if (a[i] > 1)
                                break;
                            if (a[i] == 0)
                                continue;
                            c++;
                        }
                        if (i > n && c % 2) {
                            print("MACHINE LOSES\n");
                            break;
                        }
                    }
                }
                for (i = 1; i <= n; i++) {
                    e = a[i];
                    for (j = 0; j <= 10; j++) {
                        f = e / 2;
                        b[i][j] = 2 * (f - Math.floor(f));
                        e = Math.floor(f);
                    }
                }
                for (j = 10; j >= 0; j--) {
                    c = 0;
                    h = 0;
                    for (i = 1; i <= n; i++) {
                        if (b[i][j] == 0)
                            continue;
                        c++;
                        if (a[i] <= h)
                            continue;
                        h = a[i];
                        g = i;
                    }
                    if (c % 2)
                        break;
                }
                if (j < 0) {
                    do {
                        e = Math.floor(n * Math.random() + 1);
                    } while (a[e] == 0) ;
                    f = Math.floor(a[e] * Math.random() + 1);
                    a[e] -= f;
                } else {
                    a[g] = 0;
                    for (j = 0; j <= 10; j++) {
                        b[g][j] = 0;
                        c = 0;
                        for (i = 1; i <= n; i++) {
                            if (b[i][j] == 0)
                                continue;
                            c++;
                        }
                        a[g] = a[g] + (c % 2) * Math.pow(2, j);
                    }
                    if (w != 1) {
                        c = 0;
                        for (i = 1; i <= n; i++) {
                            if (a[i] > 1)
                                break;
                            if (a[i] == 0)
                                continue;
                            c++;
                        }
                        if (i > n && c % 2 == 0)
                            a[g] = 1 - a[g];
                    }
                }
                print("PILE  SIZE\n");
                for (i = 1; i <= n; i++)
                    print(" " + i + "  " + a[i] + "\n");
                if (w != 2) {
                    if (game_completed()) {
                        print("MACHINE WINS");
                        break;
                    }
                }
            } else {
                player_first = false;
            }
            while (1) {
                print("YOUR MOVE - PILE , NUMBER TO BE REMOVED");
                str = await input();
                x = parseInt(str);
                y = parseInt(str.substr(str.indexOf(",") + 1));
                if (x < 1 || x > n)
                    continue;
                if (y < 1 || y > a[x])
                    continue;
                break;
            }
            a[x] -= y;
            if (game_completed()) {
                print("MACHINE LOSES");
                break;
            }
        }
        print("DO YOU WANT TO PLAY ANOTHER GAME");
        while (1) {
            str = await input();
            str = str.toUpperCase();
            if (str == "YES" || str == "NO")
                break;
            print("PLEASE ANSWER YES OR NO.\n");
        }
        if (str == "NO")
            break;
    }
}

```

这是一个 JavaScript 函数，名为 `game_completed()`。函数的作用是判断游戏是否完成，即所有隐藏的按钮都已经被点击过。

函数的实现过程如下：

1. 通过一个 `for` 循环，从 1 到 `n`(假设 `n` 是游戏中的按钮数)进行遍历。
2. 在每次遍历过程中，使用一个 `if` 语句判断当前是否已经点击过该按钮。如果是，函数返回 `false`。
3. 循环结束后，如果仍然没有找到任何按钮被点击过，函数返回 `true`。

因此，这个函数的输出结果是布尔值 `true` 或 `false`。


```
function game_completed()
{
    for (var i = 1; i <= n; i++) {
        if (a[i] != 0)
            return false;
    }
    return true;
}

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


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


# `65_Nim/python/Traditional_NIM.py`

This is a game where the player has to pick up all the objects ( represented by the "O" on a peg) from a pile, but they can only pick up one object at a time. The player has to follow certain rules, such as taking at least one object and not taking the same object twice. The machine can remove one of the objects at any pile and the pile can be passed on to the next player.

There is also a "win" condition, which is defined as the player picking the last remaining object.


```
import random
from typing import Tuple


class NIM:
    def __init__(self) -> None:
        self.piles = {1: 7, 2: 5, 3: 3, 4: 1}

    def remove_pegs(self, command) -> None:
        try:

            pile, num = command.split(",")
            num = int(num)
            pile = int(pile)

        except Exception as e:

            if "not enough values" in str(e):
                print(
                    '\nNot a valid command. Your command should be in the form of "1,3", Try Again\n'
                )

            else:
                print("\nError, Try again\n")
            return None

        if self._command_integrity(num, pile):
            self.piles[pile] -= num
        else:
            print("\nInvalid value of either Peg or Pile\n")

    def get_ai_move(self) -> Tuple[int, int]:
        possible_pile = []
        for k, v in self.piles.items():
            if v != 0:
                possible_pile.append(k)

        pile = random.choice(possible_pile)

        num = random.randint(1, self.piles[pile])

        return pile, num

    def _command_integrity(self, num, pile) -> bool:
        return pile <= 4 and pile >= 1 and num <= self.piles[pile]

    def print_pegs(self) -> None:
        for pile, peg in self.piles.items():
            print("Pile {} : {}".format(pile, "O " * peg))

    def help(self) -> None:
        print("-" * 10)
        print('\nThe Game is player with a number of Piles of Objects("O" == one peg)')
        print("\nThe Piles are arranged as given below(Tradional NIM)\n")
        self.print_pegs()
        print(
            '\nAny Number of of Objects are removed one pile by "YOU" and the machine alternatively'
        )
        print("\nOn your turn, you may take all the objects that remain in any pile")
        print("but you must take ATLEAST one object")
        print("\nAnd you may take objects from only one pile on a single turn.")
        print("\nThe winner is defined as the one that picks the last remaning object")
        print("-" * 10)

    def check_for_win(self) -> bool:
        sum = 0
        for v in self.piles.values():
            sum += v

        return sum == 0


```

这段代码是一个Python游戏程序，它包括了游戏初始化、游戏循环、玩家和计算机的操作等基本功能。下面是这个程序的具体解释：

1. 游戏初始化：在程序开始时，创建了一个名为`NIM`的游戏引擎对象，并输出了一条消息，告诉玩家这是一个NIM游戏。
2. 帮助功能：当玩家输入"YES"时，程序会输出一些游戏帮助信息，告诉玩家如何使用这个游戏。
3. 游戏循环：在游戏循环中，程序会先输出一些游戏界面信息，然后等待玩家的操作。玩家可以输入"YES"来获得帮助信息，或者输入命令来移动游戏中的棋子。程序会处理玩家的操作，并检查游戏是否结束。如果游戏结束，程序会输出相应的信息并提示玩家再次输入命令。
4. 玩家操作：当玩家输入命令时，程序会尝试执行相应的操作，包括移动棋子和检查游戏是否结束。如果玩家赢得了游戏，程序会输出一条消息并提示玩家再次输入命令。如果游戏没有结束，程序会继续执行下一次操作。
5. 计算机操作：程序会从游戏引擎中获取人工智能的移动棋子命令，并输出相应的信息。然后程序会执行相应的操作，并检查游戏是否结束。如果游戏结束，程序会提示玩家。

总之，这个程序是一个基本的NIM游戏，包括了游戏初始化、游戏循环、玩家和计算机的操作等功能。


```
def main() -> None:
    # Game initialization
    game = NIM()

    print("Hello, This is a game of NIM")
    help = input("Do You Need Instruction (YES or NO): ")

    if help.lower() == "yes":
        game.help()

    # Start game loop
    input("\nPress Enter to start the Game:\n")
    end = False
    while True:
        game.print_pegs()

        # Players Move
        command = input("\nYOUR MOVE - Number of PILE, Number of Object? ")
        game.remove_pegs(command)
        end = game.check_for_win()
        if end:
            print("\nPlayer Wins the Game, Congratulations!!")
            input("\nPress any key to exit")
            break

        # Computers Move
        ai_command = game.get_ai_move()
        print(
            "\nA.I MOVE - A.I Removed {} pegs from Pile {}".format(
                ai_command[1], ai_command[0]
            )
        )
        game.remove_pegs(str(ai_command[0]) + "," + str(ai_command[1]))
        end = game.check_for_win()
        if end:
            print("\nComputer Wins the Game, Better Luck Next Time\n")
            input("Press any key to exit")
            break


```

这段代码是一个Python程序中的一个if语句，它的作用是判断当前程序是否作为主程序运行。如果当前程序作为主程序运行，那么程序将跳转到if语句的内部，否则程序将继续执行if语句内部的代码。

if __name__ == "__main__":
   main()

在Python中，__name__是一个特殊字符，用于标识当前程序是否作为主程序运行。如果当前程序的__name__和 "__main__" 等于同一个字符串，那么程序将跳转到__main__函数内部，否则程序将继续执行if语句内部的代码。

因此，这段代码的作用是检查当前程序是否作为主程序运行，如果是，就执行main函数内部的代码，否则就跳过if语句内部的代码。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Number

In contrast to other number guessing games where you keep guessing until you get the random number selected by the computer (GUESS, TRAP, STARS, etc.), in this game you only get one guess per play and you gain or lose points depending upon how close your guess is to the random number selected by the computer. You occasionally get a jackpot which will double your point count. You win when you get 500 points.

Tom Adametx wrote this program while a student at Curtis Junior High School in Sudbury, Massachusetts.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=121)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=136)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

Contrary to the description, the computer picks *five* random numbers per turn, not one.  You are not rewarded based on how close your guess is to one number, but rather to which of these five random numbers (if any) it happens to match exactly.


# `66_Number/csharp/program.cs`

This is a C# program that simulates a game of猜数字. The game starts with a random number of points and a goal of 500 points. The player is prompted to guess a number, and based on the number they guess, the game adjusts the points awarded or removed. If the player correctly guesses the number, the game will end with a win. Otherwise, the game will continue until the player runs out of points or they guess incorrectly.

The `Number` class in this program uses the `NextDouble` method from the `System.Random` namespace to generate random numbers. The class also has methods for getting the random numbers for the four additional numbers.

The `Play` method in this class is the main method for the game. It displays the Introduction screen and then prompts the player to enter their guess. The player can then continue guessing until they correctly guess the number or run out of points.

The `PlayTheGame` method is the same as the `Play` method but it calls the `Play` method in the `Number` class.

The `main` method in this program is the entry point for the program. It starts by displaying the Introduction screen and then calls the `PlayTheGame` method.


```
using System.Text;

namespace Number
{
    class Number
    {
        private void DisplayIntro()
        {
            Console.WriteLine();
            Console.WriteLine("NUMBER".PadLeft(23));
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("You have 100 points.  By guessing numbers from 1 to 5, you");
            Console.WriteLine("can gain or lose points depending upon how close you get to");
            Console.WriteLine("a random number selected by the computer.");
            Console.WriteLine();
            Console.WriteLine("You occaisionally will get a jackpot which will double(!)");
            Console.WriteLine("your point count.  You win when you get 500 points.");
            Console.WriteLine();

        }
        private int PromptForGuess()
        {
            bool Success = false;
            int Guess = 0;

            while (!Success)
            {
                Console.Write("Guess a number from 1 to 5? ");
                string LineInput = Console.ReadLine().Trim().ToLower();

                if (int.TryParse(LineInput, out Guess))
                {
                    if (Guess >= 0 && Guess <= 5)
                        Success = true;
                }
                else
                    Console.WriteLine("Please enter a number between 1 and 5.");
            }

            return Guess;
        }

        private void GetRandomNumbers(out int Random1, out int Random2, out int Random3, out int Random4, out int Random5)
        {
            Random rand = new Random();

            // Get a unique set of random numbers between 1 and 5
            // I assume this is what the original BASIC  FNR(X)=INT(5*RND(1)+1) is doing
            Random1 = (int)(5 * rand.NextDouble() + 1);
            do
            {
                Random2 = (int)(5 * rand.NextDouble() + 1);
            } while (Random2 == Random1);
            do
            {
                Random3 = (int)(5 * rand.NextDouble() + 1);
            } while (Random3 == Random1 || Random3 == Random2);
            do
            {
                Random4 = (int)(5 * rand.NextDouble() + 1);
            } while (Random4 == Random1 || Random4 == Random2 || Random4 == Random3);
            do
            {
                Random5 = (int)(5 * rand.NextDouble() + 1);
            } while (Random5 == Random1 || Random5 == Random2 || Random5 == Random3 || Random5 == Random4);

        }
        private void Play()
        {

            int Points = 100;
            bool Win = false;
            int Random1, Random2, Random3, Random4, Random5;
            int Guess = 0;

            GetRandomNumbers(out Random1, out Random2, out Random3, out Random4, out Random5);

            while (!Win)
            {

                Guess = PromptForGuess();

                if (Guess == Random1)
                    Points -= 5;
                else if (Guess == Random2)
                    Points += 5;
                else if (Guess == Random3)
                {
                    Points += Points;
                    Console.WriteLine("You hit the jackpot!!!");
                }
                else if (Guess == Random4)
                    Points += 1;
                else if (Guess == Random5)
                    Points -= (int)(Points * 0.5);

                if (Points > 500)
                {
                    Console.WriteLine("!!!!You Win!!!! with {0} points.", Points);
                    Win = true;
                }
                else
                    Console.WriteLine("You have {0} points.", Points);
            }
        }

        public void PlayTheGame()
        {
            DisplayIntro();

            Play();
        }
    }
    class Program
    {
        static void Main(string[] args)
        {

            new Number().PlayTheGame();

        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `66_Number/java/1/Number.java`

This is a Java program that simulates a game of Pick 3, where the user tries to match a random number with a sequence of predefined numbers. The user can win the game if any number in the sequence matches the one they guessed. The game continues until the user correctly guesses a number or hits the jackpot, which will double the user's score.

The program first explains the rules of the game and then enters a loop where the user is prompted to guess a number. The program uses a `Scanner` object to read the user's input and a `Random` object to generate random numbers.

If the user enters a valid number, the program checks if it matches any of the predefined numbers and updates the user's score accordingly. If the user enters an invalid number, the program prints a message and continues to the next iteration.

The game continues until the user correctly guesses a number or hits the jackpot, at which point the program prints a message and returns.

Note that this program may not work correctly if there are issues with the user's input or if there are not enough numbers in the predefined sequence for the user to correctly guess a number.


```

import java.time.temporal.ValueRange;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;

public class Number {

    public static int points = 0;

    public static void printempty() { System.out.println(" "); }

    public static void print(String toprint) { System.out.println(toprint); }

    public static void main(String[] args) {
        print("YOU HAVE 100 POINTS.  BY GUESSING NUMBERS FROM 1 TO 5, YOU");
        print("CAN GAIN OR LOSE POINTS DEPENDING UPON HOW CLOSE YOU GET TO");
        print("A RANDOM NUMBER SELECTED BY THE COMPUTER.");
        printempty();
        print("YOU OCCASIONALLY WILL GET A JACKPOT WHICH WILL DOUBLE(!)");
        print("YOUR POINT COUNT.  YOU WIN WHEN YOU GET 500 POINTS.");
        printempty();

        try {
            while (true) {
                print("GUESS A NUMBER FROM 1 TO 5");


                Scanner numbersc = new Scanner(System.in);
                String numberstring = numbersc.nextLine();

                int number = Integer.parseInt(numberstring);

                if (!(number < 1| number > 5)) {

                    Random rand = new Random();

                    int randomNum = rand.nextInt((5 - 1) + 1) + 1;

                    if (randomNum == number) {
                        print("YOU HIT THE JACKPOT!!!");
                        points = points * 2;
                    } else if(ValueRange.of(randomNum, randomNum + 1).isValidIntValue(number)) {
                        print("+5");
                        points = points + 5;
                    } else if(ValueRange.of(randomNum - 1, randomNum + 2).isValidIntValue(number)) {
                        print("+1");
                        points = points + 1;
                    } else if(ValueRange.of(randomNum - 3, randomNum + 1).isValidIntValue(number)) {
                        print("-1");
                        points = points - 1;
                    } else {
                        print("-half");
                        points = (int) (points * 0.5);
                    }

                    print("YOU HAVE " + points + " POINTS.");
                }

                if (points >= 500) {
                    print("!!!!YOU WIN!!!! WITH " + points + " POINTS.");
                    return;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

```

# `66_Number/java/2/Number.java`

This is a Java program that simulates a game of猜测1到5之间的随机数，并根据猜测结果获得或失去相应的游戏币，玩家可以选择继续猜测直到猜对或得到500个游戏币为止。

程序首先引入随机数生成器和两个变量，用于跟踪玩家猜测的点数和游戏币数。

然后，程序调用了一个名为randomNumber的静态方法来生成一个随机的随机数，并将其与1到5之间的随机数进行比较，生成了一个值，该值将作为游戏币数。

接下来，程序中的if-else语句用于处理玩家猜测的数字与游戏中的随机数的关系。如果玩家猜测的数字与随机数相等，则游戏币数将增加；如果玩家猜测的数字比随机数大或小很多，则游戏币数将减少。否则，如果玩家猜测的数字在1到5之间，则继续猜测。

程序还使用了一个布尔变量done来指示游戏是否结束，如果点数猜测超过500，则done变量将变为true，游戏结束。

最后，程序打印出游戏开始的信息，生成随机数，并让玩家开始猜测。


```
import java.util.Scanner;

public class Number {

	public static void main(String[] args) {
		printIntro();
		int points = 100; //start with 100 points for the user

		Scanner scan = new Scanner(System.in);
		boolean done = false;
		while (!done) {
			System.out.print("GUESS A NUMBER FROM 1 TO 5? ");
			int g = scan.nextInt();

			//Initialize 5 random numbers between 1-5
			var r = randomNumber(1);
			var s = randomNumber(1);
			var t = randomNumber(1);
			var u = randomNumber(1);
			var v = randomNumber(1);

			if (r == g) {
				points -= 5;
			} else if (s == g) {
				points += 5;
			} else if (t == g) {
				points += points;
			} else if (u == g) {
				points += 1;
			} else if (v == g) {
				points -= points * 0.5;
			} else {
				continue; //Doesn't match any of our random numbers, so just ask for another guess
			}

			if (points > 500) {
				done = true;
			} else {
				System.out.println("YOU HAVE " + points + " POINTS.");
			}
		}

		System.out.println("!!!!YOU WIN!!!! WITH " + points + " POINTS.\n");
	}

	private static int randomNumber(int x) {
		//Note: 'x' is totally ignored as was in the original basic listing
		return (int) (5 * Math.random() + 1);
	}

	private static void printIntro() {
		System.out.println("                                NUMBER");
		System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
		System.out.println("\n\n\n");
		System.out.println("YOU HAVE 100 POINTS.  BY GUESSING NUMBERS FROM 1 TO 5, YOU");
		System.out.println("CAN GAIN OR LOSE POINTS DEPENDING UPON HOW CLOSE YOU GET TO");
		System.out.println("A RANDOM NUMBER SELECTED BY THE COMPUTER.");
		System.out.println("\n");
		System.out.println("YOU OCCASIONALLY WILL GET A JACKPOT WHICH WILL DOUBLE(!)");
		System.out.println("YOUR POINT COUNT.  YOU WIN WHEN YOU GET 500 POINTS.");
	}
}

```