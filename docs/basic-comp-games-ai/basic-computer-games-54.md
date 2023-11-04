# BasicComputerGames源码解析 54

# `53_King/python/king_variable_update.py`

很抱歉，我无法解释任何没有提供完整代码的请求。请提供完整的代码，以便我可以为您提供帮助。


```

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


## Porting notes

Variables:

* A: Available rallods (money)
* B: Current countrymen
* C: foreign_workers
* C1: foreign_workers_influx
* D: Available land (farmland=D-1000)
* F1: polution_deaths (last round)
* B5: died_contrymen (starvation + pollution)
* H: sm_sell_to_industry
* I: distributed_rallods
* J: planted_sq in a round
* K: pollution_control_spendings in a round
* X5: years in office
* N5: YEARS_IN_TERM - how many years one term in office has
* P1: population_change (positive means people come, negative means people leave)
* W: land_buy_price
* V9: planting_cost
* U2: crop_loss
* V1-V2: Earnings from tourist trade
* V3: tourism_earnings
* T1: crop_loss_last_year
* W: land_buy_price
* X: only show an error message once

Functions:

* `RND(1)`: `random.random()`
* `INT(...)`: `int(...)`
* `ABS(...)`: `abs(...)`

Bugs: See [53 King README](../README.md)

Implicit knowledge:

* `COST_OF_LIVING`: One countryman needs 100 for food. Otherwise they will die of starvation
* `COST_OF_FUNERAL`: A funeral costs 9


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


King
====

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [rust](https://www.rust-lang.org/).

Porting Notes
-------------

### Floats

The original code implicitly uses floating point numbers in many places which are explicitly cast to integers. In this port, I avoided using floats and tried to replicate the behaviour using just integers. It is possible that I missed some places where rounding a value would have made a difference. If you find such a bug, please notify me or make implement a fix yourself.

### Signed Numbers

I used unsigned integers for most of the program because it was easier than to check for negative values all the time. Unfortunately, that made the code a bit whacky in one or two places.

Since I only allow input of positive numbers, it is not possible to exit the game when entering the stats to resume a game, which would be possible by entering negative numbers in the original game.

### Bugs

I tried to fix all bugs listed in the [main README for King](../README.md). I have tested this implementation a bit but not extensively, so there may be some portation bugs. If you find them, you are free to fix them.

Future Development
------------------

I plan to add some tests and tidy up the code a bit, but this version should be feature-complete.


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Letter

LETTER is similar to the game GUESS in which you guess a number chosen by the computer; in this program, the computer picks a random letter of the alphabet and you must guess which one it is using the clues provided as you go along. It should not take you more than five guesses to get the mystery letter.

The program which appears here is loosely based on the original written by Bob Albrect of People’s Computer Company.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=99)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=114)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `54_Letter/csharp/Game.cs`

This is a class that defines a game where the player has to guess a randomly generated number within a certain number of guesses. The class includes methods for displaying information about the game, such as the number of guesses remaining and the number of successful guesses. It also includes a method for displaying a success message and a method for getting valid input from the keyboard.

The `DisplaySuccessMessage` method displays a success message and the number of guesses made to defeat the message. If the player has more than the maximum number of guesses allowed, it will display a message and a red color to indicate failure. If the player has made a successful guess within the maximum number of guesses, it will display a green color and a message indicating that they got it.

The `GetCharacterFromKeyboard` method reads a valid input from the keyboard and converts it to upper case. It returns the character as a `char` variable.

Overall, this class is useful for displaying information about a game and getting valid input from the keyboard.


```
﻿namespace Letter
{
    internal static class Game
    {
        /// <summary>
        /// Maximum number of guesses.
        /// Note the program doesn't enforce this - it just displays a message if this is exceeded.
        /// </summary>
        private const int MaximumGuesses = 5;

        /// <summary>
        /// Main game loop.
        /// </summary>
        public static void Play()
        {
            DisplayIntroductionText();

            // Keep playing forever, or until the user quits.
            while (true)
            {
                PlayRound();
            }
        }

        /// <summary>
        /// Play a single round.
        /// </summary>
        internal static void PlayRound()
        {
            var gameState = new GameState();
            DisplayRoundIntroduction();

            char letterInput = '\0'; // Set the initial character to something that's not A-Z.
            while (letterInput != gameState.Letter)
            {
                letterInput = GetCharacterFromKeyboard();
                gameState.GuessesSoFar++;
                DisplayGuessResult(gameState.Letter, letterInput);
            }
            DisplaySuccessMessage(gameState);
        }

        /// <summary>
        /// Display an introduction when the game loads.
        /// </summary>
        internal static void DisplayIntroductionText()
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("LETTER");
            Console.WriteLine("Creative Computing, Morristown, New Jersey.");
            Console.WriteLine("");

            Console.ForegroundColor = ConsoleColor.DarkGreen;
            Console.WriteLine("Letter Guessing Game");
            Console.WriteLine("I'll think of a letter of the alphabet, A to Z.");
            Console.WriteLine("Try to guess my letter and I'll give you clues");
            Console.WriteLine("as to how close you're getting to my letter.");
            Console.WriteLine("");

            Console.ResetColor();
        }

        /// <summary>
        /// Display introductionary text for each round.
        /// </summary>
        internal static void DisplayRoundIntroduction()
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("O.K., I have a letter. Start guessing.");

            Console.ResetColor();
        }

        /// <summary>
        /// Display text depending whether the guess is lower or higher.
        /// </summary>
        internal static void DisplayGuessResult(char letterToGuess, char letterInput)
        {
            Console.BackgroundColor = ConsoleColor.White;
            Console.ForegroundColor = ConsoleColor.Black;
            Console.Write(" " + letterInput + " ");

            Console.ResetColor();
            Console.ForegroundColor = ConsoleColor.Gray;
            Console.Write(" ");
            if (letterInput != letterToGuess)
            {
                if (letterInput > letterToGuess)
                {
                    Console.WriteLine("Too high. Try a lower letter");
                }
                else
                {
                    Console.WriteLine("Too low. Try a higher letter");
                }
            }
            Console.ResetColor();
        }

        /// <summary>
        /// Display success, and the number of guesses.
        /// </summary>
        internal static void DisplaySuccessMessage(GameState gameState)
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"You got it in {gameState.GuessesSoFar} guesses!!");
            if (gameState.GuessesSoFar > MaximumGuesses)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"But it shouldn't take more than {MaximumGuesses} guesses!");
            }
            else
            {
                Console.WriteLine("Good job !!!!!");
            }
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("");
            Console.WriteLine("Let's play again.....");

            Console.ResetColor();
        }

        /// <summary>
        /// Get valid input from the keyboard: must be an alpha character. Converts to upper case if necessary.
        /// </summary>
        internal static char GetCharacterFromKeyboard()
        {
            char letterInput;
            do
            {
                var keyPressed = Console.ReadKey(true);
                letterInput = Char.ToUpper(keyPressed.KeyChar); // Convert to upper case immediately.
            } while (!Char.IsLetter(letterInput)); // If the input is not a letter, wait for another letter to be pressed.
            return letterInput;
        }
    }
}

```

# `54_Letter/csharp/GameState.cs`

这段代码是一个C#类，名为"Letter"，描述了一个游戏中的当前状态。这个游戏状态包括一个随机生成的字母，"guessesSoFar"计数器，以及一个"Letter"属性，用于存储当前正在猜测的字母。

在代码中，还定义了一个名为"GetRandomLetter"的静态方法，用于生成一个随机字母，该字母范围为A-Z。

这个游戏的初始化方法是构造函数，使用了一个随机数生成器，随机生成一个字母并将其赋值给"Letter"属性，然后将"GuessesSoFar"属性初始化为0。

游戏循环使用了一个无限循环，每次循环将用户输入的字母转换为字符，然后检查用户是否猜对了。如果猜对了，"GuessesSoFar"计数器加1。如果猜错了，就重新生成一个随机字母，让用户继续猜测。

总之，这段代码描述了一个简单的游戏，用户每次猜测时，程序会随机生成一个字母，然后根据用户猜测的字母是否正确来判断猜测成功与否。


```
﻿namespace Letter
{
    /// <summary>
    /// Holds the current state.
    /// </summary>
    internal class GameState
    {
        /// <summary>
        /// Initialise the game state with a random letter.
        /// </summary>
        public GameState()
        {
            Letter = GetRandomLetter();
            GuessesSoFar = 0;
        }

        /// <summary>
        /// The letter that the user is guessing.
        /// </summary>
        public char Letter { get; set; }

        /// <summary>
        /// The number of guesses the user has had so far.
        /// </summary>
        public int GuessesSoFar { get; set; }

        /// <summary>
        /// Get a random character (A-Z) for the user to guess.
        /// </summary>
        internal static char GetRandomLetter()
        {
            var random = new Random();
            var randomNumber = random.Next(0, 26);
            return (char)('A' + randomNumber);
        }
    }
}

```

# `54_Letter/csharp/Program.cs`

这是一个使用JavaScript的代码片段，其中包括以下几个主要部分：

1. `using Letter;`：这是一个导入Letter包的指令，这个包可能包含了JavaScript库或者第三方组件。

2. `Game.Play();`：这个指令调用了一个名为Game的实例的Play方法。根据作用域，这个Game可能是一个全局变量或者一个函数，它们提供了一些游戏相关的服务或者功能。调用Play方法可能是为了启动一个游戏或者模拟游戏场景。

3. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。

4. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。

5. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。

6. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。

7. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。

8. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。

9. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。

10. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。

11. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。

12. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。

13. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。

14. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。

15. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。

16. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。

17. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。

18. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。

19. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。

20. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。

21. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。

22. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。

23. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。

24. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。

25. `{}`：这个部分可能是注释，但是根据上下文来看，它并没有提供太多有用信息。


```
﻿using Letter;

Game.Play();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `54_Letter/java/src/Letter.java`

This is a Java class that represents a game of "LetterGuessingGame".

The game starts with an initial state of "GameOver", and the player is provided with a hint of "3猜测内不会超过 optimalGuesses"。

The game then enters a loop where the player is provided with one letter each time, and the player must guess the letter by trying to figure out the number of spaces between the letter and the number of spaces provided.

The game provides clues to the player to help them guess the letter, and after a certain number of guesses, the game will check if the player's guess is correct.

It should be noted that the game will end when the player达成了 optimalGuesses。


```
import java.awt.*;
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Letter
 * <p>
 * Based on the Basic game of Letter here
 * https://github.com/coding-horror/basic-computer-games/blob/main/54%20Letter/letter.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Letter {

    public static final int OPTIMAL_GUESSES = 5;
    public static final int ASCII_A = 65;
    public static final int ALL_LETTERS = 26;

    private enum GAME_STATE {
        STARTUP,
        INIT,
        GUESSING,
        RESULTS,
        GAME_OVER
    }

    // Used for keyboard input
    private final Scanner kbScanner;

    // Current game state
    private GAME_STATE gameState;

    // Players guess count;
    private int playerGuesses;

    // Computers ascii code for a random letter between A..Z
    private int computersLetter;

    public Letter() {

        gameState = GAME_STATE.STARTUP;

        // Initialise kb scanner
        kbScanner = new Scanner(System.in);
    }

    /**
     * Main game loop
     */
    public void play() {

        do {
            switch (gameState) {

                // Show an introduction the first time the game is played.
                case STARTUP:
                    intro();
                    gameState = GAME_STATE.INIT;
                    break;

                case INIT:
                    playerGuesses = 0;
                    computersLetter = ASCII_A + (int) (Math.random() * ALL_LETTERS);
                    System.out.println("O.K., I HAVE A LETTER.  START GUESSING.");
                    gameState = GAME_STATE.GUESSING;
                    break;

                // Player guesses the number until they get it or run out of guesses
                case GUESSING:
                    String playerGuess = displayTextAndGetInput("WHAT IS YOUR GUESS? ").toUpperCase();

                    // Convert first character of input string to ascii
                    int toAscii = playerGuess.charAt(0);
                    playerGuesses++;
                    if (toAscii == computersLetter) {
                        gameState = GAME_STATE.RESULTS;
                        break;
                    }

                    if (toAscii > computersLetter) {
                        System.out.println("TOO HIGH.  TRY A LOWER LETTER.");
                    } else {
                        System.out.println("TOO LOW.  TRY A HIGHER LETTER.");
                    }
                    break;

                // Play again, or exit game?
                case RESULTS:
                    System.out.println();
                    System.out.println("YOU GOT IT IN " + playerGuesses + " GUESSES!!");
                    if (playerGuesses <= OPTIMAL_GUESSES) {
                        System.out.println("GOOD JOB !!!!!");
                        // Original game beeped 15 tims if you guessed in the optimal guesses or less
                        // Changed this to do a single beep only
                        Toolkit.getDefaultToolkit().beep();
                    } else {
                        // Took more than optimal number of guesses
                        System.out.println("BUT IT SHOULDN'T TAKE MORE THAN " + OPTIMAL_GUESSES + " GUESSES!");
                    }
                    System.out.println();
                    System.out.println("LET'S PLAN AGAIN.....");
                    gameState = GAME_STATE.INIT;
                    break;
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    public void intro() {
        System.out.println(simulateTabs(33) + "LETTER");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("LETTER GUESSING GAME");
        System.out.println();
        System.out.println("I'LL THINK OF A LETTER OF THE ALPHABET, A TO Z.");
        System.out.println("TRY TO GUESS MY LETTER AND I'LL GIVE YOU CLUES");
        System.out.println("AS TO HOW CLOSE YOU'RE GETTING TO MY LETTER.");
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
}

```

# `54_Letter/java/src/LetterGame.java`

这是一个Java程序，名为"LetterGame"。该程序定义了一个名为"LetterGame"的类。在这个类中，有一个名为"main"的静态方法。

在"main"方法中，创建了一个名为"Letter"的类对象，并调用该对象的"play"方法。

"Letter"类可能是一个自定义类，实现了Java中的"LetterGame"接口。在这里，我们创建了一个名为"Letter"的类，但它的具体实现并不影响我们理解代码的作用。

总的来说，这段代码在创建一个名为"LetterGame"的类对象，并使用该对象的方法"play"，使得程序开始输出一系列字母。


```
public class LetterGame {

    public static void main(String[] args) {

        Letter letter = new Letter();
        letter.play();
    }
}

```

# `54_Letter/javascript/letter.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是在文档中创建一个输出文本框（output 元素），然后将一个字符串（str）添加到该文本框中。

`input` 函数的作用是接收用户输入的字符串（inputStr），并在输入框中添加一个输入元素（inputElement）。然后将输入元素设置为输入类型为 text、长度为 50 的文本框，并将其添加到文档中的输出元素（output）中。接着，将输入元素的焦点设置，并添加一个 keydown 事件监听器，以便在用户按下回车键时接收输入并将其添加到输出中，并输出字符串。


```
// LETTER
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



This is a implementation of the alphabetical guessing game. The game will randomly choose a letter from the alphabet and the player will have to guess it. The player will have 26 attempts to guess the letter, and each guess will be evaluated by checking if the letter is higher or lower than the chosen letter. If the player makes too many guesses, the game will end and the player will be told that they should try again. If the player correctly guesses the letter, the game will end and the player will be given feedback on how close they were to the chosen letter.


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
    print(tab(33) + "LETTER\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("LETTER GUESSING GAME\n");
    print("\n");
    print("I'LL THINK OF A LETTER OF THE ALPHABET, A TO Z.\n");
    print("TRY TO GUESS MY LETTER AND I'LL GIVE YOU CLUES\n");
    print("AS TO HOW CLOSE YOU'RE GETTING TO MY LETTER.\n");
    while (1) {
        l = 65 + Math.floor(26 * Math.random());
        g = 0;
        print("\n");
        print("O.K., I HAVE A LETTER.  START GUESSING.\n");
        while (1) {

            print("\n");
            print("WHAT IS YOUR GUESS");
            g++;
            str = await input();
            a = str.charCodeAt(0);
            print("\n");
            if (a == l)
                break;
            if (a < l) {
                print("TOO LOW.  TRY A HIGHER LETTER.\n");
            } else {
                print("TOO HIGH.  TRY A LOWER LETTER.\n");
            }
        }
        print("\n");
        print("YOU GOT IT IN " + g + " GUESSES!!\n");
        if (g > 5) {
            print("BUT IT SHOULDN'T TAKE MORE THAN 5 GUESSES!\n");
        } else {
            print("GOOD JOB !!!!!\n");
        }
        print("\n");
        print("LET'S PLAY AGAIN.....");
    }
}

```

这道题的代码是 `main()`，它是程序的入口函数。在 `main()` 函数中，程序会首先查找并运行主函数，然后会创建一个根对象 `root`，并将其赋值为 `1`。接下来，会创建一个根节点 `root_node`，并将 `root_node` 的 `value` 属性设置为 `2`。然后，会遍历 `root_node` 的所有子节点，并将它们的 `value` 属性也设置为 `3`。最后，程序会输出 `root_node` 的 `value` 属性，结果为 `3`。


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


# `54_Letter/python/letter.py`

这段代码是一个简单的 Python 程序，用于制作一个猜字母游戏。游戏规则是，玩家需要在规定时间内猜出一个 5 字母以下的字母，如果猜对，就可以获得一个数字，如果猜错，则重新选择一个字母猜测。程序会根据玩家的猜测结果，给出提示信息，直到猜对为止。

程序中使用了两个变量：BELLS_ON_SUCCESS 和LETTER。BELLS_ON_SUCCESS 表示一个布尔值，表示在玩家猜测成功时，是否显示 "BELL" 字符。LETTER 变量则是游戏开始时随机生成的字母。

程序的主要逻辑在第一行中，使用import random模块从随机数生成器中获取一个 5 字母以下的随机字母，然后将其赋值给字符串LETTER。

紧接着的一行，程序定义了一个名为BELLS_ON_SUCCESS的布尔变量，将其设置为False。

在接下来的行中，程序开始一个循环，每次循环程序都会随机生成一个字母，然后将这个字母与LETTER比较，如果它们相等，则程序会提示玩家猜测下一个字母。如果它们不相等，程序就会重新生成一个字母，并继续提示。程序会重复这个步骤，直到玩家猜中为止。

最后，程序使用if语句检查BELLS_ON_SUCCESS是否为True，如果是，则程序会输出"BELL"，否则不会输出任何内容。


```
"""
LETTER

A letter guessing game.

Ported by Dave LeCompte
"""

import random

# The original code printed character 7, the "BELL" character 15 times
# when the player won. Many modern systems do not support this, and in
# any case, it can quickly become annoying, so it is disabled here.

BELLS_ON_SUCCESS = False


```

这道题目是一个猜字母游戏，游戏规则是要猜测一个随机生成的字母，每次猜测后，程序会告诉玩家猜大了还是猜小了，直到猜对为止。猜对的玩家可以得到分数，分数大于5的话，程序会提示玩家猜测的字母偏移了一些，让玩家继续猜测。当玩家猜测5次之内没有猜对，程序会提示玩家失败，并给出提示。


```
def print_instructions() -> None:
    print("LETTER GUESSING GAME")
    print()
    print("I'LL THINK OF A LETTER OF THE ALPHABET, A TO Z.")
    print("TRY TO GUESS MY LETTER AND I'LL GIVE YOU CLUES")
    print("AS TO HOW CLOSE YOU'RE GETTING TO MY LETTER.")


def play_game() -> None:
    target_value = random.randint(ord("A"), ord("Z"))
    num_guesses = 0
    print()
    print("O.K., I HAVE A LETTER.  START GUESSING.")
    print()
    while True:
        print("WHAT IS YOUR GUESS?")
        num_guesses += 1
        guess = ord(input())
        print()
        if guess == target_value:
            print()
            print(f"YOU GOT IT IN {num_guesses} GUESSES!!")
            if num_guesses > 5:
                print("BUT IT SHOULDN'T TAKE MORE THAN 5 GUESSES!")
                # goto 515
            print("GOOD JOB !!!!!")

            if BELLS_ON_SUCCESS:
                bell_str = chr(7) * 15
                print(bell_str)

            print()
            print("LET'S PLAY AGAIN.....")
            return
        elif guess > target_value:
            print("TOO HIGH. TRY A LOWER LETTER.")
            continue
        else:
            print("TOO LOW. TRY A HIGHER LETTER.")
            continue


```

这段代码定义了一个名为`main`的函数，它是一个Python内置函数中的`None`类型，表示函数不会执行任何操作，也不会返回任何值。函数内部通过`print`函数输出了一些字符，包括33个空格和15个左括号。

接下来，定义了一个包含两行内容的字符串变量`instructions_str`。第一行输出的是游戏指导，第二行输出的是游戏开发者、日期和一些关于 Morristown, New Jersey的介绍信息，通过`print_instructions`函数输出的。

接下来，定义了一个`print_instructions`函数，该函数只是调用`print`函数，但是会在字符串前面加上%贝符号`%`。调用`print_instructions`函数后，再次调用`print`函数，输出了一些字符，包括一个空格和一些关于 Morristown, New Jersey的介绍信息。

接下来，进入了一个无限循环，调用了一个名为`play_game`的函数。由于这个函数没有定义，因此不会执行任何操作，也不会返回任何值。

最后，由于`main`函数没有任何返回值，因此程序会无限循环，直到被强制终止。


```
def main() -> None:
    print(" " * 33 + "LETTER")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

    print_instructions()

    while True:
        play_game()


if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Life

The Game of Life was originally described in _Scientific American_, October 1970, in an article by Martin Gardner. The game itself was originated by John Conway of Gonville and Caius College, University of Cambridge England.

In the “manual” game, organisms exist in the form of counters (chips or checkers) on a large checkerboard and die or reproduce according to some simple genetic rules. Conway’s criteria for choosing his genetic laws were carefully delineated as follows:
1. There should be no initial pattern for which there is a simple proof that the population can grow without limit.
2. There should be simple initial patterns that apparently do grow without limit.
3. There should be simple initial patterns that grow and change for a considerable period of time before coming to an end in three possible ways:
    1. Fading away completely (from overcrowding or from becoming too sparse)
    2. Settling into a stable configuration that remains unchanged thereafter
    3. Entering an oscillating phase in which they repeat an endless cycle of two or more periods

In brief, the rules should be such as to make the behavior of the population relatively unpredictable. Conway’s genetic laws are delightfully simple. First note that each cell of the checkerboard (assumed to be an infinite plane) has eight neighboring cells, four adjacent orthogonally, four adjacent diagonally. The rules are:
1. Survivals. Every counter with two or three neighboring counters survives for the next generation.
2. Deaths. Each counter with four or more neighbors dies (is removed) from overpopulation. Every counter with one neighbor or none dies from isolation.
3. Births. Each empty cell adjacent to exactly three neighbors — no more — is a birth cell. A counter is placed on it at the next move.

It is important to understand that all births and deaths occur simultaneously. Together they constitute a single generation or, as we shall call it, a “move” in the complete “life history” of the initial configuration.

You will find the population constantly undergoing unusual, sometimes beautiful and always unexpected change. In a few cases the society eventually dies out (all counters vanishing), although this may not happen until after a great many generations. Most starting patterns either reach stable figures — Conway calls them “still lifes” — that cannot change or patterns that oscillate forever. Patterns with no initial symmetry tend to become symmetrical. Once this happens the symmetry cannot be lost, although it may increase in richness.

Conway used a DEC PDP-7 with a graphic display to observe long-lived populations. You’ll probably find this more enjoyable to watch on a CRT than a hard-copy terminal.

Since MITS 8K BASIC does not have LINE INPUT, to enter leading blanks in the patter, type a “.” at the start of the line. This will be converted to a space by BASIC, but it permits you to type leading spaces. Typing DONE indicates that you are finished entering the pattern. See sample run.

Clark Baker of Project DELTA originally wrote this version of LIFE which was further modified by Steve North of Creative Computing.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=100)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=115)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html


#### Porting Notes

- To make sense of the code, it's important to understand what the values in the A(X,Y) array mean:
  - 0: dead cell
  - 1: live cell
  - 2: currently live, but dead next cycle
  - 3: currently dead, but alive next cycle


(please note any difficulties or challenges in porting here)


# `55_Life/csharp/Program.cs`

这段代码的作用是模拟一个平面上的生长的菌落。程序首先读取一个限制高度和宽度的模板，并将其存储在变量pattern中。然后，程序定义了菌落中心在二维数组中的最小坐标，并将模板中的图案平方后存储在变量matrix中。接下来，程序使用InitializeSimulation函数设置一个模拟菌落的实验，该实验将使用给定的模板和模拟菌落矩阵。最后，程序在无限循环中，等待用户输入并读取图案，然后模拟菌落的生长过程。


```
﻿using System.Text;

const int maxWidth = 70;
const int maxHeight = 24;

Console.WriteLine("ENTER YOUR PATTERN:");
var pattern = new Pattern(ReadPattern(limitHeight: maxHeight).ToArray());

var minX = 10 - pattern.Height / 2;
var minY = 34 - pattern.Width / 2;
var maxX = maxHeight - 1;
var maxY = maxWidth - 1;

var matrix = new Matrix(height: maxHeight, width: maxWidth);
var simulation = InitializeSimulation(pattern, matrix);

```

这段代码的主要目的是读取并打印输出在一个限制高度的游戏中的所有可能的模式。

在 `PrintHeader()` 函数中，主要输出了游戏的标题头信息，用于在游戏开始时显示。

在 `ProcessSimulation()` 函数中，运行了模拟游戏过程的代码。这个函数可能是通过 `SimulateGame()` 函数来实现的，这个函数需要进一步了解。

`IEnumerable<string> ReadPattern(int limitHeight)` 函数的作用是读取并返回一个输入模式序列，限制高度为 `limitHeight`。在函数内部，使用了一个循环来读取用户输入，如果输入是 `DONE`，则跳出循环。

循环的具体内容是在循环开始时将输入的所有空格前置一个星号 `*`。如果输入开始时包含一个星号 `.`，则跳过输入的前星号，因为星号 `.` 会匹配到输入中的任意位置。

在循环中，将输入的字符串 `i` 带回到一个 `yield` 语句中，用于输出到列表中。最后，在循环的结束，将字符串 `input` 输出来，作为模式序列的结束标志。


```
PrintHeader();
ProcessSimulation();

IEnumerable<string> ReadPattern(int limitHeight)
{
    for (var i = 0; i < limitHeight; i++)
    {
        var input = Console.ReadLine();
        if (input.ToUpper() == "DONE")
        {
            break;
        }

        // In the original version, BASIC would trim the spaces in the beginning of an input, so the original
        // game allowed you to input an '.' before the spaces to circumvent this limitation. This behavior was
        // kept for compatibility.
        if (input.StartsWith('.'))
            yield return input.Substring(1, input.Length - 1);

        yield return input;
    }
}

```

这段代码定义了一个名为 "PrintHeader" 的函数，其作用是输出一段文本并将其居中。

具体来说，函数内部定义了一个名为 "PrintCentered" 的函数，它接受一个字符串参数 "text"。函数内部使用变量 "pageWidth" 来获取纸张的宽度，然后计算出字符串 "text" 中包含的字符数量 "count"。接下来，函数使用变量 "spaceCount" 来计算出 "pageWidth" 减去 "text" 长度后剩余的空间数量，然后使用 Console.Write 函数在字符串的两端输出指定的字符数。最后，函数输出四个空行，以便在输出字符串时能够居中。

函数 "PrintHeader" 在定义时先输出 "LIFE" 和 "CREATIVE COMPUTING"，然后循环输出四个空行，最后再次输出 "MORRISTOWN, NEW JERSEY"。


```
void PrintHeader()
{
    void PrintCentered(string text)
    {
        const int pageWidth = 64;

        var spaceCount = (pageWidth - text.Length) / 2;
        Console.Write(new string(' ', spaceCount));
        Console.WriteLine(text);
    }

    PrintCentered("LIFE");
    PrintCentered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    Console.WriteLine();
    Console.WriteLine();
    Console.WriteLine();
}

```

这段代码的主要作用是创建一个新的Simulation对象，并在给定的矩阵上初始化种群。

具体来说，代码首先创建一个名为newSimulation的新Simulation对象。接着，使用两个嵌套的for循环来将模式中的每个元素复制到矩阵的中间位置，并计数初始种群数量。如果模式中的某个元素是' '(空格)，则跳过该行的计数。

在嵌套的for循环中，将矩阵ToInitialize中的对应行设置为' '。然后，调用Simulation的IncreasePopulation方法来增加当前区域的个体数量。最后，返回创建的新的Simulation对象。


```
Simulation InitializeSimulation(Pattern pattern, Matrix matrixToInitialize) {
    var newSimulation = new Simulation();

    // transcribes the pattern to the middle of the simulation and counts initial population
    for (var x = 0; x < pattern.Height; x++)
    {
        for (var y = 0; y < pattern.Width; y++)
        {
            if (pattern.Content[x][y] == ' ')
                continue;

            matrixToInitialize[minX + x, minY + y] = CellState.Stable;
            newSimulation.IncreasePopulation();
        }
    }

    return newSimulation;
}

```

这段代码是一个方法 `GetPauseBetweenIterations`，它接受两个参数 `args` 和 `args[0]`。如果 `args` 的长度不等于 2，则返回 `TimeSpan.Zero`。否则，它将检查第一个参数中的参数是否包含 `"wait"`。如果是，它将尝试从第二个参数中加载一个整数，并将其转换为 `TimeSpan` 类型的值。最后，如果第二个参数为 `null` 或 `string` 类型，或者参数中包含的 `"wait"` 不存在，则返回 `TimeSpan.Zero`。


```
TimeSpan GetPauseBetweenIterations()
{
    if (args.Length != 2) return TimeSpan.Zero;

    var parameter = args[0].ToLower();
    if (parameter.Contains("wait"))
    {
        var value = args[1];
        if (int.TryParse(value, out var sleepMilliseconds))
            return TimeSpan.FromMilliseconds(sleepMilliseconds);
    }

    return TimeSpan.Zero;
}

```



This is a Python implementation of a simple SimCity game. The game starts by initializing the space where the cells are placed. The cells are initialized with a state of either "Stable", "Dying", or "Empty". The SimCity game then searches for improvements in the cell state, and updates the cells accordingly. The game continues until the user ends the simulation or a new simulation is started.

The `Cell` class represents a single cell in the game. It has a `state` property that indicates the current state of the cell, as well as `neighbors` property that indicates the number of cells that are connected to the current cell.

The `SimCity` class is the main class that runs the simulation. It has a `pauseBetweenIterations` property that determines how long to pause between iterations.

The `RunSimCity` function is the main method for starting a new simulation. It creates a new instance of the `SimCity` class, initializes the cells, and runs the simulation until it is stopped or a new simulation is started.

The `OnCellClick` function is called when a user clicks on a cell. It updates the state of the cell based on the user's click.

The `OnCellStateChanged` function is called when the state of a cell is changed. It updates the `neighbors` property of the cell and sends the change to the `SimCity` instance.


```
void ProcessSimulation()
{
    var pauseBetweenIterations = GetPauseBetweenIterations();
    var isInvalid = false;

    while (true)
    {
        var invalidText = isInvalid ? "INVALID!" : "";
        Console.WriteLine($"GENERATION: {simulation.Generation}\tPOPULATION: {simulation.Population} {invalidText}");

        simulation.StartNewGeneration();

        var nextMinX = maxHeight - 1;
        var nextMinY = maxWidth - 1;
        var nextMaxX = 0;
        var nextMaxY = 0;

        var matrixOutput = new StringBuilder();

        // prints the empty lines before search area
        for (var x = 0; x < minX; x++)
        {
            matrixOutput.AppendLine();
        }

        // refreshes the matrix and updates search area
        for (var x = minX; x <= maxX; x++)
        {
            var printedLine = Enumerable.Repeat(' ', maxWidth).ToList();
            for (var y = minY; y <= maxY; y++)
            {
                if (matrix[x, y] == CellState.Dying)
                {
                    matrix[x, y] = CellState.Empty;
                    continue;
                }
                if (matrix[x, y] == CellState.New)
                {
                    matrix[x, y] = CellState.Stable;
                }
                else if (matrix[x, y] != CellState.Stable)
                {
                    continue;
                }

                printedLine[y] = '*';

                nextMinX = Math.Min(x, nextMinX);
                nextMaxX = Math.Max(x, nextMaxX);
                nextMinY = Math.Min(y, nextMinY);
                nextMaxY = Math.Max(y, nextMaxY);
            }

            matrixOutput.AppendLine(string.Join(separator: null, values: printedLine));
        }

        // prints empty lines after search area
        for (var x = maxX + 1; x < maxHeight; x++)
        {
            matrixOutput.AppendLine();
        }
        Console.Write(matrixOutput);

        void UpdateSearchArea()
        {
            minX = nextMinX;
            maxX = nextMaxX;
            minY = nextMinY;
            maxY = nextMaxY;

            const int limitX = 21;
            const int limitY = 67;

            if (minX < 2)
            {
                minX = 2;
                isInvalid = true;
            }

            if (maxX > limitX)
            {
                maxX = limitX;
                isInvalid = true;
            }

            if (minY < 2)
            {
                minY = 2;
                isInvalid = true;
            }

            if (maxY > limitY)
            {
                maxY = limitY;
                isInvalid = true;
            }
        }
        UpdateSearchArea();

        for (var x = minX - 1; x <= maxX + 1; x++)
        {
            for (var y = minY - 1; y <= maxY + 1; y++)
            {
                int CountNeighbors()
                {
                    var neighbors = 0;
                    for (var i = x - 1; i <= x + 1; i++)
                    {
                        for (var j = y - 1; j <= y + 1; j++)
                        {
                            if (matrix[i, j] == CellState.Stable || matrix[i, j] == CellState.Dying)
                                neighbors++;
                        }
                    }

                    return neighbors;
                }

                var neighbors = CountNeighbors();
                if (matrix[x, y] == CellState.Empty)
                {
                    if (neighbors == 3)
                    {
                        matrix[x, y] = CellState.New;
                        simulation.IncreasePopulation();
                    }
                }
                else if (neighbors is < 3 or > 4)
                {
                    matrix[x, y] = CellState.Dying;
                }
                else
                {
                    simulation.IncreasePopulation();
                }
            }
        }

        // expands search area to accommodate new cells
        minX--;
        minY--;
        maxX++;
        maxY++;

        if (pauseBetweenIterations > TimeSpan.Zero)
            Thread.Sleep(pauseBetweenIterations);
    }
}

```



该代码定义了一个名为“Pattern”的类，该类包含一个名为“Content”的字符串数组、一个名为“Height”的整数变量和一个名为“Width”的整数变量。

在构造函数中，该类接受一个字符串字符串数组，使用该数组中的所有字符来填充该类的“Content”字段。

构造函数初始化“Height”变量为字符串字符串数组中的元素长度最大值，初始化“Width”变量为字符串字符串数组中的最大长度。

“NormalizeWidth”方法使用字符串的“PadRight”方法来获取该字符串字符串数组中所有字符的长度之和，然后返回一个字符串数组，该数组中的所有字符都被填充到字符串的字符数组长度上，使其长度与该字符串字符串数中所有字符的长度之和相等。


```
public class Pattern
{
    public string[] Content { get; }
    public int Height { get; }
    public int Width { get; }

    public Pattern(IReadOnlyCollection<string> patternLines)
    {
        Height = patternLines.Count;
        Width = patternLines.Max(x => x.Length);
        Content = NormalizeWidth(patternLines);
    }

    private string[] NormalizeWidth(IReadOnlyCollection<string> patternLines)
    {
        return patternLines
            .Select(x => x.PadRight(Width, ' '))
            .ToArray();
    }
}

```

这段代码定义了一个名为CellState的枚举类型，用于描述细胞在 simulation 中的状态，包括Empty、Stable、Dying和New四种状态。

在Simulation类中，定义了一个内部变量Generation，用于记录当前 simulation 的代数。还定义了一个内部变量Population，用于记录当前 simulation 中的个体数量。

在StartNewGeneration方法中，通过调用父类的构造函数，初始化了Generation变量为0,Population变量为0。

在IncreasePopulation方法中，调用了自身的一个增殖方法，用于在 simulation 中增加个体数量。


```
/// <summary>
/// Indicates the state of a given cell in the simulation.
/// </summary>
internal enum CellState
{
    Empty = 0,
    Stable = 1,
    Dying = 2,
    New = 3
}

public class Simulation
{
    public int Generation { get; private set; }

    public int Population { get; private set; }

    public void StartNewGeneration()
    {
        Generation++;
        Population = 0;
    }

    public void IncreasePopulation()
    {
        Population++;
    }
}

```

这段代码定义了一个名为Matrix的类，用于 aid debugging。这个类实现了两个方法：

1. 一个ToString()方法，这个方法通过字符串拼接单元格中的内容，并将结果返回。
2. 一个空格的占位符。

这个类有一个private的二维数组_matrix，用于存储单元格状态。这个类有一个this[int x, int y]方法，它返回矩阵中(x, y)单元格的引用。这个方法还有一个get和set方法，用于获取和设置(x, y)单元格的值。

这个类的构造函数接收两个整数参数，表示矩阵的高度和宽度。在ToString()方法中，这个类定义了一个循环来遍历矩阵，并将每个单元格中的内容转换成字符串并拼接到字符串Builder中。最终，这个方法返回一个字符串，它包含了所有单元格中的字符串。


```
/// <summary>
/// This class was created to aid debugging, through the implementation of the ToString() method.
/// </summary>
class Matrix
{
    private readonly CellState[,] _matrix;

    public Matrix(int height, int width)
    {
        _matrix = new CellState[height, width];
    }

    public CellState this[int x, int y]
    {
        get => _matrix[x, y];
        set => _matrix[x, y] = value;
    }

    public override string ToString()
    {
        var stringBuilder = new StringBuilder();
        for (var x = 0; x < _matrix.GetLength(0); x++)
        {
            for (var y = 0; y < _matrix.GetLength(1); y++)
            {
                var character = _matrix[x, y] == 0 ? " ": ((int)_matrix[x, y]).ToString();
                stringBuilder.Append(character);
            }

            stringBuilder.AppendLine();
        }
        return stringBuilder.ToString();
    }
}

```

# Life

An implementation of John Conway's popular cellular automaton, also know as **Conway's Game of Life**. The original source was downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html).

Ported by Dyego Alekssander Maas.

## How to run

This program requires you to install [.NET 6 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/6.0). After installed, you just need to run `dotnet run` from this directory in the terminal.

## Know more about Conway's Game of Life

You can find more about Conway's Game of Life on this page of the [Cornell Math Explorers' Club](http://pi.math.cornell.edu/~lipa/mec/lesson6.html), alongside many examples of patterns you can try.

### Optional parameters

Optionally, you can run this program with the `--wait 1000` argument, the number being the time in milliseconds
that the application will pause between each iteration. This is enables you to watch the simulation unfolding. By default, there is no pause between iterations.

The complete command would be `dotnet run --wait 1000`.

## Entering patterns

Once running the game, you are expected to enter a pattern. This pattern consists of multiple lines of text with either **spaces** or **some character**, usually an asterisk (`*`).

Spaces represent empty cells. Asterisks represent alive cells.

After entering the pattern, you need to enter the word "DONE". It is not case sensitive. An example of pattern would be:

```
 *
***
DONE
```

### Some patterns you could try

```
 *
***
```

```
*
***
```

```
**
**
```

```
  *
 *
*
```

This one is known as **glider**:

```
***
*
 *
```

## Instructions to the port

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# Game of Life - Java version

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)

## Requirements

* Requires Java 17 (or later)

## Notes

The Java version of Game of Life tries to mimics the behaviour of the BASIC version.
However, the Java code does not have much in common with the original.

**Differences in behaviour:**
* Input supports the ```.``` character, but it's optional.
* Evaluation of ```DONE``` input string is case insensitive.
* Run with the ```-s``` command line argument to halt the program after each generation, and continue when ```ENTER``` is pressed.


# `55_Life/java/src/java/Life.java`

这段代码定义了一个名为 "Game of Life" 的类，它扩展了 Java 标准库中的 List 类。这个类包含了一些方法，用于在控制台输出游戏中每个单元格的状态（是 "活" 还是 "死"）以及整个游戏的状态。

该代码的主要作用是模拟 Game of Life 游戏的玩法。该游戏的基本规则是，对于每个细胞，如果它的 2 周围细胞中有 "活" 细胞，那么它就保持 "活" 状态，否则就变成 "死" 状态。Game of Life 游戏可以通过控制台输入 "." 字符来控制每个单元格的状态，也可以输入 "DONE" 字符来表示游戏已经结束。


```
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * The Game of Life class.<br>
 * <br>
 * Mimics the behaviour of the BASIC version, however the Java code does not have much in common with the original.
 * <br>
 * Differences in behaviour:
 * <ul>
 *     <li>Input supports the "." character, but it's optional.</li>
 *     <li>Input regarding the "DONE" string is case insensitive.</li>
 * </ul>
 */
```

This is a Java program that simulates a game of life. The game is played on a 2D matrix, with each cell being either "alive" or "invalid". The program takes command line arguments to control the number of generations and the output format.

The program has several main methods:

* `main`: This method is called when the program starts. It initializes the variables needed for the game, starts a new game, and calls the `start` method.
* `start`: This method is called each time the program starts a new game. It sets the initial population，花数， and the number of cells in the matrix to 1. It then sets the matrix to all "alive".
* `printGameHeader`: This method prints the header of the game to the console.
* `printIndented`: This method prints the console to the left by a specified number of spaces.
* `printGeneration`: This method prints the generation of the game.
* `printGenerationHeader`: This method prints the header of the `printGeneration` method.
* `main`: This method is the entry point of the program. It parses the command line arguments and calls the appropriate method.

The program also has a `stop` method that can be called to stop the program after each generation.

Note: This program assumes that a 2D `char` array is used to represent the matrix, with a single character representing each cell. The ALIVE cell is represented by the character '*', and the non-ALIVE cell is represented by the character ' '.


```
public class Life {

    private static final byte DEAD  = 0;
    private static final byte ALIVE = 1;
    private static final String NEWLINE = "\n";

    private final Scanner consoleReader = new Scanner(System.in);

    private final byte[][] matrix = new byte[21][67];
    private int generation = 0;
    private int population = 0;
    boolean stopAfterGen = false;
    boolean invalid = false;

    /**
     * Constructor.
     *
     * @param args the command line arguments
     */
    public Life(String[] args) {
        parse(args);
    }

    private void parse(String[] args) {
        for (String arg : args) {
            if ("-s".equals(arg)) {
                stopAfterGen = true;
                break;
            }
        }
    }

    /**
     * Starts the game.
     */
    public void start() {
        printGameHeader();
        readPattern();
        while (true) {
            printGeneration();
            advanceToNextGeneration();
            if (stopAfterGen) {
                System.out.print("PRESS ENTER TO CONTINUE");
                consoleReader.nextLine();
            }
        }
    }

    private void advanceToNextGeneration() {
        // store all cell transitions in a list, i.e. if a dead cell becomes alive, or a living cell dies
        List<Transition> transitions = new ArrayList<>();
        // there's still room for optimization: instead of iterating over all cells in the matrix,
        // we could consider only the section containing the pattern(s), as in the BASIC version
        for (int y = 0; y < matrix.length; y++) {
            for (int x = 0; x < matrix[y].length; x++) {
                int neighbours = countNeighbours(y, x);
                if (matrix[y][x] == ALIVE) {
                    if (neighbours < 2 || neighbours > 3) {
                        transitions.add(new Transition(y, x, DEAD));
                        population--;
                    }
                } else { // cell is dead
                    if (neighbours == 3) {
                        if (x < 2 || x > 67 || y < 2 || y > 21) {
                            invalid = true;
                        }
                        transitions.add(new Transition(y, x, ALIVE));
                        population++;
                    }
                }
            }
        }
        // apply all transitions to the matrix
        transitions.forEach(t -> matrix[t.y()][t.x()] = t.newState());
        generation++;
    }

    private int countNeighbours(int y, int x) {
        int neighbours = 0;
        for (int row = Math.max(y - 1, 0); row <= Math.min(y + 1, matrix.length - 1); row++) {
            for (int col = Math.max(x - 1, 0); col <= Math.min(x + 1, matrix[row].length - 1); col++) {
                if (row == y && col == x) {
                    continue;
                }
                if (matrix[row][col] == ALIVE) {
                    neighbours++;
                }
            }
        }
        return neighbours;
    }

    private void readPattern() {
        System.out.println("ENTER YOUR PATTERN:");
        List<String> lines = new ArrayList<>();
        String line;
        int maxLineLength = 0;
        boolean reading = true;
        while (reading) {
            System.out.print("? ");
            line = consoleReader.nextLine();
            if (line.equalsIgnoreCase("done")) {
                reading = false;
            } else {
                // optional support for the '.' that is needed in the BASIC version
                lines.add(line.replace('.', ' '));
                maxLineLength = Math.max(maxLineLength, line.length());
            }
        }
        fillMatrix(lines, maxLineLength);
    }

    private void fillMatrix(List<String> lines, int maxLineLength) {
        float xMin = 33 - maxLineLength / 2f;
        float yMin = 11 - lines.size() / 2f;
        for (int y = 0; y < lines.size(); y++) {
            String line = lines.get(y);
            for (int x = 1; x <= line.length(); x++) {
                if (line.charAt(x-1) == '*') {
                    matrix[floor(yMin + y)][floor(xMin + x)] = ALIVE;
                    population++;
                }
            }
        }
    }

    private int floor(float f) {
        return (int) Math.floor(f);
    }

    private void printGameHeader() {
        printIndented(34, "LIFE");
        printIndented(15, "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println(NEWLINE.repeat(3));
    }

    private void printIndented(int spaces, String str) {
        System.out.println(" ".repeat(spaces) + str);
    }

    private void printGeneration() {
        printGenerationHeader();
        for (int y = 0; y < matrix.length; y++) {
            for (int x = 0; x < matrix[y].length; x++) {
                System.out.print(matrix[y][x] == 1 ? "*" : " ");
            }
            System.out.println();
        }
    }

    private void printGenerationHeader() {
        String invalidText = invalid ? "INVALID!" : "";
        System.out.printf("GENERATION: %-13d POPULATION: %d %s\n", generation, population, invalidText);
    }

    /**
     * Main method that starts the program.
     *
     * @param args the command line arguments:
     *             <pre>-s: Stop after each generation (press enter to continue)</pre>
     * @throws Exception if something goes wrong.
     */
    public static void main(String[] args) throws Exception {
        new Life(args).start();
    }

}

```

这段代码定义了一个名为`Transition`的元组类型，用于表示单个单元格在矩阵中的状态更改。该元组类型包含三个整数成员变量：`y`表示单元格的y坐标，`x`表示单元格的x坐标，`newState`表示单元格的新状态，可以是`DEAD`或`ALIVE`中的一个。

当我们向一个矩阵中的单个单元格发送一个状态更改请求时，我们使用`Transition`类型来传递这个信息。例如，如果我们想要在一个3x3的矩阵中某个单元格中，让第2行第3列的单元格从`DEAD`状态变为`ALIVE`状态，我们可以这样调用该函数：
```scss
Transition(2, 3, 1);
```
该函数的参数传递给该函数时，将分别传递给`Transition`类型的三个成员变量。


```
/**
 * Represents a state change for a single cell within the matrix.
 *
 * @param y the y coordinate (row) of the cell
 * @param x the x coordinate (column) of the cell
 * @param newState the new state of the cell (either DEAD or ALIVE)
 */
record Transition(int y, int x, byte newState) { }

```