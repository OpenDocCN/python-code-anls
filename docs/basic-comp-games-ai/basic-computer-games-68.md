# BasicComputerGames源码解析 68

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


# `73_Reverse/python/reverse.py`

这段代码是一个用于Morristown, New Jersey地区的数学游戏。它由两个主要的函数组成：`main`函数和`game_loop`函数。

`main`函数用于生成`NUMCNT`个随机整数，并打印出游戏的标题以及一些文本。然后，它等待用户输入是否想要游戏规则，如果用户选择"否"，则调用`print_rules`函数。

如果用户选择"是",`main`函数将打印游戏的规则。游戏规则是在有限的时间内，玩家需要尝试在指定时间内解决一组数学问题。

`game_loop`函数用于让玩家依次解决游戏中的数学问题。首先，它随机生成一些数学问题，并打印出问题以及答案。然后，玩家需要尝试解决问题，如果回答正确，则游戏继续进行下一题。如果回答错误，游戏结束并返回。游戏将继续进行直到玩家选择“退出”，或者当尝试所有问题后，游戏也结束。


```
#!/usr/bin/env python3
import random
import textwrap

NUMCNT = 9  # How many numbers are we playing with?


def main() -> None:
    print("REVERSE".center(72))
    print("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY".center(72))
    print()
    print()
    print("REVERSE -- A GAME OF SKILL")
    print()

    if not input("DO YOU WANT THE RULES? (yes/no) ").lower().startswith("n"):
        print_rules()

    while True:
        game_loop()

        if not input("TRY AGAIN? (yes/no) ").lower().startswith("y"):
            return


```

这段代码是一个Python游戏循环，主要作用是让学生猜测数字列表中的随机数字，并尝试在有限次数内猜出所有的数字。如果猜中了，程序会打印出获胜的步数，否则会提示猜测失败，并重新开始游戏。

具体来说，代码的作用可以概括为以下几个步骤：

1. 创建一个数字列表，从1开始，直到达到指定的最大数量(NUMCNT)，然后对列表进行随机排序。
2. 打印原始数字列表，并开始游戏。
3. 让用户循环输入猜测数字的数量(1到NUMCNT)，每次增加1，直到达到指定的最大数量(NUMCNT)。
4. 如果用户猜测的数字数量为0，程序会提示用户，或者在猜测数字数量超过最大数量时打印错误消息并继续游戏。
5. 在每次猜测后，程序会随机翻转指定数量的数字，并将它们添加到数字列表的末尾。
6. 循环打印数字列表，并检查是否所有数字都已经被翻转过，如果是，那么用户就赢得了游戏。
7. 如果所有数字都被翻转过，程序会打印获胜的步数，并结束游戏。否则，程序会继续循环，直到用户结束游戏。


```
def game_loop() -> None:
    """Play the main game."""
    # Make a random list from 1 to NUMCNT
    numbers = list(range(1, NUMCNT + 1))
    random.shuffle(numbers)

    # Print original list and start the game
    print()
    print("HERE WE GO ... THE LIST IS:")
    print_list(numbers)

    turns = 0
    while True:
        try:
            howmany = int(input("HOW MANY SHALL I REVERSE? "))
            assert howmany >= 0
        except (ValueError, AssertionError):
            continue

        if howmany == 0:
            return

        if howmany > NUMCNT:
            print("OOPS! WRONG! I CAN REVERSE AT MOST", NUMCNT)
            continue

        turns += 1

        # Reverse as many items as requested.
        newnums = numbers[0:howmany]
        newnums.reverse()
        newnums.extend(numbers[howmany:])
        numbers = newnums

        print_list(numbers)

        # Check for a win
        if all(numbers[i] == i + 1 for i in range(NUMCNT)):
            print(f"YOU WON IT IN {turns} MOVES!")
            print()
            return


```

这段代码定义了两个函数，`print_list()` 和 `print_rules()`。这两个函数的功能是打印一个数字列表。

`print_list()` 函数接收一个数字列表 `numbers`，并将其中的所有数字转换为字符串并连接起来。最后，函数通过 `print()` 函数将结果输出到屏幕上。

`print_rules()` 函数接收一个字符串 `help`，该字符串包含了游戏的规则。函数通过 `textwrap.dedent()` 方法将字符串中的缩进字符删除，并打印出来。最后，函数通过 `print()` 函数将结果输出到屏幕上。


```
def print_list(numbers) -> None:
    print(" ".join(map(str, numbers)))


def print_rules() -> None:
    help = textwrap.dedent(
        """
        THIS IS THE GAME OF "REVERSE".  TO WIN, ALL YOU HAVE
        TO DO IS ARRANGE A LIST OF NUMBERS (1 THROUGH {})
        IN NUMERICAL ORDER FROM LEFT TO RIGHT.  TO MOVE, YOU
        TELL ME HOW MANY NUMBERS (COUNTING FROM THE LEFT) TO
        REVERSE.  FOR EXAMPLE, IF THE CURRENT LIST IS:

        2 3 4 5 1 6 7 8 9

        AND YOU REVERSE 4, THE RESULT WILL BE:

        5 4 3 2 1 6 7 8 9

        NOW IF YOU REVERSE 5, YOU WIN!

        1 2 3 4 5 6 7 8 9

        NO DOUBT YOU WILL LIKE THIS GAME, BUT
        IF YOU WANT TO QUIT, REVERSE 0 (ZERO).
        """.format(
            NUMCNT
        )
    )
    print(help)
    print()


```

这段代码是一个 if 语句，它判断当前脚本是否为__main__.如果这个脚本被当作主程序运行，那么它将执行 if 语句中的代码。

在这个 if 语句中，有一个 try 块，这个块中包含了一个函数 main。如果 main 函数在执行过程中发生了异常，那么将会执行 except 块中的代码。

if 语句的 else 块中并没有代码，它只是起到了一个分行代码的作用，表示如果 if 语句的判断条件为真，那么将跳过 else 块中的代码，否则执行 if 语句中的代码。

总结起来，这段代码的作用是判断当前脚本是否为__main__，如果是，就执行 main() 函数，如果不是，就跳过 else 块中的代码。


```
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Rock, Scissors, Paper

Remember the game of rock-scissors-paper. You and your opponent make a motion three times with your fists and then either show:
- a flat hand (paper)
- fist (rock)
- two fingers (scissors)

Depending upon what is shown, the game is a tie (both show the same) or one person wins. Paper wraps up rock, so it wins. Scissors cut paper, so they win. And rock breaks scissors, so it wins.

In this computerized version of rock-scissors-paper, you can play up to ten games vs. the computer.

Charles Lund wrote this game while at the American School in The Hague, Netherlands.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=137)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=152)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `74_Rock_Scissors_Paper/csharp/Choice.cs`

这段代码定义了一个名为`Choice`的类，表示游戏中的选择棋选择包。

类中包含一个`string`类型的属性`Selector`，表示棋的选择器，以及一个`string`类型的属性`Name`，表示棋子的名称。

还有一个内部类型的属性`CanBeat`，表示该棋是否能够打败其他棋子。

类中还有一个构造函数，用于初始化棋的选择器、名称以及该棋是否能够打败其他棋子。

还有一个名为`Beats`的接口，用于定义比较两个`Choice`对象的大小。

该代码的主要作用是创建一个`Choice`实例，并实现`Beats`接口，用于比较两个`Choice`对象的大小。


```
namespace RockScissorsPaper
{
    public class Choice
    {
        public string Selector {get; private set; }
        public string Name { get; private set; }
        internal Choice CanBeat { get; set; }

        public Choice(string selector, string name) {
            Selector = selector;
            Name = name;
        }

        public bool Beats(Choice choice)
        {
            return choice == CanBeat;
        }
    }
}

```

# `74_Rock_Scissors_Paper/csharp/Choices.cs`

这段代码定义了一个名为 `Choices` 的类，代表了一个 Rock(棋子)、Scissors(飞镖) 和 Paper(纸牌) 三种基本选择的随机选择。在 `GetRandom` 方法中，使用了一个 `Random` 对象来生成一个随机的选择，然后从 `_allChoices` 数组中返回它的第一个元素，这个数组是在定义 `Choices` 类时创建的。

`TryGetBySelector` 方法尝试根据传入的选择器来查找一个随机选择，如果没有找到匹配的选择，则返回 `false`。这个方法接受两个参数：要查找的选择器和返回参数 `out`。

该代码的作用是创建一个 `Choices` 类来代表三种基本选择，并定义了一些方法来生成随机选择和查找特定选择。这个类可以被用来在游戏或其他需要随机选择的时候进行操作。


```
using System;

namespace RockScissorsPaper
{
    public class Choices
    {
        public static readonly Choice Rock = new Choice("3", "Rock");
        public static readonly Choice Scissors = new Choice("2", "Scissors");
        public static readonly Choice Paper = new Choice("1", "Paper");

        private static readonly Choice[] _allChoices;
        private static readonly Random _random = new Random();

        static Choices()
        {
            Rock.CanBeat = Scissors;
            Scissors.CanBeat = Paper;
            Paper.CanBeat = Rock;

            _allChoices = new[] { Rock, Scissors, Paper };
        }

        public static Choice GetRandom()
        {
            return _allChoices[_random.Next(_allChoices.GetLength(0))];
        }

        public static bool TryGetBySelector(string selector, out Choice choice)
        {
            foreach (var possibleChoice in _allChoices)
            {
                if (string.Equals(possibleChoice.Selector, selector))
                {
                    choice = possibleChoice;
                    return true;
                }
            }
            choice = null;
            return false;
        }
    }
}

```

# `74_Rock_Scissors_Paper/csharp/Game.cs`

This code defines a game class `Game` that simulates the game of Rock-Paper-Scissors. The game uses the `Choices` class to get a random choice for the computer and the `GetHumanChoice` method to get a random choice for the human. The `ComputerWins`, `HumanWins`, and `TieGames` properties keep track of the number of wins, ties, and game score for each player, respectively. The `PlayGame` method simulates a game, the `WriteFinalScore` method displays the final game score, and the `GetRandomChoices` method generates a random choice for the computer or human.


```
using System;
using System.Linq;

namespace RockScissorsPaper
{
    public class Game
    {
        public int ComputerWins { get; private set; }
        public int HumanWins { get; private set; }
        public int TieGames { get; private set; }

        public void PlayGame()
        {
            var computerChoice = Choices.GetRandom();
            var humanChoice = GetHumanChoice();

            Console.WriteLine("This is my choice...");
            Console.WriteLine("...{0}", computerChoice.Name);

            if (humanChoice.Beats(computerChoice))
            {
                Console.WriteLine("You win!!!");
                HumanWins++;
            }
            else if (computerChoice.Beats(humanChoice))
            {
                Console.WriteLine("Wow!  I win!!!");
                ComputerWins++;
            }
            else
            {
                Console.WriteLine("Tie game.  No winner.");
                TieGames++;
            }
        }

        public void WriteFinalScore()
        {
            Console.WriteLine();
            Console.WriteLine("Here is the final game score:");
            Console.WriteLine("I have won {0} game(s).", ComputerWins);
            Console.WriteLine("You have one {0} game(s).", HumanWins);
            Console.WriteLine("And {0} game(s) ended in a tie.", TieGames);
        }

        public Choice GetHumanChoice()
        {
            while (true)
            {
                Console.WriteLine("3=Rock...2=Scissors...1=Paper");
                Console.WriteLine("1...2...3...What's your choice");
                if (Choices.TryGetBySelector(Console.ReadLine(), out var choice))
                    return choice;
                Console.WriteLine("Invalid.");
            }
        }
    }
}

```

# `74_Rock_Scissors_Paper/csharp/Program.cs`

这段代码是一个 RockScissorsPaper 游戏的程序。这个程序的主要作用是接受玩家输入的游戏数量，然后循环进行游戏，并输出最终的得分。

具体来说，这段代码首先会定义一个名为 Program 的类，其中包含一个 Main 方法。Main 方法是程序的入口点，当程序运行时，它首先会输出 "GAME OF ROCK, SCISSORS, PAPER" 这样的提示信息，然后进入一个无限循环。在循环中，程序会调用一个名为 GetNumberOfGames 的方法来获取玩家输入的游戏数量。如果玩家输入的数量小于 11，程序会输出 "Sorry, but we aren't allowed to play that many。" 的提示信息，如果玩家输入的数量大于 0，程序会继续执行。

一旦获取了游戏数量后，程序会创建一个 Game 类，并使用循环来调用 Game.PlayGame 方法来开始玩游戏。在玩游戏的过程中，程序会不断输出当前的游戏分数。当所有的游戏都结束时，程序会调用 Game.WriteFinalScore 方法来输出最终的得分。

最后，程序会输出 "Thanks for playing!!" 的提示信息，以感谢玩家参与游戏。


```
﻿using System;

namespace RockScissorsPaper
{
    static class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("GAME OF ROCK, SCISSORS, PAPER");
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();

            var numberOfGames = GetNumberOfGames();

            var game = new Game();
            for (var gameNumber = 1; gameNumber <= numberOfGames; gameNumber++) {
                Console.WriteLine();
                Console.WriteLine("Game number {0}", gameNumber);

                game.PlayGame();
            }

            game.WriteFinalScore();

            Console.WriteLine();
            Console.WriteLine("Thanks for playing!!");
        }

        static int GetNumberOfGames()
        {
            while (true) {
                Console.WriteLine("How many games");
                if (int.TryParse(Console.ReadLine(), out var number))
                {
                    if (number < 11 && number > 0)
                        return number;
                    Console.WriteLine("Sorry, but we aren't allowed to play that many.");
                }
                else
                {
                    Console.WriteLine("Sorry, I didn't understand.");
                }
            }
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `74_Rock_Scissors_Paper/java/src/RockScissors.java`

This is a Java program that simulates a Rock-Paper-Scissors game. It starts by displaying the game board and the current player.

The game logic is implemented in a separate class called `RockScissors`. When a player plays, the `RockScissors` class is responsible for checking the game state and making the necessary decisions.

The `displayTextAndGetNumber` method displays a message on the screen and accepts a number from the keyboard. The `displayTextAndGetInput` method displays a message on the screen and accepts a string from the keyboard.

The game will end when one of the players wins or the game is a tie.

I hope this helps! Let me know if you have any questions.


```
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Rock Scissors Paper
 * <p>
 * Based on the Basic game of Rock Scissors here
 * https://github.com/coding-horror/basic-computer-games/blob/main/74%20Rock%20Scissors%20Paper/rockscissors.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */

public class RockScissors {

    public static final int MAX_GAMES = 10;

    public static final int PAPER = 1;
    public static final int SCISSORS = 2;
    public static final int ROCK = 3;

    // Used for keyboard input
    private final Scanner kbScanner;

    private enum GAME_STATE {
        START_GAME,
        GET_NUMBER_GAMES,
        START_ROUND,
        PLAY_ROUND,
        GAME_RESULT,
        GAME_OVER
    }

    // Current game state
    private GAME_STATE gameState;

    private enum WINNER {
        COMPUTER,
        PLAYER
    }

    private WINNER gameWinner;

    int playerWins;
    int computerWins;
    int numberOfGames;
    int currentGameCount;
    int computersChoice;

    public RockScissors() {
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
                    currentGameCount = 0;
                    gameState = GAME_STATE.GET_NUMBER_GAMES;

                    break;

                case GET_NUMBER_GAMES:
                    numberOfGames = displayTextAndGetNumber("HOW MANY GAMES? ");
                    if (numberOfGames <= MAX_GAMES) {
                        gameState = GAME_STATE.START_ROUND;
                    } else {
                        System.out.println("SORRY, BUT WE AREN'T ALLOWED TO PLAY THAT MANY.");
                    }
                    break;

                case START_ROUND:
                    currentGameCount++;
                    if (currentGameCount > numberOfGames) {
                        gameState = GAME_STATE.GAME_RESULT;
                        break;
                    }
                    System.out.println("GAME NUMBER: " + (currentGameCount));
                    computersChoice = (int) (Math.random() * 3) + 1;
                    gameState = GAME_STATE.PLAY_ROUND;

                case PLAY_ROUND:
                    System.out.println("3=ROCK...2=SCISSORS...1=PAPER");
                    int playersChoice = displayTextAndGetNumber("1...2...3...WHAT'S YOUR CHOICE? ");
                    if (playersChoice >= PAPER && playersChoice <= ROCK) {
                        switch (computersChoice) {
                            case PAPER:
                                System.out.println("...PAPER");
                                break;
                            case SCISSORS:
                                System.out.println("...SCISSORS");
                                break;
                            case ROCK:
                                System.out.println("...ROCK");
                                break;
                        }

                        if (playersChoice == computersChoice) {
                            System.out.println("TIE GAME.  NO WINNER.");
                        } else {
                            switch (playersChoice) {
                                case PAPER:
                                    if (computersChoice == SCISSORS) {
                                        gameWinner = WINNER.COMPUTER;
                                    } else if (computersChoice == ROCK) {
                                        // Don't need to re-assign here, as its initialized to
                                        // false I'd argue this aids readability.
                                        gameWinner = WINNER.PLAYER;
                                    }
                                    break;
                                case SCISSORS:
                                    if (computersChoice == ROCK) {
                                        gameWinner = WINNER.COMPUTER;
                                    } else if (computersChoice == PAPER) {
                                        // Don't need to re-assign here, as its initialized to
                                        // false I'd argue this aids readability.
                                        gameWinner = WINNER.PLAYER;
                                    }
                                    break;
                                case ROCK:
                                    if (computersChoice == PAPER) {
                                        gameWinner = WINNER.COMPUTER;
                                    } else if (computersChoice == SCISSORS) {
                                        // Don't need to re-assign here, as its initialized to
                                        // false I'd argue this aids readability.
                                        gameWinner = WINNER.PLAYER;
                                    }
                                    break;
                            }

                            if (gameWinner == WINNER.COMPUTER) {
                                System.out.println("WOW!  I WIN!!!");
                                computerWins++;
                            } else {
                                System.out.println("YOU WIN!!!");
                                playerWins++;
                            }
                        }
                        gameState = GAME_STATE.START_ROUND;
                    } else {
                        System.out.println("INVALID.");
                    }

                    break;

                case GAME_RESULT:
                    System.out.println();
                    System.out.println("HERE IS THE FINAL GAME SCORE:");
                    System.out.println("I HAVE WON " + computerWins + " GAME" + (computerWins != 1 ? "S." : "."));
                    System.out.println("YOU HAVE WON " + playerWins + " GAME" + (playerWins != 1 ? "S." : "."));
                    int tiedGames = numberOfGames - (computerWins + playerWins);
                    System.out.println("AND " + tiedGames + " GAME" + (tiedGames != 1 ? "S " : " ") + "ENDED IN A TIE.");
                    System.out.println();
                    System.out.println("THANKS FOR PLAYING!!");
                    gameState = GAME_STATE.GAME_OVER;
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    private void intro() {
        System.out.println(addSpaces(21) + "GAME OF ROCK, SCISSORS, PAPER");
        System.out.println(addSpaces(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
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

    public static void main(String[] args) {

        RockScissors rockScissors = new RockScissors();
        rockScissors.play();
    }
}

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


# `74_Rock_Scissors_Paper/python/rockscissors.py`

This appears to be a Python program that allows the user to play rock-paper-scissors against a computer. The user can choose to play as either the computer or the human, and the computer has a 1-3 range, while the human has a fixed choice of 1, 2, or 3. The computer wins if its choice is different from the human's, and the game is a tie if the computer or the human wins.

The program uses the `random` module to generate random numbers for the user's `guess_human` variable, which is the user's choice. The `guess_computer` variable is the computer's choice, which is determined by the user's `guess_human` number.

The program also keeps track of the number of wins for each player and outputs that information at the end of the game.

Note that the program is missing some print statements, so you may want to add some additional statements to complete it.


```
#!/usr/bin/env python3
# ROCKSCISSORS
#
# Converted from BASIC to Python by Trevor Hobson

import random


def play_game() -> None:
    """Play one round of the game"""

    while True:
        try:
            games = int(input("How many games? "))
            if games >= 11:
                print("Sorry, but we aren't allowed to play that many.")
            else:
                break

        except ValueError:
            print("Please enter a number.")

    won_computer = 0
    won_human = 0

    for game in range(games):
        print("\nGame number", game + 1)
        guess_computer = random.randint(1, 3)
        print("3=Rock...2=Scissors...1=Paper")
        guess_human = 0
        while guess_human == 0:
            try:
                guess_human = int(input("1...2...3...What's your choice "))
                if guess_human not in [1, 2, 3]:
                    guess_human = 0
                    print("Invalid")

            except ValueError:
                print("Please enter a number.")
        print("This is my choice...")
        if guess_computer == 1:
            print("...Paper")
        elif guess_computer == 2:
            print("...Scissors")
        elif guess_computer == 3:
            print("...Rock")
        if guess_computer == guess_human:
            print("Tie Game. No winner")
        elif guess_computer > guess_human:
            if guess_human != 1 or guess_computer != 3:
                print("Wow! I win!!!")
                won_computer = won_computer + 1
            else:
                print("You win!!!")
                won_human = won_human + 1
        elif guess_computer == 1:
            if guess_human != 3:
                print("You win!!!")
                won_human = won_human + 1
            else:
                print("Wow! I win!!!")
                won_computer = won_computer + 1
    print("\nHere is the final game score:")
    print("I have won", won_computer, "game(s).")
    print("You have won", won_human, "game(s).")
    print("and", games - (won_computer + won_human), "game(s) ended in a tie.")
    print("\nThanks for playing!!\n")


```

这段代码定义了一个名为 `main` 的函数，它返回一个名为 `None` 的空值。函数内部使用多种字符串生成器(print statement)来输出文本字符串。

首先，使用 `print` 函数输出一行游戏规则，然后使用另一行输出游戏的背景信息。接着，使用两个循环来控制玩家是否继续玩这个游戏，并且在循环内部调用 `play_game` 函数来让玩家玩一次游戏。

游戏循环内部，调用 `print` 函数输出 "Play again?(yes or no)" 消息，用来询问玩家是否继续玩，如果玩家输入 "yes"，那么循环将会再次运行，否则退出循环。

最后，如果游戏没有被手动中断，函数会一直运行，直到程序被关闭。


```
def main() -> None:
    print(" " * 21 + "GAME OF ROCK, SCISSORS, PAPER")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")

    keep_playing = True
    while keep_playing:

        play_game()

        keep_playing = input("Play again? (yes or no) ").lower().startswith("y")


if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Roulette

This game simulates an American Roulette wheel; “American” because it has 38 number compartments (1 to 36, 0 and 00). The European wheel has 37 numbers (1 to 36 and 0). The Bahamas, Puerto Rico, and South American countries are slowly switching to the American wheel because it gives the house a bigger percentage. Odd and even numbers alternate around the wheel, as do red and black. The layout of the wheel insures a highly random number pattern. In fact, roulette wheels are sometimes used to generate tables of random numbers.

In this game, you may bet from $5 to $500 and you may bet on red or black, odd or even, first or second 18 numbers, a column, or single number. You may place any number of bets on each spin of the wheel.

There is no long-range winning strategy for playing roulette. However, a good strategy is that of “doubling.” First spin, bet $1 on an even/odds bet (odd, even, red, or black). If you lose, double your bet again to $2. If you lose again, double to $4. Continue to double until you win (i.e, you break even on a losing sequence). As soon as you win, bet $1 again, and after every win, bet $1. Do not ever bet more than $1 unless you are recuperating losses by doubling. Do not ever bet anything but the even odds bets. Good luck!

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=138)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=153)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

- The program keeps a count of how often each number comes up in array `X`, but never makes use of this information.


# `75_Roulette/csharp/Bet.cs`



这段代码定义了一个名为Bet的结构体，表示赌博中的下注。Bet结构体包含三个整型字段，分别为BetType类型、Number下注数量和Wager下注金额。另外，还有一个包含三个字段的Bet结构体变量。

Bet结构体中的三个字段分别是：

1. Payout，表示下注赢得的钱数，是Wager和Type字段相乘的结果。
2. Number，表示下注的数量，是一个整型字段。
3. Wager，表示下注的金额，是一个整型字段。

Bet结构体变量可以让用户输入一个Bet对象，例如：

```
Bet myBet = new Bet(BetType.Low, 10, 100);
```

这将创建一个Bet对象，BetType为Low,Number为10,Wager为100。


```
namespace Roulette;

internal record struct Bet(BetType Type, int Number, int Wager)
{
    public int Payout => Wager * Type.Payout;
}

```

# `75_Roulette/csharp/BetType.cs`

这段代码定义了一个BetType结构体，用于表示赌博中的下注类型，每个BetType结构体包含一个整型成员Value，以及一个Payout成员变量，用于表示下注的赔率。

同时，还定义了一个BetType类型的函数operator，用于将一个int类型的参数转换为BetType类型的结构体，并返回结构体。

该代码的主要作用是定义一个BetType结构体，用于表示赌博中的下注类型，以及一个BetType类型的函数operator，用于将一个int类型的参数转换为BetType类型的结构体。


```
namespace Roulette;

internal record struct BetType(int Value)
{
    public static implicit operator BetType(int value) => new(value);

    public int Payout => Value switch
        {
            <= 36 or >= 49 => 35,
            <= 42 => 2,
            <= 48 => 1
        };
}

```

# `75_Roulette/csharp/Croupier.cs`



这段代码是一个C++语言的类，名为Croupier。Croupier是Roulette游戏中的一个角色，代表的是赌客。

该类包含以下方法：

- public string Totals => Strings.Totals(_house, _player); 
- public bool PlayerIsBroke => _player <= 0; 
- public bool HouseIsBroke => _house <= 0;
- internal string Pay(Bet bet)
{
   _house -= bet.Payout;
   _player += bet.Payout;

   if (_house <= 0)
   {
       _player = _initialHouse + _initialPlayer;
   }

   return Strings.Win(bet);
}

- internal string Take(Bet bet)
{
   _house += bet.Wager;
   _player -= bet.Wager;

   return Strings.Lose(bet);
}

- public void CutCheck(IReadWrite io, IRandom random)
{
   var name = io.ReadString(Prompts.Check);
   io.Write(Strings.Check(random, name, _player));
}

代码的作用是模拟赌场游戏中的情况。通过这个类，玩家可以决定是否要下注、加倍下注或者退出游戏，而赌场则会根据玩家的下注情况来决定是否胜利或失败。在游戏开始时，赌场的初始资金和玩家的初始筹码为100,000和1,000, respectively。当玩家下注时，赌场会减少玩家的筹码，并相应地更新玩家的下注数和赌场的资金数。如果玩家的筹码用完了，或者赌场的资金数也为0，游戏就结束了。


```
namespace Roulette;

internal class Croupier
{
    private const int _initialHouse = 100_000;
    private const int _initialPlayer = 1_000;

    private int _house = _initialHouse;
    private int _player = _initialPlayer;

    public string Totals => Strings.Totals(_house, _player);
    public bool PlayerIsBroke => _player <= 0;
    public bool HouseIsBroke => _house <= 0;

    internal string Pay(Bet bet)
    {
        _house -= bet.Payout;
        _player += bet.Payout;

        if (_house <= 0)
        {
            _player = _initialHouse + _initialPlayer;
        }

        return Strings.Win(bet);
    }

    internal string Take(Bet bet)
    {
        _house += bet.Wager;
        _player -= bet.Wager;

        return Strings.Lose(bet);
    }

    public void CutCheck(IReadWrite io, IRandom random)
    {
        var name = io.ReadString(Prompts.Check);
        io.Write(Strings.Check(random, name, _player));
    }
}

```

# `75_Roulette/csharp/Game.cs`



这段代码是一个赌博游戏，由一个内部类Game组成。

Game类包含四个私有变量：

- IReadWrite io：读写流，用于向玩家显示游戏界面和接收玩家输入的指令。
- IRandom random：随机数生成器，用于生成随机数。
- Table table：游戏牌的Table对象，包含所有玩家可以看到的游戏牌面信息。
- Croupier croupier：轮盘赌中的赌徒，负责决定游戏是否继续进行和计算玩家的赌注。

游戏 Play方法中，首先创建了一个游戏牌的Table对象，并使用 io.Write(Streams.Title) 将游戏标题显示在屏幕上。然后，游戏会提示玩家输入指令，如果玩家输入的不是“n”，则游戏将继续进行。

在 while 循环中，游戏将不断从 table.Play() 方法中获取新的游戏牌面信息，并使用 croupier.CutCheck(io, random) 方法来决定是否应该宣布赌注。如果 croupier 认为玩家已经破产，游戏将使用 io.Write(Streams.LastDollar) 和 io.Write(Streams.Thanks) 方法来向玩家宣布胜利并结束游戏。如果 croupier 认为游戏没有破产，游戏将继续进行，并使用 io.Write(Streams.BrokeHouse) 方法来宣布破产。


```
namespace Roulette;

internal class Game
{
    private readonly IReadWrite _io;
    private readonly IRandom _random;
    private readonly Table _table;
    private readonly Croupier _croupier;

    public Game(IReadWrite io, IRandom random)
    {
        _io = io;
        _random = random;
        _croupier = new();
        _table = new(_croupier, io, random);
    }

    public void Play()
    {
        _io.Write(Streams.Title);
        if (!_io.ReadString(Prompts.Instructions).ToLowerInvariant().StartsWith('n'))
        {
            _io.Write(Streams.Instructions);
        }

        while (_table.Play());

        if (_croupier.PlayerIsBroke)
        {
            _io.Write(Streams.LastDollar);
            _io.Write(Streams.Thanks);
            return;
        }

        if (_croupier.HouseIsBroke)
        {
            _io.Write(Streams.BrokeHouse);
        }

        _croupier.CutCheck(_io, _random);
    }
}

```

# `75_Roulette/csharp/IOExtensions.cs`



这段代码是一个名为 "Roulette.IOExtensions" 的命名空间类，其中包含两个方法，名为 "ReadBetCount" 和 "ReadBet"。

"ReadBetCount" 方法从用户那里获取 "HowManyBets" 的数量，并返回该数量的整数表示。如果输入的值不是有效的整数，该方法将一直循环直到输入有效为止。

"ReadBet" 方法从用户那里获取一个整数 "number"，然后循环直到找到有效的赌注或直到循环结束。对于有效的赌注，该方法返回一个赌注对象，其中包含赌注类型、数量和赌注金额。如果输入的值不是有效的整数或赌注金额超出了允许的范围，该方法将返回 false。

"IsValidInt" 方法是一个辅助方法，用于检查给定的值是否为有效的整数。它的参数是一个浮点数和一个最低值和一个最高值，它返回如果给定的值等于有效的整数并且值在最低值和最高值之间。


```
namespace Roulette;

internal static class IOExtensions
{
    internal static int ReadBetCount(this IReadWrite io)
    {
        while (true)
        {
            var betCount = io.ReadNumber(Prompts.HowManyBets);
            if (betCount.IsValidInt(1)) { return (int)betCount; }
        }
    }

    internal static Bet ReadBet(this IReadWrite io, int number)
    {
        while (true)
        {
            var (type, amount) = io.Read2Numbers(Prompts.Bet(number));

            if (type.IsValidInt(1, 50) && amount.IsValidInt(5, 500))
            {
                return new()
                {
                    Type = (int)type, 
                    Number = number, 
                    Wager = (int)amount
                };
            }
        }
    }

    internal static bool IsValidInt(this float value, int minValue, int maxValue = int.MaxValue)
        => value == (int)value && value >= minValue && value <= maxValue;
}
```

# `75_Roulette/csharp/Program.cs`

这段代码使用了三个Assembly引用：

1. Games.Common.IO：用于文件输入和输出。
2. Games.Common.Randomness：用于生成随机数。
3. Roulette.Resources.Resource：用于在游戏中的资源。

另外，还使用了Roulette.AI.RandomBot这个类，它是一个基于随机模拟的AI游戏组件。

接着，创建了一个新的Game实例，这个实例使用了两个构造函数，第一个构造函数接收一个ConsoleIO作为输入参数，第二个构造函数接收一个RandomNumberGenerator作为输入参数。

最后，调用这个Game实例的Play方法，这个方法开始一个新的游戏，并返回一个Game2048实例。

总体来说，这段代码是一个用C#编写的游戏组件，可以在Game2048游戏中使用。


```
global using Games.Common.IO;
global using Games.Common.Randomness;
global using static Roulette.Resources.Resource;
using Roulette;

new Game(new ConsoleIO(), new RandomNumberGenerator()).Play();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `75_Roulette/csharp/Slot.cs`



这段代码定义了一个名为 Slot 的类，用于表示赌桌上的小赌注。

内部类 Slot 实现了 ImmutableHashSet 接口，用于存储所有被覆盖的赌注类型。

构造函数接收两个参数，第一个参数是一个赌注类型的字符串，第二个参数是一个 BetType 数组，用于存储所有被覆盖的赌注类型。

在内部类中，成员变量 name 用于存储小赌注的名字，_coveringBets 存储了所有被覆盖的赌注类型，都存储在 ImmutableHashSet 中。

slot 类还实现了两个方法：

IsCoveredBy(Bet bet) -> 检查给定的赌注类型是否在覆盖赌注类型中，如果是，则返回 true，否则返回 false。

public Slot (string name, params BetType[] coveringBets)
{
   Name = name;
   _coveringBets = coveringBets.ToImmutableHashSet();
}

这个类的作用是定义一个赌桌上的小赌注。通过接受一个赌注类型字符串和一个BetType数组参数，这个小赌注可能有哪些覆盖的赌注类型，这个小赌注的名字和覆盖的赌注类型存储在同一个 ImmutableHashSet 中。


```
using System.Collections.Immutable;

namespace Roulette;

internal class Slot
{
    private readonly ImmutableHashSet<BetType> _coveringBets;

    public Slot (string name, params BetType[] coveringBets)
    {
        Name = name;
        _coveringBets = coveringBets.ToImmutableHashSet();
    }

    public string Name { get; }

    public bool IsCoveredBy(Bet bet) => _coveringBets.Contains(bet.Type);
}

```

# `75_Roulette/csharp/Table.cs`

This is a simple Python implementation of a game of Monte Carlo巡视。 It uses the Monte Carlo simulation method to determine if the player is broke or not based on the actions they take.

To run this program, you will need to install the `random` module and the `croupier` module. You can install these modules using `pip install random croupier`.

The `Table` class represents the game table. It takes three arguments:

* `croupier`: The `Croupier` object that will determine the game.
* `io`: An `Io` object that will handle the game output.
* `random`: An `Rand` object that will generate random numbers for the game.

The `Play` method is the main method that runs the game. It uses the `AcceptBets` and `SpinWheel` methods to determine the outcome of


```
namespace Roulette;

internal class Table
{
    private readonly IReadWrite _io;
    private readonly Wheel _wheel;
    private readonly Croupier _croupier;

    public Table(Croupier croupier, IReadWrite io, IRandom random)
    {
        _croupier = croupier;
        _io = io;
        _wheel = new(random);
    }

    public bool Play()
    {
        var bets = AcceptBets();
        var slot = SpinWheel();
        SettleBets(bets, slot);

        _io.Write(_croupier.Totals);

        if (_croupier.PlayerIsBroke || _croupier.HouseIsBroke) { return false; }

        return _io.ReadString(Prompts.Again).ToLowerInvariant().StartsWith('y');
    }

    private Slot SpinWheel()
    {
        _io.Write(Streams.Spinning);
        var slot = _wheel.Spin();
        _io.Write(slot.Name);
        return slot;
    }

    private IReadOnlyList<Bet> AcceptBets()
    {
        var betCount = _io.ReadBetCount();
        var betTypes = new HashSet<BetType>();
        var bets = new List<Bet>();
        for (int i = 1; i <= betCount; i++)
        {
            while (!TryAdd(_io.ReadBet(i)))
            {
                _io.Write(Streams.BetAlready);
            }
        }

        return bets.AsReadOnly();

        bool TryAdd(Bet bet)
        {
            if (betTypes.Add(bet.Type))
            {
                bets.Add(bet);
                return true;
            }

            return false;
        }
    }

    private void SettleBets(IReadOnlyList<Bet> bets, Slot slot)
    {
        foreach (var bet in bets)
        {
            _io.Write(slot.IsCoveredBy(bet) ? _croupier.Pay(bet) : _croupier.Take(bet));
        }
    }
}

```

# `75_Roulette/csharp/Wheel.cs`

This is a Java class that represents a slot for a physical wheel. The wheel has 26 slots, and each slot is labeled with a black and a red string representing the position of the slot on the wheel. The class has a spin method that randomly selects one of the slots based on the _random object. It also has a constructor that takes an IRandom object to be used for generating random numbers.

Note that this class is in no way related to the operating system of any programming language, it is just a class that holds a variable number of string objects to represent the position of each slot on the physical wheel, and a spin method that randomly selects one of the slots.


```
using System.Collections.Immutable;

namespace Roulette;

internal class Wheel
{
    private static readonly ImmutableArray<Slot> _slots = ImmutableArray.Create(
        new Slot(Strings.Red(1), 1, 37, 40, 43, 46, 47),
        new Slot(Strings.Black(2), 2, 37, 41, 43, 45, 48),
        new Slot(Strings.Red(3), 3, 37, 42, 43, 46, 47),
        new Slot(Strings.Black(4), 4, 37, 40, 43, 45, 48),
        new Slot(Strings.Red(5), 5, 37, 41, 43, 46, 47),
        new Slot(Strings.Black(6), 6, 37, 42, 43, 45, 48),
        new Slot(Strings.Red(7), 7, 37, 40, 43, 46, 47),
        new Slot(Strings.Black(8), 8, 37, 41, 43, 45, 48),
        new Slot(Strings.Red(9), 9, 37, 42, 43, 46, 47),
        new Slot(Strings.Black(10), 10, 37, 40, 43, 45, 48),
        new Slot(Strings.Black(11), 11, 37, 41, 43, 46, 48),
        new Slot(Strings.Red(12), 12, 37, 42, 43, 45, 47),
        new Slot(Strings.Black(13), 13, 38, 40, 43, 46, 48),
        new Slot(Strings.Red(14), 14, 38, 41, 43, 45, 47),
        new Slot(Strings.Black(15), 15, 38, 42, 43, 46, 48),
        new Slot(Strings.Red(16), 16, 38, 40, 43, 45, 47),
        new Slot(Strings.Black(17), 17, 38, 41, 43, 46, 48),
        new Slot(Strings.Red(18), 18, 38, 42, 43, 45, 47),
        new Slot(Strings.Red(19), 19, 38, 40, 44, 46, 47),
        new Slot(Strings.Black(20), 20, 38, 41, 44, 45, 48),
        new Slot(Strings.Red(21), 21, 38, 42, 44, 46, 47),
        new Slot(Strings.Black(22), 22, 38, 40, 44, 45, 48),
        new Slot(Strings.Red(23), 23, 38, 41, 44, 46, 47),
        new Slot(Strings.Black(24), 24, 38, 42, 44, 45, 48),
        new Slot(Strings.Red(25), 25, 39, 40, 44, 46, 47),
        new Slot(Strings.Black(26), 26, 39, 41, 44, 45, 48),
        new Slot(Strings.Red(27), 27, 39, 42, 44, 46, 47),
        new Slot(Strings.Black(28), 28, 39, 40, 44, 45, 48),
        new Slot(Strings.Black(29), 29, 39, 41, 44, 46, 48),
        new Slot(Strings.Red(30), 30, 39, 42, 44, 45, 47),
        new Slot(Strings.Black(31), 31, 39, 40, 44, 46, 48),
        new Slot(Strings.Red(32), 32, 39, 41, 44, 45, 47),
        new Slot(Strings.Black(33), 33, 39, 42, 44, 46, 48),
        new Slot(Strings.Red(34), 34, 39, 40, 44, 45, 47),
        new Slot(Strings.Black(35), 35, 39, 41, 44, 46, 48),
        new Slot(Strings.Red(36), 36, 39, 42, 44, 45, 47),
        new Slot("0", 49),
        new Slot("00", 50));
    
    private readonly IRandom _random;

    public Wheel(IRandom random) => _random = random;

    public Slot Spin() => _slots[_random.Next(_slots.Length)];
}

```

# `75_Roulette/csharp/Resources/Resource.cs`

This is a code snippet for a无所不知的 AI language model. It appears to define a number of helper classes and methods for generating random numbers and stringslots for displaying different pieces of information in a赌场-like game.

The `Strings` class defines several methods for generating random numbers with a given number of decimal places and different formats for displaying the numbers. The `Prompts` class defines several methods for generating random numbers, displaying the number of bets, and displaying the outcome of a game.

The `GetStream` class defines a method for generating a stream of information for a given name.

The `Assembly` class is used to dynamically load an executable assembly that contains the type definition for this AI language model.

It appears that this code snippet is intended for use in a game where the user can place bets and the AI will generate information about the game and display it to the user.


```
using System.Reflection;
using System.Runtime.CompilerServices;
using Games.Common.Randomness;

namespace Roulette.Resources;

internal static class Resource
{
    internal static class Streams
    {
        public static Stream Title => GetStream();
        public static Stream Instructions => GetStream();
        public static Stream BetAlready => GetStream();
        public static Stream Spinning => GetStream();
        public static Stream LastDollar => GetStream();
        public static Stream BrokeHouse => GetStream();
        public static Stream Thanks => GetStream();
    }

    internal static class Strings
    {
        public static string Black(int number) => Slot(number);
        public static string Red(int number) => Slot(number);
        private static string Slot(int number, [CallerMemberName] string? colour = null)
            => string.Format(GetString(), number, colour);

        public static string Lose(Bet bet) => Outcome(bet.Wager, bet.Number);
        public static string Win(Bet bet) => Outcome(bet.Payout, bet.Number);
        private static string Outcome(int amount, int number, [CallerMemberName] string? winlose = null)
            => string.Format(GetString(), winlose, amount, number);

        public static string Totals(int me, int you) => string.Format(GetString(), me, you);

        public static string Check(IRandom random, string payee, int amount)
            => string.Format(GetString(), random.Next(100), DateTime.Now, payee, amount);
    }

    internal static class Prompts
    {
        public static string Instructions => GetPrompt();
        public static string HowManyBets => GetPrompt();
        public static string Bet(int number) => string.Format(GetPrompt(), number);
        public static string Again => GetPrompt();
        public static string Check => GetPrompt();
    }

    private static string GetPrompt([CallerMemberName] string? name = null) => GetString($"{name}Prompt");

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

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)

Two versions of Roulette has been contributed. They are indicated within given sub-folders

- [oop](./oop) - Conversion by Andrew McGuinness (andrew@arobeia.co.uk)
- [iterative](./iterative) - Conversion by Thomas Kwashnak ([Github](https://github.com/LittleTealeaf)).
    - Implements features from JDK 17.
    - Does make use of some object oriented programming, but acts as a more iterative solution.


# `75_Roulette/java/iterative/Roulette.java`

This is a simple text-based game where the player can choose to play again or exit. If the player chooses to play again, the game will start again with a random bet amount.

The game uses the `Random` class to generate a random number between 0 and 100, which is used to determine whether the player has won or not. The `DateTimeFormatter` class is used to format the current date and time in ISO format for the name of the order.

The `Bet` class represents a bet made by the player, with the `num` and `amount` attributes representing the number of the bet and the amount, respectively.

The game also uses a `Scanner` object to read input from the player.


```
import java.io.InputStream;
import java.io.PrintStream;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

public class Roulette {

    private static Set<Integer> RED_NUMBERS;

    static {
        RED_NUMBERS = Set.of(1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36);
    }

    private PrintStream out;
    private Scanner scanner;
    private int houseBalance, playerBalance;
    private Random random;

    public Roulette(PrintStream out, InputStream in) {
        this.out = out;
        this.scanner = new Scanner(in);
        houseBalance = 100000;
        playerBalance = 1000;
        random = new Random();
    }

    public static void main(String[] args) {
        new Roulette(System.out, System.in).play();
    }

    public void play() {
        out.println("                                ROULETTE");
        out.println("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        out.println("WELCOME TO THE ROULETTE TABLE\n");
        out.print("DO YOU WANT INSTRUCTIONS? ");
        if (scanner.nextLine().toLowerCase().charAt(0) != 'n') {
            printInstructions();
        }

        do {

            Bet[] bets = queryBets();

            out.print("SPINNING...\n\n");
            int result = random.nextInt(1, 39);

            /*
            Equivalent to following line
            if(result == 37) {
                out.println("00");
            } else if(result == 38) {
                out.println("0");
            } else if(RED_NUMBERS.contains(result)) {
                out.println(result + " RED");
            } else {
                out.println(result + " BLACK");
            }
             */
            out.println(switch (result) {
                case 37 -> "00";
                case 38 -> "0";
                default -> result + (RED_NUMBERS.contains(result) ? " RED" : " BLACK");
            });

            betResults(bets, result);
            out.println();

            out.println("TOTALS:\tME\t\tYOU");
            out.format("\t\t%5d\t%d\n", houseBalance, playerBalance);
        } while (playAgain());
        if (playerBalance <= 0) {
            out.println("THANKS FOR YOUR MONEY\nI'LL USE IT TO BUY A SOLID GOLD ROULETTE WHEEL");
        } else {
            printCheck();
        }
        out.println("COME BACK SOON!");
    }

    public void printInstructions() {
        out.println();
        out.println("THIS IS THE BETTING LAYOUT");
        out.println("  (*=RED)");
        out.println();
        out.println(" 1*    2     3*");
        out.println(" 4     5*    6 ");
        out.println(" 7*    8     9*");
        out.println("10    11    12*");
        out.println("---------------");
        out.println("13    14*   15 ");
        out.println("16*   17    18*");
        out.println("19*   20    21*");
        out.println("22    23*   24 ");
        out.println("---------------");
        out.println("25*   26    27*");
        out.println("28    29    30*");
        out.println("31    32*   33 ");
        out.println("34*   35    36*");
        out.println("---------------");
        out.println("    00    0    ");
        out.println();
        out.println("TYPES OF BETS");
        out.println();
        out.println("THE NUMBERS 1 TO 36 SIGNIFY A STRAIGHT BET");
        out.println("ON THAT NUMBER.");
        out.println("THESE PAY OFF 35:1");
        out.println();
        out.println("THE 2:1 BETS ARE:");
        out.println(" 37) 1-12     40) FIRST COLUMN");
        out.println(" 38) 13-24    41) SECOND COLUMN");
        out.println(" 39) 25-36    42) THIRD COLUMN");
        out.println();
        out.println("THE EVEN MONEY BETS ARE:");
        out.println(" 43) 1-18     46) ODD");
        out.println(" 44) 19-36    47) RED");
        out.println(" 45) EVEN     48) BLACK");
        out.println();
        out.println(" 49)0 AND 50)00 PAY OFF 35:1");
        out.println(" NOTE: 0 AND 00 DO NOT COUNT UNDER ANY");
        out.println("       BETS EXCEPT THEIR OWN.");
        out.println();
        out.println("WHEN I ASK FOR EACH BET, TYPE THE NUMBER");
        out.println("AND THE AMOUNT, SEPARATED BY A COMMA.");
        out.println("FOR EXAMPLE: TO BET $500 ON BLACK, TYPE 48,500");
        out.println("WHEN I ASK FOR A BET.");
        out.println();
        out.println("THE MINIMUM BET IS $5, THE MAXIMUM IS $500.");
    }

    private Bet[] queryBets() {
        int numBets = -1;
        while (numBets < 1) {
            out.print("HOW MANY BETS? ");
            try {
                numBets = Integer.parseInt(scanner.nextLine());
            } catch (NumberFormatException ignored) {
            }
        }

        Bet[] bets = new Bet[numBets];

        for (int i = 0; i < numBets; i++) {
            while (bets[i] == null) {
                try {
                    out.print("NUMBER" + (i + 1) + "? ");
                    String[] values = scanner.nextLine().split(",");
                    int betNumber = Integer.parseInt(values[0]);
                    int betValue = Integer.parseInt(values[1]);

                    for (int j = 0; j < i; j++) {
                        if (bets[j].num == betNumber) {
                            out.println("YOU MADE THAT BET ONCE ALREADY,DUM-DUM");
                            betNumber = -1; //Since -1 is out of the range, this will throw it out at the end
                        }
                    }

                    if (betNumber > 0 && betNumber <= 50 && betValue >= 5 && betValue <= 500) {
                        bets[i] = new Bet(betNumber,betValue);
                    }
                } catch (Exception ignored) {
                }
            }
        }
        return bets;
    }

    private void betResults(Bet[] bets, int num) {
        for (int i = 0; i < bets.length; i++) {
            Bet bet = bets[i];
            /*
            Using a switch statement of ternary operators that check if a certain condition is met based on the bet value
            Returns the coefficient that the bet amount should be multiplied by to get the resulting value
             */
            int coefficient = switch (bet.num) {
                case 37 -> (num <= 12) ? 2 : -1;
                case 38 -> (num > 12 && num <= 24) ? 2 : -1;
                case 39 -> (num > 24 && num < 37) ? 2 : -1;
                case 40 -> (num < 37 && num % 3 == 1) ? 2 : -1;
                case 41 -> (num < 37 && num % 3 == 2) ? 2 : -1;
                case 42 -> (num < 37 && num % 3 == 0) ? 2 : -1;
                case 43 -> (num <= 18) ? 1 : -1;
                case 44 -> (num > 18 && num <= 36) ? 1 : -1;
                case 45 -> (num % 2 == 0) ? 1 : -1;
                case 46 -> (num % 2 == 1) ? 1 : -1;
                case 47 -> RED_NUMBERS.contains(num) ? 1 : -1;
                case 48 -> !RED_NUMBERS.contains(num) ? 1 : -1;
                case 49 -> (num == 37) ? 35 : -1;
                case 50 -> (num == 38) ? 35 : -1;
                default -> (bet.num < 49 && bet.num == num) ? 35 : -1;
            };

            int betResult = bet.amount * coefficient;

            if (betResult < 0) {
                out.println("YOU LOSE " + -betResult + " DOLLARS ON BET " + (i + 1));
            } else {
                out.println("YOU WIN " + betResult + " DOLLARS ON BET " + (i + 1));
            }

            playerBalance += betResult;
            houseBalance -= betResult;
        }
    }

    private boolean playAgain() {

        if (playerBalance <= 0) {
            out.println("OOPS! YOU JUST SPENT YOUR LAST DOLLAR!");
            return false;
        } else if (houseBalance <= 0) {
            out.println("YOU BROKE THE HOUSE!");
            playerBalance = 101000;
            houseBalance = 0;
            return false;
        } else {
            out.println("PLAY AGAIN?");
            return scanner.nextLine().toLowerCase().charAt(0) == 'y';
        }
    }

    private void printCheck() {
        out.print("TO WHOM SHALL I MAKE THE CHECK? ");
        String name = scanner.nextLine();

        out.println();
        for (int i = 0; i < 72; i++) {
            out.print("-");
        }
        out.println();

        for (int i = 0; i < 50; i++) {
            out.print(" ");
        }
        out.println("CHECK NO. " + random.nextInt(0, 100));

        for (int i = 0; i < 40; i++) {
            out.print(" ");
        }
        out.println(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE));
        out.println();

        out.println("PAY TO THE ORDER OF -----" + name + "----- $" + (playerBalance));
        out.println();

        for (int i = 0; i < 40; i++) {
            out.print(" ");
        }
        out.println("THE MEMORY BANK OF NEW YORK");

        for (int i = 0; i < 40; i++) {
            out.print(" ");
        }
        out.println("THE COMPUTER");

        for (int i = 0; i < 40; i++) {
            out.print(" ");
        }
        out.println("----------X-----");

        for (int i = 0; i < 72; i++) {
            out.print("-");
        }
        out.println();
    }

    public class Bet {

        final int num, amount;

        public Bet(int num, int amount) {
            this.num = num;
            this.amount = amount;
        }
    }
}

```