# BasicComputerGames源码解析 86

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by Uğur Küpeli [ugurkupeli](https://github.com/ugurkupeli)

Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### 23 Matches

In the game of twenty-three matches, you start with 23 matches lying on a table. On each turn, you may take 1, 2, or 3 matches. You alternate moves with the computer and the one who has to take the last match loses.

The easiest way to devise a winning strategy is to start at the end of the game. Since your wish to leave the last match to your opponent, you would like to have either 4, 3, or 2 on your last turn you so can take away 3, 2, or 1 and leave 1. Consequently, you would like to leave your opponent with 5 on his next to last turn so, no matter what his move, you are left with 4, 3, or 2. Work this backwards to the beginning and you’ll find the game can effectively be won on the first move. Fortunately, the computer gives you the first move, so if you play wisely, you can win.

After you’ve mastered 23 Matches, move on to BATNUM and then to NUM.

This version of 23 Matches was originally written by Bob Albrecht of People’s Computer Company.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=177)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=192)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

There is an oddity (you can call it a bug, but it is no big deal) in the original code. If there are only two or three matches left at the player's turn and the player picks all of them (or more), the game would still register that as a win for the player.


# `93_23_Matches/csharp/23Matches.cs`



This is a sample code written in C# that demonstrates a simple game of pool where two players take turns trying to消除 all their scorers. The player starts with a score of zero and the computer starts with a score of four. If the player removes a match, the computer removes one match. If the player removes all their matches before the computer, the player wins. If the computer removes all their matches before the player, the computer wins.

The code uses a while loop to keep the game going until one of the players wins or the game is ended. The while loop has a condition that checks if the player has won or if the computer has won. If the player wins, the code displays a win message and the game ends. If the computer wins, the code displays a win message and the game ends.

The code also includes a method called `ReadPlayerInput()` that is used to handle the player input. This method reads the player input and converts it to an integer. It then tries to read the player input again. If there is an error in the player input, the code reads the player input again.

Overall, this code is simple and easy to understand, but it does not provide a lot of depth or functionality. It is intended for educational purposes only and can be easily modified to create a more complex game.



```
﻿using System;

namespace Program
{
  class Program
  {

    // Initialize 3 public variables so that they can be ascessed anywhere in the code
    public static int numberOfMatches;
    public static int numberOfMatchesRemovedByPlayer;
    public static bool playerGoesFirst = false; // a flag to show if the player won the coin toss
    static void Main(string[] args)
    {
      // Print introduction text

      // Prints the title with 31 spaces placed in front of the text using the PadLeft() string function
      Console.WriteLine("23 MATCHES".PadLeft(31));
      Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY".PadLeft(15));
      
      // Print 3 blank lines with \n escape sequence
      Console.Write("\n\n\n");
      Console.WriteLine(" THIS IS A GAME CALLED '23 MATCHES'.");
      Console.Write("\n");

      Console.WriteLine("WHEN IT IS YOUR TURN, YOU MAY TAKE ONE, TWO, OR THREE");
      Console.WriteLine("MATCHES. THE OBJECT OF THE GAME IS NOT TO HAVE TO TAKE");
      Console.WriteLine("THE LAST MATCH.");
      Console.Write("\n");
      Console.WriteLine("LET'S FLIP A COIN TO SEE WHO GOES FIRST.");
      Console.WriteLine("IF IT COMES UP HEADS, I WILL WIN THE TOSS.");
      Console.Write("\n");

      // Set the number of matches to 23
      numberOfMatches = 23;

      // Create a random class object to generate the coin toss
      Random random = new Random();
      // Generates a random number between 0.0 and 1.0
      // Multiplies that number by 2 and then
      // Converts it into an integer giving either a 0 or a 1
      int coinTossResult = (int)(2 * random.NextDouble()); 

      if (coinTossResult == 1)
      {
        Console.WriteLine("TAILS! YOU GO FIRST. ");
        // Sets the player coin toss flag to true
        playerGoesFirst = true;
        PlayerTurn();
      }
      else
      {
        Console.WriteLine("HEADS! I WIN! HA! HA!");
        Console.WriteLine("PREPARE TO LOSE, MEATBALL-NOSE!!");
        Console.Write("\n");
        Console.WriteLine("I TAKE 2 MATCHES");
        numberOfMatches = numberOfMatches - 2;
      }

      // loops the code until there is 1 or fewer matches
      do
      {
        // Checks if the player has already gone 
        // because they won the coin toss
        // if they have not then the player can go
        if (playerGoesFirst == false)
        {
          Console.Write("THE NUMBER OF MATCHES IS NOW " + numberOfMatches);
          PlayerTurn();
        }
        // sets the coint toss flag to false since
        // this is only needed on the first loop of the code
        playerGoesFirst = false;
        ComputerTurn();        
      } while (numberOfMatches > 1);

    }

    static void PlayerTurn()
    {
      Console.WriteLine("\n");
      Console.WriteLine("YOUR TURN -- YOU MAY TAKE 1, 2, OR 3 MATCHES.");
      Console.Write("HOW MANY DO YOU WISH TO REMOVE ?? ");
      // Get player input
      numberOfMatchesRemovedByPlayer = ReadPlayerInput();
      // If the input is invalid (not 1, 2, or 3)
      // then ask the player to input again
      while (numberOfMatchesRemovedByPlayer > 3 || numberOfMatchesRemovedByPlayer <= 0)
      {
        Console.WriteLine("VERY FUNNY! DUMMY!");
        Console.WriteLine("DO YOU WANT TO PLAY OR GOOF AROUND?");
        Console.Write("NOW, HOW MANY MATCHES DO YOU WANT                 ?? ");
        numberOfMatchesRemovedByPlayer = ReadPlayerInput();
      }

      // Remove the player specified number of matches
      numberOfMatches = numberOfMatches - numberOfMatchesRemovedByPlayer;

      Console.WriteLine("THE ARE NOW " + numberOfMatches + " MATCHES REMAINING");      
    }
    static void ComputerTurn()
    {
      // Initialize the numberOfMatchesRemovedByComputer
      int numberOfMatchesRemovedByComputer = 0;
      switch (numberOfMatches)
      {
        case 4:
          numberOfMatchesRemovedByComputer = 3;
          break;
        case 3:
          numberOfMatchesRemovedByComputer = 2;
          break;
        case 2:
          numberOfMatchesRemovedByComputer = 1;
          break;
        case 1: case 0: // If the computer losses call this case
          Console.WriteLine("YOU WON, FLOPPY EARS !");
          Console.WriteLine("THING YOU'RE PRETTY SMART !");
          Console.WriteLine("LETS PLAY AGAIN AND I'LL BLOW YOUR SHOES OFF !!");
          break;
        default: // If there are > than 4 matches call this case
          numberOfMatchesRemovedByComputer = 4 - numberOfMatchesRemovedByPlayer;
          break;
      }
      // If the numberOfMatchesRemovedByComputer has been updated run this code,
      // if not them the computer has lost
      if (numberOfMatchesRemovedByComputer != 0)
      {
        Console.WriteLine("MY TURN ! I REMOVE " + numberOfMatchesRemovedByComputer + " MATCHES");
        numberOfMatches = numberOfMatches - numberOfMatchesRemovedByComputer;
        // If there are less than or equal to 1 matches
        // then the player has lost        
        if (numberOfMatches <= 1)
        {
          Console.Write("\n");
          Console.WriteLine("YOU POOR BOOB! YOU TOOK THE LAST MATCH! I GOTCHA!!");
          Console.WriteLine("HA ! HA ! I BEAT YOU !!!");
          Console.Write("\n");
          Console.WriteLine("GOOD BYE LOSER!");
        }
      }
    }


    // This method handles the player input 
    // and will handle inncorrect input
    static int ReadPlayerInput()
    {
      // Read user input and convert to integer
      int playerInput = 0;
      // Try to read player input
      try
      {
        playerInput = Convert.ToInt32(Console.ReadLine());
      }
      // If there is an error in the player input
      catch (System.Exception)
      {
        Console.WriteLine("?REENTER");
        Console.Write("?? ");
        // Ask the player to reenter their input
        playerInput = ReadPlayerInput();
      }
      return playerInput;      
    }

  }
}
```

# `93_23_Matches/java/CoinSide.java`

这段代码定义了一个枚举类型CoinSide，它有两个枚举值，分别为HEADS和TAILS。这个枚举类型用来表示硬币的正反两面，HEADS表示正面，TAILS表示反面。

枚举类型是一种数据类型，它可以用来表示一些有限或者无限制的属性或者状态。在这个例子中，CoinSide枚举类型表示硬币的正反两面，它只有两个枚举值，因此这个枚举类型可以用来描述硬币的两面。

在实际的应用中，CoinSide枚举类型可能会被用来进行一些判断或者选择，例如在游戏或者赌博中，需要选择硬币的正反两面来进行赌博或者游戏。


```
public enum CoinSide {
    HEADS,
    TAILS
}

```

# `93_23_Matches/java/Messages.java`

Based on the information provided, it appears that the game is a simple Rock, Paper, Scissors match. The objective is to be the first to win a match. The code does not seem to have any implementation of strategy or logic that would suggest that there is more to the game. It appears that the code is intended to be a simple read-aloud of the possible moves and the outcome of each match.


```
public class Messages {

    // This is a utility class and contains only static members.
    // Utility classes are not meant to be instantiated.
    private Messages() {
        throw new IllegalStateException("Utility class");
    }

    public static final String INTRO = """
                                          23 MATCHES
                          CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY



             THIS IS A GAME CALLED '23 MATCHES'.

            WHEN IT IS YOUR TURN, YOU MAY TAKE ONE, TWO, OR THREE
            MATCHES. THE OBJECT OF THE GAME IS NOT TO HAVE TO TAKE
            THE LAST MATCH.

            LET'S FLIP A COIN TO SEE WHO GOES FIRST.
            IF IT COMES UP HEADS, I WILL WIN THE TOSS.
            """;

    public static final String HEADS = """
            HEADS! I WIN! HA! HA!
            PREPARE TO LOSE, MEATBALL-NOSE!!

            I TAKE 2 MATCHES
            """;

    public static final String TAILS = """
            TAILS! YOU GO FIRST.
            """;

    public static final String MATCHES_LEFT = """
            THE NUMBER OF MATCHES IS NOW %d

            YOUR TURN -- YOU MAY TAKE 1, 2 OR 3 MATCHES.
            """;

    public static final String REMOVE_MATCHES_QUESTION = "HOW MANY DO YOU WISH TO REMOVE? ";

    public static final String REMAINING_MATCHES = """
            THERE ARE NOW %d MATCHES REMAINING.
            """;

    public static final String INVALID = """
            VERY FUNNY! DUMMY!
            DO YOU WANT TO PLAY OR GOOF AROUND?
            NOW, HOW MANY MATCHES DO YOU WANT?
            """;

    public static final String WIN = """
            YOU WON, FLOPPY EARS !
            THINK YOU'RE PRETTY SMART !
            LETS PLAY AGAIN AND I'LL BLOW YOUR SHOES OFF !!
            """;

    public static final String CPU_TURN = """
            MY TURN ! I REMOVE %d MATCHES.
            """;

    public static final String LOSE = """
            YOU POOR BOOB! YOU TOOK THE LAST MATCH! I GOTCHA!!
            HA ! HA ! I BEAT YOU !!!

            GOOD BYE LOSER!
            """;
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `93_23_Matches/java/TwentyThreeMatches.java`

This is a Java class that simulates a game of rock-paper-scissors between two players. The code implements the rock-paper-scissors algorithm.

The `TurnOfPlayer` method handles the choice of which player will play. It uses the `scan` method to get the number entered by the user. If the number is invalid (less than 1 or greater than 4), it will return without handling the situation. If the user enters 1, the CPU's turn is simulated. The `remainingMatches` variable is updated based on the number entered by the user.

The `matchesLeft` variable is updated based on the result of the `remainingMatches` variable and the number entered by the user. If there are 1 or 2 matches left, the CPU has to take the match. If there are 3 or 4 matches left, the Player has to take the match.

The `轮您出`与 `您出对了` methods are called to return the result of the game.

The code中还定义了一个 `flipCoin` method that flips the coin.


```
import java.util.Random;
import java.util.Scanner;

public class TwentyThreeMatches {

    private static final int MATCH_COUNT_START = 23;
    private static final Random RAND = new Random();
    private final Scanner scan = new Scanner(System.in);

    public void startGame() {
        //Initialize values
        int cpuRemoves = 0;
        int matchesLeft = MATCH_COUNT_START;
        int playerRemoves = 0;

        //Flip coin and decide who goes first.
        CoinSide coinSide = flipCoin();
        if (coinSide == CoinSide.HEADS) {
            System.out.println(Messages.HEADS);
            matchesLeft -= 2;
        } else {
            System.out.println(Messages.TAILS);
        }

        // Game loop
        while (true) {
            //Show matches left if CPU went first or Player already removed matches
            if (coinSide == CoinSide.HEADS) {
                System.out.format(Messages.MATCHES_LEFT, matchesLeft);
            }
            coinSide = CoinSide.HEADS;

            // Player removes matches
            System.out.println(Messages.REMOVE_MATCHES_QUESTION);
            playerRemoves = turnOfPlayer();
            matchesLeft -= playerRemoves;
            System.out.format(Messages.REMAINING_MATCHES, matchesLeft);

            // If 1 match is left, the CPU has to take it. You win!
            if (matchesLeft <= 1) {
                System.out.println(Messages.WIN);
                return;
            }

            // CPU removes matches
            // At least two matches are left, because win condition above was not triggered.
            if (matchesLeft <= 4) {
                cpuRemoves = matchesLeft - 1;
            } else {
                cpuRemoves = 4 - playerRemoves;
            }
            System.out.format(Messages.CPU_TURN, cpuRemoves);
            matchesLeft -= cpuRemoves;

            // If 1 match is left, the Player has to take it. You lose!
            if (matchesLeft <= 1) {
                System.out.println(Messages.LOSE);
                return;
            }
        }
    }

    private CoinSide flipCoin() {
        return RAND.nextBoolean() ? CoinSide.HEADS : CoinSide.TAILS;
    }

    private int turnOfPlayer() {
        while (true) {
            int playerRemoves = scan.nextInt();
            // Handle invalid entries
            if ((playerRemoves > 3) || (playerRemoves <= 0)) {
                System.out.println(Messages.INVALID);
                continue;
            }
            return playerRemoves;
        }
    }

}

```

# `93_23_Matches/java/TwentyThreeMatchesGame.java`

这是一个名为"TwentyThreeMatchesGame"的Java类，它基于23 Matches游戏。这个游戏的目的是创建一个与1970年BASIC游戏相似的版本，但没有引入新的功能，如文本或错误检查。代码中包含一个main方法，用于启动游戏。main方法中创建了一个"TwentyThreeMatches"类的实例，然后调用该实例的startGame方法来开始游戏。showIntro方法用于在游戏开始时输出一条欢迎消息。


```
/**
 * Game of 23 Matches
 * <p>
 * Based on the BASIC game of 23 Matches here
 * https://github.com/coding-horror/basic-computer-games/blob/main/93%2023%20Matches/23matches.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 * <p>
 * Converted from BASIC to Java by Darren Cardenas.
 */
public class TwentyThreeMatchesGame {

    public static void main(String[] args) {
        showIntro();
        TwentyThreeMatches game = new TwentyThreeMatches();
        game.startGame();
    }

    private static void showIntro() {
        System.out.println(Messages.INTRO);
    }

}

```

# `93_23_Matches/javascript/23matches.js`

这段代码定义了两个函数，分别是`print()`和`input()`。它们的主要作用如下：

1. `print()`函数将一个字符串打印到页面上，字符串由`console.log()`函数传递给`document.getElementById()`的`output`元素。这里的`console.log()`函数将字符串转换为JavaScript，`document.getElementById()`获取了指定的HTML元素，并将字符串作为文本节点添加到该元素中。

2. `input()`函数用于接收用户的输入，并将其存储在变量`input_str`中。它通过向用户提示输入字符串，并监听键盘事件来获取用户输入。当用户按下键盘上的数字13时，它会将用户输入的字符串打印到页面上，并使用`console.log()`函数将其保存在一个变量中。

在这里，`console.log()`函数将字符串打印为字符串对象，`document.getElementById()`获取了指定的HTML元素，并将该字符串作为文本节点添加到元素的`appendChild()`方法中。`console.log()`函数的`\n`字符代表一个换行符，用于在字符串之间添加新行。


```
// 23 MATCHES
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

This appears to be a game of Tw锐HFB, a type of abstraction-based programming language.

The game starts by printing the number of matches remaining and the user's turn. The user is then prompted to choose how many matches they want to remove.

If the user chooses to remove matches, the number of matches remaining is decremented by the specified number of matches.

If the user wins, the game prints a message and the user is prompted to play again. If the user loses, the game prints a different message and the user is prompted to exit.

It is not clear how the user's input is being stored or processed, and it is not clear how the matches are being implemented. It is also not clear what the purpose of the game is, or what the expected input/output for the user is.



```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// Main control section
async function main()
{
    print(tab(31) + "23 MATCHES\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print(" THIS IS A GAME CALLED '23 MATCHES'.\n");
    print("\n");
    print("WHEN IT IS YOUR TURN, YOU MAY TAKE ONE, TWO, OR THREE\n");
    print("MATCHES. THE OBJECT OF THE GAME IS NOT TO HAVE TO TAKE\n");
    print("THE LAST MATCH.\n");
    print("\n");
    print("LET'S FLIP A COIN TO SEE WHO GOES FIRST.\n");
    print("IF IT COMES UP HEADS, I WILL WIN THE TOSS.\n");
    print("\n");
    n = 23;
    q = Math.floor(2 * Math.random());
    if (q != 1) {
        print("TAILS! YOU GO FIRST. \n");
        print("\n");
    } else {
        print("HEADS! I WIN! HA! HA!\n");
        print("PREPARE TO LOSE, MEATBALL-NOSE!!\n");
        print("\n");
        print("I TAKE 2 MATCHES\n");
        n -= 2;
    }
    while (1) {
        if (q == 1) {
            print("THE NUMBER OF MATCHES IS NOW " + n + "\n");
            print("\n");
            print("YOUR TURN -- YOU MAY TAKE 1, 2 OR 3 MATCHES.\n");
        }
        print("HOW MANY DO YOU WISH TO REMOVE ");
        while (1) {
            k = parseInt(await input());
            if (k <= 0 || k > 3) {
                print("VERY FUNNY! DUMMY!\n");
                print("DO YOU WANT TO PLAY OR GOOF AROUND?\n");
                print("NOW, HOW MANY MATCHES DO YOU WANT ");
            } else {
                break;
            }
        }
        n -= k;
        print("THERE ARE NOW " + n + " MATCHES REMAINING.\n");
        if (n == 4) {
            z = 3;
        } else if (n == 3) {
            z = 2;
        } else if (n == 2) {
            z = 1;
        } else if (n > 1) {
            z = 4 - k;
        } else {
            print("YOU WON, FLOPPY EARS !\n");
            print("THINK YOU'RE PRETTY SMART !\n");
            print("LETS PLAY AGAIN AND I'LL BLOW YOUR SHOES OFF !!\n");
            break;
        }
        print("MY TURN ! I REMOVE " + z + " MATCHES\n");
        n -= z;
        if (n <= 1) {
            print("\n");
            print("YOU POOR BOOB! YOU TOOK THE LAST MATCH! I GOTCHA!!\n");
            print("HA ! HA ! I BEAT YOU !!!\n");
            print("\n");
            print("GOOD BYE LOSER!\n");
            break;
        }
        q = 1;
    }

}

```

这道题目缺少上下文，无法给出具体的解释。通常来说，在编程中，`main()` 函数是程序的入口点，也是程序的控制中心。在 main 函数中，程序会首先被编译，然后被加载到内存中执行。main 函数可以包含程序中的任何代码，包括数据定义、变量初始化、函数定义等。


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


# `93_23_Matches/python/23matches.py`

It looks like the code is trying to simulate a game of Rock, Paper, Scissors where the computer is the AI and the human is the player. The code is逐行分析可得：
```
print("Welcome, little meatball! Let's play some Rock, Paper, Scissors!")
print("回合1：")
print("1. Rock")
print("2. Paper")
print("3. Scissors")
print("输入1.2.3分别输入对应回合")
choice_human = 2
matches = 3
```
从代码中可得知，游戏开始时，输出欢迎消息，然后进入第一个回合，接着玩家可以选择1、2、3三种出拳方式，然后计算出拳结果，比较大小，若玩家胜利，则显示玩家获胜信息，若失败则显示电脑获胜信息，最后进入下一个回合。代码使用的是循环结构，重复执行出拳操作直到游戏结束。


```
#!/usr/bin/env python3
# 23 Matches
#
# Converted from BASIC to Python by Trevor Hobson

import random


def play_game() -> None:
    """Play one round of the game"""

    matches = 23
    humans_turn = random.randint(0, 1) == 1
    if humans_turn:
        print("Tails! You go first.\n")
        prompt_human = "How many do you wish to remove "
    else:
        print("Heads! I win! Ha! Ha!")
        print("Prepare to lose, meatball-nose!!")

    choice_human = 2
    while matches > 0:
        if humans_turn:
            choice_human = 0
            if matches == 1:
                choice_human = 1
            while choice_human == 0:
                try:
                    choice_human = int(input(prompt_human))
                    if choice_human not in [1, 2, 3] or choice_human > matches:
                        choice_human = 0
                        print("Very funny! Dummy!")
                        print("Do you want to play or goof around?")
                        prompt_human = "Now, how many matches do you want "
                except ValueError:
                    print("Please enter a number.")
                    prompt_human = "How many do you wish to remove "
            matches = matches - choice_human
            if matches == 0:
                print("You poor boob! You took the last match! I gotcha!!")
                print("Ha ! Ha ! I beat you !!\n")
                print("Good bye loser!")
            else:
                print("There are now", matches, "matches remaining.\n")
        else:
            choice_computer = 4 - choice_human
            if matches == 1:
                choice_computer = 1
            elif 1 < matches < 4:
                choice_computer = matches - 1
            matches = matches - choice_computer
            if matches == 0:
                print("You won, floppy ears !")
                print("Think you're pretty smart !")
                print("Let's play again and I'll blow your shoes off !!")
            else:
                print("My turn ! I remove", choice_computer, "matches")
                print("The number of matches is now", matches, "\n")
        humans_turn = not humans_turn
        prompt_human = "Your turn -- you may take 1, 2 or 3 matches.\nHow many do you wish to remove "


```

这段代码是一个Python程序，主要目的是让用户通过掷硬币来决定游戏规则并开始游戏。程序中定义了一个函数`main()`，它返回一个`None`，即没有返回值。

在函数内部，首先输出一段文本，占据了31个字符。接着输出一段文本，占据了15个字符。然后输出一段文本，包含了游戏的名称。

接下来是游戏规则的说明。程序提示用户在每次游戏时可以选择1、2或3个匹配，但无论用户选择什么，游戏的目标都不是要赢得最后一个匹配。程序还提示用户进行掷硬币，以决定游戏开始谁先手。

程序使用了一个名为`keep_playing`的布尔变量，决定是否继续游戏。在游戏过程中，程序调用了一个名为`play_game()`的函数，但这个函数并未在代码中定义。推测`play_game()`函数可能会实现游戏规则的相关逻辑，例如从用户那里获取游戏ID等。

游戏规则的解释可能较为复杂，因为这段代码只提供了简单的游戏玩法，并没有提供详细的游戏规则说明。


```
def main() -> None:
    print(" " * 31 + "23 MATCHHES")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")
    print("This is a game called '23 Matches'.\n")
    print("When it is your turn, you may take one, two, or three")
    print("matches. The object of the game is not to have to take")
    print("the last match.\n")
    print("Let's flip a coin to see who goes first.")
    print("If it comes up heads, I will win the toss.\n")

    keep_playing = True
    while keep_playing:
        play_game()
        keep_playing = input("\nPlay again? (yes or no) ").lower().startswith("y")


```

这段代码是一个if语句，判断当前脚本是否是Python标准模块中的一个可执行文件(即.py文件)。如果是可执行文件，则执行main函数。

if __name__ == "__main__":
```是if语句的条件，它表示当条件为真时，代码块内的内容会被执行。在这个例子中，它判断当前脚本是否是Python标准模块中的一个可执行文件。

if __name__ == "__main__":
```如果是，执行main函数

``` 

如果当前脚本不是一个可执行文件，则条件为假，代码块内的内容不会被执行。

```


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [rust](https://www.rust-lang.org/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### War

This program plays the card game of War. In War, the card deck is shuffled, then two cards are dealt, one to each player. Players compare cards and the higher card (numerically) wins. In case of a tie, no one wins. The game ends when you have gone through the whole deck (52 cards, 26 games) or when you decide to quit.

The computer gives cards by suit and number, for example, S-7 is the 7 of spades.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=178)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=193)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html


#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `94_War/csharp/War/Cards.cs`

This is a C# class that represents a deck of cards. The `Deck` class has a number of methods for working with the cards, including `ToString()`, which returns a string representation of the deck.

The `operator >=` method is an overload for the `>` operator, which is called when comparing two cards. The `ToString()` method is also overridden to return a string representation of the deck.

The `Deck` class has a `DeckSize` property, which specifies the maximum number of cards in the deck. The deck is initialized when a new `Deck` object is created, with all the cards being assigned their corresponding suits and ranks.

The `GetCard()` method returns a card at a particular position in the deck.

The `Shuffle()` method shuffles the cards in the deck.


```
﻿using System;
using System.Collections.Generic;



namespace War
{
    // These enums define the card's suit and rank.
    public enum Suit
    {
        clubs,
        diamonds,
        hearts,
        spades
    }

    public enum Rank
    {
        // Skip 1 because ace is high.
        two = 2,
        three,
        four,
        five,
        six,
        seven,
        eight,
        nine,
        ten,
        jack,
        queen,
        king,
        ace
    }

    // A class to represent a playing card.
    public class Card
    {
        // A card is an immutable object (i.e. it can't be changed) so its suit
        // and rank value are readonly; they can only be set in the constructor.
        private readonly Suit suit;
        private readonly Rank rank;

        // These dictionaries are used to convert a suit or rank value into a string.
        private readonly Dictionary<Suit, string> suitNames = new Dictionary<Suit, string>()
        {
            { Suit.clubs, "C"},
            { Suit.diamonds, "D"},
            { Suit.hearts, "H"},
            { Suit.spades, "S"},
        };

        private readonly Dictionary<Rank, string> rankNames = new Dictionary<Rank, string>()
        {
            { Rank.two, "2"},
            { Rank.three, "3"},
            { Rank.four, "4"},
            { Rank.five, "5"},
            { Rank.six, "6"},
            { Rank.seven, "7"},
            { Rank.eight, "8"},
            { Rank.nine, "9"},
            { Rank.ten, "10"},
            { Rank.jack, "J"},
            { Rank.queen, "Q"},
            { Rank.king, "K"},
            { Rank.ace, "A"},
        };

        public Card(Suit suit, Rank rank)
        {
            this.suit = suit;
            this.rank = rank;
        }

        // Relational Operator Overloading.
        //
        // You would normally expect the relational operators to consider both the suit and the
        // rank of a card, but in this program suit doesn't matter so we define the operators to just
        // compare rank.

        // When adding relational operators we would normally include == and != but they are not
        // relevant to this program so haven't been defined. Note that if they were defined we
        // should also override the Equals() and GetHashCode() methods. See, for example:
        // http://www.blackwasp.co.uk/CSharpRelationalOverload.aspx

        // If the == and != operators were defined they would look like this:
        //
        //public static bool operator ==(Card lhs, Card rhs)
        //{
        //    return lhs.rank == rhs.rank;
        //}
        //
        //public static bool operator !=(Card lhs, Card rhs)
        //{
        //    return !(lhs == rhs);
        //}

        public static bool operator <(Card lhs, Card rhs)
        {
            return lhs.rank < rhs.rank;
        }

        public static bool operator >(Card lhs, Card rhs)
        {
            return rhs < lhs;
        }

        public static bool operator <=(Card lhs, Card rhs)
        {
            return !(lhs > rhs);
        }

        public static bool operator >=(Card lhs, Card rhs)
        {
            return !(lhs < rhs);
        }

        public override string ToString()
        {
            // N.B. We are using string interpolation to create the card name.
            return $"{suitNames[suit]}-{rankNames[rank]}";
        }
    }

    // A class to represent a deck of cards.
    public class Deck
    {
        public const int deckSize = 52;

        private Card[] theDeck = new Card[deckSize];

        public Deck()
        {
            // Populate theDeck with all the cards in order.
            int i = 0;
            for (Suit suit = Suit.clubs; suit <= Suit.spades; suit++)
            {
                for (Rank rank = Rank.two; rank <= Rank.ace; rank++)
                {
                    theDeck[i] = new Card(suit, rank);
                    i++;
                }
            }
        }

        // Return the card at a particular position in the deck.
        // N.B. As this is such a short method, we make it an
        // expression-body method.
        public Card GetCard(int i) => theDeck[i];

        // Shuffle the cards, this uses the modern version of the
        // Fisher-Yates shuffle, see:
        // https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm
        public void Shuffle()
        {
            var rand = new Random();

            // Iterate backwards through the deck.
            for (int i = deckSize - 1; i >= 1; i--)
            {
                int j = rand.Next(0, i);

                // Swap the cards at i and j
                Card temp = theDeck[j];
                theDeck[j] = theDeck[i];
                theDeck[i] = temp;
            }
        }
    }
}

```

# `94_War/csharp/War/Program.cs`

这段代码是一个文本输入输出程序，用于玩一个名为“war”的游戏。程序中定义了一个用户界面（UI）类，用于显示游戏结果，包括玩家的得分和电脑的得分。游戏中有两张牌（A、B），每张牌有两个面，玩家可以选择其中一张，而电脑则不能选择。游戏会随机洗牌，然后玩家和电脑交替进行游戏。在游戏过程中，玩家可以选择继续或结束游戏。如果玩家选择结束游戏，程序会统计玩家和电脑的得分并输出。


```
﻿namespace War
{
    class Program
    {
        static void Main(string[] args)
        {
            var ui = new UserInterface();
            ui.WriteIntro();

            var deck = new Deck();
            deck.Shuffle();

            int yourScore = 0;
            int computersScore = 0;
            bool usedAllCards = true;

            for (int i = 0; i < Deck.deckSize; i += 2)
            {
                // Play the next hand.
                var yourCard = deck.GetCard(i);
                var computersCard = deck.GetCard(i + 1);

                ui.WriteAResult(yourCard, computersCard, ref computersScore, ref yourScore);

                if (!ui.AskAQuestion("DO YOU WANT TO CONTINUE? "))
                {
                    usedAllCards = false;
                    break;
                }
            }

            ui.WriteClosingRemarks(usedAllCards, yourScore, computersScore);
        }
    }
}

```

# `94_War/csharp/War/UserInterface.cs`

This looks like a class that simulates a game of cards where one player is the computer and the other player is the player who is trying to win. The class has several methods, including a method to write a message to the console indicating the winner of the game, a method to write the results of the game to the console, a method to ask the computer to choose a card and a method to write the closing remarks of the game. The class also has a method to initialize the game and a method to write the results of the game.


```
﻿using System;



namespace War
{
    // This class displays all the text that the user sees when playing the game.
    // It also handles asking the user a yes/no question and returning their answer.
    public class UserInterface
    {
        public void WriteIntro()
        {
            Console.WriteLine("                                 WAR");
            Console.WriteLine("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine();

            Console.WriteLine("THIS IS THE CARD GAME OF WAR.  EACH CARD IS GIVEN BY SUIT-#");
            Console.Write("AS S-7 FOR SPADE 7.  ");

            if (AskAQuestion("DO YOU WANT DIRECTIONS? "))
            {
                Console.WriteLine("THE COMPUTER GIVES YOU AND IT A 'CARD'.  THE HIGHER CARD");
                Console.WriteLine("(NUMERICALLY) WINS.  THE GAME ENDS WHEN YOU CHOOSE NOT TO");
                Console.WriteLine("CONTINUE OR WHEN YOU HAVE FINISHED THE PACK.");
            }

            Console.WriteLine();
            Console.WriteLine();
        }

        public void WriteAResult(Card yourCard, Card computersCard, ref int computersScore, ref int yourScore)
        {
            Console.WriteLine($"YOU: {yourCard}     COMPUTER: {computersCard}");
            if (yourCard < computersCard)
            {
                computersScore++;
                Console.WriteLine($"THE COMPUTER WINS!!! YOU HAVE {yourScore} AND THE COMPUTER HAS {computersScore}");
            }
            else if (yourCard > computersCard)
            {
                yourScore++;
                Console.WriteLine($"YOU WIN. YOU HAVE {yourScore} AND THE COMPUTER HAS {computersScore}");
            }
            else
            {
                Console.WriteLine("TIE.  NO SCORE CHANGE");
            }
        }

        public bool AskAQuestion(string question)
        {
            // Repeat asking the question until the user answers "YES" or "NO".
            while (true)
            {
                Console.Write(question);
                string result = Console.ReadLine();

                if (result.ToLower() == "yes")
                {
                    Console.WriteLine();
                    return true;
                }
                else if (result.ToLower() == "no")
                {
                    Console.WriteLine();
                    return false;
                }

                Console.WriteLine("YES OR NO, PLEASE.");
            }
        }

        public void WriteClosingRemarks(bool usedAllCards, int yourScore, int computersScore)
        {
            if (usedAllCards)
            {
                Console.WriteLine("WE HAVE RUN OUT OF CARDS.");
            }
            Console.WriteLine($"FINAL SCORE:  YOU: {yourScore}  THE COMPUTER: {computersScore}");
            Console.WriteLine("THANKS FOR PLAYING.  IT WAS FUN.");
        }
    }
}

```

# `94_War/csharp/WarTester/Tests.cs`

It looks like the `DeckTest` class is the testing class for the `Deck` class.  The `InitialDeckContainsCardsInOrder` test method checks that the initial deck contains the cards in order.  The `ShufflingChangesDeck` test method checks that the deck is shuffled and that the cards are not in the initial order.

The `Deck` class is defined with a constructor that takes a deck object and a shuffle method.  The deck object is used to store the state of the deck.  The `Shuffle` method is used to shuffle the cards in the deck.

It looks like the `DeckTest` class has two main methods: `InitialDeckContainsCardsInOrder` and `ShufflingChangesDeck`.  The `InitialDeckContainsCardsInOrder` method checks that the initial deck contains the cards in order.  The `ShufflingChangesDeck` method checks that the deck is shuffled and that the cards are not in the initial order.

It is not clear from the code provided how the `Deck` class is implemented.  It is possible that the `Deck` class has


```
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Text;
using War;



namespace WarTester
{
    [TestClass]
    public class CardTest
    {
        private Card c1 = new Card(Suit.clubs, Rank.two);
        private Card c2 = new Card(Suit.clubs, Rank.ten);
        private Card c3 = new Card(Suit.diamonds, Rank.ten);
        private Card c4 = new Card(Suit.diamonds, Rank.ten);

        // Test the relational operators.

        [TestMethod]
        public void LessThanIsValid()
        {
            Assert.IsTrue(c1 < c2, "c1 < c2");  // Same suit, different rank.
            Assert.IsFalse(c2 < c1, "c2 < c1"); // Same suit, different rank.

            Assert.IsFalse(c3 < c4, "c3 < c4"); // Same suit, same rank.

            Assert.IsTrue(c1 < c3, "c1 < c3");  // Different suit, different rank.
            Assert.IsFalse(c3 < c1, "c3 < c1"); // Different suit, different rank.

            Assert.IsFalse(c2 < c4, "c2 < c4"); // Different suit, same rank.
            Assert.IsFalse(c4 < c2, "c4 < c2"); // Different suit, same rank.
        }

        [TestMethod]
        public void GreaterThanIsValid()
        {
            Assert.IsFalse(c1 > c2, "c1 > c2"); // Same suit, different rank.
            Assert.IsTrue(c2 > c1, "c2 > c1");  // Same suit, different rank.

            Assert.IsFalse(c3 > c4, "c3 > c4"); // Same suit, same rank.

            Assert.IsFalse(c1 > c3, "c1 > c3"); // Different suit, different rank.
            Assert.IsTrue(c3 > c1, "c3 > c1");  // Different suit, different rank.

            Assert.IsFalse(c2 > c4, "c2 > c4"); // Different suit, same rank.
            Assert.IsFalse(c4 > c2, "c4 > c2"); // Different suit, same rank.
        }

        [TestMethod]
        public void LessThanEqualsIsValid()
        {
            Assert.IsTrue(c1 <= c2, "c1 <= c2");  // Same suit, different rank.
            Assert.IsFalse(c2 <= c1, "c2 <= c1"); // Same suit, different rank.

            Assert.IsTrue(c3 <= c4, "c3 <= c4");  // Same suit, same rank.

            Assert.IsTrue(c1 <= c3, "c1 <= c3");  // Different suit, different rank.
            Assert.IsFalse(c3 <= c1, "c3 <= c1"); // Different suit, different rank.

            Assert.IsTrue(c2 <= c4, "c2 <= c4");  // Different suit, same rank.
            Assert.IsTrue(c4 <= c2, "c4 <= c2");  // Different suit, same rank.
        }

        [TestMethod]
        public void GreaterThanEqualsIsValid()
        {
            Assert.IsFalse(c1 >= c2, "c1 >= c2"); // Same suit, different rank.
            Assert.IsTrue(c2 >= c1, "c2 >= c1");  // Same suit, different rank.

            Assert.IsTrue(c3 >= c4, "c3 >= c4");  // Same suit, same rank.

            Assert.IsFalse(c1 >= c3, "c1 >= c3"); // Different suit, different rank.
            Assert.IsTrue(c3 >= c1, "c3 >= c1");  // Different suit, different rank.

            Assert.IsTrue(c2 >= c4, "c2 >= c4");  // Different suit, same rank.
            Assert.IsTrue(c4 >= c2, "c4 >= c2");  // Different suit, same rank.
        }

        [TestMethod]
        public void ToStringIsValid()
        {
            var s1 = c1.ToString();
            var s2 = c3.ToString();
            var s3 = new Card(Suit.hearts, Rank.queen).ToString();
            var s4 = new Card(Suit.spades, Rank.ace).ToString();

            Assert.IsTrue(s1 == "C-2", "s1 invalid");
            Assert.IsTrue(s2 == "D-10", "s2 invalid");
            Assert.IsTrue(s3 == "H-Q", "s3 invalid");
            Assert.IsTrue(s4 == "S-A", "s4 invalid");
        }
    }

    [TestClass]
    public class DeckTest
    {
        private readonly string cardNamesInOrder = "C-2C-3C-4C-5C-6C-7C-8C-9C-10C-JC-QC-KC-AD-2D-3D-4D-5D-6D-7D-8D-9D-10D-JD-QD-KD-AH-2H-3H-4H-5H-6H-7H-8H-9H-10H-JH-QH-KH-AS-2S-3S-4S-5S-6S-7S-8S-9S-10S-JS-QS-KS-A";

        //Helper method. Adds the names of all the cards together into a single string.
        private string ConcatenateTheDeck(Deck d)
        {
            StringBuilder sb = new StringBuilder();

            for (int i = 0; i < Deck.deckSize; i++)
            {
                sb.Append(d.GetCard(i));
            }

            return sb.ToString();
        }

        [TestMethod]
        public void InitialDeckContainsCardsInOrder()
        {
            Deck d = new Deck();
            string allTheCards = ConcatenateTheDeck(d);

            Assert.IsTrue(allTheCards == cardNamesInOrder);
        }

        [TestMethod]
        public void ShufflingChangesDeck()
        {
            // I'm not sure how to test that shuffling has worked other than to check that the cards aren't in the initial order.
            Deck d = new Deck();
            d.Shuffle();
            string allTheCards = ConcatenateTheDeck(d);

            Assert.IsTrue(allTheCards != cardNamesInOrder);
        }
    }
}

```

﻿Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html)

Converted to [D](https://dlang.org/) by [Bastiaan Veelo](https://github.com/veelo).

## Running the code

Assuming the reference [dmd](https://dlang.org/download.html#dmd) compiler:
```shell
dmd -preview=dip1000 -run war.d
```

[Other compilers](https://dlang.org/download.html) also exist.

## Specialties explained

This game code contains some specialties that you might want to know more about. Here goes.

### Suits

Most modern consoles are capable of displaying more than just ASCII, and so I have chosen to display the actual ♠, ♥, ♦
and ♣ instead of substituting them by letters like the BASIC original did. Only the Windows console needs a nudge in
the right direction with these instructions:
```d
SetConsoleOutputCP(CP_UTF8); // Set code page
SetConsoleOutputCP(GetACP);  // Restore the default
```
Instead of cluttering the `main()` function with these lesser important details, we can move them into a
[module constructor and module destructor](https://dlang.org/spec/module.html#staticorder), which run before and after
`main()` respectively. And because order of declaration is irrelevant in a D module, we can push those all the way
down to the bottom of the file. This is of course only necessary on Windows (and won't even work anywhere else) so
we'll need to wrap this in a `version (Windows)` conditional code block:
```d
version (Windows)
{
    import core.sys.windows.windows;

    shared static this() @trusted
    {
        SetConsoleOutputCP(CP_UTF8);
    }

    shared static ~this() @trusted
    {
        SetConsoleOutputCP(GetACP);
    }
}
```
Although it doesn't matter much in this single-threaded program, the `shared` attribute makes that these
constructors/destructors are run once per program invocation; non-shared module constructors and module destructors are
run for every thread. The `@trusted` annotation is necessary because these are system API calls; The compiler cannot
check these for memory-safety, and so we must indicate that we have reviewed the safety manually.

### Uniform Function Call Syntax

In case you wonder why this line works:
```d
if ("Do you want instructions?".yes)
    // ...
```
then it is because this is equivalent to
```d
if (yes("Do you want instructions?"))
    // ...
```
where `yes()` is a Boolean function that is defined below `main()`. This is made possible by the language feature that
is called [uniform function call syntax (UFCS)](https://dlang.org/spec/function.html#pseudo-member). UFCS works by
passing what is in front of the dot as the first parameter to the function, and it was invented to make it possible to
call free functions on objects as if they were member functions. UFCS can also be used to obtain a more natural order
of function calls, such as this line inside `yes()`:
```d
return trustedReadln.strip.toLower.startsWith("y");
```
which reads easier than the equivalent
```d
return startsWith(toLower(strip(trustedReadln())), "y");
```

### Type a lot or not?

It would have been straight forward to define the `cards` array explicitly like so:
```d
const cards = ["2♠", "2♥", "2♦", "2♣", "3♠", "3♥", "3♦", "3♣",
               "4♠", "4♥", "4♦", "4♣", "5♠", "5♥", "5♦", "5♣",
               "6♠", "6♥", "6♦", "6♣", "7♠", "7♥", "7♦", "7♣",
               "8♠", "8♥", "8♦", "8♣", "9♠", "9♥", "9♦", "9♣",
               "10♠", "10♥", "10♦", "10♣", "J♥", "J♦", "J♣", "J♣",
               "Q♠", "Q♥", "Q♦", "Q♣", "K♠", "K♥", "K♦", "K♣",
               "A♠", "A♥", "A♦", "A♣"];
```
but that's tedious, difficult to spot errors in (*can you?*) and looks like something a computer can automate. Indeed
it can:
```d
static const cards = cartesianProduct(["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"],
                                      ["♠", "♥", "♦", "♣"]).map!(a => a.expand.only.join).array;
```
The function [`cartesianProduct`](https://dlang.org/phobos/std_algorithm_setops.html#cartesianProduct) takes two
ranges, like the horizontal and vertical headers of a spreadsheet, and fills the table with the combinations that form
the coordinates of the cells. But the output of that function is in the form of an array of
[`Tuple`](https://dlang.org/phobos/std_typecons.html#Tuple)s, which looks like `[Tuple!(string, string)("2", "♠"),
Tuple!(string, string)("2", "♥"), ... etc]`. [`map`](https://dlang.org/phobos/std_algorithm_iteration.html#map)
comes to the rescue, converting each Tuple to a string, by calling
[`expand`](https://dlang.org/phobos/std_typecons.html#.Tuple.expand), then
[`only`](https://dlang.org/phobos/std_range.html#only) and then [`join`](https://dlang.org/phobos/std_array.html#join)
on them. The result is a lazily evaluated range of strings. Finally,
[`array`](https://dlang.org/phobos/std_array.html#array) turns the range into a random access array. The `static`
attribute makes that all this is performed at compile-time, so the result is exactly the same as the manually entered
data, but without the typo's.

### Shuffle the cards or not?

The original BASIC code works with a constant array of cards, ordered by increasing numerical value, and indexing it
with indices that have been shuffled. This is efficient because in comparing who wins, the indices can be compared
directly, since a higher index correlates to a card with a higher numerical value (when divided by the number of suits,
4). Some of the other reimplementations in other languages have been written in a lesser efficient way by shuffling the
array of cards itself. This then requires the use of a lookup table or searching for equality in an auxiliary array
when comparing cards.

I find the original more elegant, so that's what you see here:
```d
const indices = iota(0, cards.length).array.randomShuffle;
```
[`iota`](https://dlang.org/phobos/std_range.html#iota) produces a range of integers, in this case starting at 0 and
increasing up to the number of cards in the deck (exclusive). [`array`](https://dlang.org/phobos/std_array.html#array)
turns the range into an array, so that [`randomShuffle`](https://dlang.org/phobos/std_random.html#randomShuffle) can
do its work.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `94_War/java/War.java`

This code appears to be a card game where the player and computer take turns rolling a die and answering a yes or no question. The code has two main functions, `endGameOutput` and `playerWonRound`, which are used to show the final score of the game when it ends early or when the player wins a round.

The `endGameOutput` function takes a boolean value indicating whether the game was ended early or not. If the game was ended early, it prints out a message indicating the winner of the game. If the game was not ended early, it prints out a message and the score of the game.

The `playerWonRound` function increments the player's total score by 1 if they won the round. It does not seem to have any影响的 on the computer's score.

Overall, this code is a simple game that allows the player to roll a die and answer yes or no questions to try to win a round.


```
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Scanner;

/**
 * Converted FROM BASIC to Java by Nahid Mondol.
 *
 * Based on Trevor Hobsons approach.
 */
public class War {

    private static final int CARD_DECK_SIZE = 52;
    private static int playerTotalScore = 0;
    private static int computerTotalScore = 0;
    private static boolean invalidInput;
    private static Scanner userInput = new Scanner(System.in);

    // Simple approach for storing a deck of cards.
    // Suit-Value, ex: 2 of Spades = S-2, King of Diamonds = D-K, etc...
    private static ArrayList<String> deckOfCards = new ArrayList<String>(
            Arrays.asList("S-2", "H-2", "C-2", "D-2", "S-3", "H-3", "C-3", "D-3", "S-4", "H-4", "C-4", "D-4", "S-5",
                    "H-5", "C-5", "D-5", "S-6", "H-6", "C-6", "D-6", "S-7", "H-7", "C-7", "D-7", "S-8", "H-8", "C-8",
                    "D-8", "S-9", "H-9", "C-9", "D-9", "S-10", "H-10", "C-10", "D-10", "S-J", "H-J", "C-J", "D-J",
                    "S-Q", "H-Q", "C-Q", "D-Q", "S-K", "H-K", "C-K", "D-K", "S-A", "H-A", "C-A", "D-A"));

    public static void main(String[] args) {
        introMessage();
        showDirectionsBasedOnInput();
        playGame();
    }

    private static void introMessage() {
        System.out.println("\t         WAR");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println("THIS IS THE CARD GAME OF WAR. EACH CARD IS GIVEN BY SUIT-#");
        System.out.print("AS S-7 FOR SPADE 7. DO YOU WANT DIRECTIONS? ");
    }

    private static void showDirectionsBasedOnInput() {
        // Stay in loop until player chooses an option.
        invalidInput = true;
        while (invalidInput) {
            switch (userInput.nextLine().toLowerCase()) {
                case "yes":
                    System.out.println("THE COMPUTER GIVES YOU AND IT A 'CARD'. THE HIGHER CARD");
                    System.out.println("(NUMERICALLY) WINS. THE GAME ENDS WHEN YOU CHOOSE NOT TO ");
                    System.out.println("CONTINUE OR WHEN YOU HAVE FINISHED THE PACK.\n");
                    invalidInput = false;
                    break;
                case "no":
                    System.out.println();
                    invalidInput = false;
                    break;
                default:
                    System.out.print("YES OR NO, PLEASE.   ");
            }
        }
    }

    private static void playGame() {

        // Checks to see if the player ends the game early.
        // Ending early will cause a different output to appear.
        boolean gameEndedEarly = false;

        // Shuffle the deck of cards.
        Collections.shuffle(deckOfCards);

        // Since the deck is already suffled, pull each card until the deck is empty or
        // until the user quits.
        outerloop:
        for (int i = 1; i <= CARD_DECK_SIZE; i += 2) {
            System.out.println("YOU: " + deckOfCards.get(i - 1) + "\t " + "COMPUTER: " + deckOfCards.get(i));
            getWinner(deckOfCards.get(i - 1), deckOfCards.get(i));

            invalidInput = true;
            while (invalidInput) {
                if (endedEarly()) {
                    // Player ended game early.
                    // Break out of game loop and show end game output.
                    gameEndedEarly = true;
                    break outerloop;
                }
            }
        }

        endGameOutput(gameEndedEarly);
    }

    /**
     * Outputs the winner of the current round.
     *
     * @param playerCard   Players card.
     * @param computerCard Computers card.
     */
    private static void getWinner(String playerCard, String computerCard) {

        // Return the number value of the card pulled.
        String playerCardScore = (playerCard.length() == 3) ? Character.toString(playerCard.charAt(2))
                : playerCard.substring(2, 4);
        String computerCardScore = (computerCard.length() == 3) ? Character.toString(computerCard.charAt(2))
                : computerCard.substring(2, 4);

        if (checkCourtCards(playerCardScore) > checkCourtCards(computerCardScore)) {
            System.out.println("YOU WIN.   YOU HAVE " + playerWonRound() + "   COMPUTER HAS " + getComputerScore());
        } else if (checkCourtCards(playerCardScore) < checkCourtCards(computerCardScore)) {
            System.out.println(
                    "COMPUTER WINS!!!   YOU HAVE " + getPlayerScore() + "   COMPUTER HAS " + computerWonRound());
        } else {
            System.out.println("TIE.  NO SCORE CHANGE");
        }

        System.out.print("DO YOU WANT TO CONTINUE? ");
    }

    /**
     * @param cardScore Score of the card being pulled.
     * @return an integer value of the current card's score.
     */
    private static int checkCourtCards(String cardScore) {
        switch (cardScore) {
            case "J":
                return Integer.parseInt("11");
            case "Q":
                return Integer.parseInt("12");
            case "K":
                return Integer.parseInt("13");
            case "A":
                return Integer.parseInt("14");
            default:
                return Integer.parseInt(cardScore);
        }
    }

    /**
     * @return true if the player ended the game early. false otherwise.
     */
    private static boolean endedEarly() {
        switch (userInput.nextLine().toLowerCase()) {
            case "yes":
                invalidInput = false;
                return false;
            case "no":
                invalidInput = false;
                return true;
            default:
                invalidInput = true;
                System.out.print("YES OR NO, PLEASE.   ");
                return false;
        }
    }

    /**
     * Show output based on if the game was ended early or not.
     *
     * @param endedEarly true if the game was ended early, false otherwise.
     */
    private static void endGameOutput(boolean endedEarly) {
        if (endedEarly) {
            System.out.println("YOU HAVE ENDED THE GAME. FINAL SCORE:  YOU: " + getPlayerScore() + " COMPUTER: "
                    + getComputerScore());
            System.out.println("THANKS FOR PLAYING.  IT WAS FUN.");
        } else {
            System.out.println("WE HAVE RUN OUT OF CARDS. FINAL SCORE:  YOU: " + getPlayerScore() + " COMPUTER: "
                    + getComputerScore());
            System.out.println("THANKS FOR PLAYING.  IT WAS FUN.");
        }
    }

    /**
     * Increment the player's total score if they have won the round.
     */
    private static int playerWonRound() {
        return playerTotalScore += 1;
    }

    /**
     * Get the player's total score.
     */
    private static int getPlayerScore() {
        return playerTotalScore;
    }

    /**
     * Increment the computer's total score if they have won the round.
     */
    private static int computerWonRound() {
        return computerTotalScore += 1;
    }

    /**
     * Get the computer's total score.
     */
    private static int getComputerScore() {
        return computerTotalScore;
    }
}

```