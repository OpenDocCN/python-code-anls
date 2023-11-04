# BasicComputerGames源码解析 52

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Hurkle

Hurkle? A Hurkle is a happy beast and lives in another galaxy on a planet named Lirht that has three moons. Hurkle are favorite pets of the Gwik, the dominant race of Lihrt and … well, to find out more, read “The Hurkle is a Happy Beast,” a story in the book _A Way Home_ by Theodore Sturgeon.

In this program a shy hurkle is hiding on a 10 by 10 grid. Homebase is point 0,0 in the _Southwest_ corner. Your guess as to the gridpoint where the hurkle is hiding should be a pair of whole numbers, separated by a comma. After each try, the computer will tell you the approximate direction to go look for the Hurkle. You get five guesses to find him; you may change this number, although four guesses is actually enough.

This program was written by Bob Albrecht of People’s Computer Company.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=94)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=109)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `51_Hurkle/csharp/CardinalDirection.cs`

这段代码定义了一个名为 "CardinalDirection" 的内部枚举类型，该类型定义了 9 种方向，分别为：

North(北方)
NorthEast(北方东)
East(东方)
SouthEast(东方)
South(南方)
SouthWest(南方西)
West(西方)
NorthWest(北方西)

这个枚举类型可能用于表示流向或者位置，例如在地图上使用，或者在计算机图形学中使用。


```
namespace hurkle
{
    internal enum CardinalDirection
    {
        None,
        North,
        NorthEast,
        East,
        SouthEast,
        South,
        SouthWest,
        West,
        NorthWest
    }
}

```

# `51_Hurkle/csharp/ConsoleHurkleView.cs`

This is a class that simulates a game of cards where one player is the AI (人工智能) and the other is the human (human player). The AI has a set of clues to give the human player, and the human player has a limited number of guesses to guess where the AI thinks the card is.

The `CardinalDirection` enum defines the four cardinal directions: `NORTH`, `SOUTH`, `EAST`, and `WEST`.

The `FailedGuessViewModel` class represents the view model for the failed guess. It has a `Direction` property that indicates the current direction of the AI (north, south, east, or west), and a `MaxGuesses` property that stores the number of guesses the AI has allowed the human player.

The `LossViewModel` class represents the view model for the loss. It has a `MaxGuesses` property that stores the maximum number of guesses the AI has allowed the human player, and a `HurkleLocation` property that stores the coordinates of the Hurkle (a randomly chosen location on the card) that the AI has marked.

The `VictoryViewModel` class represents the view model for the victory. It has a `CurrentGuessNumber` property that stores the number of guesses the human player has made, and a `Message` property that stores the message the AI should display when it wins.

The `ShowDirection`, `ShowDirection`, `ShowLoss`, and `ShowVictory` methods are used to display information about the game to the human player. These methods are called by the `ShowDirection`, `ShowLoss`, and `ShowVictory` methods in the `CardinalDirection` enum.

The `ShowDirection` method displays the direction the AI is looking (north, south, east, or west).

The `ShowLoss` method displays the number of guesses the AI has allowed the human player and the location of the Hurkle that the AI has marked.

The `ShowVictory` method displays a message when the AI wins.


```
using System;

namespace hurkle
{
    internal class ConsoleHurkleView : IHurkleView
    {
        public GamePoint GetGuess(GuessViewModel guessViewModel)
        {
            Console.WriteLine($"GUESS #{guessViewModel.CurrentGuessNumber}");
            var inputLine = Console.ReadLine();
            var seperateStrings = inputLine.Split(',', 2, StringSplitOptions.TrimEntries);
            var guessPoint = new GamePoint{
                X = int.Parse(seperateStrings[0]),
                Y = int.Parse(seperateStrings[1])
            };

            return guessPoint;
        }

        public void ShowDirection(FailedGuessViewModel failedGuessViewModel)
        {
            Console.Write("GO ");
            switch(failedGuessViewModel.Direction)
            {
                case CardinalDirection.East:
                    Console.WriteLine("EAST");
                    break;
                case CardinalDirection.North:
                    Console.WriteLine("NORTH");
                    break;
                case CardinalDirection.South:
                    Console.WriteLine("SOUTH");
                    break;
                case CardinalDirection.West:
                    Console.WriteLine("WEST");
                    break;
                case CardinalDirection.NorthEast:
                    Console.WriteLine("NORTHEAST");
                    break;
                case CardinalDirection.NorthWest:
                    Console.WriteLine("NORTHWEST");
                    break;
                case CardinalDirection.SouthEast:
                    Console.WriteLine("SOUTHEAST");
                    break;
                case CardinalDirection.SouthWest:
                    Console.WriteLine("SOUTHWEST");
                    break;
            }

            Console.WriteLine();
        }

        public void ShowLoss(LossViewModel lossViewModel)
        {
            Console.WriteLine();
            Console.WriteLine($"SORRY, THAT'S {lossViewModel.MaxGuesses} GUESSES");
            Console.WriteLine($"THE HURKLE IS AT {lossViewModel.HurkleLocation.X},{lossViewModel.HurkleLocation.Y}");
        }

        public void ShowVictory(VictoryViewModel victoryViewModel)
        {
            Console.WriteLine();
            Console.WriteLine($"YOU FOUND HIM IN {victoryViewModel.CurrentGuessNumber} GUESSES!");
        }
    }
}

```

# `51_Hurkle/csharp/FailedGuessViewModel.cs`

这段代码定义了一个名为 "FailedGuessViewModel" 的类，它是一个封装类，实现了 "FailedGuessViewModel" 接口。这个类包含一个名为 "Direction" 的成员变量，它是一个 "CardinalDirection" 类型的成员变量，属于 "FailedGuessViewModel" 类。

具体来说，"FailedGuessViewModel" 类的作用是提供一种记录猜测结果的方式，当猜测结果与预期结果不一致时，记录下来。当猜测结果正确时，则显示猜测结果。

在这段代码中，我们定义了一个 "FailedGuessViewModel" 类，其中包含一个 "Direction" 属性。这个属性用于记录当前猜测的方向，例如，如果当前猜测向下，则 "Direction" 属性应该是一个 "Down" 的二进制值。当谜底被猜出来时，可以更新 "Direction" 属性的值，从而实现显示猜结果的目的。


```
namespace hurkle
{
    internal class FailedGuessViewModel
    {
        public CardinalDirection Direction { get; init; }
    }
}

```

# `51_Hurkle/csharp/GamePoint.cs`



这段代码定义了一个名为 "GamePoint" 的内部类，该类包含一个 "GamePoint" 类型的成员变量 "X", "Y"，以及一个 "GetDirectionTo" 方法。

"GetDirectionTo" 方法接收一个名为 "target" 的 "GamePoint" 类型的成员变量，并返回一个 "CardinalDirection" 类型的变量，表示从当前 "GamePoint" 类型的成员变量 "X" 出发，朝哪个方向可以到达 "target" 类型的成员变量 "GamePoint"。

具体来说，如果当前 "GamePoint" 类型的成员变量 "X" 等于 "target.X"，并且 "Y" 比 "target.Y" 大，那么函数将返回 "CardinalDirection.South" 表示为南方向。如果 "X" 大于 "target.X"，并且 "Y" 小于 "target.Y"，那么函数将返回 "CardinalDirection.North" 表示为北方向。如果 "X" 和 "Y" 都不相等，那么函数将返回 "CardinalDirection.None" 表示为没有方向。

如果 "X" 大于 "target.X"，并且 "Y" 大于 "target.Y"，那么函数将返回 "CardinalDirection.SouthWest" 表示为东南方向。如果 "X" 小于 "target.X"，并且 "Y" 大于 "target.Y"，那么函数将返回 "CardinalDirection.NorthWest" 表示为西北方向。如果 "X" 大于 "target.X"，并且 "Y" 小于 "target.Y"，那么函数将返回 "CardinalDirection.SouthEast" 表示为东南方向。


```
namespace hurkle
{
    internal class GamePoint
    {
        public int X {get;init;}
        public int Y {get;init;}

        public CardinalDirection GetDirectionTo(GamePoint target)
        {
            if(X == target.X)
            {
                if(Y > target.Y)
                {
                    return CardinalDirection.South;
                }
                else if(Y < target.Y)
                {
                    return CardinalDirection.North;
                }
                else
                {
                    return CardinalDirection.None;
                }
            }
            else if(X > target.X)
            {
                if(Y == target.Y)
                {
                    return CardinalDirection.West;
                }
                else if(Y > target.Y)
                {
                    return CardinalDirection.SouthWest;
                }
                else
                {
                    return CardinalDirection.NorthWest;
                }
            }
            else
            {
                if(Y == target.Y)
                {
                    return CardinalDirection.East;
                }
                else if(Y > target.Y)
                {
                    return CardinalDirection.SouthEast;
                }
                else{
                    return CardinalDirection.NorthEast;
                }
            }
        }
    }
}

```

# `51_Hurkle/csharp/GuessViewModel.cs`

这段代码定义了一个名为“GuessViewModel”的内部类，该类包含一个名为“CurrentGuessNumber”的公共整型成员变量。

具体来说，这个类表示一个猜数字游戏中的“当前猜测值”。在游戏玩家输入一个数字后，这个类将用这个数字作为猜测值，并在屏幕上显示出来。当玩家再次猜测时，程序将检查当前猜测值是否与之前输入的数字相同。如果当前猜测值与之前输入的数字相同，则认为玩家已经猜中了，否则继续猜测。

当玩家结束游戏时，这个类将保存玩家输入的最后一个猜测值，以便在游戏结束时比较两个猜测值，并输出获胜者。


```
namespace hurkle
{
    internal class GuessViewModel
    {
        public int CurrentGuessNumber {get;init;}
    }
}

```

# `51_Hurkle/csharp/HurkleGame.cs`



这段代码是一个用于玩 Hurkle 游戏的内部类。Hurkle 游戏是一种猜数字游戏，通常由一个玩家和一个随机生成的初始位置的 "0" 开始。每个玩家需要猜测一个位置的 "0" 值，直到他们猜中了为止。游戏的胜利和失败都是由游戏自己来决定的。

具体来说，这段代码包括以下几个方法：

1. `PlayGame` 方法：这个方法会根据玩家猜测的数字，生成一个 Hurkle 点（X 和 Y 坐标），并使用游戏提供的视角来判断玩家猜测的数字是否正确，如果正确就显示胜利，否则显示错误。
2. `GetGuess` 方法：这个方法会根据玩家猜测的数字，获取与其位置对应的猜测点，并返回一个 `GuessViewModel` 对象，用于显示玩家当前猜测的位置和方向。
3. `GetDirectionTo` 方法：这个方法会根据玩家当前猜测的位置，获取到位置 `hurklePoint` 所处的方向，并返回一个 `CardinalDirection` 对象，用于显示玩家猜测的方向。
4. `ShowVictory` 方法：这个方法会根据玩家猜测的数字，显示胜利，并设置 `CurrentGuessNumber` 为 `K`（参数），其中 `K` 是当前猜测的数字。
5. `ShowDirection` 方法：这个方法会根据玩家猜测的方向，显示错误信息，并设置 `Direction` 参数为 `Direction`（参数）。
6. `ShowLoss` 方法：这个方法会根据玩家猜测的数字，显示失败，并设置 `CurrentGuessNumber` 为 `MaxGuesses`，设置 `HurkleLocation` 为 `hurklePoint`（参数）。


```
using System;

namespace hurkle
{
    internal class HurkleGame
    {
        private readonly Random _random = new Random();
        private readonly IHurkleView _view;
        private readonly int guesses;
        private readonly int gridSize;

        public HurkleGame(int guesses, int gridSize, IHurkleView view)
        {
            _view = view;
            this.guesses = guesses;
            this.gridSize = gridSize;
        }

        public void PlayGame()
        {
            // BASIC program was generating a float between 0 and 1
            // then multiplying by the size of the grid to to a number
            // between 1 and 10. C# allows you to do that directly.
            var hurklePoint = new GamePoint{
                X = _random.Next(0, gridSize),
                Y = _random.Next(0, gridSize)
            };

            for(var K=1;K<=guesses;K++)
            {
                var guessPoint = _view.GetGuess(new GuessViewModel{CurrentGuessNumber = K});

                var direction = guessPoint.GetDirectionTo(hurklePoint);
                switch(direction)
                {
                    case CardinalDirection.None:
                        _view.ShowVictory(new VictoryViewModel{CurrentGuessNumber = K});
                        return;
                    default:
                        _view.ShowDirection(new FailedGuessViewModel{Direction = direction});
                        continue;
                }
            }

            _view.ShowLoss(new LossViewModel{MaxGuesses = guesses, HurkleLocation = hurklePoint } );
        }
    }
}

```

# `51_Hurkle/csharp/IHurkleView.cs`

这段代码定义了一个名为 "hurkle" 的namespace，其中包含一个名为 "IHurkleView" 的内部接口，以及四个内部函数，分别对应于 "GetGuess"、"ShowVictory"、"ShowDirection" 和 "ShowLoss" 功能。

具体来说，这段代码描述了一个 "GuessViewModel" 和 "VictoryViewModel" 两个模型类，以及它们所继承的 "FailedGuessViewModel" 和 "LossViewModel" 类。这些模型类实现了 "IHurkleView" 接口，并且可以通过 "IHurkleView.GetGuess"、"IHurkleView.ShowVictory"、"IHurkleView.ShowDirection" 和 "IHurkleView.ShowLoss" 函数来获取猜测、胜利、方向和失败的信息。

因此，这段代码定义了一个用于处理玩家猜测的 API，可以输出 "GuessViewModel" 和 "VictoryViewModel" 类，以及 "FailedGuessViewModel" 和 "LossViewModel" 类中实现的 "GetGuess"、"ShowVictory"、"ShowDirection" 和 "ShowLoss" 函数。


```
namespace hurkle
{
    internal interface IHurkleView
    {
        GamePoint GetGuess(GuessViewModel guessViewModel);
        void ShowVictory(VictoryViewModel victoryViewModel);
        void ShowDirection(FailedGuessViewModel failedGuessViewModel);
        void ShowLoss(LossViewModel lossViewModel);
    }
}

```

# `51_Hurkle/csharp/LossViewModel.cs`

这段代码定义了一个名为 "LossViewModel" 的内部类 LossViewModel，它具有两个成员变量：MaxGuesses 和 HurkleLocation。

MaxGuesses 是一个整型变量，它用于存储玩家在游戏中最多能猜中多少次的答案。

HurkleLocation 是一个名为 "GamePoint" 的内部类，它存储了玩家当前所处的游戏位置，包括其行、列和颜色等信息。

LossViewModel 类的作用是为了解决一个猜测游戏的问题。它存储了玩家在游戏中最多能猜中多少次的答案，以及他们当前所处的游戏位置。这些信息可以帮助游戏开发者进行游戏逻辑的决策，例如决定是否允许玩家继续猜测、如何评分玩家的猜测结果等等。


```
namespace hurkle
{
    internal class LossViewModel
    {
        public int MaxGuesses { get; init; }
        public GamePoint HurkleLocation { get; init; }
    }
}

```

# `51_Hurkle/csharp/Program.cs`

The Hura许可证游戏（Hurky's Gridpoint）是一个基于网格（grid）的益智游戏。在这个游戏中，玩家需要在给定的网格上找到隐藏的宝藏（Hurkle）。宝藏的位置是一个整数，用逗号分隔。玩家需要猜测这个整数，同时游戏也会告诉玩家在宝藏的位置，让玩家可以尝试去寻找宝藏。

这个程序的主要思路是先创建一个Hurkle游戏对象，然后让玩家在游戏中进行猜测。当玩家猜测正确时，游戏会告诉玩家猜测的宝藏大约在哪个方向，然后让玩家继续猜测。如果玩家猜错，游戏会提示玩家再猜一次，或者直接告诉玩家正确的答案。

在这个程序中，使用了一个ConsoleHurkleView类来输出游戏信息，包括游戏当前状态、宝藏位置等信息。此外，程序还使用了string格式ting来输出游戏信息，通过一个'$'符号进行格式化输出。


```
﻿using System;

namespace hurkle
{
    class Program
    {
        static void Main(string[] args)
        {
            /*
            Original source transscription
            10 PRINT TAB(33);"HURKLE"
            20 PRINT TAB(15);"CREATIVE COMPUTING NORRISTOWN, NEW JERSEY"
            30 PRINT;PRINT;PRINT
            */
            Console.WriteLine(new string(' ', 33) + @"HURKLE");
            Console.WriteLine(new string(' ', 15) + @"CREATIVE COMPUTING NORRISTOWN, NEW JERSEY");
            /*
            110 N=5
            120 G=10
            */
            var N=5;
            var G=10;
            /*
            210 PRINT
            220 PRINT "A HURKLE IS HIDING ON A";G;"BY";G;"GRID. HOMEBASE"
            230 PRINT "ON THE GRID IS POINT 0,0 AND ANY GRIDPOINT IS A"
            240 PRINT "PAIR OF WHOLE NUMBERS SEPERATED BY A COMMA. TRY TO"
            250 PRINT "GUESS THE HURKLE'S GRIDPOINT. YOU GET";N;"TRIES."
            260 PRINT "AFTER EACH TRY, I WILL TELL YOU THE APPROXIMATE"
            270 PRINT "DIRECTION TO GO TO LOOK FOR THE HURKLE."
            280 PRINT
            */
            // Using string formatting via the '$' string
            Console.WriteLine();
            Console.WriteLine($"A HURKLE IS HIDING ON A {G} BY {G} GRID. HOMEBASE");
            Console.WriteLine(@"ON THE GRID IS POINT 0,0 AND ANY GRIDPOINT IS A");
            Console.WriteLine(@"PAIR OF WHOLE NUMBERS SEPERATED BY A COMMA. TRY TO");
            Console.WriteLine($"GUESS THE HURKLE'S GRIDPOINT. YOU GET {N} TRIES.");
            Console.WriteLine(@"AFTER EACH TRY, I WILL TELL YOU THE APPROXIMATE");
            Console.WriteLine(@"DIRECTION TO GO TO LOOK FOR THE HURKLE.");
            Console.WriteLine();

            var view = new ConsoleHurkleView();
            var hurkle = new HurkleGame(N,G, view);
            while(true)
            {
                hurkle.PlayGame();

                Console.WriteLine("PLAY AGAIN? (Y)ES/(N)O");
                var playAgainResponse = Console.ReadLine();
                if(playAgainResponse.Trim().StartsWith("y", StringComparison.InvariantCultureIgnoreCase))
                {
                    Console.WriteLine();
                    Console.WriteLine("LET'S PLAY AGAIN. HURKLE IS HIDING");
                    Console.WriteLine();
                }else{
                    Console.WriteLine("THANKS FOR PLAYING!");
                    break;
                }

            }
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)

This is demonstrating seperating the user interface from the application logic through the
use of the View/ViewModel/Controller pattern.

It also makes an effort to be relatively immutable.


# `51_Hurkle/csharp/VictoryViewModel.cs`

这段代码定义了一个名为`VictoryViewModel`的内部类，它有一个名为`CurrentGuessNumber`的公共整型成员变量。

这个`VictoryViewModel`类的作用是代表一个游戏的胜利视图模型。`CurrentGuessNumber`成员变量表示玩家当前猜测的数字，它用于记录玩家在游戏中猜测的数字，直到玩家猜中正确的数字，这个数字将增加。

在游戏过程中，玩家可以猜数字，每当玩家猜中正确的数字，`CurrentGuessNumber`的值将增加，并且游戏将继续进行直到玩家猜不中为止。当玩家猜不中数字时，游戏将停止并显示错误消息。


```
namespace hurkle
{
    internal class VictoryViewModel
    {
        public int CurrentGuessNumber {get; init;}
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `51_Hurkle/java/src/Hurkle.java`

This is a Java class that appears to be part of a video game. It is used to create a grid of numbers, where the numbers are displayed on the screen and the player is prompted to guess the h urkles gridpoint. The class has several methods including randomNumber, displayTextAndGetInput, and getDelimitedValue. The displayTextAndGetInput method displays a message on the screen and accepts input from the keyboard. The getDelimitedValue method takes a string and a position (count 0) and returns the int representation of the value.


```
import java.util.Scanner;

/**
 * Game of Hurkle
 * <p>
 * Based on the Basic game of Hurkle here
 * https://github.com/coding-horror/basic-computer-games/blob/main/51%20Hurkle/hurkle.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Hurkle {

    public static final int GRID_SIZE = 10;
    public static final int MAX_GUESSES = 5;

    private enum GAME_STATE {
        STARTING,
        START_GAME,
        GUESSING,
        PLAY_AGAIN,
        GAME_OVER
    }

    private GAME_STATE gameState;

    // Used for keyboard input
    private final Scanner kbScanner;

    private int guesses;

    // hurkle position
    private int hurkleXPos;
    private int hurkleYPos;

    // player guess
    private int playerGuessXPos;
    private int playerGuessYPos;

    public Hurkle() {

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

                // Show an introduction the first time the game is played.
                case STARTING:
                    intro();
                    gameState = GAME_STATE.START_GAME;
                    break;

                // Start the game, set the number of players, names and round
                case START_GAME:

                    hurkleXPos = randomNumber();
                    hurkleYPos = randomNumber();

                    guesses = 1;
                    gameState = GAME_STATE.GUESSING;

                    break;

                // Guess an x,y position of the hurkle
                case GUESSING:
                    String guess = displayTextAndGetInput("GUESS #" + guesses + "? ");
                    playerGuessXPos = getDelimitedValue(guess, 0);
                    playerGuessYPos = getDelimitedValue(guess, 1);
                    if (foundHurkle()) {
                        gameState = GAME_STATE.PLAY_AGAIN;
                    } else {
                        showDirectionOfHurkle();
                        guesses++;
                        if (guesses > MAX_GUESSES) {
                            System.out.println("SORRY, THAT'S "
                                    + MAX_GUESSES + " GUESSES.");
                            System.out.println("THE HURKLE IS AT "
                                    + hurkleXPos + "," + hurkleYPos);
                            System.out.println();
                            gameState = GAME_STATE.PLAY_AGAIN;
                        }
                    }

                    break;

                case PLAY_AGAIN:
                    System.out.println("LET'S PLAY AGAIN, HURKLE IS HIDING.");
                    System.out.println();
                    gameState = GAME_STATE.START_GAME;
                    break;
            }
            // Effectively an endless loop because the game never quits as per
            // the original basic code.
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    private void showDirectionOfHurkle() {
        System.out.print("GO ");
        if (playerGuessYPos == hurkleYPos) {
            // don't print North or South because the player has chosen the
            // same y grid pos as the hurkle
        } else if (playerGuessYPos < hurkleYPos) {
            System.out.print("NORTH");
        } else if (playerGuessYPos > hurkleYPos) {
            System.out.print("SOUTH");
        }

        if (playerGuessXPos == hurkleXPos) {
            // don't print East or West because the player has chosen the
            // same x grid pos as the hurkle
        } else if (playerGuessXPos < hurkleXPos) {
            System.out.print("EAST");
        } else if (playerGuessXPos > hurkleXPos) {
            System.out.print("WEST");
        }
        System.out.println();
    }

    private boolean foundHurkle() {
        if ((playerGuessXPos - hurkleXPos)
                - (playerGuessYPos - hurkleYPos) == 0) {
            System.out.println("YOU FOUND HIM IN " + guesses + " GUESSES.");
            return true;
        }

        return false;
    }

    /**
     * Display info about the game
     */
    private void intro() {
        System.out.println("HURKLE");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("A HURKLE IS HIDING ON A " + GRID_SIZE + " BY "
                + GRID_SIZE + " GRID. HOMEBASE");
        System.out.println("ON THE GRID IS POINT 0,0 IN THE SOUTHWEST CORNER,");
        System.out.println("AND ANY POINT ON THE GRID IS DESIGNATED BY A");
        System.out.println("PAIR OF WHOLE NUMBERS SEPERATED BY A COMMA. THE FIRST");
        System.out.println("NUMBER IS THE HORIZONTAL POSITION AND THE SECOND NUMBER");
        System.out.println("IS THE VERTICAL POSITION. YOU MUST TRY TO");
        System.out.println("GUESS THE HURKLE'S GRIDPOINT. YOU GET "
                + MAX_GUESSES + " TRIES.");
        System.out.println("AFTER EACH TRY, I WILL TELL YOU THE APPROXIMATE");
        System.out.println("DIRECTION TO GO TO LOOK FOR THE HURKLE.");
    }

    /**
     * Generate random number
     * Used to create one part of an x,y grid position
     *
     * @return random number
     */
    private int randomNumber() {
        return (int) (Math.random()
                * (GRID_SIZE) + 1);
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
     * Accepts a string delimited by comma's and returns the pos'th delimited
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

# `51_Hurkle/java/src/HurkleGame.java`



该代码创建了一个名为 "HurkleGame" 的类，其中包含一个名为 "main" 的方法。

在 "main" 方法中，使用关键字 "new" 创建了一个名为 "hurkle" 的对象，类型为 "Hurkle"。

接着，使用 "hurkle.play()" 方法将 "Hurkle" 对象开始演奏。由于 "Hurkle" 对象中包含一个 "play" 方法，因此它将使用该方法开始演奏。

此代码的作用是创建一个 "Hurkle" 对象并开始演奏它。


```
public class HurkleGame {

    public static void main(String[] args) {
        Hurkle hurkle = new Hurkle();
        hurkle.play();
    }
}

```

# `51_Hurkle/javascript/hurkle.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是在页面上输出一个字符串，由一个 JavaScript 函数转换而来。这个函数将会在页面上创建一个输入框，要求用户输入一个字符串，然后将输入的字符串通过 `document.getElementById` 获取到的一个元素（这里指的是 "output" 元素）追加到页面上。

`input` 函数的作用是获取用户输入的字符串，并将其存储在变量 `input_str` 中。该函数会首先创建一个输入框元素，要求用户输入一个字符串，然后返回该字符串。函数会在输入框元素上添加事件监听器，当用户按下回车键时，函数会将用户输入的字符串打印到页面上，并返回该字符串。然后函数会继续等待用户输入，以便在后续获取用户输入。


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

The HURKLE game is a guessing game where the player is trying to find a gridpoint by giving hints. The gridpoint is a designated point on a grid, and the player is given a number of attempts to guess it. After each guess, the player will receive feedback from the game, which will tell them the direction to go in to find the gridpoint. The game will continue until the player has guessed the location of the gridpoint or a certain number of attempts have been made.



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
    print(tab(33) + "HURKLE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    n = 5;
    g = 10;
    print("\n");
    print("A HURKLE IS HIDING ON A " + g + " BY " + g + " GRID. HOMEBASE\n");
    print("ON THE GRID IS POINT 0,0 IN THE SOUTHWEST CORNER,\n");
    print("AND ANY POINT ON THE GRID IS DESIGNATED BY A\n");
    print("PAIR OF WHOLE NUMBERS SEPERATED BY A COMMA. THE FIRST\n");
    print("NUMBER IS THE HORIZONTAL POSITION AND THE SECOND NUMBER\n");
    print("IS THE VERTICAL POSITION. YOU MUST TRY TO\n");
    print("GUESS THE HURKLE'S GRIDPOINT. YOU GET " + n + " TRIES.\n");
    print("AFTER EACH TRY, I WILL TELL YOU THE APPROXIMATE\n");
    print("DIRECTION TO GO TO LOOK FOR THE HURKLE.\n");
    print("\n");
    while (1) {
        a = Math.floor(g * Math.random());
        b = Math.floor(g * Math.random());
        for (k = 1; k <= n; k++) {
            print("GUESS #" + k + " ");
            str = await input();
            x = parseInt(str);
            y = parseInt(str.substr(str.indexOf(",") + 1));
            if (x == a && y == b) {
                print("\n");
                print("YOU FOUND HIM IN " + k + " GUESSES!\n");
                break;
            }
            print("GO ");
            if (y < b) {
                print("NORTH");
            } else if (y > b) {
                print("SOUTH");
            }
            if (x < a) {
                print("EAST\n");
            } else {
                print("WEST\n");
            }
        }
        if (k > n) {
            print("\n");
            print("SORRY, THAT'S " + n + " GUESSES.\n");
            print("THE HURKLE IS AT " + a + "," + b + "\n");
        }
        print("\n");
        print("LET'S PLAY AGAIN, HURKLE IS HIDING.\n");
        print("\n");
    }
}

```

这道题目缺少上下文，无法得知代码的具体作用。一般来说，在没有任何上下文的情况下，程序的名称是 "main()"，而 main() 函数是许多程序的入口点。在 main() 函数中，程序会加载到内存中并开始执行。对于大多数程序，main() 函数的主要作用是加载到内存中的代码，并确保程序在内存中成功启动。


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


# `51_Hurkle/python/hurkle.py`

该代码是一个Python脚本，作用是打印出给定点A(x1, y1)和点B(x2, y2)之间的方向提示，以帮助用户找到Hurley树。

具体来说，该脚本使用Python标准库中的random模块生成随机数，并通过direction函数来打印出朝向信息。direction函数的参数包括点A(x1, y1)和点B(x2, y2)的坐标值，以及当前点A的朝向信息(东、南、西、北)。

在direction函数中，首先使用print函数打印出"GO "，然后根据点B的坐标值Y相对于点A的坐标值X的大小关系，将“NORTH”、“SOUTH”或“EAST”、“WEST”打印出来，以指示当前点A的朝向。最后，使用print函数打印一个空行，以使输出更易于阅读。


```
#!/usr/bin/env python3

"""Ported to Python by @iamtraction"""

from random import random


def direction(A, B, X, Y) -> None:
    """Print the direction hint for finding the hurkle."""

    print("GO ", end="")
    if Y < B:
        print("NORTH", end="")
    elif Y > B:
        print("SOUTH", end="")

    if X < A:
        print("EAST", end="")
    elif X > A:
        print("WEST", end="")

    print()


```

This is a Python program that simulates the game of H奴古。在游戏中，玩家需要猜测一个坐标为 (A,B) 的宝藏(隐藏在网格中的指定位置)，而游戏难度将随着游戏的进行而逐渐增加。如果玩家在一定次数内没有找到宝藏，游戏结束并显示 "YOU FOUND HIM IN NUMBERS" 消息。

程序的主要部分包括：

1. 打印游戏场景和地图，以及宝藏的位置和数量。
2. 猜测宝藏的位置，并尝试寻找。如果猜测的位置与宝藏的实际位置相同，游戏难度将减少。
3. 如果猜测的位置与宝藏的实际位置不同，程序将随机改变宝藏的位置，并重新猜测。
4. 循环尝试猜测的位置，直到玩家找到宝藏或达到游戏规定的次数为止。
5. 在每次猜测失败后，程序将告诉玩家他们猜测的宝藏的位置和方向。
6. 在游戏开始前，程序将询问玩家是否要尝试猜测，如果玩家选择尝试猜测，程序将随机选择一个难度级别。

程序中使用了一些辅助函数，例如 random() 函数用于生成随机数，int() 函数用于将字符串转换为整数。程序还使用了一定数量的 while 循环来控制玩家猜测宝藏的次数，并使用了一些控制台输出来显示游戏结果和玩家信息。


```
def main() -> None:
    print(" " * 33 + "HURKLE")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")

    print("\n\n\n")

    N = 5
    G = 10

    print()
    print("A HURKLE IS HIDING ON A", G, "BY", G, "GRID. HOMEBASE")
    print("ON THE GRID IS POINT 0,0 IN THE SOUTHWEST CORNER,")
    print("AND ANY POINT ON THE GRID IS DESIGNATED BY A")
    print("PAIR OF WHOLE NUMBERS SEPERATED BY A COMMA. THE FIRST")
    print("NUMBER IS THE HORIZONTAL POSITION AND THE SECOND NUMBER")
    print("IS THE VERTICAL POSITION. YOU MUST TRY TO")
    print("GUESS THE HURKLE'S GRIDPOINT. YOU GET", N, "TRIES.")
    print("AFTER EACH TRY, I WILL TELL YOU THE APPROXIMATE")
    print("DIRECTION TO GO TO LOOK FOR THE HURKLE.")
    print()

    while True:
        A = int(G * random())
        B = int(G * random())

        for k in range(0, N):
            print("\nGUESS #" + str(k))

            # read coordinates in `X, Y` format, split the string
            # at `,`, and then parse the coordinates to `int` and
            # store them in `X` and `Y` respectively.
            [X, Y] = [int(c) for c in input("X,Y? ").split(",")]

            if abs(X - A) + abs(Y - B) == 0:
                print("\nYOU FOUND HIM IN", k + 1, "GUESSES!")
                break
            else:
                direction(A, B, X, Y)
                continue

        print("\n\nLET'S PLAY AGAIN, HURKLE IS HIDING.\n")


```

这段代码是一个条件判断语句，它的作用是在程序运行时检查当前操作系统的名称是否为 "__main__"。如果是 "__main__"，那么程序会执行 main() 函数；如果不是 "__main__"，那么程序将跳过条件判断并直接执行默认的出口（通常是 None）。

换句话说，这段代码会检查当前程序是否被当作主程序运行，如果是，那么程序会执行 main() 函数，否则会直接跳过条件判断并执行默认操作。


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


### Kinema

This program tests your fundamental knowledge of kinematics. It presents a simple problem: a ball is thrown straight up into the air at some random velocity. You then must answer three questions about the flight of the ball:
1. How high will it go?
2. How long until it returns to earth?
3. What will be its velocity after a random number of seconds?

The computer evaluates your performance; within 15% of the correct answer is considered close enough. After each run, the computer gives you another problem until you interrupt it.

KINEMA was shorted from the original Huntington Computer Project Program, KINERV, by Richard Pav of Patchogue High School, Patchogue, New York.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=95)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=110)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `52_Kinema/java/src/Kinema.java`

This program appears to simulate a Turing machine, where the player is asked to guess a series of bids. The program uses the `simulateTabs` method to simulate the space required for each bid, based on the number of spaces specified. The `calculate` method is used to determine if the player's guess is correct. If the player answers incorrectly, the program prints a message and tries again. If the player answers correctly, the program prints a message and accepts the player's input again.

The program also uses the `displayTextAndGetNumber` method to display a message to the screen and the `displayTextAndGetInput` method to accept the player's input. The `displayTextAndGetInput` method uses a `kbScanner` object to read the player's input.

Overall, the program simulates a Turing machine, where the player is asked to guess a series of bids. The player must use the `simulateTabs` method to simulate the space required for each bid, based on the number of spaces specified. The `calculate` method is used to determine if the player's guess is correct. If the player answers incorrectly, the program prints a message and tries again. If the player answers correctly, the program prints a message and accepts the player's input again.


```
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Kinema
 * <p>
 * Based on the Basic game of Kinema here
 * https://github.com/coding-horror/basic-computer-games/blob/main/52%20Kinema/kinema.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Kinema {

    // Used for keyboard input
    private final Scanner kbScanner;

    private enum GAME_STATE {
        STARTUP,
        INIT,
        HOW_HIGH,
        SECONDS_TILL_IT_RETURNS,
        ITS_VELOCITY,
        RESULTS,
        GAME_OVER
    }

    // Current game state
    private GAME_STATE gameState;

    private int numberAnswersCorrect;

    // How many meters per second a ball is thrown
    private int velocity;

    public Kinema() {
        kbScanner = new Scanner(System.in);

        gameState = GAME_STATE.STARTUP;
    }

    /**
     * Main game loop
     */
    public void play() {

        double playerAnswer;
        double correctAnswer;
        do {
            switch (gameState) {

                case STARTUP:
                    intro();
                    gameState = GAME_STATE.INIT;
                    break;

                case INIT:
                    numberAnswersCorrect = 0;

                    // calculate a random velocity for the player to use in the calculations
                    velocity = 5 + (int) (35 * Math.random());
                    System.out.println("A BALL IS THROWN UPWARDS AT " + velocity + " METERS PER SECOND.");
                    gameState = GAME_STATE.HOW_HIGH;
                    break;

                case HOW_HIGH:

                    playerAnswer = displayTextAndGetNumber("HOW HIGH WILL IT GO (IN METERS)? ");

                    // Calculate the correct answer to how high it will go
                    correctAnswer = 0.05 * Math.pow(velocity, 2);
                    if (calculate(playerAnswer, correctAnswer)) {
                        numberAnswersCorrect++;
                    }
                    gameState = GAME_STATE.ITS_VELOCITY;
                    break;

                case ITS_VELOCITY:

                    playerAnswer = displayTextAndGetNumber("HOW LONG UNTIL IT RETURNS (IN SECONDS)? ");

                    // Calculate current Answer for how long until it returns to the ground in seconds
                    correctAnswer = (double) velocity / 5;
                    if (calculate(playerAnswer, correctAnswer)) {
                        numberAnswersCorrect++;
                    }
                    gameState = GAME_STATE.SECONDS_TILL_IT_RETURNS;
                    break;

                case SECONDS_TILL_IT_RETURNS:

                    // Calculate random number of seconds for 3rd question
                    double seconds = 1 + (Math.random() * (2 * velocity)) / 10;

                    // Round to one decimal place.
                    double scale = Math.pow(10, 1);
                    seconds = Math.round(seconds * scale) / scale;

                    playerAnswer = displayTextAndGetNumber("WHAT WILL ITS VELOCITY BE AFTER " + seconds + " SECONDS? ");

                    // Calculate the velocity after the given number of seconds
                    correctAnswer = velocity - (10 * seconds);
                    if (calculate(playerAnswer, correctAnswer)) {
                        numberAnswersCorrect++;
                    }
                    gameState = GAME_STATE.RESULTS;
                    break;

                case RESULTS:
                    System.out.println(numberAnswersCorrect + " RIGHT OUT OF 3");
                    if (numberAnswersCorrect > 1) {
                        System.out.println(" NOT BAD.");
                    }
                    gameState = GAME_STATE.STARTUP;
                    break;
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    private void intro() {
        System.out.println(simulateTabs(33) + "KINEMA");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
    }

    private boolean calculate(double playerAnswer, double correctAnswer) {

        boolean gotItRight = false;

        if (Math.abs((playerAnswer - correctAnswer) / correctAnswer) < 0.15) {
            System.out.println("CLOSE ENOUGH");
            gotItRight = true;
        } else {
            System.out.println("NOT EVEN CLOSE");
        }
        System.out.println("CORRECT ANSWER IS " + correctAnswer);
        System.out.println();

        return gotItRight;
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     * Converts input to a Double
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private double displayTextAndGetNumber(String text) {
        return Double.parseDouble(displayTextAndGetInput(text));
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

# `52_Kinema/java/src/KinemaGame.java`

这段代码的作用是创建一个名为 "Kinema" 的游戏对象，并调用该对象的 "play" 方法来开始游戏。

在代码中，首先定义了一个名为 "KinemaGame" 的类。该类包含一个名为 "main" 的方法，该方法是程序的入口点。在 "main" 方法中，创建了一个名为 "kinema" 的对象，并将其赋值为创建 "Kinema" 对象时调用构造函数的参数。

接着，使用 "kinema.play()" 方法来开始游戏。在此处，"play" 方法可能指定了游戏的目标，例如屏幕、输出流或网络流等。具体实现可能会因游戏设计而异。


```
public class KinemaGame {
    public static void main(String[] args) {

        Kinema kinema = new Kinema();
        kinema.play();
    }
}

```

# `52_Kinema/javascript/kinema.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是将一个字符串打印到页面上，并将其存储在 `output` 元素中。这个字符串由另一个函数 `input` 生成，该函数会在页面上接收用户的输入，并将其存储在 `input_str` 变量中。

`input` 函数的作用是获取用户输入的字符串，并将其存储在 `input_str` 变量中。该函数会在页面上接收用户输入，然后将其存储在 `input_str` 变量中。函数会在输入框中添加一个 `INPUT` 元素，设置其 `type` 属性为 `text`，设置其 `length` 属性为 `50`，这样它就能接收长度为 50 的字符串输入。函数还会在 `input_str` 变量上添加一个 `keydown` 事件监听器，以便在用户按下键盘上的键时接收相应的输入并将其存储在 `input_str` 变量中。

总的来说，这两个函数一起工作，使得用户可以在页面上输入字符串，并将其打印到页面上。


```
// KINEMA
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

这两段代码定义了一个名为`tab`的函数和一个名为`evaluate_answer`的函数。

`tab`函数的作用是在字符串变量`str`中添加指定的空格，直到空格的数量减少到0为止。然后，函数返回`str`中包含的所有空格。

`evaluate_answer`函数的作用是接受一个字符串参数`str`和一个浮点数参数`a`，然后计算它们之间的差的绝对值与差的绝对值的比值。如果这个比值小于0.15（即95%的准确度），函数将输出“CLOSE ENOUGH.”；否则，函数将输出“NOT EVEN CLOSE....”。如果计算结果准确度小于0.15，函数将输出“NOT EVEN CLOSE....”。如果计算结果准确度大于等于0.15，函数将输出“CORRECT ANSWER IS ” followed by the input value `a`。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var q;

function evaluate_answer(str, a)
{
    g = parseFloat(str);
    if (Math.abs((g - a) / a) < 0.15) {
        print("CLOSE ENOUGH.\n");
        q++;
    } else {
        print("NOT EVEN CLOSE....\n");
    }
    print("CORRECT ANSWER IS " + a + "\n\n");
}

```

这段代码是一个用于Windows系统的多用户交互命令行程序。其主要目的是在用户面前显示一些有趣的文本，并接受他们的输入。程序会生成一些随机数，然后根据用户输入的单词计数，在计数达到2个时停止。

程序可以被分为以下几个部分：

1. 输出一些有趣的文本，包含tab()函数输出的字符串。
2. 在字符串中插入一些有趣的信息，包含一些自定义生成的字符。
3. 在循环中接受用户输入的单词，并使用evaluate_answer函数来计算它们在程序中的解释。
4. 在循环中根据用户输入的单词，尝试计算出它们在程序中的速度，并输出结果。
5. 在循环外输出一行消息，告诉用户输入的单词中有多少个字符。
6. 如果用户输入的单词数量小于2个，就停止循环并继续执行下一个命令。
7. 如果用户输入的单词数量大于等于2个，就停止循环并提示用户输入更多的单词。


```
// Main program
async function main()
{
    print(tab(33) + "KINEMA\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    while (1) {
        print("\n");
        print("\n");
        q = 0;
        v = 5 + Math.floor(35 * Math.random());
        print("A BALL IS THROWN UPWARDS AT " + v + " METERS PER SECOND.\n");
        print("\n");
        a = 0.5 * Math.pow(v, 2);
        print("HOW HIGH WILL IT GO (IN METERS)");
        str = await input();
        evaluate_answer(str, a);
        a = v / 5;
        print("HOW LONG UNTIL IT RETURNS (IN SECONDS)");
        str = await input();
        evaluate_answer(str, a);
        t = 1 + Math.floor(2 * v * Math.random()) / 10;
        a = v - 10 * t;
        print("WHAT WILL ITS VELOCITY BE AFTER " + t + " SECONDS");
        str = await input();
        evaluate_answer(str, a);
        print("\n");
        print(q + " RIGHT OUT OF 3.");
        if (q < 2)
            continue;
        print("  NOT BAD.\n");
    }
}

```

这道题的代码是 `main()`，这是一个程序的入口函数。在大部分程序中，`main()` 函数是必须的，因为它负责启动程序并处理任何异常终止程序的运行。

然而，对于小型简单的程序，有时候可能不需要 `main()` 函数。在这种情况下，你可以将所有功能都放入一个函数中，例如：

```
function main() {
 // 你的程序代码
}
```

但是，即使在这种情况下，`main()` 函数仍然是一个独立的函数，用于启动程序和处理任何异常终止程序的运行。


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


# `52_Kinema/python/kinema.py`

这段代码是一个用于模拟物理学的问卷，它涉及到重力、随机数和近似计算。它的主要目的是让用户回答一些物理学问题，并通过用户的回答来计算得分。

具体来说，这段代码的功能如下：

1. 引入Python标准库中的random模块，用于生成随机数。
2. 定义了一个名为g的变量，并将其设置为10，这意味着我们假设重力约为10牛顿。这个值在实际应用中可能会有所不同，但它是一个方便的起点。
3. 在函数中使用random.uniform()函数生成一个0到1之间的随机数，用作用户的回答。
4. 通过调用print()函数将得分打印出来。
5. 使用if语句检查用户是否已经回答了所有问题。如果用户已经回答了所有问题，那么程序会计算总分并打印结果。
6. 最后，使用continue()函数来跳过已完成的题目，并重新生成所有题目，以便用户继续答题。

总体来说，这段代码是一个简单的Python程序，旨在向用户展示物理学的基本概念，以及如何使用Python来编写实用的程序。


```
"""
KINEMA

A kinematics physics quiz.

Ported by Dave LeCompte
"""

import random

# We approximate gravity from 9.8 meters/second squared to 10, which
# is only off by about 2%. 10 is also a lot easier for people to use
# for mental math.

g = 10

```

这段代码是一个Python的函数，名为do_quiz，它进行了一次模拟测验，以评估学生回答问题的正确性。

do_quiz函数的作用是让玩家进行多次测验，然后统计出他们答对题目的数量。在这个特定的测验中，我们希望答案的准确度不超过预期的准确度百分比（设置为15%）。

具体来说，do_quiz函数会执行以下操作：

1. 随机生成一个向上抛出篮球的速度值v0，然后进行两次计算，第一次计算是求出篮球到达最高点的速度，第二次计算是计算出篮球在空中的时间。
2. 使用玩家输入的信息，包括高度和时间，计算出篮球的竖直上抛运动方程中的参数g和answer。
3. 根据时间间隔，使用公式v=v0-gt计算出竖直上抛运动的垂直速度，然后计算出篮球在空中的总时间。
4. 输出正确答案的数量，并判断是否超过2个，如果是，输出一个鼓励性的评价。

最后，do_quiz函数没有返回任何值，而是直接执行了上述操作。


```
# We only expect the student to get within this percentage of the
# correct answer. This isn't rocket science.

EXPECTED_ACCURACY_PERCENT = 15


def do_quiz() -> None:
    print()
    print()
    num_questions_correct = 0

    # pick random initial velocity
    v0 = random.randint(5, 40)
    print(f"A BALL IS THROWN UPWARDS AT {v0} METERS PER SECOND.")
    print()

    answer = v0**2 / (2 * g)
    num_questions_correct += ask_player("HOW HIGH WILL IT GO (IN METERS)?", answer)

    answer = 2 * v0 / g
    num_questions_correct += ask_player(
        "HOW LONG UNTIL IT RETURNS (IN SECONDS)?", answer
    )

    t = 1 + random.randint(0, 2 * v0) // g
    answer = v0 - g * t
    num_questions_correct += ask_player(
        f"WHAT WILL ITS VELOCITY BE AFTER {t} SECONDS?", answer
    )

    print()
    print(f"{num_questions_correct} right out of 3.")
    if num_questions_correct >= 2:
        print("  NOT BAD.")


```

这段代码定义了一个名为 `ask_player` 的函数，用于向玩家提出一个问题并接收玩家的回答。函数有两个参数：一个字符串类型的 `question` 参数和一个浮点数类型的 `answer` 参数。

函数首先打印问题，然后提示玩家输入答案。接着，函数计算玩家答案与预期答案之间的百分比误差，如果误差小于预设的准确率，函数打印 "CLOSE ENOUGH." 并返回得分为 1。否则，函数打印 "NOT EVEN CLOSE...." 并返回得分为 0。最后，函数打印出正确答案并输出结果。

该函数可以作为一个简单的工具，让玩家输入答案，然后根据答案的准确度评定得分。


```
def ask_player(question: str, answer) -> int:
    print(question)
    player_answer = float(input())

    accuracy_frac = EXPECTED_ACCURACY_PERCENT / 100.0
    if abs((player_answer - answer) / answer) < accuracy_frac:
        print("CLOSE ENOUGH.")
        score = 1
    else:
        print("NOT EVEN CLOSE....")
        score = 0
    print(f"CORRECT ANSWER IS {answer}")
    print()
    return score


```

这段代码是一个Python程序，名为“main”。程序的主要作用是输出一个带有33个空格的字符串，以及一个15个空格的字符串。然后，程序会输出一行“KINEMA”和一行“CREATIVE COMPUTING”紧接着两个空格，之后是两行空白。接着程序会进入一个无限循环，调用一个名为“do_quiz”的函数。

在程序外部（即在文件外部），如果运行这个Python文件，它将首先调用“do_quiz”函数。do_quiz函数的功能没有在代码中定义，因此在运行程序之前，我们无法知道它具体做了什么。


```
def main() -> None:
    print(" " * 33 + "KINEMA")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

    while True:
        do_quiz()


if __name__ == "__main__":
    main()

```