# BasicComputerGames源码解析 35

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Craps

This game simulates the game of craps played according to standard Nevada craps table rules. That is:
1. A 7 or 11 on the first roll wins
2. A 2, 3, or 12 on the first roll loses
3. Any other number rolled becomes your “point.”
    - You continue to roll, if you get your point, you win.
    - If you roll a 7, you lose and the dice change hands when this happens.

This version of craps was modified by Steve North of Creative Computing. It is based on an original which appeared one day on a computer at DEC.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=52)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=67)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

    15 LET R=0

`R` is a variable that tracks winnings and losings.  Unlike other games that
start out with a lump sum of cash to spend this game assumes the user has as
much money as they want and we only track how much they lost or won.

      21 LET T=1
      22 PRINT "PICK A NUMBER AND INPUT TO ROLL DICE";
      23 INPUT Z
      24 LET X=(RND(0))
      25 LET T =T+1
      26 IF T<=Z THEN 24

This block of code does nothing other than try to scramble the random number
generator. Random number generation is not random, they are generated from the
previous generated number. Because of the slow speed of these systems back then,
gaming random number generators was a concern, mostly for gameplay quality.
If you could know the "seed value" to the generator then you could effectively
know how to get the exact same dice rolls to happen and change your bet to
maximize your winnings and minimize your losses.

The first reason this is an example of bad coding practice is the user is asked
to input a number but no clue is given as to the use of this number. This number
has no bearing on the game and as we'll see only has bearing on the internal
implementation of somehow trying to get an un-game-able seed for the random number
generator (since all future random numbers generated are based off this seed value.)

The `RND(1)` command generates a number from a seed value that is always
the same, everytime, from when the machine is booted up (old C64 behavior). In
order to avoid the same dice rolls being generated, a special call to `RND(-TI)`
would initialize the random generator with something else. But RND(-TI) is not
a valid command on all systems. So `RND(0)`, which generates a random number
from the system clock is used. But technically this could be gamed because the
system clock was driven by the bootup time, there wasn't a BIOS battery on these
systems that kept an internal real time clock going even when the system was
turned off, unlike your regular PC. Therefore, in order to ensure as true
randomness as possible, insert human reaction time by asking for human input.

But a human could just be holding down the enter key on bootup and that would
just skip any kind of multi-millisecond variance assigned by a natural human
reaction time. So, paranoia being a great motivator, a number is asked of the
user to avoid just holding down the enter key which negates the timing variance
of a human reaction.

What comes next is a bit of nonsense. The block of code loops a counter, recalling
the `RND(0)` function (and thus reseeding it with the system clock value)
and then comparing the counter to the user's number input
in order to bail out of the loop. Because the `RND(0)` function is based off the
system clock and the loop of code has no branching other than the bailout
condition, the loop also takes a fixed amount of time to execute, thus making
repeated calls to `RND(0)` predictive and this scheming to get a better random
number is pointless. Furthermore, the loop is based on the number the user inputs
so a huge number like ten million causes a very noticable delay and leaves the
user wondering if the program has errored. The author could have simply called
`RND(0)` once and used a prompt that made more sense like asking for the users
name and then using that name in the game's replies.

It is advised that you use whatever your languages' random number generator
provides and simply skip trying to recreate this bit of nonsense including
the user input.

      27 PRINT"INPUT THE AMOUNT OF YOUR WAGER.";
      28 INPUT F
      30 PRINT "I WILL NOW THROW THE DICE"
      40 LET E=INT(7*RND(1))
      41 LET S=INT(7*RND(1))
      42 LET X=E+S
      .... a bit later ....
      60 IF X=1 THEN 40
      65 IF X=0 THEN 40

`F` is a variable that represents the users wager for this betting round.
`E` and `S` represent the two individual and random dice being rolled.
This code is actually wrong because it returns a value between 0 and 6.
`X` is the sum of these dice rolls. As you'll see though further down in the
code, if `X` is zero or one it re-rolls the dice to maintain a potential
outcome of the sum of two dice between 2 and 12. This skews the normal distribution
of dice values to favor lower numbers because it does not consider that `E`
could be zero and `S` could be 2 or higher. To show this skewing of values
you can run the `distribution.bas` program which creates a histogram of the
distribution of the bad dice throw code and proper dice throw code.

Here are the results:

      DISTRIBUTION OF DICE ROLLS WITH  INT(7*RND(1))  VS  INT(6*RND(1)+1)
      THE INT(7*RND(1)) DISTRIBUTION:
      2             3             4             5             6             7             8             9             10            11            12
      6483          8662          10772         13232         15254         13007         10746         8878          6486          4357          2123
      THE INT(6*RND(1)+1) DISTRIBUTION
      2             3             4             5             6             7             8             9             10            11            12
      2788          5466          8363          11072         13947         16656         13884         11149         8324          5561          2790
If the dice rolls are fair then we should see the largest occurrence be a 7 and
the smallest should be 2 and 12. Furthermore the occurrences should be
symetrical meaning there should be roughly the same amount of 2's as 12's, the
same amount of 3's as 11's, 4's as 10's and so on until you reach the middle, 7.
But notice in the skewed dice roll, 6 is the most rolled number not 7, and the
rest of the numbers are not symetrical, there are many more 2's than 12's.
So the lesson is test your code.

The proper way to model a dice throw, in almost every language is
    `INT(6*RND(1)+1)` or `INT(6*RND(1))+1`

SideNote: `X` was used already in the
previous code block discussed but its value was never used. This is another
poor coding practice: **Don't reuse variable names for different purposes.**

      50 IF X=7 THEN 180
      55 IF X=11 THEN 180
      60 IF X=1 THEN 40
      62 IF X=2 THEN 195
      65 IF X=0 THEN 40
      70 IF X=2 THEN 200
      80 IF X=3 THEN 200
      90 IF X=12 THEN 200
      125 IF X=5 THEN 220
      130 IF X =6 THEN 220
      140 IF X=8 THEN 220
      150 IF X=9 THEN 220
      160 IF X =10 THEN 220
      170 IF X=4 THEN 220

This bit of code determines the routing of where to go for payout, or loss.
Of course, line 60 and 65 are pointless as we've just shown and should be removed
as long as the correct dice algorithm is also changed.

      62 IF X=2 THEN 195
      ....
      70 IF X=2 THEN 200
The check for a 2 has already been made and the jump is done. Line 70 is
therefore redundant and can be left out. The purpose of line 62 is only to
print a special output, "SNAKE EYES!" which we'll see in the next block creates
duplicate code.

Lines 125-170 are also pointlessly checked because we know previous values have
been ruled out, only these last values must remain, and they are all going to
the same place, line 220. Line 125-170 could have simply been replaced with
`GOTO 220`



      180 PRINT X "- NATURAL....A WINNER!!!!"
      185 PRINT X"PAYS EVEN MONEY, YOU WIN"F"DOLLARS"
      190 GOTO 210
      195 PRINT X"- SNAKE EYES....YOU LOSE."
      196 PRINT "YOU LOSE"F "DOLLARS."
      197 LET F=0-F
      198 GOTO 210
      200 PRINT X " - CRAPS...YOU LOSE."
      205 PRINT "YOU LOSE"F"DOLLARS."
      206 LET F=0-F
      210 LET R= R+F
      211 GOTO 320

This bit of code manages instant wins or losses due to 7,11 or 2,3,12. As
mentioned previously, lines 196 and 197 are essentially the same as lines
205 and 206. A simpler code would be just to jump after printing the special
message of "SNAKE EYES!" to line 205.

Lines 197 and 206 just negate the wager by subtracting it from zero. Just saying
`F = -F` would have sufficed. Line 210 updates your running total of winnings
or losses with this bet.

      220 PRINT X "IS THE POINT. I WILL ROLL AGAIN"
      230 LET H=INT(7*RND(1))
      231 LET Q=INT(7*RND(1))
      232 LET O=H+Q
      240 IF O=1 THEN 230
      250 IF O=7 THEN 290
      255 IF O=0 THEN 230

This code sets the point, the number you must re-roll to win without rolling
a 7, the most probable number to roll. Except in this case again, it has the
same incorrect dice rolling code and therefore 6 is the most probable number
to roll. The concept of DRY (don't repeat yourself) is a coding practice which
encourages non-duplication of code because if there is an error in the code, it
can be fixed in one place and not multiple places like in this code. The scenario
might be that a programmer sees some wrong code, fixes it, but neglects to
consider that there might be duplicates of the same wrong code elsewhere.  If
you practice DRY then you never worry much about behaviors in your code diverging
due to duplicate code snippets.

      260 IF O=X THEN 310
      270 PRINT O " - NO POINT. I WILL ROLL AGAIN"
      280 GOTO 230
      290 PRINT O "- CRAPS. YOU LOSE."
      291 PRINT "YOU LOSE $"F
      292 F=0-F
      293 GOTO 210
      300 GOTO 320
      310 PRINT X"- A WINNER.........CONGRATS!!!!!!!!"
      311 PRINT X "AT 2 TO 1 ODDS PAYS YOU...LET ME SEE..."2*F"DOLLARS"
      312 LET F=2*F
      313 GOTO 210

This is the code to keep rolling until the point is made or a seven is rolled.
Again we see the negated `F` wager and lose message duplicated. This code could
have been reorganized using a subroutine, or in BASIC, the GOSUB command, but
in your language its most likely just known as a function or method. You can
do a `grep -r 'GOSUB'` from the root directory to see other BASIC programs in
this set that use GOSUB.

The rest of the code if fairly straight forward, replay the game or end with
a report of your winnings or losings.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `29_Craps/csharp/Craps/CrapsGame.cs`

Craps is a card game where two players take turns rolling two dice and trying to outscore their opponent by calling a point or a win. The game is won by either calling point or winning. The Craps game implemented in the code you provided is a simple example that uses two dice and has the properties of a craps game. It has a constructor that takes a ref to a UserInterface object and initializes the dice1 and dice2 as Dice objects. It has a Play method that roll the two dice, check the result of the roll and return the appropriate outcome.


```
﻿namespace Craps
{
    public enum Result
    {
        // It's not used in this program but it's often a good idea to include a "none"
        // value in an enum so that you can set an instance of the enum to "invalid" or
        // initialise it to "none of the valid values".
        noResult,
        naturalWin,
        snakeEyesLoss,
        naturalLoss,
        pointLoss,
        pointWin,
    };

    class CrapsGame
    {
        private readonly UserInterface ui;
        private Dice dice1 = new Dice();
        private Dice dice2 = new Dice();

        public CrapsGame(ref UserInterface ui)
        {
            this.ui = ui;
        }

        public Result Play(out int diceRoll)
        {
            diceRoll = dice1.Roll() + dice2.Roll();

            if (Win(diceRoll))
            {
                return Result.naturalWin;
            }
            else if (Lose(diceRoll))
            {
                return (diceRoll == 2) ? Result.snakeEyesLoss : Result.naturalLoss;
            }
            else
            {
                var point = diceRoll;
                ui.Point(point);

                while (true)
                {
                    var newRoll = dice1.Roll() + dice2.Roll();
                    if (newRoll == point)
                    {
                        diceRoll = newRoll;
                        return Result.pointWin;
                    }
                    else if (newRoll == 7)
                    {
                        diceRoll = newRoll;
                        return Result.pointLoss;
                    }

                    ui.NoPoint(newRoll);
                }
            }
        }

        private bool Lose(int diceRoll)
        {
            return diceRoll == 2 || diceRoll == 3 || diceRoll == 12;
        }

        private bool Win(int diceRoll)
        {
            return diceRoll == 7 || diceRoll == 11;
        }
    }
}

```

# `29_Craps/csharp/Craps/Dice.cs`

这段代码定义了一个名为Dice的类，其作用是创建一个随机的6面骰子。这个类有两个构造函数，一个是无参构造函数，用于创建一个骰子并设置其侧面数为6，另一个是有参构造函数，用于创建一个骰子并提供一个侧面数。骰子类有一个名为Roll的方法，用于生成一个1到侧面数+1之间的随机整数，并返回其上一次生成的结果。通过创建一个Dice实例，可以方便地使用其Roll方法生成随机数。


```
﻿using System;


namespace Craps
{
    public class Dice
    {
        private Random rand = new Random();
        public readonly int sides;

        public Dice()
        {
            sides = 6;
        }

        public Dice(int sides)
        {
            this.sides = sides;
        }

        public int Roll() => rand.Next(1, sides + 1);
    }
}

```

# `29_Craps/csharp/Craps/Program.cs`

这段代码是一个Craps游戏的程序，其中包含了以下主要步骤：

1. 导入System.Diagnostics命名空间；
2. 定义一个UserInterface类，用于与用户进行交互；
3. 定义一个CrapsGame类，用于处理游戏逻辑；
4. 在Main方法中创建一个UserInterface实例，一个CrapsGame实例，以及一个计数器变量winnings；
5. 循环让用户继续游戏，直到用户选择结束；
6. 在每次游戏结束时，根据游戏结果输出UI的信息，包括胜赔和奖金，并输出结果；
7. 在开始游戏前，先询问用户是否想要继续游戏；
8. 编写一个通过return来返回一个int类型的结果，用于在循环中使用；
9. 编写default类型的分支，用于在结果变更时输出错误消息，但不会对此进行任何操作；
10. 编写一个静态的invoke方法，用于调用CrapsGame的Play方法，以执行游戏。


```
﻿using System.Diagnostics;



namespace Craps
{
    class Program
    {
        static void Main(string[] args)
        {
            var ui = new UserInterface();
            var game = new CrapsGame(ref ui);
            int winnings = 0;

            ui.Intro();

            do
            {
	            var bet = ui.PlaceBet();
                var result = game.Play(out int diceRoll);

                switch (result)
                {
                    case Result.naturalWin:
                        winnings += bet;
                        break;

                    case Result.naturalLoss:
                    case Result.snakeEyesLoss:
                    case Result.pointLoss:
                        winnings -= bet;
                        break;

                    case Result.pointWin:
                        winnings += (2 * bet);
                        break;

                    // Include a default so that we will be warned if the values of the enum
                    // ever change and we forget to add code to handle the new value.
                    default:
                        Debug.Assert(false); // We should never get here.
                        break;
                }

                ui.ShowResult(result, diceRoll, bet);
            } while (ui.PlayAgain(winnings));

            ui.GoodBye(winnings);
        }
    }
}

```

# `29_Craps/csharp/Craps/UserInterface.cs`

This is a class written in C# that simulates a game of snake and新增s a random element to it. The class has a variable of a Snake game object and a Random variable. It starts by randomly choosing one of the four outcomes (natural win, natural loss, snake eyes loss, or point loss) for the game. If the player loses, it will display the amount they lost and the score. If the player wins, it will display the score and the amount they won. If the game is a draw, it will display "CRAPS...YOU LOSE" or "A WINNER.........CONGRATS AT 2 TO 1 ODDS PAYS YOU...LET ME SEE...". It also includes a default outcome for when the values of the enum change and the player forgets to add code to handle the new value.


```
using System;
using System.Diagnostics;



namespace Craps
{
    public class UserInterface
	{
        public void Intro()
        {
            Console.WriteLine("                                 CRAPS");
            Console.WriteLine("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n");
            Console.WriteLine("2,3,12 ARE LOSERS; 4,5,6,8,9,10 ARE POINTS; 7,11 ARE NATURAL WINNERS.");

            // In the original game a random number would be generated and then thrown away for as many
            // times as the number the user entered. This is presumably something to do with ensuring
            // that different random numbers will be generated each time the program is run.
            //
            // This is not necessary in C#; the random number generator uses the current time as a seed
            // so the results will always be different every time it is run.
            //
            // So that the game exactly matches the original game we ask the question but then ignore
            // the answer.
            Console.Write("PICK A NUMBER AND INPUT TO ROLL DICE ");
            GetInt();
        }

        public int PlaceBet()
        {
            Console.Write("INPUT THE AMOUNT OF YOUR WAGER. ");
            int n = GetInt();
            Console.WriteLine("I WILL NOW THROW THE DICE");

            return n;
        }

        public bool PlayAgain(int winnings)
        {
            // Goodness knows why we have to enter 5 to play
            // again but that's what the original game asked.
            Console.Write("IF YOU WANT TO PLAY AGAIN PRINT 5 IF NOT PRINT 2 ");

            bool playAgain = (GetInt() == 5);

            if (winnings < 0)
            {
                Console.WriteLine($"YOU ARE NOW UNDER ${-winnings}");
            }
            else if (winnings > 0)
            {
                Console.WriteLine($"YOU ARE NOW OVER ${winnings}");
            }
            else
            {
                Console.WriteLine($"YOU ARE NOW EVEN AT ${winnings}");
            }

            return playAgain;
        }

        public void GoodBye(int winnings)
        {
            if (winnings < 0)
            {
                Console.WriteLine("TOO BAD, YOU ARE IN THE HOLE. COME AGAIN.");
            }
            else if (winnings > 0)
            {
                Console.WriteLine("CONGRATULATIONS---YOU CAME OUT A WINNER. COME AGAIN!");
            }
            else
            {
                Console.WriteLine("CONGRATULATIONS---YOU CAME OUT EVEN, NOT BAD FOR AN AMATEUR");
            }
        }

        public void NoPoint(int diceRoll)
        {
            Console.WriteLine($"{diceRoll} - NO POINT. I WILL ROLL AGAIN ");
        }

        public void Point(int point)
        {
            Console.WriteLine($"{point} IS THE POINT. I WILL ROLL AGAIN");
        }

        public void ShowResult(Result result, int diceRoll, int bet)
        {
            switch (result)
            {
                case Result.naturalWin:
                    Console.WriteLine($"{diceRoll} - NATURAL....A WINNER!!!!");
                    Console.WriteLine($"{diceRoll} PAYS EVEN MONEY, YOU WIN {bet} DOLLARS");
                    break;

                case Result.naturalLoss:
                    Console.WriteLine($"{diceRoll} - CRAPS...YOU LOSE.");
                    Console.WriteLine($"YOU LOSE {bet} DOLLARS.");
                    break;

                case Result.snakeEyesLoss:
                    Console.WriteLine($"{diceRoll} - SNAKE EYES....YOU LOSE.");
                    Console.WriteLine($"YOU LOSE {bet} DOLLARS.");
                    break;

                case Result.pointLoss:
                    Console.WriteLine($"{diceRoll} - CRAPS. YOU LOSE.");
                    Console.WriteLine($"YOU LOSE ${bet}");
                    break;

                case Result.pointWin:
                    Console.WriteLine($"{diceRoll} - A WINNER.........CONGRATS!!!!!!!!");
                    Console.WriteLine($"AT 2 TO 1 ODDS PAYS YOU...LET ME SEE... {2 * bet} DOLLARS");
                    break;

                // Include a default so that we will be warned if the values of the enum
                // ever change and we forget to add code to handle the new value.
                default:
                    Debug.Assert(false); // We should never get here.
                    break;
            }
        }

        private int GetInt()
        {
            while (true)
            {
	            string input = Console.ReadLine();
                if (int.TryParse(input, out int n))
                {
                    return n;
                }
                else
                {
                    Console.Write("ENTER AN INTEGER ");
                }
            }
        }
    }
}

```

# `29_Craps/csharp/CrapsTester/CrapsTests.cs`

This appears to be a code snippet written in C# that simulates rolling a die. It initializes a variable to keep track of the number of ones, twos, threes, and sixes rolled, as well as the number of errors.

It then uses a nested loop to simulate rolling the die 600000 times. For each type of die, it keeps track of the number of corresponding dice rolls. It then checks whether the number of rolls is within the specified minimum and maximum roll range, and if not, it increments the error counter.

Finally, it assigns the value 10 to the variable numErrors, which is assumed to be the maximum number of errors that can occur.

It is important to note that this code snippet simulates rolling a die, which is a game with a fixed outcome. The outcome of the game is predetermined, and the game is not meant to be a simulation of real-world decision-making scenarios.



```
using Craps;
using Microsoft.VisualStudio.TestTools.UnitTesting;



namespace CrapsTester
{
    [TestClass]
    public class DiceTests
    {
        [TestMethod]
        public void SixSidedDiceReturnsValidRolls()
        {
            var dice = new Dice();
            for (int i = 0; i < 100000; i++)
            {
                var roll = dice.Roll();
                Assert.IsTrue(roll >= 1 && roll <= dice.sides);
            }
        }

        [TestMethod]
        public void TwentySidedDiceReturnsValidRolls()
        {
            var dice = new Dice(20);
            for (int i = 0; i < 100000; i++)
            {
                var roll = dice.Roll();
                Assert.IsTrue(roll >= 1 && roll <= dice.sides);
            }
        }

        [TestMethod]
        public void DiceRollsAreRandom()
        {
            // Roll 600,000 dice and count how many rolls there are for each side.

            var dice = new Dice();

            int numOnes = 0;
            int numTwos = 0;
            int numThrees = 0;
            int numFours = 0;
            int numFives = 0;
            int numSixes = 0;
            int numErrors = 0;

            for (int i = 0; i < 600000; i++)
            {
                switch (dice.Roll())
                {
                    case 1:
                        numOnes++;
                        break;

                    case 2:
                        numTwos++;
                        break;

                    case 3:
                        numThrees++;
                        break;

                    case 4:
                        numFours++;
                        break;

                    case 5:
                        numFives++;
                        break;

                    case 6:
                        numSixes++;
                        break;

                    default:
                        numErrors++;
                        break;
                }
            }

            // We'll assume that a variation of 10% in rolls for the different numbers is random enough.
            // Perfectly random rolling would produce 100000 rolls per side, +/- 5% of this gives the
            // range 90000..110000.
            const int minRolls = 95000;
            const int maxRolls = 105000;
            Assert.IsTrue(numOnes >= minRolls && numOnes <= maxRolls);
            Assert.IsTrue(numTwos >= minRolls && numTwos <= maxRolls);
            Assert.IsTrue(numThrees >= minRolls && numThrees <= maxRolls);
            Assert.IsTrue(numFours >= minRolls && numFours <= maxRolls);
            Assert.IsTrue(numFives >= minRolls && numFives <= maxRolls);
            Assert.IsTrue(numSixes >= minRolls && numSixes <= maxRolls);
            Assert.AreEqual(numErrors, 0);
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `29_Craps/java/src/Craps.java`

This looks like a Java class called `Winnings`. It appears to be a simulator for a video game that awards players based on how well they do in the game. The player is prompted to enter their name and they are given a choice of whether they want to play again or not. If they choose to play again, they are asked to enter a bet and they are told how much they won. If they win, they are given a message, if they lose or they quit, they are given a message. It also appears that there is a `getWager` method for the player to enter their bet and a `getInput` method for the player to enter their name.

It's important to note that this is just a basic example of how the code might look like and it may not be complete or fully functional.


```
import java.util.Random;
import java.util.Scanner;

/**
 *  Port of Craps from BASIC to Java 17.
 */
public class Craps {
  public static final Random random = new Random();

  public static void main(String[] args) {
    System.out.println("""
                                                            CRAPS
                                          CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY


                           2,3,12 ARE LOSERS; 4,5,6,8,9,10 ARE POINTS; 7,11 ARE NATURAL WINNERS.
                           """);
    double winnings = 0.0;
    do {
      winnings = playCraps(winnings);
    } while (stillInterested(winnings));
    winningsReport(winnings);
  }

  public static double playCraps(double winnings) {
    double wager = getWager();
    System.out.println("I WILL NOW THROW THE DICE");
    int roll = rollDice();
    double payout = switch (roll) {
      case 7, 11 -> naturalWin(roll, wager);
      case 2, 3, 12 -> lose(roll, wager);
      default -> setPoint(roll, wager);
    };
    return winnings + payout;
  }

  public static int rollDice() {
    return random.nextInt(1, 7) + random.nextInt(1, 7);
  }

  private static double setPoint(int point, double wager) {
    System.out.printf("%1$ d IS THE POINT. I WILL ROLL AGAIN%n",point);
    return makePoint(point, wager);
  }

  private static double makePoint(int point, double wager) {
    int roll = rollDice();
    if (roll == 7)
      return lose(roll, wager);
    if (roll == point)
      return win(roll, wager);
    System.out.printf("%1$ d - NO POINT. I WILL ROLL AGAIN%n", roll);
    return makePoint(point, wager);  // recursive
  }

  private static double win(int roll, double wager) {
    double payout = 2 * wager;
    System.out.printf("%1$ d - A WINNER.........CONGRATS!!!!!!!!%n", roll);
    System.out.printf("%1$ d AT 2 TO 1 ODDS PAYS YOU...LET ME SEE...$%2$3.2f%n",
                      roll, payout);
    return payout;
  }

  private static double lose(int roll, double wager) {
    String msg = roll == 2 ? "SNAKE EYES.":"CRAPS";
    System.out.printf("%1$ d - %2$s...YOU LOSE.%n", roll, msg);
    System.out.printf("YOU LOSE $%3.2f%n", wager);
    return -wager;
  }

  public static double naturalWin(int roll, double wager) {
    System.out.printf("%1$ d - NATURAL....A WINNER!!!!%n", roll);
    System.out.printf("%1$ d PAYS EVEN MONEY, YOU WIN $%2$3.2f%n", roll, wager);
    return wager;
  }

  public static void winningsUpdate(double winnings) {
    System.out.println(switch ((int) Math.signum(winnings)) {
      case 1 -> "YOU ARE NOW AHEAD $%3.2f".formatted(winnings);
      case 0 -> "YOU ARE NOW EVEN AT 0";
      default -> "YOU ARE NOW UNDER $%3.2f".formatted(-winnings);
    });
  }

  public static void winningsReport(double winnings) {
    System.out.println(
        switch ((int) Math.signum(winnings)) {
          case 1 -> "CONGRATULATIONS---YOU CAME OUT A WINNER. COME AGAIN!";
          case 0 -> "CONGRATULATIONS---YOU CAME OUT EVEN, NOT BAD FOR AN AMATEUR";
          default -> "TOO BAD, YOU ARE IN THE HOLE. COME AGAIN.";
        }
    );
  }

  public static boolean stillInterested(double winnings) {
    System.out.print(" IF YOU WANT TO PLAY AGAIN PRINT 5 IF NOT PRINT 2 ");
    int fiveOrTwo = (int)getInput();
    winningsUpdate(winnings);
    return fiveOrTwo == 5;
  }

  public static double getWager() {
    System.out.print("INPUT THE AMOUNT OF YOUR WAGER. ");
    return getInput();
  }

  public static double getInput() {
    Scanner scanner = new Scanner(System.in);
    System.out.print("> ");
    while (true) {
      try {
        return scanner.nextDouble();
      } catch (Exception ex) {
        try {
          scanner.nextLine(); // flush whatever this non number stuff is.
        } catch (Exception ns_ex) { // received EOF (ctrl-d or ctrl-z if windows)
          System.out.println("END OF INPUT, STOPPING PROGRAM.");
          System.exit(1);
        }
      }
      System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE");
      System.out.print("> ");
    }
  }
}

```

# `29_Craps/javascript/craps.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是将一个字符串 `str` 输出到网页的 `#output` 元素中，例如：
```javascript
function print(str) {
   document.getElementById("output").appendChild(document.createTextNode(str));
}
```
将一个字符串 `str` 作为参数，使用 `document.getElementById("output").appendChild(document.createTextNode(str))` 将该字符串转换为文本节点并添加到 `#output` 元素中，这样就可以在网页上显示该字符串了。

`input` 函数的作用是从用户那里获取输入的值，例如：
```javascript
function input() {
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
                      input_element.addEventListener("keyup", function (event) {
                                                     if (event.keyCode == 27) {
                                                     resolve(null);
                                                     }
                                                     });
                      });
                      input_element.focus();
                      print("请输入：");
                      input_element.required = true;
                      input_element.required = true;
                      input_str = undefined;
                      input_element.addEventListener("change", function (event) {
                                                     if (event.target.value != null) {
                                                     input_str = event.target.value;
                                                     print(input_str);
                                                     print("\n");
                                                     resolve(input_str);
                                                     }
                                                     });
                      });
                      document.getElementById("output").appendChild(input_element);
                      input_element.focus();
                      print("请输入：");
                      input_element.required = true;
                      input_element.addEventListener("change", function (event) {
                                                     if (event.target.value != null) {
                                                     resolve(event.target.value);
                                                     }
                                                     });
                      });
                      input_element.focus();
                      input_str = undefined;
                      input_element.addEventListener("keyup", function (event) {
                                                     if (event.keyCode == 27) {
                                                     resolve(null);
                                                     }
                                                     });
                      });
                      document.getElementById("output").appendChild(input_element);
                      input_element.focus();
                      input_str = undefined;
                      input_element.addEventListener("keyup", function (event) {
                                                     if (event.keyCode == 27) {
```


```
// CRAPS
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

It looks like you're trying to write a program that simulates a game of 20 questions where one person, the "questioner," is trying to guess the location of a "$20 Questions" button, and the other person, the "teller," is the one who is trying to protect the button from the questioner. The program should keep track of the number of questions the questioner has answered correctly and the number of questions the questioner has answered incorrectly, and it should print out a message to the teller when the questioner has run out of questions or when they have answered the location correctly.

Here's some sample code to get that started:
``` 
#include <stdio.h>
#include <string.h>
#include <time.h>

int main() {
   int questioner_score = 0;
   int teller_score = 0;
   int round = 1;
   int max_attempts = 10;

   char key;
   while (1) {
       key = roll();
       if (key == 7) {
           max_attempts++;
       } else {
           int guess;
           scanf("%d", &guess);

           if (guess == 2) {
               // The questioner has answered the location correctly.
               questioner_score++;
               print("问对啦！ ");
               print(" 你现在胜利了！ ");
               print(" Congratulations! ");
               print(" OUT 胜利了！ ");
               break;
           } else if (guess == 1) {
               // The questioner has answered incorrectly.
               questioner_score++;
               print("很遗憾，答错了。 ");
               print(" 你还有 $questioner_score$ 的问题。 ");
               print(" 请问你要再问多少次呢？ ");
               print(" ");
               print(" - 再来一次吗？ ");
               scanf("%d", &round);
               if (round <= max_attempts) {
                   max_attempts--;
                   print("好的，再来一次。");
               } else {
                   print("答问题的次数不能超过 $max_attempts$。");
               }
           } else {
               // The questioner has answered the location incorrectly.
               print("很遗憾，答错了。 ");
               print(" 你还有 $questioner_score$ 的问题。 ");
               print(" 请问你要再问多少次呢？ ");
               print(" ");
               print(" - 再来一次吗？ ");
               scanf("%d", &round);
               if (round <= max_attempts) {
                   max_attempts--;
                   print("好的，再来一次。");
               } else {
                   print("答问题的次数不能超过 $max_attempts$。");
               }
           }
       }
   }

   if (questioner_score == max_attempts) {
       print("最终答案是：");
       print(roll() == 7 ? "是在中国" : "是在美国");
   } else if (teller_score > questioner_score) {
       print("非常遗憾，你失败了。");
       print("恭喜你赢得了比赛。");
       print("你答对了，获得冠军的是你。");
   } else {
       print("恭喜你，答对的次数比答错的次数多。");
       print("你答对的次数是：");
       print(questioner_score);
       print(", ");
       print(teller_score);
```


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

function roll()
{
    return Math.floor(6 * Math.random())+1 + Math.floor(6 * Math.random())+1;
}

// Main program
async function main()
{
    print(tab(33) + "CRAPS\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    r = 0;
    print("2,3,12 ARE LOSERS: 4,5,6,8,9,10 ARE POINTS: 7,11 ARE NATURAL WINNERS.\n");
    while (1) {
        print("INPUT THE AMOUNT OF YOUR WAGER.");
        f = parseInt(await input());
        print("I WILL NOW THROW THE DICE\n");
        x = roll();
        if (x == 7 || x == 11) {
            print(x + " - NATURAL....A WINNER!!!!\n");
            print(x + " PAYS EVEN MONEY, YOU WIN " + f + " DOLLARS\n");
            r += f;
        } else if (x == 2) {
            print(x + " - SNAKE EYES....YOU LOSE.\n");
            print("YOU LOSE " + f + " DOLLARS.\n");
            r -= f;
        } else if (x == 3 || x == 12) { // Original duplicates comparison in line 70
            print(x + " - CRAPS....YOU LOSE.\n");
            print("YOU LOSE " + f + " DOLLARS.\n");
            r -= f;
        } else {
            print(x + " IS THE POINT. I WILL ROLL AGAIN\n");
            while (1) {
                o = roll();
                if (o == 7) {
                    print(o + " - CRAPS, YOU LOSE.\n");
                    print("YOU LOSE $" + f + "\n");
                    r -= f;
                    break;
                }
                if (o == x) {
                    print(x + " - A WINNER.........CONGRATS!!!!!!!!\n");
                    print(x + " AT 2 TO 1 ODDS PAYS YOU...LET ME SEE..." + 2 * f + " DOLLARS\n");
                    r += f * 2;
                    break;
                }
                print(o + " - NO POINT. I WILL ROLL AGAIN\n");
            }
        }
        print(" IF YOU WANT TO PLAY AGAIN PRINT 5 IF NOT PRINT 2");
        m = parseInt(await input());
        if (r < 0) {
            print("YOU ARE NOW UNDER $" + -r + "\n");
        } else if (r > 0) {
            print("YOU ARE NOW AHEAD $" + r + "\n");
        } else {
            print("YOU ARE NOW EVEN AT 0\n");
        }
        if (m != 5)
            break;
    }
    if (r < 0) {
        print("TOO BAD, YOU ARE IN THE HOLE. COME AGAIN.\n");
    } else if (r > 0) {
        print("CONGRATULATIONS---YOU CAME OUT A WINNER. COME AGAIN!\n");
    } else {
        print("CONGRATULATIONS---YOU CAME OUT EVEN, NOT BAD FOR AN AMATEUR\n");
    }

}

```

这道题的代码是一个 C 语言的主函数，也就是程序的入口点。在 C 语言中，每个程序都必须包含一个 main 函数，程序的控制权在 main 函数内。

当程序运行时，首先会检查 main 函数是否存在。如果存在，程序会进入 main 函数内，否则会输出“程序未定义”的错误信息。

所以，这段代码的作用是定义了一个 C 语言程序的 main 函数，但该函数并未做任何实际的逻辑处理。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/) by Alex Conconi

---

### Lua porting notes

-  The `craps_main` function contains the main game loop, which iteratively
plays craps rounds by calling `play_round` and tracks winnings and losings.
- Replaced the original routine that tries to scramble the random number
generator with a proper seed initializer in Lua: `math.randomseed(os.time())`
(as advised in the general porting notes). 
- Added basic input validation to accept only positive integers for the
wager and the answer to the "If you want to play again print 5" question.
- "If you want to play again print 5 if not print 2" reads a bit odd but
we decided to leave it as is and stay true to the BASIC original version.

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `29_Craps/python/craps.py`

这段代码是一个用于模拟 craps 游戏的脚本，按照标准的赌博游戏规则进行模拟。具体来说，该脚本实现了以下功能：

1. 在第一次掷骰子时，如果出现 7 或 11，则赢得了游戏，否则失败。
2. 在第一次掷骰子时，如果出现 2、3 或 12，则输掉了游戏，并且需要换手。
3. 如果出现任何其他数字，则被认为是“得分”。在继续投骰子之后，如果得到了得分，则赢得了游戏，否则更换了手。
4. 如果第一次掷骰子出现了 7，则需要更换手，并且游戏结束。

另外，该脚本还使用了 Python 3 环境。


```
#!/usr/bin/env python3
"""This game simulates the games of craps played according to standard Nevada craps table rules.

That is:

1. A 7 or 11 on the first roll wins
2. A 2, 3, or 12 on the first roll loses
3. Any other number rolled becomes your "point." You continue to roll; if you get your point you win. If you
   roll a 7, you lose and the dice change hands when this happens.

This version of craps was modified by Steve North of Creative Computing. It is based on an original which
appeared one day one a computer at DEC.
"""
from random import randint


```

It looks like you have defined a game where the player roll a pair of dice, and if the outcome is 2 or 7, the player loses. Depending on the outcome, the winnings are added to the player's score. If the player wins, they get to keep rolling until they choose to stop. The game keeps track of the player's score, and the winner is determined when the winnings have been added to the score.


```
def throw_dice() -> int:
    return randint(1, 6) + randint(1, 6)


def main() -> None:
    print(" " * 33 + "Craps")
    print(" " * 15 + "Creative Computing  Morristown, New Jersey\n\n\n")

    winnings = 0
    print("2,3,12 are losers; 4,5,6,8,9,10 are points; 7,11 are natural winners.")

    play_again = True
    while play_again:
        wager = int(input("Input the amount of your wager: "))

        print("I will now throw the dice")
        roll_1 = throw_dice()

        if roll_1 in [7, 11]:
            print(f"{roll_1} - natural.... a winner!!!!")
            print(f"{roll_1} pays even money, you win {wager} dollars")
            winnings += wager
        elif roll_1 == 2:
            print(f"{roll_1} - snake eyes.... you lose.")
            print(f"You lose {wager} dollars")
            winnings -= wager
        elif roll_1 in [3, 12]:
            print(f"{roll_1} - craps.... you lose.")
            print(f"You lose {wager} dollars")
            winnings -= wager
        else:
            print(f"{roll_1} is the point. I will roll again")
            roll_2 = 0
            while roll_2 not in [roll_1, 7]:
                roll_2 = throw_dice()
                if roll_2 == 7:
                    print(f"{roll_2} - craps. You lose.")
                    print(f"You lose $ {wager}")
                    winnings -= wager
                elif roll_2 == roll_1:
                    print(f"{roll_1} - a winner.........congrats!!!!!!!!")
                    print(
                        f"{roll_1} at 2 to 1 odds pays you...let me see... {2 * wager} dollars"
                    )
                    winnings += 2 * wager
                else:
                    print(f"{roll_2} - no point. I will roll again")

        m = input("  If you want to play again print 5 if not print 2: ")
        if winnings < 0:
            print(f"You are now under ${-winnings}")
        elif winnings > 0:
            print(f"You are now ahead ${winnings}")
        else:
            print("You are now even at 0")
        play_again = m == "5"

    if winnings < 0:
        print("Too bad, you are in the hole. Come again.")
    elif winnings > 0:
        print("Congratulations---you came out a winner. Come again.")
    else:
        print("Congratulations---you came out even, not bad for an amateur")


```

这段代码是一个if语句，它的判断条件是(__name__ == "__main__")。如果这个条件为真，代码将跳转到__main__函数内执行。

"__name__"是一个特殊变量，它是用来保存当前程序的名称的。在Python中，程序的名称通常就是它所在的文件名，所以 "__name__"保存的就是当前程序的名称。

"__main__"是一个内置函数，用于执行程序的入口点。也就是说，当程序运行时，首先会执行__main__函数，这个函数内部就是程序的代码。

因此，如果 __name__ == "__main__"，那么这段代码将会执行程序的入口点，也就是执行main函数。


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


### Cube

CUBE is a game played on the facing sides of a cube with a side dimension of 2. A location is designated by three numbers — e.g., 1, 2, 1. The object is to travel from 1, 1, 1 to 3, 3, 3 by moving one horizontal or vertical (not diagonal) square at a time without striking one of 5 randomly placed landmines. You are staked to $500; prior to each play of the game you may make a wager whether you will reach your destination. You lose if you hit a mine or try to make an illegal move — i.e., change more than one digit from your previous position.

Cube was created by Jerimac Ratliff of Fort Worth, Texas.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=53)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=68)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)

##### Known Bugs

This program does very little validation of its input, enabling the user to cheat in two ways:
- One can enter a large negative wager, purposely lose, and gain that much money.
- One can move outside the cube (using coordinates 0 or 4), then safely walk "around" the standard play volume to the destination square.

It's remotely possible that these are clever solutions the user is intended to find, solving an otherwise purely random game.

##### Randomization Logic

The BASIC code uses an interesting technique for choosing the random coordinates for the mines. The first coordinate is
chosen like this:

```basic
380 LET A=INT(3*(RND(X)))
390 IF A<>0 THEN 410
400 LET A=3
```

where line 410 is the start of a similar block of code for the next coordinate. The behaviour of `RND(X)` depends on the
value of `X`. If `X` is greater than zero then it returns a random value between 0 and 1. If `X` is zero it returns the
last random value generated, or 0 if no value has yet been generated.

If `X` is 1, therefore, the first line above set `A` to 0, 1, or 2. The next 2 lines replace a 0 with a 3. The
replacement values varies for the different coordinates with the result that the random selection is biased towards a
specific set of points. If `X` is 0, the `RND` calls all return 0, so the coordinates are the known. It appears that
this technique was probably used to allow testing the game with a well-known set of locations for the mines. However, in
the code as it comes to us, the value of `X` is never set and is thus 0, so the mine locations are never randomized.

The C# port implements the biased randomized mine locations, as seems to be the original intent, but includes a
command-line switch to enable the deterministic execution as well.


# `30_Cube/csharp/Game.cs`

This appears to be a programming language implementation of a game of snake. The `SnakeGame` class represents the game logic, while the `Snake` class represents the AI.

The game starts with a randomly chosen initial location and an initial balance. The player can then choose to try again to choose a new location for the game.

The `PlayGame` method reads the initial position of the snake, the goal location, and the current score from the console. It then repeatedly chooses a new location for the snake to move to, using the `MoveIsLegal` method to check if the new location is valid.

If the snake reaches the goal location, the game is considered won, and the player is prompted to enter their name to receive a congratulatory message. If the snake does not reach the goal location and runs out of money, the game is considered lost, and the player is prompted to enter a new name to try again.

The `Lose` method prints out the final score and杀死 the snake. The `Win` method does the same thing but also prints out a congratulatory message.


```
namespace Cube;

internal class Game
{
    private const int _initialBalance = 500;
    private readonly IEnumerable<(int, int, int)> _seeds = new List<(int, int, int)>
    {
        (3, 2, 3), (1, 3, 3), (3, 3, 2), (3, 2, 3), (3, 1, 3)
    };
    private readonly (float, float, float) _startLocation = (1, 1, 1);
    private readonly (float, float, float) _goalLocation = (3, 3, 3);

    private readonly IReadWrite _io;
    private readonly IRandom _random;

    public Game(IReadWrite io, IRandom random)
    {
        _io = io;
        _random = random;
    }

    public void Play()
    {
        _io.Write(Streams.Introduction);

        if (_io.ReadNumber("") != 0)
        {
            _io.Write(Streams.Instructions);
        }

        PlaySeries(_initialBalance);

        _io.Write(Streams.Goodbye);
    }

    private void PlaySeries(float balance)
    {
        while (true)
        {
            var wager = _io.ReadWager(balance);

            var gameWon = PlayGame();

            if (wager.HasValue)
            {
                balance = gameWon ? (balance + wager.Value) : (balance - wager.Value);
                if (balance <= 0)
                {
                    _io.Write(Streams.Bust);
                    return;
                }
                _io.WriteLine(Formats.Balance, balance);
            }

            if (_io.ReadNumber(Prompts.TryAgain) != 1) { return; }
        }
    }

    private bool PlayGame()
    {
        var mineLocations = _seeds.Select(seed => _random.NextLocation(seed)).ToHashSet();
        var currentLocation = _startLocation;
        var prompt = Prompts.YourMove;

        while (true)
        {
            var newLocation = _io.Read3Numbers(prompt);

            if (!MoveIsLegal(currentLocation, newLocation)) { return Lose(Streams.IllegalMove); }

            currentLocation = newLocation;

            if (currentLocation == _goalLocation) { return Win(Streams.Congratulations); }

            if (mineLocations.Contains(currentLocation)) { return Lose(Streams.Bang); }

            prompt = Prompts.NextMove;
        }
    }

    private bool Lose(Stream text)
    {
        _io.Write(text);
        return false;
    }

    private bool Win(Stream text)
    {
        _io.Write(text);
        return true;
    }

    private bool MoveIsLegal((float, float, float) from, (float, float, float) to)
        => (to.Item1 - from.Item1, to.Item2 - from.Item2, to.Item3 - from.Item3) switch
        {
            ( > 1, _, _) => false,
            (_, > 1, _) => false,
            (_, _, > 1) => false,
            (1, 1, _) => false,
            (1, _, 1) => false,
            (_, 1, 1) => false,
            _ => true
        };
}

```

# `30_Cube/csharp/IOExtensions.cs`

这段代码是一个名为 "IOExtensions" 的内部类，属于 "Cube" 命名空间。它包含一个名为 "ReadWager" 的方法，其接收一个 "IReadWrite" 类型的 io 参数和一个 float 类型的参数 balance。

"ReadWager" 的作用是向一个名为 "Wager" 的浮点数中写入数据，然后从 "balance" 参数中读取一个浮点数，前提是 "Wager" 不小于 "balance"。这个方法使用了 C# 中的 "Write" 和 "ReadNumber" 方法，以及 "Prompts" 类中的 "HowMuch" 和 "BetAgain" 方法。

总之，这个方法允许用户输入一个投注额（balance），然后根据用户的选择，或者再次输入投注额并比较两个结果，输出的结果是投注额或者输出的投注结果。


```
namespace Cube;

internal static class IOExtensions
{
    internal static float? ReadWager(this IReadWrite io, float balance)
    {
        io.Write(Streams.Wager);
        if (io.ReadNumber("") == 0) { return null; }

        var prompt = Prompts.HowMuch;

        while(true)
        {
            var wager = io.ReadNumber(prompt);
            if (wager <= balance) { return wager; }

            prompt = Prompts.BetAgain;
        }
    }
}

```