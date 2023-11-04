# BasicComputerGames源码解析 72

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `78_Sine_Wave/java/src/SineWave.java`

这段代码是一个基于Sine Wave的程序，主要目的是创建一个与1970年代BASIC程序相似的Java程序。它主要通过以下方式来实现这个目标：

1. 在程序中引入了一个名为SineWave的类。
2. 在main方法中，系统地输出字符，从而创建出Sine Wave图形。
3. 通过循环40次，每次间隔0.25，来计算出曲线上的点数。
4. 根据isCreative变量，分别打印"CREATIVE"和"COMPUTING"单词。
5. 在循环中，系统会改变输出的内容，从而实现Sine Wave曲线的平移和缩放。

总之，这段代码主要实现了Sine Wave曲线的绘制，其中包括了平移、缩放、循环、输出等操作。


```
/**
 * Sine Wave
 *
 * Based on the Sine Wave program here
 * https://github.com/coding-horror/basic-computer-games/blob/main/78%20Sine%20Wave/sinewave.bas
 *
 * Note:  The idea was to create a version of the 1970's Basic program in Java, without introducing
 *        new features - no additional text, error checking, etc has been added.
 */
public class SineWave {

    public static void main(String[] args) {
        System.out.println("""
           SINE WAVE
           CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY
           """);
        var isCreative = true;
        for(var t = 0d; t<40; t += .25) {
            //Indent output
            var indentations = 26 + (int) (25 * Math.sin(t));
            System.out.print(" ".repeat(indentations));
            //Change output every iteration
            var word = isCreative ? "CREATIVE" : "COMPUTING";
            System.out.println(word);
            isCreative = !isCreative ;
        }
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


# `78_Sine_Wave/python/sinewave.py`

这段代码是一个简单的程序，它绘制了一个正弦波形的图形。正弦波形是一种常见的波形，它的值随着时间的推移而变化，但是它的形状和振幅保持不变。

该程序使用了BASIC编程语言，这是1978年由BASIC计算机游戏开发公司开发的一种简单的编程语言。该程序是在此语言的背景下开发的，并用于展示计算机的背景活动。

该程序的核心部分是使用SIN函数来计算正弦波形的值，然后使用显示函数将这个值显示出来。函数的参数是时间（in seconds）和振幅（in value）。

该程序没有其他的功能，但是它为使用者提供了一个简单的可视化工具，用于观察和理解计算机的基本原理。


```
########################################################
#
# Sine Wave
#
# From: BASIC Computer Games (1978)
#       Edited by David H. Ahl
#
# "Did you ever go to a computer show and see a bunch of
#  CRT terminals just sitting there waiting forlornly
#  for someone to give a demo on them.  It was one of
#  those moments when I was at DEC that I decided there
#  should be a little bit of background activity.  And
#  why not plot with words instead of the usual X's?
#  Thus SINE WAVE was born and lives on in dozens of
#  different versions.  At least those CRTs don't look
```

这段代码是一个Python程序，它将绘制一个正弦波形图案，并在其中心显示"Creative Computing"和"Computing"两行文本。程序的主要目的是通过循环和数学函数来绘制正弦波形，并使用DELAY参数来控制每行之间的延迟。

具体来说，程序首先定义了一系列常量，包括用于在屏幕上显示文本和绘图的字体、最大行数、每行中的字符数量、控制正弦波形输出的字符和控制文本中心位置的字符。然后，程序创建了一个名为main的函数，它包含程序的主要操作。

在main函数中，程序首先打印出两行文本，这些文本将在屏幕上显示。然后，程序使用循环来获取每个字符，并使用数学函数sin来计算每个字符的正弦值。程序还定义了一个STEP_SIZE变量，用于控制每行之间的延迟，并使用DELAY变量来控制循环的频率。

接下来，程序使用while循环来创建一个无限的循环，该循环将在屏幕上绘制正弦波形图案。在每次循环中，程序将使用计算得到的角度来绘制字符中心处的字符，并使用CENTER变量来确定字符的位置。然后，程序将用rjust函数来将字符居中，并使用len函数来检查还有多少个字符。如果字符串中有字符，则程序还将计算并增加字符之间的步长。

最后，程序使用time.sleep函数来让程序在绘制字符时稍微休息一下，以免绘制的过程太快，导致效果不佳。


```
#  so lifeless anymore."
#
# Original BASIC version by David Ahl
#
# Python port by Jeff Jetton, 2019
#
########################################################

import math
import time


def main() -> None:
    # Constants
    STRINGS = ("Creative", "Computing")  # Text to display
    MAX_LINES = 160
    STEP_SIZE = 0.25  # Number of radians to increase at each
    # line. Controls speed of horizontal
    # printing movement.
    CENTER = 26  # Controls left edge of "middle" string
    DELAY = 0.05  # Amount of seconds to wait between lines

    # Display "intro" text
    print("\n                        Sine Wave")
    print("         Creative Computing  Morristown, New Jersey")
    print("\n\n\n\n")
    # "REMarkable program by David Ahl"

    string_index = 0
    radians: float = 0
    width = CENTER - 1

    # "Start long loop"
    for _line_num in range(MAX_LINES):

        # Get string to display on this line
        curr_string = STRINGS[string_index]

        # Calculate how far over to print the text
        sine = math.sin(radians)
        padding = int(CENTER + width * sine)
        print(curr_string.rjust(padding + len(curr_string)))

        # Increase radians and increment our tuple index
        radians += STEP_SIZE
        string_index += 1
        if string_index >= len(STRINGS):
            string_index = 0

        # Make sure the text doesn't fly by too fast...
        time.sleep(DELAY)


```

这段代码是一个用于在 Python 中执行本地 Basic 版本程序的控制台 if 语句。if 语句的语法是在 Python 2.0 或更高版本中定义的，因此对于之前的版本，可能需要进行一些修改。

在这段注释中，作者解释了这段代码的作用。这段代码的作用是判断当前程序是否作为主程序运行，如果是，那么执行 main 函数。

具体来说，这段代码定义了一个名为 __main__ 的内置函数，如果当前程序作为主程序运行，那么将执行该函数体内的代码。在这个例子中，作者使用 if 语句来决定程序是否要输出一个字符串，该字符串存储在名为 "main" 的变量中。


```
if __name__ == "__main__":
    main()

########################################################
#
# Porting Notes
#
#   The original BASIC version hardcoded two words in
#   the body of the code and then used a sentinel flag
#   (flipping between 0 and 1) with IF statements to
#   determine the word to display next.
#
#   Here, the words have been moved to a Python tuple,
#   which is iterated over without any assumptions about
#   how long it is.  The STRINGS tuple can therefore be
```

这段代码是一个Python脚本，它的目的是让程序输出任何数量行的文本序列，而不受当前计算机的性能限制。这个脚本有一个延迟组件，旨在使输出更接近历史数据。

延迟组件中的一些想法，例如让用户输入要输出的行数和/或步长，目前尚未实现。


```
#   modified to have to program print out any sequence
#   of any number of lines of text.
#
#   Since a modern computer running Python will print
#   to the screen much more quickly than a '70s-era
#   computer running BASIC would, a delay component
#   has been introduced in this version to make the
#   output more historically accurate.
#
#
# Ideas for Modifications
#
#   Ask the user for desired number of lines (perhaps
#   with an "infinite" option) and/or step size.
#
```

这段代码的主要目的是让用户输入文本字符串以显示，而不是让它们在代码中预先定义为一个常量。

在代码中，首先通过用户输入获取两个文本字符串作为参数。然后，使用内置函数`str.center()`计算它们之间的合适居中。居中的方式是根据最长的字符串的长度来确定的。

接着，代码将两个获取到的字符串连接起来，并将它们放置在居中位置，生成一个新的字符串。如果新字符串的长度大于两个原始字符串的长度，则它将使用其中一个原始字符串进行填充，以使其长度与另一个原始字符串匹配。

最后，代码还设置了一个变量`STINGS`，用于存储用户输入的文本字符串。


```
#   Let the user input the text strings to display,
#   rather than having it pre-defined in a constant.
#   Calculate an appropriate CENTER based on length of
#   longest string.
#
#   Try changing STINGS so that it only includes a
#   single string, just like this:
#
#       STRINGS = ('Howdy!')
#
#   What happens? Why? How would you fix it?
#
########################################################

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by Anthony Rubick [AnthonyMichaelTDM](https://github.com/AnthonyMichaelTDM)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Slalom

This game simulates a slalom run down a course with one to 25 gates. The user picks the number of gates and has some control over his speed down the course.

If you’re not a skier, here’s your golden opportunity to try it with minimal risk. If you are a skier, here’s something to do while your leg is in a cast.

SLALOM was written by J. Panek while a student at Dartmouth College.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=147)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=162)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- In the original version, the data pointer doesn't reset after a race is completed. This causes subsequent races to error at some future point at line 540, `READ Q'.

- It also doesn't restore the data pointer after executing the MAX command to see the gate speeds, meaning that if you use this command, it effectively skips those gates, and the speeds shown are completely incorrect.

#### Porting Notes


# `79_Slalom/csharp/program.cs`

This is a program written in C# that simulates a game of都是由火车通过一座大桥。The program takes command-line arguments to specify the number of gates to train, the initial rating of the game, and whether to keep playing the game after a certain number of wins.

The program uses two main classes:

* `Runner` class that performs the actual race simulation. This class has methods for displaying instructions, displaying the current gate speeds, and keeping track of the player's score.
* `Game` class that displays the instructions for the game, displays the gate information for the current round, and prompts the player to keep playing.

The `Runner` class has several methods:

* `DisplayInstructions()`: displays the instructions for the game.
* `DisplayGate()`: displays the current speed of each gate.
* `CreateRace()`: starts a new race and returns the gate number where the player failed.
* `RunRace()`: performs the race and updates the gate score, the player's score and the train's score.
* `GetGateScore()`: returns the score of the gate.
* `GetTrainScore()`: returns the score of the train.
* `SetTrainScore(int score)`: sets the score of the train.
* `SetGateScore(int score)`: sets the score of the gate.

The `Game` class has several methods:

* `DisplayInstructions()`: displays the instructions for the game.
* `SetGateNumber(int gateNumber)`: sets the gate number for the player.
* `SetRating(int rating)`: sets the player's rating.
* `SetIsKeepPlaying(bool isKeepPlaying)`: sets whether the player wants to keep playing.
* `Race()`: starts the race and updates the gate score, the player's score, and the train's score.
* `GetGateScore(int gateNumber)`: returns the score of the gate.
* `GetTrainScore()`: returns the score of the train.
* `SetTrainScore(int score)`: sets the score of the train.
* `SetGateScore(int score)`: sets the score of the gate.
* `DisplayRace()`: displays the race information.

This is just a basic example, the game has much more features and options to improve the game.


```
using System.Text;

namespace Slalom
{
    class Slalom
    {
        private int[] GateMaxSpeed = { 14,18,26,29,18,25,28,32,29,20,29,29,25,21,26,29,20,21,20,
                                       18,26,25,33,31,22 };

        private int GoldMedals = 0;
        private int SilverMedals = 0;
        private int BronzeMedals = 0;
        private void DisplayIntro()
        {
            Console.WriteLine("");
            Console.WriteLine("SLALOM".PadLeft(23));
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine("");
        }

        private void DisplayInstructions()
        {
            Console.WriteLine();
            Console.WriteLine("*** Slalom: This is the 1976 Winter Olympic Giant Slalom.  You are");
            Console.WriteLine("            the American team's only hope of a gold medal.");
            Console.WriteLine();
            Console.WriteLine("     0 -- Type this if you want to see how long you've taken.");
            Console.WriteLine("     1 -- Type this if you want to speed up a lot.");
            Console.WriteLine("     2 -- Type this if you want to speed up a little.");
            Console.WriteLine("     3 -- Type this if you want to speed up a teensy.");
            Console.WriteLine("     4 -- Type this if you want to keep going the same speed.");
            Console.WriteLine("     5 -- Type this if you want to check a teensy.");
            Console.WriteLine("     6 -- Type this if you want to check a litte.");
            Console.WriteLine("     7 -- Type this if you want to check a lot.");
            Console.WriteLine("     8 -- Type this if you want to cheat and try to skip a gate.");
            Console.WriteLine();
            Console.WriteLine(" The place to use these options is when the computer asks:");
            Console.WriteLine();
            Console.WriteLine("Option?");
            Console.WriteLine();
            Console.WriteLine("               Good Luck!");
            Console.WriteLine();
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
                    Console.WriteLine("Please type 'YES' or 'NO'");
            }

            return false;
        }

        private int PromptForGates()
        {
            bool Success = false;
            int NumberOfGates = 0;

            while (!Success)
            {
                Console.Write("How many gates does this course have (1 to 25) ");
                string LineInput = Console.ReadLine().Trim().ToLower();

                if (int.TryParse(LineInput, out NumberOfGates))
                {
                    if (NumberOfGates >= 1 && NumberOfGates <= 25)
                    {
                        Success = true;
                    }
                    else if (NumberOfGates < 1)
                    {
                        Console.WriteLine("Try again,");
                    }
                    else // greater than 25
                    {
                        Console.WriteLine("25 is the limit.");
                        NumberOfGates = 25;
                        Success = true;
                    }
                }
                else
                {
                    Console.WriteLine("Try again,");
                }
            }

            return NumberOfGates;
        }

        private int PromptForRate()
        {
            bool Success = false;
            int Rating = 0;

            while (!Success)
            {
                Console.Write("Rate yourself as a skier, (1=worst, 3=best) ");
                string LineInput = Console.ReadLine().Trim().ToLower();

                if (int.TryParse(LineInput, out Rating))
                {
                    if (Rating >= 1 && Rating <= 3)
                    {
                        Success = true;
                    }
                    else
                    {
                        Console.WriteLine("The bounds are 1-3");
                    }
                }
                else
                {
                    Console.WriteLine("The bounds are 1-3");
                }
            }

            return Rating;
        }

        private int PromptForOption()
        {
            bool Success = false;
            int Option = 0;

            while (!Success)
            {
                Console.Write("Option? ");
                string LineInput = Console.ReadLine().Trim().ToLower();

                if (int.TryParse(LineInput, out Option))
                {
                    if (Option >= 0 && Option <= 8)
                    {
                        Success = true;
                    }
                    else if (Option > 8)
                    {
                        Console.WriteLine("What?");
                    }
                }
                else
                {
                    Console.WriteLine("What?");
                }
            }

            return Option;
        }

        private string PromptForCommand()
        {
            bool Success = false;
            string Result = "";

            Console.WriteLine();
            Console.WriteLine("Type \"INS\" for intructions");
            Console.WriteLine("Type \"MAX\" for approximate maximum speeds");
            Console.WriteLine("Type \"RUN\" for the beginning of the race");

            while (!Success)
            {

                Console.Write("Command--? ");
                string LineInput = Console.ReadLine().Trim().ToLower();

                if (LineInput.Equals("ins") || LineInput.Equals("max") || LineInput.Equals("run"))
                {
                    Result = LineInput;
                    Success = true;
                }
                else
                {
                    Console.WriteLine();
                    Console.WriteLine();
                    Console.WriteLine("\"{0}\" is an illegal command--retry", LineInput);
                }
            }

            return Result;
        }

        private bool ExceedGateSpeed(double MaxGateSpeed, double MPH, double Time)
        {
            Random rand = new Random();

            Console.WriteLine("{0:N0} M.P.H.", MPH);
            if (MPH > MaxGateSpeed)
            {
                Console.Write("You went over the maximum speed ");
                if (rand.NextDouble() < ((MPH - (double)MaxGateSpeed) * 0.1) + 0.2)
                {
                    Console.WriteLine("and made it!");
                }
                else
                {
                    if (rand.NextDouble() < 0.5)
                    {
                        Console.WriteLine("snagged a flag!");
                    }
                    else
                    {
                        Console.WriteLine("wiped out!");
                    }

                    Console.WriteLine("You took {0:N2} seconds", rand.NextDouble() + Time);

                    return false;
                }
            }
            else if (MPH > (MaxGateSpeed - 1))
            {
                Console.WriteLine("Close one!");
            }

            return true;
        }
        private void DoARun(int NumberOfGates, int Rating)
        {
            Random rand = new Random();
            double MPH = 0;
            double Time = 0;
            int Option = 0;
            double MaxGateSpeed = 0; // Q
            double PreviousMPH = 0;
            double Medals = 0;

            Console.WriteLine("The starter counts down...5...4...3...2...1...GO!");

            MPH = rand.NextDouble() * (18-9)+9;

            Console.WriteLine();
            Console.WriteLine("You're off!");

            for (int GateNumber = 1; GateNumber <= NumberOfGates; GateNumber++)
            {
                MaxGateSpeed = GateMaxSpeed[GateNumber-1];

                Console.WriteLine();
                Console.WriteLine("Here comes Gate # {0}:", GateNumber);
                Console.WriteLine("{0:N0} M.P.H.", MPH);

                PreviousMPH = MPH;

                Option = PromptForOption();
                while (Option == 0)
                {
                    Console.WriteLine("You've taken {0:N2} seconds.", Time);
                    Option = PromptForOption();
                }

                switch (Option)
                {
                    case 1:
                        MPH = MPH + (rand.NextDouble() * (10-5)+5);
                        if (ExceedGateSpeed(MaxGateSpeed, MPH, Time))
                            break;
                        else
                            return;
                    case 2:
                        MPH = MPH + (rand.NextDouble() * (5-3)+3);
                        if (ExceedGateSpeed(MaxGateSpeed, MPH, Time))
                            break;
                        else
                            return;
                    case 3:
                        MPH = MPH + (rand.NextDouble() * (4-1)+1);
                        if (ExceedGateSpeed(MaxGateSpeed, MPH, Time))
                            break;
                        else
                            return;
                    case 4:
                        if (ExceedGateSpeed(MaxGateSpeed, MPH, Time))
                            break;
                        else
                            return;
                    case 5:
                        MPH = MPH - (rand.NextDouble() * (4-1)+1);
                        if (ExceedGateSpeed(MaxGateSpeed, MPH, Time))
                            break;
                        else
                            return;
                    case 6:
                        MPH = MPH - (rand.NextDouble() * (5-3)+3);
                        if (ExceedGateSpeed(MaxGateSpeed, MPH, Time))
                            break;
                        else
                            return;
                    case 7:
                        MPH = MPH - (rand.NextDouble() * (10-5)+5);
                        if (ExceedGateSpeed(MaxGateSpeed, MPH, Time))
                            break;
                        else
                            return;
                    case 8:  // Cheat!
                        Console.WriteLine("***Cheat");
                        if (rand.NextDouble() < 0.7)
                        {
                            Console.WriteLine("An official caught you!");
                            Console.WriteLine("You took {0:N2} seconds.", Time);

                            return;
                        }
                        else
                        {
                            Console.WriteLine("You made it!");
                            Time = Time + 1.5;
                        }
                        break;
                }

                if (MPH < 7)
                {
                    Console.WriteLine("Let's be realistic, OK?  Let's go back and try again...");
                    MPH = PreviousMPH;
                }
                else
                {
                    Time = Time + (MaxGateSpeed - MPH + 1);
                    if (MPH > MaxGateSpeed)
                    {
                        Time = Time + 0.5;

                    }
                }
            }

            Console.WriteLine();
            Console.WriteLine("You took {0:N2} seconds.", Time);

            Medals = Time;
            Medals = Medals / NumberOfGates;

            if (Medals < (1.5 - (Rating * 0.1)))
            {
                Console.WriteLine("You won a gold medal!");
                GoldMedals++;
            }
            else if (Medals < (2.9 - (Rating * 0.1)))
            {
                Console.WriteLine("You won a silver medal!");
                SilverMedals++;
            }
            else if (Medals < (4.4 - (Rating * 0.01)))
            {
                Console.WriteLine("You won a bronze medal!");
                BronzeMedals++;
            }
        }

        private void PlayOneRound()
        {
            int NumberOfGates = 0;
            string Command = "first";
            bool KeepPlaying = false;
            int Rating = 0;

            Console.WriteLine("");

            NumberOfGates = PromptForGates();

            while (!Command.Equals(""))
            {
                Command = PromptForCommand();

                // Display instructions
                if (Command.Equals("ins"))
                {
                    DisplayInstructions();
                }
                else if (Command.Equals("max"))
                {
                    Console.WriteLine("Gate Max");
                    Console.WriteLine(" #  M.P.H.");
                    Console.WriteLine("----------");
                    for (int i = 0; i < NumberOfGates; i++)
                    {
                        Console.WriteLine(" {0}     {1}", i+1, GateMaxSpeed[i]);
                    }
                }
                else // do a run!
                {
                    Rating = PromptForRate();

                    do
                    {
                        DoARun(NumberOfGates, Rating);

                        KeepPlaying = PromptYesNo("Do you want to race again? ");
                    }
                    while (KeepPlaying);

                    Console.WriteLine("Thanks for the race");

                    if (GoldMedals > 0)
                        Console.WriteLine("Gold Medals: {0}", GoldMedals);
                    if (SilverMedals > 0)
                        Console.WriteLine("Silver Medals: {0}", SilverMedals);
                    if (BronzeMedals > 0)
                        Console.WriteLine("Bronze Medals: {0}", BronzeMedals);

                    return;
                }
            }
        }

        public void PlayTheGame()
        {
            DisplayIntro();

            PlayOneRound();
        }
    }
    class Program
    {
        static void Main(string[] args)
        {

            new Slalom().PlayTheGame();

        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `79_Slalom/java/Slalom.java`

This appears to be a program written in Creative Comp就像是平台er. It appears to have a main menu with three options: INS, MAX, and RUN. The user is then prompted to enter the number of gates for the chosen race.

The program then appear to have a while loop that runs until the user chooses to exit. Inside the loop, the program prints a message asking the user how many gates the course has, with the number of gates maximum out of the possible number of gates (25). The program then converts the number of gates entered by the user to an integer and returns it.

It looks like there is a subclass called DisqualifiedException which extends the built-in Exception class. It appears to be a throwable that is caught if the program tries to read more than the maximum number of gates specified by the user.

There is also a subclass called WipedOutOrSnaggedAFlag which extends the built-in Exception class. It appears to be a throwable that is caught if the program tries to read more than the maximum number of gates specified by the user.

There is also a subclass called Medals which has a gold, silver, and bronze fields. These fields appear to be used to keep track of the user's medals and are used to determine the winner of the race.

It looks like there is a constructor for the Medals class which initializes the gold, silver, and bronze fields with the values 0.

There is also a printIntro method which appears to be printing an introduction message for the race.

It looks like there are several methods in the program including printMenu, readNumberOfGatesChoice, and printIntro which are not listed asstatic. It would be the responsibility of the user to decide which one to use.


```
import java.util.Arrays;
import java.util.InputMismatchException;
import java.util.Random;
import java.util.Scanner;

/**
 * Slalom
 * <p>
 * Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
 *
 * There is a bug in the original version where the data pointer doesn't reset after a race is completed. This causes subsequent races to error at
 * some future point on line "540    READ Q"
 */
public class Slalom {

    private static final int MAX_NUM_GATES = 25;
    private static final int[] MAX_SPEED = {
            14, 18, 26, 29, 18,
            25, 28, 32, 29, 20,
            29, 29, 25, 21, 26,
            29, 20, 21, 20, 18,
            26, 25, 33, 31, 22
    };

    public static void main(String[] args) {
        var random = new Random();

        printIntro();
        Scanner scanner = new Scanner(System.in);

        int numGates = readNumberOfGatesChoice(scanner);

        printMenu();
        MenuChoice menuChoice;
        do {
            menuChoice = readMenuOption(scanner);
            switch (menuChoice) {
                case INS:
                    printInstructions();
                    break;
                case MAX:
                    printApproxMaxSpeeds(numGates);
                    break;
                case RUN:
                    run(numGates, scanner, random);
                    break;
            }
        } while (menuChoice != MenuChoice.RUN);
    }

    private static void run(int numGates, Scanner scan, Random random) {
        int rating = readSkierRating(scan);
        boolean gameInProgress = true;
        var medals = new Medals(0, 0, 0);

        while (gameInProgress) {
            System.out.println("THE STARTER COUNTS DOWN...5...4...3...2...1...GO!");
            System.out.println("YOU'RE OFF!");

            int speed = random.nextInt(18 - 9) + 9;

            float totalTimeTaken = 0;
            try {
                totalTimeTaken = runThroughGates(numGates, scan, random, speed);
                System.out.printf("%nYOU TOOK %.2f SECONDS.%n", totalTimeTaken + random.nextFloat());

                medals = evaluateAndUpdateMedals(totalTimeTaken, numGates, rating, medals);
            } catch (WipedOutOrSnaggedAFlag | DisqualifiedException e) {
                //end of this race! Print time taken and stop
                System.out.printf("%nYOU TOOK %.2f SECONDS.%n", totalTimeTaken + random.nextFloat());
            }

            gameInProgress = readRaceAgainChoice(scan);
        }

        System.out.println("THANKS FOR THE RACE");
        if (medals.getGold() >= 1) System.out.printf("GOLD MEDALS: %d%n", medals.getGold());
        if (medals.getSilver() >= 1) System.out.printf("SILVER MEDALS: %d%n", medals.getSilver());
        if (medals.getBronze() >= 1) System.out.printf("BRONZE MEDALS: %d%n", medals.getBronze());
    }

    private static Medals evaluateAndUpdateMedals(float totalTimeTaken, int numGates, int rating,
                                                  Medals medals) {
        var m = totalTimeTaken;
        m = m / numGates;
        int goldMedals = medals.getGold();
        int silverMedals = medals.getSilver();
        int bronzeMedals = medals.getBronze();
        if (m < 1.5 - (rating * 0.1)) {
            System.out.println("YOU WON A GOLD MEDAL!");
            goldMedals++;
        } else if (m < 2.9 - rating * 0.1) {
            System.out.println("YOU WON A SILVER MEDAL");
            silverMedals++;
        } else if (m < 4.4 - rating * 0.01) {
            System.out.println("YOU WON A BRONZE MEDAL");
            bronzeMedals++;
        }
        return new Medals(goldMedals, silverMedals, bronzeMedals);
    }

    /**
     * @return the total time taken through all the gates.
     */
    private static float runThroughGates(int numGates, Scanner scan, Random random, int speed) throws DisqualifiedException, WipedOutOrSnaggedAFlag {
        float totalTimeTaken = 0.0f;
        for (int i = 0; i < numGates; i++) {
            var gateNum = i + 1;
            boolean stillInRace = true;
            boolean gateCompleted = false;
            while (!gateCompleted) {
                System.out.printf("%nHERE COMES GATE # %d:%n", gateNum);
                printSpeed(speed);

                var tmpSpeed = speed;

                int chosenOption = readOption(scan);
                switch (chosenOption) {
                    case 0:
                        //how long
                        printHowLong(totalTimeTaken, random);
                        break;
                    case 1:
                        //speed up a lot
                        speed = speed + random.nextInt(10 - 5) + 5;
                        break;
                    case 2:
                        //speed up a little
                        speed = speed + random.nextInt(5 - 3) + 3;
                        break;
                    case 3:
                        //speed up a teensy
                        speed = speed + random.nextInt(4 - 1) + 1;
                        break;
                    case 4:
                        //keep going at the same speed
                        break;
                    case 5:
                        //check a teensy
                        speed = speed - random.nextInt(4 - 1) + 1;
                        break;
                    case 6:
                        //check a little
                        speed = speed - random.nextInt(5 - 3) + 3;
                        break;
                    case 7:
                        //check a lot
                        speed = speed - random.nextInt(10 - 5) + 5;
                        break;
                    case 8:
                        //cheat
                        System.out.println("***CHEAT");
                        if (random.nextFloat() < 0.7) {
                            System.out.println("AN OFFICIAL CAUGHT YOU!");
                            stillInRace = false;
                        } else {
                            System.out.println("YOU MADE IT!");
                            totalTimeTaken = totalTimeTaken + 1.5f;
                        }
                        break;
                }

                if (stillInRace) {
                    printSpeed(speed);
                    stillInRace = checkAndProcessIfOverMaxSpeed(random, speed, MAX_SPEED[i]);
                    if (!stillInRace) throw new WipedOutOrSnaggedAFlag();
                } else {
                    throw new DisqualifiedException();//we've been dis-qualified
                }

                if (speed < 7) {
                    System.out.println("LET'S BE REALISTIC, OK?  LET'S GO BACK AND TRY AGAIN...");
                    speed = tmpSpeed;
                    gateCompleted = false;
                } else {
                    totalTimeTaken = totalTimeTaken + (MAX_SPEED[i] - speed + 1);
                    if (speed > MAX_SPEED[i]) {
                        totalTimeTaken = totalTimeTaken + 0.5f;
                    }
                    gateCompleted = true;
                }
            }

        }
        return totalTimeTaken;
    }

    private static boolean checkAndProcessIfOverMaxSpeed(Random random, int speed, int maxSpeed) {
        boolean stillInRace = true;
        if (speed > maxSpeed) {
            if (random.nextFloat() >= (speed - maxSpeed) * 0.1 + 0.2) {
                System.out.println("YOU WENT OVER THE MAXIMUM SPEED AND MADE IT!");
            } else {
                System.out.print("YOU WENT OVER THE MAXIMUM SPEED AND ");
                if (random.nextBoolean()) {
                    System.out.println("WIPED OUT!");
                } else {
                    System.out.println("SNAGGED A FLAG!");
                }
                stillInRace = false;
            }
        } else if (speed > maxSpeed - 1) {
            System.out.println("CLOSE ONE!");
        }
        return stillInRace;
    }

    private static boolean readRaceAgainChoice(Scanner scan) {
        System.out.print("\nDO YOU WANT TO RACE AGAIN? ");
        String raceAgain = "";
        final String YES = "YES";
        final String NO = "NO";
        while (!YES.equals(raceAgain) && !NO.equals(raceAgain)) {
            raceAgain = scan.nextLine();
            if (!(YES.equals(raceAgain) || NO.equals(raceAgain))) {
                System.out.println("PLEASE TYPE 'YES' OR 'NO'");
            }
        }
        return raceAgain.equals(YES);
    }

    private static void printSpeed(int speed) {
        System.out.printf("%3d M.P.H.%n", speed);
    }

    private static void printHowLong(float t, Random random) {
        System.out.printf("YOU'VE TAKEN %.2f SECONDS.%n", t + random.nextFloat());
    }

    private static int readOption(Scanner scan) {
        Integer option = null;

        while (option == null) {
            System.out.print("OPTION? ");
            try {
                option = scan.nextInt();
            } catch (InputMismatchException ex) {
                System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE\n");
            }
            scan.nextLine();
            if (option != null && (option > 8 || option < 0)) {
                System.out.println("WHAT?");
                option = null;
            }
        }
        return option;
    }

    private static int readSkierRating(Scanner scan) {
        int rating = 0;

        while (rating < 1 || rating > 3) {
            System.out.print("RATE YOURSELF AS A SKIER, (1=WORST, 3=BEST)? ");
            try {
                rating = scan.nextInt();
                if (rating < 1 || rating > 3) {
                    System.out.println("THE BOUNDS ARE 1-3");
                }
            } catch (InputMismatchException ex) {
                System.out.println("!NUMBER EXPECTED - RETRY INPUT LINE\n");
            }
            scan.nextLine();
        }
        return rating;
    }

    private static void printApproxMaxSpeeds(int numGates) {
        System.out.println("GATE MAX");
        System.out.println(" #  M.P.H.");
        System.out.println("---------");
        for (int i = 0; i < numGates; i++) {
            System.out.println((i+1) + "  " + MAX_SPEED[i]);
        }
    }

    private static void printInstructions() {
        System.out.println("\n*** SLALOM: THIS IS THE 1976 WINTER OLYMPIC GIANT SLALOM.  YOU ARE");
        System.out.println("            THE AMERICAN TEAM'S ONLY HOPE OF A GOLD MEDAL.");
        System.out.println();
        System.out.println("     0 -- TYPE THIS IS YOU WANT TO SEE HOW LONG YOU'VE TAKEN.");
        System.out.println("     1 -- TYPE THIS IF YOU WANT TO SPEED UP A LOT.");
        System.out.println("     2 -- TYPE THIS IF YOU WANT TO SPEED UP A LITTLE.");
        System.out.println("     3 -- TYPE THIS IF YOU WANT TO SPEED UP A TEENSY.");
        System.out.println("     4 -- TYPE THIS IF YOU WANT TO KEEP GOING THE SAME SPEED.");
        System.out.println("     5 -- TYPE THIS IF YOU WANT TO CHECK A TEENSY.");
        System.out.println("     6 -- TYPE THIS IF YOU WANT TO CHECK A LITTLE.");
        System.out.println("     7 -- TYPE THIS IF YOU WANT TO CHECK A LOT.");
        System.out.println("     8 -- TYPE THIS IF YOU WANT TO CHEAT AND TRY TO SKIP A GATE.");
        System.out.println();
        System.out.println(" THE PLACE TO USE THESE OPTIONS IS WHEN THE COMPUTER ASKS:");
        System.out.println();
        System.out.println("OPTION?");
        System.out.println();
        System.out.println("                GOOD LUCK!");
    }

    private static MenuChoice readMenuOption(Scanner scan) {
        System.out.print("COMMAND--? ");
        MenuChoice menuChoice = null;

        while (menuChoice == null) {
            String choice = scan.next();
            if (Arrays.stream(MenuChoice.values()).anyMatch(a -> a.name().equals(choice))) {
                menuChoice = MenuChoice.valueOf(choice);
            } else {
                System.out.print("\""+ choice + "\" IS AN ILLEGAL COMMAND--RETRY? ");
            }
            scan.nextLine();
        }
        return menuChoice;
    }

    private static void printMenu() {
        System.out.println("TYPE INS FOR INSTRUCTIONS");
        System.out.println("TYPE MAX FOR APPROXIMATE MAXIMUM SPEEDS");
        System.out.println("TYPE RUN FOR THE BEGINNING OF THE RACE");
    }

    private static int readNumberOfGatesChoice(Scanner scan) {
        int numGates = 0;
        while (numGates < 1) {
            System.out.print("HOW MANY GATES DOES THIS COURSE HAVE (1 TO 25)? ");
            numGates = scan.nextInt();
            if (numGates > MAX_NUM_GATES) {
                System.out.println(MAX_NUM_GATES + " IS THE LIMIT.");
                numGates = MAX_NUM_GATES;
            }
        }
        return numGates;
    }

    private static void printIntro() {
        System.out.println("                                SLALOM");
        System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println("\n\n");
    }

    private enum MenuChoice {
        INS, MAX, RUN
    }

    private static class DisqualifiedException extends Exception {
    }

    private static class WipedOutOrSnaggedAFlag extends Exception {
    }

    private static class Medals {
        private int gold = 0;
        private int silver = 0;
        private int bronze = 0;

        public Medals(int gold, int silver, int bronze) {
            this.gold = gold;
            this.silver = silver;
            this.bronze = bronze;
        }

        public int getGold() {
            return gold;
        }

        public int getSilver() {
            return silver;
        }

        public int getBronze() {
            return bronze;
        }
    }


}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


# `79_Slalom/javascript/slalom.js`

这段代码定义了两个函数，分别是`print()`和`input()`。

`print()`函数的作用是打印一段字符串到网页的输出区域，即在网页上创建一段文本并将其添加到指定的元素中。

`input()`函数的作用是从用户接收输入的一行文本中获取输入的内容，并将其存储在变量`input_str`中。该函数通过使用`document.getElementById()`获取用户输入的元素，然后使用`focus()`方法将该元素设置为输入焦点，以便可以监听用户按键事件。当用户按下键盘上的13键时，该函数会从元素中读取当前值并将其存储在`input_str`变量中，然后将其添加到网页的输出区域中，并打印出当前的输入值。


```
// SLALOM
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

以下是两个函数，一个是用于计算SLALOM速度测试的，另一个是用于在Instruction List对话框中接收用户输入的。

```
// 计算SLALOM速度测试的函数
function calculate_speed()
{
   var test_speed = [];
   var count = 0;

   // 读取用户输入的速度
   while (speed-- > 0)
   {
       test_speed[] = speed;
       count++;
   }

   // 计算平均速度和平均速率和
   var avg_speed = Math.mean(test_speed);
   var avg_speed_per_10 = Math.ceil(avg_speed / 10);

   // 输出测试结果
   print("You have taken " + count + " seconds to calculate the speed.");
   print("Average speed: " + avg_speed_per_10 + "mph.");
   print("Average speed: " + avg_speed_per_10 + "km/h.");
   print("As you can see, your speed is very slow.\n");
}

// 在Instruction List对话框中接收用户输入的函数
function receive_input()
{
   var input = "";

   // 循环等待用户输入
   while (true)
   {
       print("Enter your choice: ");
       var choice = read_line();
       input = choice;

       // 如果用户输入了上面定义的选项，就返回相应的索引
       if (input == "0")
       {
           return [1, 2, 3, 4, 5, 6, 7, 8];
       }
       else if (input == "1")
       {
           speed++;
       }
       else if (input == "2")
       {
           speed++;
       }
       else if (input == "3")
       {
           speed++;
       }
       else if (input == "4")
       {
           speed++;
       }
       else if (input == "5")
       {
           speed++;
       }
       else if (input == "6")
       {
           speed++;
       }
       else if (input == "7")
       {
           speed++;
       }
       else if (input == "8")
       {
           speed++;
       }
       else
       {
           print("Invalid input.");
       }
   }

   return [1, 2, 3, 4, 5, 6, 7, 8];
}
```


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var speed = [,14,18,26,29,18,
             25,28,32,29,20,
             29,29,25,21,26,
             29,20,21,20,18,
             26,25,33,31,22];

function show_instructions()
{
    print("\n");
    print("*** SLALOM: THIS IS THE 1976 WINTER OLYMPIC GIANT SLALOM.  YOU ARE\n");
    print("            THE AMERICAN TEAM'S ONLY HOPE OF A GOLD MEDAL.\n");
    print("\n");
    print("     0 -- TYPE THIS IS YOU WANT TO SEE HOW LONG YOU'VE TAKEN.\n");
    print("     1 -- TYPE THIS IF YOU WANT TO SPEED UP A LOT.\n");
    print("     2 -- TYPE THIS IF YOU WANT TO SPEED UP A LITTLE.\n");
    print("     3 -- TYPE THIS IF YOU WANT TO SPEED UP A TEENSY.\n");
    print("     4 -- TYPE THIS IF YOU WANT TO KEEP GOING THE SAME SPEED.\n");
    print("     5 -- TYPE THIS IF YOU WANT TO CHECK A TEENSY.\n");
    print("     6 -- TYPE THIS IF YOU WANT TO CHECK A LITTLE.\n");
    print("     7 -- TYPE THIS IF YOU WANT TO CHECK A LOT.\n");
    print("     8 -- TYPE THIS IF YOU WANT TO CHEAT AND TRY TO SKIP A GATE.\n");
    print("\n");
    print(" THE PLACE TO USE THESE OPTIONS IS WHEN THE COMPUTER ASKS:\n");
    print("\n");
    print("OPTION?\n");
    print("\n");
    print("                GOOD LUCK!\n");
    print("\n");
}

```

This is a program that simulates a race between humans as competitors to win gold, silver, and bronze Medals based on their speed and endurance. The program takes input from the user to decide if they want to continue with the race or not.

The race is simulate based on the principle of least effort where the competitor who takes the shortest time to complete the race is the winner. The program also takes into account the competitor who takes the longest time to complete the race as they are not necessarily the winner.

The program uses the random number generator (RNG) to simulate the race and apply different winning mediaslions based on the time taken by the competitor. The winning mediaslions are displayed at the end of the race.

It is important to note that this program is not meant to be realistic or scientifically accurate. It is a simple game simulation that is meant to be used for entertainment purposes only and should not be taken seriously.

It is also worth mentioning that this program is based on a very basic algorithm of a race between two humans, it does not take into account any other factors like the environment, the weather, the other participants and the race track.


```
function show_speeds()
{
    print("GATE MAX\n");
    print(" #  M.P.H.\n");
    print("----------\n");
    for (var b = 1; b <= v; b++) {
        print(" " + b + "  " + speed[b] + "\n");
    }
}

// Main program
async function main()
{
    var gold = 0;
    var silver = 0;
    var bronze = 0;

    print(tab(33) + "SLALOM\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    while (1) {
        print("HOW MANY GATES DOES THIS COURSE HAVE (1 TO 25)");
        v = parseInt(await input());
        if (v >= 25) {
            print("25 IS THE LIMIT\n");
            v = 25;
        } else if (v < 1) {
            print("TRY AGAIN.\n");
        } else {
            break;
        }
    }
    print("\n");
    print("TYPE \"INS\" FOR INSTRUCTIONS\n");
    print("TYPE \"MAX\" FOR APPROXIMATE MAXIMUM SPEEDS\n");
    print("TYPE \"RUN\" FOR THE BEGINNING OF THE RACE\n");
    while (1) {
        print("COMMAND--");
        str = await input();
        if (str == "INS") {
            show_instructions();
        } else if (str == "MAX") {
            show_speeds();
        } else if (str == "RUN") {
            break;
        } else {
            print("\"" + str + "\" IS AN ILLEGAL COMMAND--RETRY");
        }
    }
    while (1) {
        print("RATE YOURSELF AS A SKIER, (1=WORST, 3=BEST)");
        a = parseInt(await input());
        if (a < 1 || a > 3)
            print("THE BOUNDS ARE 1-3\n");
        else
            break;
    }
    while (1) {
        print("THE STARTER COUNTS DOWN...5...4...3...2...1...GO!");
        t = 0;
        s = Math.floor(Math.random(1) * (18 - 9) + 9);
        print("\n");
        print("YOU'RE OFF!\n");
        for (o = 1; o <= v; o++) {
            q = speed[o];
            print("\n");
            print("HERE COMES GATE #" + o + " :\n");
            print(s + " M.P.H.\n");
            s1 = s;
            while (1) {
                print("OPTION");
                o1 = parseInt(await input());
                if (o1 < 0 || o1 > 8)
                    print("WHAT?\n");
                else if (o1 == 0)
                    print("YOU'VE TAKEN " + (t + Math.random()) + " SECONDS.\n");
                else
                    break;
            }
            finish = false;
            switch (o1) {
                case 1:
                    s += Math.floor(Math.random() * (10 - 5) + 5);
                    break;
                case 2:
                    s += Math.floor(Math.random() * (5 - 3) + 3);
                    break;
                case 3:
                    s += Math.floor(Math.random() * (4 - 1) + 1);
                    break;
                case 4:
                    break;
                case 5:
                    s -= Math.floor(Math.random() * (4 - 1) + 1);
                    break;
                case 6:
                    s -= Math.floor(Math.random() * (5 - 3) + 3);
                    break;
                case 7:
                    s -= Math.floor(Math.random() * (10 - 5) + 5);
                    break;
                case 8:
                    print("***CHEAT\n");
                    if (Math.random() >= 0.7) {
                        print("YOU MADE IT!\n");
                        t += 1.5;
                    } else {
                        print("AN OFFICIAL CAUGHT YOU!\n");
                        print("YOU TOOK " + (t + Math.random()) + " SECONDS.\n");
                        finish = true;
                    }
                    break;
            }
            if (!finish) {
                if (o1 != 4)
                    print(s + " M.P.H.\n");
                if (s > q) {
                    if (Math.random() < ((s - q) * 0.1) + 0.2) {
                        print("YOU WENT OVER THE MAXIMUM SPEED AND ");
                        if (Math.random() < 0.5) {
                            print("SNAGGED A FLAG!\n");
                        } else {
                            print("WIPED OUT!\n");
                        }
                        print("YOU TOOK " + (t + Math.random()) + " SECONDS.\n");
                        finish = true;
                    } else {
                        print("YOU WENT OVER THE MAXIMUM SPEED AND MADE IT!\n");
                    }
                } else if (s > q - 1) {
                    print("CLOSE ONE!\n");
                }
            }
            if (finish)
                break;
            if (s < 7) {
                print("LET'S BE REALISTIC, OK?  LET'S GO BACK AND TRY AGAIN...\n");
                s = s1;
                o--;
                continue;
            }
            t += q - s + 1;
            if (s > q) {
                t += 0.5;
            }
        }
        if (!finish) {
            print("\n");
            print("YOU TOOK " + (t + Math.random()) + " SECONDS.\n");
            m = t;
            m /= v;
            if (m < 1.5 - (a * 0.1)) {
                print("YOU WON A GOLD MEDAL!\n");
                gold++;
            } else if (m < 2.9 - (a * 0.1)) {
                print("YOU WON A SILVER MEDAL\n");
                silver++;
            } else if (m < 4.4 - (a * 0.1)) {
                print("YOU WON A BRONZE MEDAL\n");
                bronze++;
            }
        }
        while (1) {
            print("\n");
            print("DO YOU WANT TO RACE AGAIN");
            str = await input();
            if (str != "YES" && str != "NO")
                print("PLEASE TYPE 'YES' OR 'NO'\n");
            else
                break;
        }
        if (str != "YES")
            break;
    }
    print("THANKS FOR THE RACE\n");
    if (gold >= 1)
        print("GOLD MEDALS: " + gold + "\n");
    if (silver >= 1)
        print("SILVER MEDALS: " + silver + "\n");
    if (bronze >= 1)
        print("BRONZE MEDALS: " + bronze + "\n");
}

```

这是 C 语言中的一个程序，名为 `main()`，用于执行程序的所有操作。

`main()` 是程序的入口点，当程序运行时，它首先会执行这个函数。函数内部可能会包含 `printf()` 函数来输出一条消息给程序员，或者包含其他代码来设置可变参数、调用其他函数或执行其他操作。

请注意，由于我无法看到你的程序，所以无法具体了解它的工作原理。


```
main();

```