# BasicComputerGames源码解析 32

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `26_Chomp/python/chomp.py`

这段代码定义了一个名为 Canvas 的类，用于在二维画布上绘制图形并输出结果。在初始化方法 "__init__" 中，参数 `width` 和 `height` 表示画布的宽度和高度，参数 `fill` 表示在画布上填写的颜色。`_buffer` 是一个数组，用于存储绘制线程。在 `render` 方法中，将存储在 `_buffer` 中的所有内容打印出来，然后打印一个换行符。在 `chomp` 方法中，用于计算字符 "P" 在画布上的位置，并输出相应的结果。


```
#!/usr/bin/env python3

"""
CHOMP

Converted from BASIC to Python by Trevor Hobson
"""


class Canvas:
    """For drawing the cookie"""

    def __init__(self, width=9, height=9, fill="*") -> None:
        self._buffer = []
        for _ in range(height):
            line = []
            for _ in range(width):
                line.append(fill)
            self._buffer.append(line)
        self._buffer[0][0] = "P"

    def render(self) -> str:
        lines = ["       1 2 3 4 5 6 7 8 9"]
        for row, line in enumerate(self._buffer, start=1):
            lines.append(" " + str(row) + " " * 5 + " ".join(line))
        return "\n".join(lines)

    def chomp(self, r, c) -> str:
        if not 1 <= r <= len(self._buffer) or not 1 <= c <= len(self._buffer[0]):
            return "Empty"
        elif self._buffer[r - 1][c - 1] == " ":
            return "Empty"
        elif self._buffer[r - 1][c - 1] == "P":
            return "Poison"
        else:
            for row in range(r - 1, len(self._buffer)):
                for column in range(c - 1, len(self._buffer[row])):
                    self._buffer[row][column] = " "
            return "Chomp"


```

This is a Python implementation of a game where one player takes turns shooting a cookie, trying to avoid getting the cookie and getting poisoned.

There are n players, and each player has 9 columns and 9 rows. The player starts at a random row and column.

The while loop runs until one of the players quits or a new round starts.

The `Canvas` class is used to render the game board.

The game is育碧风格的，比较有趣。


```
def play_game() -> None:
    """Play one round of the game"""
    players = 0
    while players == 0:
        try:
            players = int(input("How many players "))

        except ValueError:
            print("Please enter a number.")
    rows = 0
    while rows == 0:
        try:
            rows = int(input("How many rows "))
            if rows > 9 or rows < 1:
                rows = 0
                print("Too many rows (9 is maximum).")

        except ValueError:
            print("Please enter a number.")
    columns = 0
    while columns == 0:
        try:
            columns = int(input("How many columns "))
            if columns > 9 or columns < 1:
                columns = 0
                print("Too many columns (9 is maximum).")

        except ValueError:
            print("Please enter a number.")
    cookie = Canvas(width=columns, height=rows)
    player = 0
    alive = True
    while alive:
        print()
        print(cookie.render())
        print()
        player += 1
        if player > players:
            player = 1
        while True:
            print("Player", player)
            player_row = -1
            player_column = -1
            while player_row == -1 or player_column == -1:
                try:
                    coordinates = [
                        int(item)
                        for item in input("Coordinates of chomp (Row, Column) ").split(
                            ","
                        )
                    ]
                    player_row = coordinates[0]
                    player_column = coordinates[1]

                except (ValueError, IndexError):
                    print("Please enter valid coordinates.")
            result = cookie.chomp(player_row, player_column)
            if result == "Empty":
                print("No fair. You're trying to chomp on empty space!")
            elif result == "Poison":
                print("\nYou lose player", player)
                alive = False
                break
            else:
                break


```

这段代码是一个名为“Chomp”的游戏，它在1973年的1月6日由美国科学家的Alan Kay和Simon被迫创造出来。在这段代码中，我们定义了一个名为“main”的函数，它返回一个名为“None”的对象。

在函数中，我们首先输出了一些字符，然后输出了一些关于这个游戏的信息。接着，我们询问玩家是否想要游戏规则，如果玩家输入“1”，那么我们输出了一些游戏规则的说明。

然后，我们创建了一个名为“Canvas”的类，并使用该类的“render”方法来输出游戏板。我们使用了一些字符来描述游戏板，例如“R”表示游戏板的大小，“C”表示游戏板的颜色，“P”表示毒方块的位置。

接下来，我们让玩家输入游戏板的行和列，并判断是否在毒方块内。如果玩家输入了正确的行和列，我们就显示毒方块的消除信息，并提示玩家使用“CHOMP”命令来消除该方块。

我们还有一些条件判断，例如显示游戏板的尺寸，判断是否所有行和列是否已经被访问过，以及判断玩家是否选择正确的选项。这些判断可以帮助我们确保游戏的公平性和正确性。

最后，我们通过一个无限循环来保持游戏的进行，直到玩家不再选择“CHOMP”为止。


```
def main() -> None:
    print(" " * 33 + "CHOMP")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")
    print("THIS IS THE GAME OF CHOMP (SCIENTIFIC AMERICAN, JAN 1973)")
    if input("Do you want the rules (1=Yes, 0=No!) ") != "0":
        print("Chomp is for 1 or more players (Humans only).\n")
        print("Here's how a board looks (This one is 5 by 7):")
        example = Canvas(width=7, height=5)
        print(example.render())
        print("\nThe board is a big cookie - R rows high and C columns")
        print("wide. You input R and C at the start. In the upper left")
        print("corner of the cookie is a poison square (P). The one who")
        print("chomps the poison square loses. To take a chomp, type the")
        print("row and column of one of the squares on the cookie.")
        print("All of the squares below and to the right of that square")
        print("(Including that square, too) disappear -- CHOMP!!")
        print("No fair chomping squares that have already been chomped,")
        print("or that are outside the original dimensions of the cookie.\n")
        print("Here we go...")

    keep_playing = True
    while keep_playing:
        play_game()
        keep_playing = input("\nAgain (1=Yes, 0=No!) ") == "1"


```

这段代码是一个条件判断语句，它的作用是在程序运行时判断是否是作为主程序运行。如果程序是作为主程序运行，那么程序会执行if语句块内的代码。这里的作用是输出"Hello World"。


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


### Civil War

This simulation is based on 14 battles in the Civil War. Facts and figures are based on the actual occurrence. If you follow the same strategy used in the actual battle, the results will be the same. Generally, this is a good strategy since the generals in the Civil War were fairly good military strategists. However, you can frequently outperform the Civil War generals, particularly in cases where they did not have good enemy intelligence and consequently followed a poor course of action. Naturally, it helps to know your Civil War history, although the computer gives you the rudiments.

After each of the 14 battles, your casualties are compared to the actual casualties of the battle, and you are told whether you win or lose the battle.

You may play Civil War alone in which case the program simulates the Union general. Or two players may play in which case the computer becomes the moderator.

Civil War was written in 1968 by three Students in Lexington High School, Massachusetts: L. Cram, L. Goodie, and D. Hibbard. It was modified into a 2-player game by G. Paul and R. Hess of TIES, St. Paul, Minnesota.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=46)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=61)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Known Bugs

- At the end of a single-player game, the program reports strategies used by "THE SOUTH", but these are in fact strategies used by the North (the computer player) -- the South is always a human player.

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `27_Civil_War/csharp/Army.cs`

This looks like a class written in C# that represents an AI character in a wargames simulation. This character has a chance to choose from a list of four strategies, and each strategy has a corresponding weight in the decision-making process. The weight of each strategy is determined by the relative frequency of that strategy in the data used to simulate the AI's decision-making. The character also has the ability to learn new strategies by choosing one of the four strategies and receiving a certain amount of bonus points in the decision-making process.

The class has a number of methods, including a method for calculating the losses of the AI's decisions, a method for displaying the AI's strategies, and a method for choosing a strategy. The class also has a method for learning new strategies.

It is based on the assumption that the character has a basic understanding of the wargames simulation game.


```
﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace CivilWar
{
    public class Army
    {
        private enum Resource
        {
            Food,
            Salaries,
            Ammunition
        }

        public Army(Side side)
        {
            Side = side;
        }

        public Side Side { get; }

        // Cumulative
        public int Wins { get; private set; } // W, L
        public int Losses { get; private set; } // L, W
        public int Draws { get; private set; } // W0
        public int BattlesFought => Wins + Draws + Losses;
        public bool Surrendered { get; private set; } // Y, Y2 == 5

        public int CumulativeHistoricCasualties { get; private set; } // P1, P2
        public int CumulativeSimulatedCasualties { get; private set; } // T1, T2
        public int CumulativeHistoricMen { get; private set; } // M3, M4

        private int income; // R1, R2
        private int moneySpent; // Q1, Q2

        private bool IsFirstBattle => income == 0;

        // This battle
        private int historicMen; // M1, M2
        public int HistoricCasualties { get; private set; }

        public int Money { get; private set; } // D(n)
        public int Men { get; private set; } // M5, M6
        public int Inflation { get; private set; } // I1, I2
        public int InflationDisplay => Side == Side.Confederate ? Inflation + 15 : Inflation; // Confederate inflation is shown with 15 added - no idea why!

        private readonly Dictionary<Resource, int> allocations = new(); // F(n), H(n), B(n) for food, salaries, ammunition

        public int Strategy { get; protected set; } // Y1, Y2

        public double Morale => (2.0 * allocations[Resource.Food] * allocations[Resource.Food] + allocations[Resource.Salaries] * allocations[Resource.Salaries]) / (reducedAvailableMen * reducedAvailableMen + 1); // O, O2

        public int Casualties { get; protected set; } // C5, C6
        public int Desertions { get; protected set; } // E, E2
        public int MenLost => Casualties + Desertions;
        public bool AllLost { get; private set; } // U, U2

        private double reducedAvailableMen; // F1

        protected virtual double FractionUnspent => (income - moneySpent) / (income + 1.0);

        public void PrepareBattle(int men, int casualties)
        {
            historicMen = men;
            HistoricCasualties = casualties;
            Inflation = 10 + (Losses - Wins) * 2;
            Money = 100 * (int)(men * (100 - Inflation) / 2000.0 * (1 + FractionUnspent) + 0.5);
            Men = (int)(men * 1 + (CumulativeHistoricCasualties - CumulativeSimulatedCasualties) / (CumulativeHistoricMen + 1.0));
            reducedAvailableMen = men * 5.0 / 6.0;
        }

        public virtual void AllocateResources()
        {
            Console.WriteLine($"{Side} General ---\nHow much do you wish to spend for");
            while (true)
            {
                foreach (Resource resource in Enum.GetValues<Resource>())
                {
                    if (EnterResource(resource))
                        break;
                }
                if (allocations.Values.Sum() <= Money)
                    return;
                Console.WriteLine($"Think again! You have only ${Money}");
            }
        }

        private bool EnterResource(Resource resource)
        {
            while (true)
            {
                Console.WriteLine($" - {resource}");
                switch ((int.TryParse(Console.ReadLine(), out int val), val))
                {
                    case (false, _):
                        Console.WriteLine("Not a valid number");
                        break;
                    case (_, < 0):
                        Console.WriteLine("Negative values not allowed");
                        break;
                    case (_, 0) when IsFirstBattle:
                        Console.WriteLine("No previous entries");
                        break;
                    case (_, 0):
                        Console.WriteLine("Assume you want to keep same allocations");
                        return true;
                    case (_, > 0):
                        allocations[resource] = val;
                        return false;
                }
            }
        }

        public virtual void DisplayMorale()
        {
            Console.WriteLine($"{Side} morale is {Morale switch { < 5 => "Poor", < 10 => "Fair", _ => "High" }}");
        }

        public virtual bool ChooseStrategy(bool isReplay) => EnterStrategy(true, "(1-5)");

        protected bool EnterStrategy(bool canSurrender, string hint)
        {
            Console.WriteLine($"{Side} strategy {hint}");
            while (true)
            {
                switch ((int.TryParse(Console.ReadLine(), out int val), val))
                {
                    case (false, _):
                        Console.WriteLine("Not a valid number");
                        break;
                    case (_, 5) when canSurrender:
                        Surrendered = true;
                        Console.WriteLine($"The {Side} general has surrendered");
                        return true;
                    case (_, < 1 or >= 5):
                        Console.WriteLine($"Strategy {val} not allowed.");
                        break;
                    default:
                        Strategy = val;
                        return false;
                }
            }
        }

        public virtual void CalculateLosses(Army opponent)
        {
            AllLost = false;
            int stratFactor = 2 * (Math.Abs(Strategy - opponent.Strategy) + 1);
            Casualties = (int)Math.Round(HistoricCasualties * 0.4 * (1 + 1.0 / stratFactor) * (1 + 1 / Morale) * (1.28 + reducedAvailableMen / (allocations[Resource.Ammunition] + 1)));
            Desertions = (int)Math.Round(100 / Morale);

            // If losses > men present, rescale losses
            if (MenLost > Men)
            {
                Casualties = 13 * Men / 20;
                Desertions = Men - Casualties;
                AllLost = true;
            }
        }

        public void RecordResult(Side winner)
        {
            if (winner == Side)
                Wins++;
            else if (winner == Side.Both)
                Draws++;
            else
                Losses++;

            CumulativeSimulatedCasualties += MenLost;
            CumulativeHistoricCasualties += HistoricCasualties;
            moneySpent += allocations.Values.Sum();
            income += historicMen * (100 - Inflation) / 20;
            CumulativeHistoricMen += historicMen;

            LearnStrategy();
        }

        protected virtual void LearnStrategy() { }

        public void DisplayWarResult(Army opponent)
        {
            Console.WriteLine("\n\n\n\n");
            Console.WriteLine($"The {Side} general has won {Wins} battles and lost {Losses}");
            Side winner = (Surrendered, opponent.Surrendered, Wins < Losses) switch
            {
                (_, true, _) => Side,
                (true, _, _) or (_, _, true) => opponent.Side,
                _ => Side
            };
            Console.WriteLine($"The {winner} general has won the war\n");
        }

        public virtual void DisplayStrategies() { }
    }

    class ComputerArmy : Army
    {
        public int[] StrategyProb { get; } = { 25, 25, 25, 25 }; // S(n)
        private readonly Random strategyRng = new();

        public ComputerArmy(Side side) : base(side) { }

        protected override double FractionUnspent => 0.0;

        public override void AllocateResources() { }

        public override void DisplayMorale() { }

        public override bool ChooseStrategy(bool isReplay)
        {
            if (isReplay)
                return EnterStrategy(false, $"(1-4; usually previous {Side} strategy)");

            // Basic code comments say "If actual strategy info is in  data then r-100 is extra weight given to that strategy" but there's no data or code to do it.
            int strategyChosenProb = strategyRng.Next(100); // 0-99
            int sumProbs = 0;
            for (int i = 0; i < 4; i++)
            {
                sumProbs += StrategyProb[i];
                if (strategyChosenProb < sumProbs)
                {
                    Strategy = i + 1;
                    break;
                }
            }
            Console.WriteLine($"{Side} strategy is {Strategy}");
            return false;
        }

        protected override void LearnStrategy()
        {
            // Learn  present strategy, start forgetting old ones
            // - present strategy gains 3 * s, others lose s probability points, unless a strategy falls below 5 %.
            const int s = 3;
            int presentGain = 0;
            for (int i = 0; i < 4; i++)
            {
                if (StrategyProb[i] >= 5)
                {
                    StrategyProb[i] -= s;
                    presentGain += s;
                }
            }
            StrategyProb[Strategy - 1] += presentGain;
        }

        public override void CalculateLosses(Army opponent)
        {
            Casualties = (int)(17.0 * HistoricCasualties * opponent.HistoricCasualties / (opponent.Casualties * 20));
            Desertions = (int)(5 * opponent.Morale);
        }

        public override void DisplayStrategies()
        {
            ConsoleUtils.WriteWordWrap($"\nIntelligence suggests that the {Side} general used strategies 1, 2, 3, 4 in the following percentages:");
            Console.WriteLine(string.Join(", ", StrategyProb));
        }
    }
}

```

# `27_Civil_War/csharp/Battle.cs`

It looks like you have defined a number of battles in a `Historic` class.  Each battle has a `Side` property that indicates which army won and a `Battle` property that is a list of numbers that represent the engagement details of the battle.

You have also included some example battle data, including the names of some of the battles and the anticipated results.

If you could provide more context, I might be able to give you more specific feedback.



```
﻿using System;
using System.Collections.Generic;

namespace CivilWar
{
    public enum Side { Confederate, Union, Both }
    public enum Option { Battle, Replay, Quit }

    public record Battle(string Name, int[] Men, int[] Casualties, Side Offensive, string Description)
    {
        public static readonly List<Battle> Historic = new()
        {
            new("Bull Run", new[] { 18000, 18500 }, new[] { 1967, 2708 }, Side.Union, "July 21, 1861.  Gen. Beauregard, commanding the south, met Union forces with Gen. McDowell in a premature battle at Bull Run. Gen. Jackson helped push back the union attack."),
            new("Shiloh", new[] { 40000, 44894 }, new[] { 10699, 13047 }, Side.Both, "April 6-7, 1862.  The confederate surprise attack at Shiloh failed due to poor organization."),
            new("Seven Days", new[] { 95000, 115000 }, new[] { 20614, 15849 }, Side.Both, "June 25-july 1, 1862.  General Lee (csa) upheld the offensive throughout the battle and forced Gen. McClellan and the union forces away from Richmond."),
            new("Second Bull Run", new[] { 54000, 63000 }, new[] { 10000, 14000 }, Side.Confederate, "Aug 29-30, 1862.  The combined confederate forces under Lee and Jackson drove the union forces back into Washington."),
            new("Antietam", new[] { 40000, 50000 }, new[] { 10000, 12000 }, Side.Both, "Sept 17, 1862.  The south failed to incorporate Maryland into the confederacy."),
            new("Fredericksburg", new[] { 75000, 120000 }, new[] { 5377, 12653 }, Side.Union, "Dec 13, 1862.  The confederacy under Lee successfully repulsed an attack by the union under Gen. Burnside."),
            new("Murfreesboro", new[] { 38000, 45000 }, new[] { 11000, 12000 }, Side.Union, "Dec 31, 1862.  The south under Gen. Bragg won a close battle."),
            new("Chancellorsville", new[] { 32000, 90000 }, new[] { 13000, 17197 }, Side.Confederate, "May 1-6, 1863.  The south had a costly victory and lost one of their outstanding generals, 'stonewall' Jackson."),
            new("Vicksburg", new[] { 50000, 70000 }, new[] { 12000, 19000 }, Side.Union, "July 4, 1863.  Vicksburg was a costly defeat for the south because it gave the union access to the Mississippi."),
            new("Gettysburg", new[] { 72500, 85000 }, new[] { 20000, 23000 }, Side.Both, "July 1-3, 1863.  A southern mistake by Gen. Lee at Gettysburg cost them one of the most crucial battles of the war."),
            new("Chickamauga", new[] { 66000, 60000 }, new[] { 18000, 16000 }, Side.Confederate, "Sept. 15, 1863. Confusion in a forest near Chickamauga led to a costly southern victory."),
            new("Chattanooga", new[] { 37000, 60000 }, new[] { 36700, 5800 }, Side.Confederate, "Nov. 25, 1863. After the south had sieged Gen. Rosencrans’ army for three months, Gen. Grant broke the siege."),
            new("Spotsylvania", new[] { 62000, 110000 }, new[] { 17723, 18000 }, Side.Confederate, "May 5, 1864.  Grant's plan to keep Lee isolated began to fail here, and continued at Cold Harbor and Petersburg."),
            new("Atlanta", new[] { 65000, 100000 }, new[] { 8500, 3700 }, Side.Union, "August, 1864.  Sherman and three veteran armies converged on Atlanta and dealt the death blow to the confederacy."),
        };

        public static (Option, Battle?) SelectBattle()
        {
            Console.WriteLine("\n\n\nWhich battle do you wish to simulate?");
            return int.Parse(Console.ReadLine() ?? "") switch
            {
                0 => (Option.Replay, null),
                >0 and <15 and int n  => (Option.Battle, Historic[n-1]),
                _ => (Option.Quit, null)
            };
        }
    }
}

```

# `27_Civil_War/csharp/ConsoleUtils.cs`

This is a class that represents a table data with a given set of columns and rows. It uses a csv-like format for the data, but you can use different csv separators like `,` or `;` if you want. The class has several methods for working with the data, such as `Read`, `Write`, and `ToList`.

The `Read` method reads the data from a given csv string and returns it as a list of tables. The `Write` method writes the data to a csv string. The `ToList` method returns a list of all the data in each table.

The class also has a `Table` class that represents a table with a given set of columns and rows. You can use this class to create a table with the data and then write it to a csv string using the `Write` method.

The `TableRow` class represents a table row with a given set of columns. It has a name, data, and before and after hooks for each column. You can use this class to format the data in each column and then write it to a csv string using the `Format` method.

The `Table` class also has a `Format` method for formatting the data in each column.

You can use the `Table` class to create a table with the data and then use the `Read` method to read it from a csv string, and the `Write` method to write it to a csv string.


```
﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CivilWar
{
    static class ConsoleUtils
    {
        public static bool InputYesOrNo()
        {
            while (true)
            {
                var answer = Console.ReadLine();
                switch (answer?.ToLower())
                {
                    case "no":
                        return false;
                    case "yes":
                        return true;
                    default:
                        Console.WriteLine("(Answer Yes or No)");
                        break;
                }
            }
        }

        public static void WriteWordWrap(string text)
        {
            var line = new StringBuilder(Console.WindowWidth);
            foreach (var paragraph in text.Split(Environment.NewLine))
            {
                line.Clear();
                foreach (var word in paragraph.Split(' '))
                {
                    if (line.Length + word.Length < Console.WindowWidth)
                    {
                        if (line.Length > 0)
                            line.Append(' ');
                        line.Append(word);
                    }
                    else
                    {
                        Console.WriteLine(line.ToString());
                        line.Clear().Append(word);
                    }
                }
                Console.WriteLine(line.ToString());
            }
        }

        public static void WriteTable<T>(ICollection<T> items, List<TableRow<T>> rows, bool transpose = false)
        {
            int cols = items.Count + 1;
            var content = rows.Select(r => r.Format(items)).ToList();
            if (transpose)
            {
                content = Enumerable.Range(0, cols).Select(col => content.Select(r => r[col]).ToArray()).ToList();
                cols = rows.Count;
            }
            var colWidths = Enumerable.Range(0, cols).Select(col => content.Max(c => c[col].Length)).ToArray();

            foreach (var row in content)
            {
                for (int col = 0; col < cols; col++)
                {
                    var space = new string(' ', colWidths[col] - row[col].Length);
                    row[col] = col == 0 ? row[col] + space : space + row[col]; // left-align first col; right-align other cols
                }
            }

            var sb = new StringBuilder();
            var horizBars = colWidths.Select(w => new string('═', w)).ToArray();

            void OneRow(string[] cells, char before, char between, char after)
            {
                sb.Append(before);
                sb.AppendJoin(between, cells);
                sb.Append(after);
                sb.AppendLine();
            }

            OneRow(horizBars, '╔', '╦', '╗');
            bool first = true;
            foreach (var row in content)
            {
                if (first)
                    first = false;
                else
                    OneRow(horizBars, '╠', '╬', '╣');
                OneRow(row, '║', '║', '║');
            }
            OneRow(horizBars, '╚', '╩', '╝');

            Console.WriteLine(sb.ToString());
        }

        public record TableRow<T>(string Name, Func<T, object> Data, string Before = "", string After = "")
        {
            private string FormatItem(T item) => $" {Before}{Data(item)}{After} ";

            public string[] Format(IEnumerable<T> items) => items.Select(FormatItem).Prepend($" {Name} ").ToArray();
        }
    }
}

```

# `27_Civil_War/csharp/GameOptions.cs`

这段代码是一个C#类，名为“CivilWar.ConsoleUtils”。该类使用了System命名空间，定义了一个名为“GameOptions”的类，其属性为bool类型的“TwoPlayers”和“ShowDescriptions”。

该代码的作用是定义了一个名为“GameOptions”的类，其包含两个属性的布尔值，分别表示是否为两个玩家模式和是否在游戏过程中显示描述。

由于该类使用了“System.Console”命名空间中的“Console”类，所以该类可以将一些Console应用程序内的常用功能如格式化字符串、输出控制等继承过来。


```
﻿using System;

using static CivilWar.ConsoleUtils;

namespace CivilWar
{
    public record GameOptions(bool TwoPlayers, bool ShowDescriptions)
    {
        public static GameOptions Input()
        {
            Console.WriteLine(
@"                          Civil War
               Creative Computing, Morristown, New Jersey


```

这段代码是一个选择界面，询问用户是否想要指令。它是一个关于美国内战模拟的游戏，并提供了四种选择来描述游戏中的防御和进攻策略。

当用户选择防御策略时，它可以选择炮击攻击、防御工事以抵御正面攻击、防御工事以抵御侧翼攻击或撤退。

当用户选择进攻策略时，它可以选择炮击攻击、正面攻击、侧翼攻击或合围。


```
Do you want instructions?");

            const string instructions = @"This is a civil war simulation.
To play type a response when the computer asks.
Remember that all factors are interrelated and that your responses could change history. Facts and figures used are based on the actual occurrence. Most battles tend to result as they did in the civil war, but it all depends on you!!

The object of the game is to win as many battles as possible.

Your choices for defensive strategy are:
        (1) artillery attack
        (2) fortification against frontal attack
        (3) fortification against flanking maneuvers
        (4) falling back
Your choices for offensive strategy are:
        (1) artillery attack
        (2) frontal attack
        (3) flanking maneuvers
        (4) encirclement
```

这段代码是一个文本游戏，玩家需要根据提示输入不同的数字来选择他们想要的游戏内容。以下是代码的作用解释：

1. 首先，通过一个判断语句(条件语句)判断玩家是否想继续游戏，如果玩家想继续游戏，那么将调用一个名为WriteWordWrap的函数，这个函数将用于在屏幕上显示游戏信息。

2. 如果玩家不想继续游戏，那么将显示一个确认对话框，询问玩家是否要退出游戏，如果玩家选择“是”，那么程序将退出游戏。否则，程序将继续执行下面的代码。

3. 在这一段代码中，有两个条件判断语句，第一个判断语句检查是否有两个玩家参加游戏，如果有，那么程序将为两个玩家生成一个包含两个军事统帅的布局，否则程序将生成一个包含单个军事统帅的布局。

4. 第二个条件判断语句是一个简单的输入确认对话框，询问玩家是否想要继续游戏。如果玩家选择“是”，那么程序将继续执行下面的代码否则将结束程序。

5. 最后一个条件判断语句是一个带参数的WriteWordWrap函数调用，用于在屏幕上显示玩家在游戏中选择的选项。


```
You may surrender by typing a '5' for your strategy.";

            if (InputYesOrNo())
                WriteWordWrap(instructions);

            Console.WriteLine("\n\nAre there two generals present?");
            bool twoPlayers = InputYesOrNo();
            if (!twoPlayers)
                Console.WriteLine("\nYou are the confederacy.  Good luck!\n");

            WriteWordWrap(
            @"Select a battle by typing a number from 1 to 14 on request.  Type any other number to end the simulation. But '0' brings back exact previous battle situation allowing you to replay it.

Note: a negative Food$ entry causes the program to use the entries from the previous battle

```

这段代码是一个游戏中的函数，其作用是在玩家请求一场战斗后，询问玩家是否想要查看这场战斗的描述，如果玩家选择“是”，则会返回一个新的 `GameOptions` 类，该类包含两个参数，一个是“两个玩家”，另一个是“是否显示战斗描述”。

`GameOptions` 类可能是一个用于在游戏过程中获取或设置玩家游戏选项的类。

具体的，如果玩家选择“是”，那么游戏将返回一个包含两个参数的实例，其中第一个参数是一个表示两个玩家的布尔值，第二个参数是一个表示是否要显示战斗描述的布尔值。如果玩家选择“否”，则不会返回任何实例。


```
After requesting a battle, do you wish battle descriptions (answer yes or no)");
            bool showDescriptions = InputYesOrNo();

            return new GameOptions(twoPlayers, showDescriptions);
        }
    }
}

```

# `27_Civil_War/csharp/Program.cs`

This appears to be a function written in C# that is used in a game. It takes in data about two armies and calculates the losses for each army in the battle. It then compares the calculated losses to the actual losses reported by the game and outputs this information. If the game is a two-player game, it compares the calculated losses for each army to determine the winner and outputs this information. If the game is not a two-player game, it outputs the actual losses for each army. It appears that this function is used to determine the outcome of a battle in the game.


```
﻿using System;
using System.Collections.Generic;
using System.Linq;
using CivilWar;

var options = GameOptions.Input();
var armies = new List<Army> { new Army(Side.Confederate), options.TwoPlayers ? new Army(Side.Union) : new ComputerArmy(Side.Union) };

Battle? battle = null;
while (OneBattle(ref battle)) { }
DisplayResult();

bool OneBattle(ref Battle? previous)
{
    var (option, selected) = Battle.SelectBattle();
    var (battle, isReplay, quit) = option switch
    {
        Option.Battle => (selected!, false, false),
        Option.Replay when previous != null => (previous, true, false), // can't replay if no previous battle
        _ => (null!, false, true),
    };
    if (quit)
        return false;

    if (!isReplay)
    {
        Console.WriteLine($"This is the battle of {battle.Name}.");
        if (options.ShowDescriptions)
            ConsoleUtils.WriteWordWrap(battle.Description);
        armies.ForEach(a => a.PrepareBattle(battle.Men[(int)a.Side], battle.Casualties[(int)a.Side]));
    }

    ConsoleUtils.WriteTable(armies, new()
    {
        new("", a => a.Side),
        new("Men", a => a.Men),
        new("Money", a => a.Money, Before: "$"),
        new("Inflation", a => a.InflationDisplay, After: "%")
    });

    armies.ForEach(a => a.AllocateResources());
    armies.ForEach(a => a.DisplayMorale());

    string offensive = battle.Offensive switch
    {
        Side.Confederate => "You are on the offensive",
        Side.Union => "You are on the defensive",
        _ => "Both sides are on the offensive"
    };
    Console.WriteLine($"Confederate general---{offensive}");

    if (armies.Any(a => a.ChooseStrategy(isReplay)))
    {
        return false; // someone surrendered
    }
    armies[0].CalculateLosses(armies[1]);
    armies[1].CalculateLosses(armies[0]);

    ConsoleUtils.WriteTable(armies, new()
    {
        new("", a => a.Side),
        new("Casualties", a => a.Casualties),
        new("Desertions", a => a.Desertions),
    });
    if (options.TwoPlayers)
    {
        var oneDataCol = new[] { 1 };
        Console.WriteLine($"Compared to the actual casualties at {battle.Name}");
        ConsoleUtils.WriteTable(oneDataCol, armies.Select(a => new ConsoleUtils.TableRow<int>(
            a.Side.ToString(),
            _ => $"{(double)a.Casualties / battle.Casualties[(int)a.Side]}", After: "% of the original")
        ).ToList());
    }

    Side winner;
    switch (armies[0].AllLost, armies[1].AllLost, armies[0].MenLost - armies[1].MenLost)
    {
        case (true, true, _) or (false, false, 0):
            Console.WriteLine("Battle outcome unresolved");
            winner = Side.Both; // Draw
            break;
        case (false, true, _) or (false, false, < 0):
            Console.WriteLine($"The Confederacy wins {battle.Name}");
            winner = Side.Confederate;
            break;
        case (true, false, _) or (false, false, > 0):
            Console.WriteLine($"The Union wins {battle.Name}");
            winner = Side.Union;
            break;
    }
    if (!isReplay)
    {
        armies.ForEach(a => a.RecordResult(winner));
    }
    Console.WriteLine("---------------");
    previous = battle;
    return true;
}

```

该代码的目的是在二维数组`armies`中存储所有可能的战争结果，并输出战争结果。

首先，该代码使用`DisplayWarResult()`函数来输出战争结果。

然后，该代码创建了一个名为`armies[0]`的引用，该引用代表第一支军队。接着，使用`DisplayStrategies()`函数来输出这支军队的战略。

接下来，该代码使用`armies[1]`引用来代表第二支军队。但是，由于没有提供`DisplayStrategies()`函数的实现，因此无法输出这支军队的战略。

最后，该代码通过循环遍历所有可能的战争结果，并输出每个结果。

除了循环输出每个结果外，该代码还计算了模拟战争损失中和历史上实际战争损失中的累积值。


```
void DisplayResult()
{
    armies[0].DisplayWarResult(armies[1]);

    int battles = armies[0].BattlesFought;
    if (battles > 0)
    {
        Console.WriteLine($"For the {battles} battles fought (excluding reruns)");

        ConsoleUtils.WriteTable(armies, new()
        {
            new("", a => a.Side),
            new("Historical Losses", a => a.CumulativeHistoricCasualties),
            new("Simulated Losses", a => a.CumulativeSimulatedCasualties),
            new("  % of original", a => ((double)a.CumulativeSimulatedCasualties / a.CumulativeHistoricCasualties).ToString("p2"))
        }, transpose: true);

        armies[1].DisplayStrategies();
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `27_Civil_War/java/src/CivilWar.java`

This appears to be a Java program that simulates a hypothetical World War II battle between two factions (the "Army" and the "Confederate Army"). It uses a system of OOD programming, including encapsulation, inheritance, and polymorphism.

The program includes several main classes:

* `Army`, which represents the hypothetical Army, with an implementation of the `ArmyPair` and `BattleResults` interfaces. It has a `ConfederateParty` that is responsible for representing the Confederate Army.
* `ConfederateParty`, which implements the `HistoricalDatum` interface, and contains an instance of the `Army` class.
* `Battle` and `BattleResults`, which are utility classes that manage the simulation of the battle.
* `ArmyResources`, which is an implementation of the `Army` interface, and contains classes for managing resources (such as food, salaries, andammunition) in the Army.
* `ArmyPair`, which is an implementation of the `ArmyPair` interface, and contains instances of the `ConfederateParty` and the `Army` classes.
* `HistoricalDatum`, which is an abstract class that defines the `HistoricalDatum` interface, and is used to represent the data that is stored in the `ArmyResources` class.
* `LegacyAnimalModels`, which is an interface that defines the `LegacyAnimalModels` interface. This class is used to represent the LegacyAnimalModels class, which is a legacy class that is being deprecated and will be removed in a future version of the program.

The `Battle` class includes several methods that are responsible for simulating the battle, including the methods that determine the outcome of the battle, such as `attemptToInitiateAttack()` and `assessRoutineCover()`.

The `BattleResults` class is responsible for storing the results of the battle, including the number of confederate and union losses, and the number of indeterminate results.

The `Army` class has several instance variables that are responsible for representing the different aspects of the Army, such as the ConfederateParty instance, and the ArmyResources instance. It also


```
import java.io.PrintStream;
import java.util.InputMismatchException;
import java.util.List;
import java.util.Scanner;
import java.util.function.Function;
import java.util.function.Predicate;

import static java.util.stream.Collectors.joining;
import static java.util.stream.IntStream.range;

@SuppressWarnings("SpellCheckingInspection")
public class CivilWar {

    private final PrintStream out;
    private final List<HistoricalDatum> data;

    private final BattleResults results;

    private BattleState currentBattle;
    private int numGenerals;
    private int battleNumber;
    private boolean wantBattleDescriptions;
    private final int[] strategies;

    private int confedStrategy;
    private int unionStrategy;

    private final ArmyPair<ArmyResources> resources;
    private final ArmyPair<Integer> totalExpectedCasualties;
    private final ArmyPair<Integer> totalCasualties;
    private final ArmyPair<Integer> revenue;
    private final ArmyPair<Integer> inflation;
    private final ArmyPair<Integer> totalExpenditure;
    private final ArmyPair<Integer> totalTroops;

    private boolean excessiveConfederateLosses;
    private boolean excessiveUnionLosses;

    private boolean confedSurrender;
    private boolean unionSurrender;

    private final static String YES_NO_REMINDER = "(ANSWER YES OR NO)";
    private final static Predicate<String> YES_NO_CHECKER = i -> isYes(i) || isNo(i);

    /**
     * ORIGINAL GAME DESIGN: CRAM, GOODIE, HIBBARD LEXINGTON H.S.
     * MODIFICATIONS: G. PAUL, R. HESS (TIES), 1973
     */
    public static void main(String[] args) {
        var x = new CivilWar(System.out);
        x.showCredits();

        // LET D=RND(-1) ???

        System.out.print("DO YOU WANT INSTRUCTIONS? ");

        if (isYes(inputString(YES_NO_CHECKER, YES_NO_REMINDER))) {
            x.showHelp();
        }

        x.gameLoop();
    }

    private void gameLoop() {
        out.println();
        out.println();
        out.println();
        out.print("ARE THERE TWO GENERALS PRESENT (ANSWER YES OR NO)? ");

        if (isYes(inputString(YES_NO_CHECKER, YES_NO_REMINDER))) {
            this.numGenerals = 2;
        } else {
            this.numGenerals = 1;
            out.println();
            out.println("YOU ARE THE CONFEDERACY.   GOOD LUCK!");
            out.println();
        }

        out.println("SELECT A BATTLE BY TYPING A NUMBER FROM 1 TO 14 ON");
        out.println("REQUEST.  TYPE ANY OTHER NUMBER TO END THE SIMULATION.");
        out.println("BUT '0' BRINGS BACK EXACT PREVIOUS BATTLE SITUATION");
        out.println("ALLOWING YOU TO REPLAY IT");
        out.println();
        out.println("NOTE: A NEGATIVE FOOD$ ENTRY CAUSES THE PROGRAM TO ");
        out.println("USE THE ENTRIES FROM THE PREVIOUS BATTLE");
        out.println();

        out.print("AFTER REQUESTING A BATTLE, DO YOU WISH BATTLE DESCRIPTIONS (ANSWER YES OR NO)? ");

        this.wantBattleDescriptions = isYes(inputString(YES_NO_CHECKER, YES_NO_REMINDER));

        while (true) {
            var battle = startBattle();
            if (battle == null) {
                break;
            }

            this.currentBattle = battle;

            offensiveLogic(battle.data);

            calcLosses(battle);

            reset();

            if (this.confedSurrender) {
                out.println("THE CONFEDERACY HAS SURRENDERED");
            } else if (unionSurrender) {  // FIXME Is this actually possible? 2850
                out.println("THE UNION HAS SURRENDERED.");
            }
        }

        complete();
    }

    private BattleState startBattle() {
        out.println();
        out.println();
        out.println();
        out.print("WHICH BATTLE DO YOU WISH TO SIMULATE ? ");

        var battleNumber = inputInt(i -> i >= 1 || (i == 0 && this.currentBattle != null), i -> "BATTLE " + i + " NOT ALLOWED.");

        if (battleNumber == 0) {
            out.println(this.currentBattle.data.name + " INSTANT REPLAY");
            return this.currentBattle;
        }

        if (battleNumber > this.data.size()) {  // TYPE ANY OTHER NUMBER TO END THE SIMULATION
            return null;
        }

        this.battleNumber = battleNumber;

        var battle = this.data.get(this.battleNumber - 1);
        var battleState = new BattleState(battle);


        excessiveConfederateLosses = false;

        // INFLATION CALC
        // REM - ONLY IN PRINTOUT IS CONFED INFLATION = I1+15%
        inflation.confederate = 10 + (results.union - results.confederate) * 2;
        inflation.union = 10 + (results.confederate - results.union) * 2;

        // MONEY AVAILABLE
        resources.confederate.budget = 100 * (int) Math.floor((battle.troops.confederate * (100.0 - inflation.confederate) / 2000) * (1 + (revenue.confederate - totalExpenditure.confederate) / (revenue.confederate + 1.0)) + .5);

        // MEN AVAILABLE
        battleState.F1 = 5 * battle.troops.confederate / 6.0;

        if (this.numGenerals == 2) {
            resources.union.budget = 100 * (int) Math.floor((battle.troops.union * (100.0 - inflation.union) / 2000) * (1 + (revenue.union - totalExpenditure.union) / (revenue.union + 1.0)) + .5);
        } else {
            resources.union.budget = 100 * (int) Math.floor(battle.troops.union * (100.0 - inflation.union) / 2000 + .5);
        }

        out.println();
        out.println();
        out.println();
        out.println();
        out.println();
        out.println("THIS IS THE BATTLE OF " + battle.name);

        if (this.wantBattleDescriptions) {
            for (var eachLine : battle.blurb) {
                out.println(eachLine);
            }
        }

        out.println();
        out.println("          CONFEDERACY     UNION");
        out.println("MEN         " + getConfedTroops(battle) + "          " + getUnionTroops(battle));
        out.println("MONEY     $ " + resources.confederate.budget + "       $ " + resources.union.budget);
        out.println("INFLATION   " + (inflation.confederate + 15) + "%          " + inflation.union + "%");

        // ONLY IN PRINTOUT IS CONFED INFLATION = I1+15%
        // IF TWO GENERALS, INPUT CONFED. FIRST

        var terminalInput = new Scanner(System.in);

        for (int i = 0; i < numGenerals; i++) {
            out.println();

            ArmyResources currentResources;

            if (this.numGenerals == 1 || i == 0) {
                out.print("CONFEDERATE GENERAL --- ");
                currentResources = resources.confederate;
            } else {
                out.print("UNION GENERAL --- ");
                currentResources = resources.union;
            }

            var validInputs = false;
            while (!validInputs) {
                out.println("HOW MUCH DO YOU WISH TO SPEND FOR");
                out.print("- FOOD...... ? ");
                var food = terminalInput.nextInt();
                if (food == 0) {
                    if (this.revenue.confederate != 0) {
                        out.println("ASSUME YOU WANT TO KEEP SAME ALLOCATIONS");
                        out.println();
                    }
                } else {
                    currentResources.food = food;
                }

                out.print("- SALARIES.. ? ");
                currentResources.salaries = terminalInput.nextInt();

                out.print("- AMMUNITION ? ");
                currentResources.ammunition = terminalInput.nextInt();  // FIXME Retry if -ve

                if (currentResources.getTotal() > currentResources.budget) {
                    out.println("THINK AGAIN! YOU HAVE ONLY $" + currentResources.budget);
                } else {
                    validInputs = true;
                }
            }
        }

        out.println();

        // Record Morale
        out.println(range(0, numGenerals).mapToObj(i -> moraleForArmy(battleState, i)).collect(joining(", ")));

        out.println();

        return battleState;
    }

    private int getUnionTroops(HistoricalDatum battle) {
        return (int) Math.floor(battle.troops.union * (1 + (totalExpectedCasualties.union - totalCasualties.union) / (totalTroops.union + 1.0)));
    }

    private int getConfedTroops(HistoricalDatum battle) {
        return (int) Math.floor(battle.troops.confederate * (1 + (totalExpectedCasualties.confederate - totalCasualties.confederate) / (totalTroops.confederate + 1.0)));
    }

    private String moraleForArmy(BattleState battleState, int armyIdx) {
        var builder = new StringBuilder();

        ArmyResources currentResources;

        if (this.numGenerals == 1 || armyIdx == 0) {
            builder.append("CONFEDERATE ");
            currentResources = resources.confederate;
        } else {
            builder.append("UNION ");
            currentResources = resources.union;
        }

        // FIND MORALE
        currentResources.morale = (2 * Math.pow(currentResources.food, 2) + Math.pow(currentResources.salaries, 2)) / Math.pow(battleState.F1, 2) + 1;
        if (currentResources.morale >= 10) {
            builder.append("MORALE IS HIGH");
        } else if (currentResources.morale >= 5) {
            builder.append("MORALE IS FAIR");
        } else {
            builder.append("MORALE IS POOR");
        }

        return builder.toString();
    }

    private enum OffensiveStatus {
        DEFENSIVE("YOU ARE ON THE DEFENSIVE"), OFFENSIVE("YOU ARE ON THE OFFENSIVE"), BOTH_OFFENSIVE("BOTH SIDES ARE ON THE OFFENSIVE");

        private final String label;

        OffensiveStatus(String label) {
            this.label = label;
        }
    }

    private void offensiveLogic(HistoricalDatum battle) {
        out.print("CONFEDERATE GENERAL---");
        // ACTUAL OFF/DEF BATTLE SITUATION
        out.println(battle.offensiveStatus.label);

        // CHOOSE STRATEGIES

        if (numGenerals == 2) {
            out.print("CONFEDERATE STRATEGY ? ");
        } else {
            out.print("YOUR STRATEGY ? ");
        }

        confedStrategy = inputInt(i -> i >= 1 && i <= 5, i -> "STRATEGY " + i + " NOT ALLOWED.");
        if (confedStrategy == 5) {  // 1970
            confedSurrender = true;
        }

        if (numGenerals == 2) {
            out.print("UNION STRATEGY ? ");

            unionStrategy = inputInt(i -> i >= 1 && i <= 5, i -> "STRATEGY " + i + " NOT ALLOWED.");
            if (unionStrategy == 5) {  // 1970
                unionSurrender = true;
            }
        } else {
            unionStrategy();
        }
    }

    // 2070  REM : SIMULATED LOSSES-NORTH
    private UnionLosses simulateUnionLosses(HistoricalDatum battle) {
        var losses = (2.0 * battle.expectedCasualties.union / 5) * (1 + 1.0 / (2 * (Math.abs(unionStrategy - confedStrategy) + 1)));
        losses = losses * (1.28 + (5.0 * battle.troops.union / 6) / (resources.union.ammunition + 1));
        losses = Math.floor(losses * (1 + 1 / resources.union.morale) + 0.5);
        // IF LOSS > MEN PRESENT, RESCALE LOSSES
        var moraleFactor = 100 / resources.union.morale;

        if (Math.floor(losses + moraleFactor) >= getUnionTroops(battle)) {
            losses = Math.floor(13.0 * getUnionTroops(battle) / 20);
            moraleFactor = 7 * losses / 13;
            excessiveUnionLosses = true;
        }

        return new UnionLosses((int) losses, (int) Math.floor(moraleFactor));
    }

    // 2170: CALCULATE SIMULATED LOSSES
    private void calcLosses(BattleState battle) {
        // 2190
        out.println();
        out.println("            CONFEDERACY    UNION");

        var C5 = (2 * battle.data.expectedCasualties.confederate / 5) * (1 + 1.0 / (2 * (Math.abs(unionStrategy - confedStrategy) + 1)));
        C5 = (int) Math.floor(C5 * (1 + 1.0 / resources.confederate.morale) * (1.28 + battle.F1 / (resources.confederate.ammunition + 1.0)) + .5);
        var E = 100 / resources.confederate.morale;

        if (C5 + 100 / resources.confederate.morale >= battle.data.troops.confederate * (1 + (totalExpectedCasualties.confederate - totalCasualties.confederate) / (totalTroops.confederate + 1.0))) {
            C5 = (int) Math.floor(13.0 * battle.data.troops.confederate / 20 * (1 + (totalExpectedCasualties.union - totalCasualties.confederate) / (totalTroops.confederate + 1.0)));
            E = 7 * C5 / 13.0;
            excessiveConfederateLosses = true;
        }

        /////  2270

        final UnionLosses unionLosses;

        if (this.numGenerals == 1) {
            unionLosses = new UnionLosses((int) Math.floor(17.0 * battle.data.expectedCasualties.union * battle.data.expectedCasualties.confederate / (C5 * 20)), (int) Math.floor(5 * resources.confederate.morale));
        } else {
            unionLosses = simulateUnionLosses(battle.data);
        }

        out.println("CASUALTIES:  " + rightAlignInt(C5) + "        " + rightAlignInt(unionLosses.losses));
        out.println("DESERTIONS:  " + rightAlignInt(E) + "        " + rightAlignInt(unionLosses.desertions));
        out.println();

        if (numGenerals == 2) {
            out.println("COMPARED TO THE ACTUAL CASUALTIES AT " + battle.data.name);
            out.println("CONFEDERATE: " + (int) Math.floor(100 * (C5 / (double) battle.data.expectedCasualties.confederate) + 0.5) + " % OF THE ORIGINAL");
            out.println("UNION:       " + (int) Math.floor(100 * (unionLosses.losses / (double) battle.data.expectedCasualties.union) + 0.5) + " % OF THE ORIGINAL");

            out.println();

            // REM - 1 WHO WON
            var winner = findWinner(C5 + E, unionLosses.losses + unionLosses.desertions);
            switch (winner) {
                case UNION -> {
                    out.println("THE UNION WINS " + battle.data.name);
                    results.union++;
                }
                case CONFED -> {
                    out.println("THE CONFEDERACY WINS " + battle.data.name);
                    results.confederate++;
                }
                case INDECISIVE -> {
                    out.println("BATTLE OUTCOME UNRESOLVED");
                    results.indeterminate++;
                }
            }
        } else {
            out.println("YOUR CASUALTIES WERE " + Math.floor(100 * (C5 / (double) battle.data.expectedCasualties.confederate) + 0.5) + "% OF THE ACTUAL CASUALTIES AT " + battle.data.name);

            // FIND WHO WON

            if (excessiveConfederateLosses) {
                out.println("YOU LOSE " + battle.data.name);

                if (this.battleNumber != 0) {
                    results.union++;
                }
            } else {
                out.println("YOU WIN " + battle.data.name);
                // CUMULATIVE BATTLE FACTORS WHICH ALTER HISTORICAL RESOURCES AVAILABLE.IF A REPLAY DON'T UPDATE.
                results.confederate++;
            }
        }

        if (this.battleNumber != 0) {
            totalCasualties.confederate += (int) (C5 + E);
            totalCasualties.union += unionLosses.losses + unionLosses.desertions;
            totalExpectedCasualties.confederate += battle.data.expectedCasualties.confederate;
            totalExpectedCasualties.union += battle.data.expectedCasualties.union;
            totalExpenditure.confederate += resources.confederate.getTotal();
            totalExpenditure.union += resources.union.getTotal();
            revenue.confederate += battle.data.troops.confederate * (100 - inflation.confederate) / 20;
            revenue.union += battle.data.troops.union * (100 - inflation.union) / 20;
            totalTroops.confederate += battle.data.troops.confederate;
            totalTroops.union += battle.data.troops.union;

            updateStrategies(this.confedStrategy);
        }
    }

    // 2790
    private void reset() {
        excessiveConfederateLosses = excessiveUnionLosses = false;

        out.println("---------------");
    }

    // 2820  REM------FINISH OFF
    private void complete() {
        out.println();
        out.println();
        out.println();
        out.println();
        out.println();
        out.println();
        out.println("THE CONFEDERACY HAS WON " + results.confederate + " BATTLES AND LOST " + results.union);

        if (this.unionStrategy == 5) {
            out.println("THE CONFEDERACY HAS WON THE WAR");
        }

        if (this.confedStrategy == 5 || results.confederate <= results.union) {
            out.println("THE UNION HAS WON THE WAR");
        }

        out.println();

        // FIXME 2960  IF R1=0 THEN 3100

        out.println("FOR THE " + results.getTotal() + " BATTLES FOUGHT (EXCLUDING RERUNS)");
        out.println("                       CONFEDERACY    UNION");
        out.println("HISTORICAL LOSSES      " + (int) Math.floor(totalExpectedCasualties.confederate + .5) + "          " + (int) Math.floor(totalExpectedCasualties.union + .5));
        out.println("SIMULATED LOSSES       " + (int) Math.floor(totalCasualties.confederate + .5) + "          " + (int) Math.floor(totalCasualties.union + .5));
        out.println();
        out.println("    % OF ORIGINAL      " + (int) Math.floor(100 * ((double) totalCasualties.confederate / totalExpectedCasualties.confederate) + .5) + "             " + (int) Math.floor(100 * ((double) totalCasualties.union / totalExpectedCasualties.union) + .5));

        if (this.numGenerals == 1) {
            out.println();
            out.println("UNION INTELLIGENCE SUGGESTS THAT THE SOUTH USED ");
            out.println("STRATEGIES 1, 2, 3, 4 IN THE FOLLOWING PERCENTAGES");
            out.println(this.strategies[0] + "," + this.strategies[1] + "," + this.strategies[2] + "," + this.strategies[3]);
        }
    }

    private Winner findWinner(double confLosses, double unionLosses) {
        if (this.excessiveConfederateLosses && this.excessiveUnionLosses) {
            return Winner.INDECISIVE;
        }

        if (this.excessiveConfederateLosses) {
            return Winner.UNION;
        }

        if (this.excessiveUnionLosses || confLosses < unionLosses) {
            return Winner.CONFED;
        }

        if (confLosses == unionLosses) {
            return Winner.INDECISIVE;
        }

        return Winner.UNION;  // FIXME Really? 2400-2420 ?
    }

    private enum Winner {
        CONFED, UNION, INDECISIVE
    }

    private void unionStrategy() {
        // 3130 ... so you can only input / override Union strategy on re-run??
        if (this.battleNumber == 0) {
            out.print("UNION STRATEGY ? ");
            var terminalInput = new Scanner(System.in);
            unionStrategy = terminalInput.nextInt();
            if (unionStrategy < 0) {
                out.println("ENTER 1, 2, 3, OR 4 (USUALLY PREVIOUS UNION STRATEGY)");
                // FIXME Retry Y2 input !!!
            }

            if (unionStrategy < 5) {  // 3155
                return;
            }
        }

        var S0 = 0;
        var r = 100 * Math.random();

        for (unionStrategy = 1; unionStrategy <= 4; unionStrategy++) {
            S0 += this.strategies[unionStrategy - 1];
            // IF ACTUAL STRATEGY INFO IS IN PROGRAM DATA STATEMENTS THEN R-100 IS EXTRA WEIGHT GIVEN TO THAT STATEGY.
            if (r < S0) {
                break;
            }
        }
        // IF ACTUAL STRAT. IN,THEN HERE IS Y2= HIST. STRAT.
        out.println("UNION STRATEGY IS " + unionStrategy);
    }

    public CivilWar(PrintStream out) {
        this.out = out;

        this.results = new BattleResults();

        this.totalCasualties = new ArmyPair<>(0, 0);
        this.totalExpectedCasualties = new ArmyPair<>(0, 0);
        this.totalExpenditure = new ArmyPair<>(0, 0);
        this.totalTroops = new ArmyPair<>(0, 0);

        this.revenue = new ArmyPair<>(0, 0);
        this.inflation = new ArmyPair<>(0, 0);

        this.resources = new ArmyPair<>(new ArmyResources(), new ArmyResources());

        // UNION INFO ON LIKELY CONFEDERATE STRATEGY
        this.strategies = new int[]{25, 25, 25, 25};

        // READ HISTORICAL DATA.
        // HISTORICAL DATA...CAN ADD MORE (STRAT.,ETC) BY INSERTING DATA STATEMENTS AFTER APPRO. INFO, AND ADJUSTING READ
        this.data = List.of(
                new HistoricalDatum("BULL RUN", new ArmyPair<>(18000, 18500), new ArmyPair<>(1967, 2708), OffensiveStatus.DEFENSIVE, new String[]{"JULY 21, 1861.  GEN. BEAUREGARD, COMMANDING THE SOUTH, MET", "UNION FORCES WITH GEN. MCDOWELL IN A PREMATURE BATTLE AT", "BULL RUN. GEN. JACKSON HELPED PUSH BACK THE UNION ATTACK."}),
                new HistoricalDatum("SHILOH", new ArmyPair<>(40000, 44894), new ArmyPair<>(10699, 13047), OffensiveStatus.OFFENSIVE, new String[]{"APRIL 6-7, 1862.  THE CONFEDERATE SURPRISE ATTACK AT", "SHILOH FAILED DUE TO POOR ORGANIZATION."}),
                new HistoricalDatum("SEVEN DAYS", new ArmyPair<>(95000, 115000), new ArmyPair<>(20614, 15849), OffensiveStatus.OFFENSIVE, new String[]{"JUNE 25-JULY 1, 1862.  GENERAL LEE (CSA) UPHELD THE", "OFFENSIVE THROUGHOUT THE BATTLE AND FORCED GEN. MCCLELLAN", "AND THE UNION FORCES AWAY FROM RICHMOND."}),
                new HistoricalDatum("SECOND BULL RUN", new ArmyPair<>(54000, 63000), new ArmyPair<>(10000, 14000), OffensiveStatus.BOTH_OFFENSIVE, new String[]{"AUG 29-30, 1862.  THE COMBINED CONFEDERATE FORCES UNDER", " LEE", "AND JACKSON DROVE THE UNION FORCES BACK INTO WASHINGTON."}),
                new HistoricalDatum("ANTIETAM", new ArmyPair<>(40000, 50000), new ArmyPair<>(10000, 12000), OffensiveStatus.OFFENSIVE, new String[]{"SEPT 17, 1862.  THE SOUTH FAILED TO INCORPORATE MARYLAND", "INTO THE CONFEDERACY."}),
                new HistoricalDatum("FREDERICKSBURG", new ArmyPair<>(75000, 120000), new ArmyPair<>(5377, 12653), OffensiveStatus.DEFENSIVE, new String[]{"DEC 13, 1862.  THE CONFEDERACY UNDER LEE SUCCESSFULLY", "REPULSED AN ATTACK BY THE UNION UNDER GEN. BURNSIDE."}),
                new HistoricalDatum("MURFREESBORO", new ArmyPair<>(38000, 45000), new ArmyPair<>(11000, 12000), OffensiveStatus.DEFENSIVE, new String[]{"DEC 31, 1862.  THE SOUTH UNDER GEN. BRAGG WON A CLOSE BATTLE."}),
                new HistoricalDatum("CHANCELLORSVILLE", new ArmyPair<>(32000, 90000), new ArmyPair<>(13000, 17197), OffensiveStatus.BOTH_OFFENSIVE, new String[]{"MAY 1-6, 1863.  THE SOUTH HAD A COSTLY VICTORY AND LOST", "ONE OF THEIR OUTSTANDING GENERALS, 'STONEWALL' JACKSON."}),
                new HistoricalDatum("VICKSBURG", new ArmyPair<>(50000, 70000), new ArmyPair<>(12000, 19000), OffensiveStatus.DEFENSIVE, new String[]{"JULY 4, 1863.  VICKSBURG WAS A COSTLY DEFEAT FOR THE SOUTH", "BECAUSE IT GAVE THE UNION ACCESS TO THE MISSISSIPPI."}),
                new HistoricalDatum("GETTYSBURG", new ArmyPair<>(72500, 85000), new ArmyPair<>(20000, 23000), OffensiveStatus.OFFENSIVE, new String[]{"JULY 1-3, 1863.  A SOUTHERN MISTAKE BY GEN. LEE AT GETTYSBURG", "COST THEM ONE OF THE MOST CRUCIAL BATTLES OF THE WAR."}),
                new HistoricalDatum("CHICKAMAUGA", new ArmyPair<>(66000, 60000), new ArmyPair<>(18000, 16000), OffensiveStatus.BOTH_OFFENSIVE, new String[]{"SEPT. 15, 1863. CONFUSION IN A FOREST NEAR CHICKAMAUGA LED", "TO A COSTLY SOUTHERN VICTORY."}),
                new HistoricalDatum("CHATTANOOGA", new ArmyPair<>(37000, 60000), new ArmyPair<>(36700, 5800), OffensiveStatus.BOTH_OFFENSIVE, new String[]{"NOV. 25, 1863. AFTER THE SOUTH HAD SIEGED GEN. ROSENCRANS'", "ARMY FOR THREE MONTHS, GEN. GRANT BROKE THE SIEGE."}),
                new HistoricalDatum("SPOTSYLVANIA", new ArmyPair<>(62000, 110000), new ArmyPair<>(17723, 18000), OffensiveStatus.BOTH_OFFENSIVE, new String[]{"MAY 5, 1864.  GRANT'S PLAN TO KEEP LEE ISOLATED BEGAN TO", "FAIL HERE, AND CONTINUED AT COLD HARBOR AND PETERSBURG."}),
                new HistoricalDatum("ATLANTA", new ArmyPair<>(65000, 100000), new ArmyPair<>(8500, 3700), OffensiveStatus.DEFENSIVE, new String[]{"AUGUST, 1864.  SHERMAN AND THREE VETERAN ARMIES CONVERGED", "ON ATLANTA AND DEALT THE DEATH BLOW TO THE CONFEDERACY."})
        );
    }

    private void showCredits() {
        out.println(" ".repeat(26) + "CIVIL WAR");
        out.println(" ".repeat(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        out.println();
        out.println();
        out.println();
    }

    private void updateStrategies(int strategy) {
        // REM LEARN  PRESENT STRATEGY, START FORGETTING OLD ONES
        // REM - PRESENT STRATEGY OF SOUTH GAINS 3*S, OTHERS LOSE S
        // REM   PROBABILITY POINTS, UNLESS A STRATEGY FALLS BELOW 5%.

        var S = 3;
        var S0 = 0;
        for (int i = 0; i < 4; i++) {
            if (this.strategies[i] <= 5) {
                continue;
            }

            this.strategies[i] -= S;
            S0 += S;

        }
        this.strategies[strategy - 1] += S0;

    }

    private void showHelp() {
        out.println();
        out.println();
        out.println();
        out.println();
        out.println("THIS IS A CIVIL WAR SIMULATION.");
        out.println("TO PLAY TYPE A RESPONSE WHEN THE COMPUTER ASKS.");
        out.println("REMEMBER THAT ALL FACTORS ARE INTERRELATED AND THAT YOUR");
        out.println("RESPONSES COULD CHANGE HISTORY. FACTS AND FIGURES USED ARE");
        out.println("BASED ON THE ACTUAL OCCURRENCE. MOST BATTLES TEND TO RESULT");
        out.println("AS THEY DID IN THE CIVIL WAR, BUT IT ALL DEPENDS ON YOU!!");
        out.println();
        out.println("THE OBJECT OF THE GAME IS TO WIN AS MANY BATTLES AS ");
        out.println("POSSIBLE.");
        out.println();
        out.println("YOUR CHOICES FOR DEFENSIVE STRATEGY ARE:");
        out.println("        (1) ARTILLERY ATTACK");
        out.println("        (2) FORTIFICATION AGAINST FRONTAL ATTACK");
        out.println("        (3) FORTIFICATION AGAINST FLANKING MANEUVERS");
        out.println("        (4) FALLING BACK");
        out.println(" YOUR CHOICES FOR OFFENSIVE STRATEGY ARE:");
        out.println("        (1) ARTILLERY ATTACK");
        out.println("        (2) FRONTAL ATTACK");
        out.println("        (3) FLANKING MANEUVERS");
        out.println("        (4) ENCIRCLEMENT");
        out.println("YOU MAY SURRENDER BY TYPING A '5' FOR YOUR STRATEGY.");
    }

    private static final int MAX_NUM_LENGTH = 6;

    private String rightAlignInt(int number) {
        var s = String.valueOf(number);
        return " ".repeat(MAX_NUM_LENGTH - s.length()) + s;
    }

    private String rightAlignInt(double number) {
        return rightAlignInt((int) Math.floor(number));
    }

    private static String inputString(Predicate<String> validator, String reminder) {
        while (true) {
            try {
                var input = new Scanner(System.in).nextLine();
                if (validator.test(input)) {
                    return input;
                }
            } catch (InputMismatchException e) {
                // Ignore
            }
            System.out.println(reminder);
        }
    }

    private static int inputInt(Predicate<Integer> validator, Function<Integer, String> reminder) {
        while (true) {
            try {
                var input = new Scanner(System.in).nextInt();
                if (validator.test(input)) {
                    return input;
                }
                System.out.println(reminder.apply(input));
            } catch (InputMismatchException e) {
                System.out.println(reminder.apply(0));
            }
        }
    }

    private static boolean isYes(String s) {
        if (s == null) {
            return false;
        }
        var uppercase = s.toUpperCase();
        return uppercase.equals("Y") || uppercase.equals("YES");
    }

    private static boolean isNo(String s) {
        if (s == null) {
            return false;
        }
        var uppercase = s.toUpperCase();
        return uppercase.equals("N") || uppercase.equals("NO");
    }

    private static class BattleState {
        private final HistoricalDatum data;
        private double F1;

        public BattleState(HistoricalDatum data) {
            this.data = data;
        }
    }

    private static class ArmyPair<T> {
        private T confederate;
        private T union;

        public ArmyPair(T confederate, T union) {
            this.confederate = confederate;
            this.union = union;
        }
    }

    private static class BattleResults {
        private int confederate;
        private int union;
        private int indeterminate;

        public int getTotal() {
            return confederate + union + indeterminate;
        }
    }

    private static class ArmyResources {
        private int food;
        private int salaries;
        private int ammunition;
        private int budget;

        private double morale;  // TODO really here?

        public int getTotal() {
            return this.food + this.salaries + this.ammunition;
        }
    }

    private record HistoricalDatum(String name, ArmyPair<Integer> troops,
                                   ArmyPair<Integer> expectedCasualties,
                                   OffensiveStatus offensiveStatus, String[] blurb) {
    }

    private record UnionLosses(int losses, int desertions) {
    }
}

```