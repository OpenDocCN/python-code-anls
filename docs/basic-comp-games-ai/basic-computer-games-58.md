# BasicComputerGames源码解析 58

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


README.md

Original source downloaded from Vintage Basic

This folder for chapter #59 contains three different games.  Three folders here contain the three games:

 - Rocket
 - LEM
 - lunar

Conversion to [Rust](https://www.rust-lang.org)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### MasterMind

In that Match-April 1976 issue of _Creative_ we published a computerized version of Master Mind, a logic game. Master Mind is played by two people—one is called the code-maker; the other, the code-breaker. At the beginning of the game the code-maker forms a code, or combination of colored pegs. He hides these from the code-breaker. The code-breaker then attempts to deduce the code, by placing his own guesses, one at a time, on the board. After he makes a guess (by placing a combination of colored pegs on the board) the code-maker then gives the code-breaker clues to indicate how close the guess was to the code. For every peg in the guess that’s the right color but not in the right position, the code-breaker gets a white peg. Note that these black and white pegs do not indicate _which_ pegs in the guess are correct, but merely that they exist. For example, if the code was:
```
Yellow Red Red Green
```

and my guess was
```
Red Red Yellow Black
```
I would receive two white pegs and one black peg for the guess. I wouldn’t know (except by comparing previous guesses) which one of the pegs in my guess was the right color in the right position.

Many people have written computer programs to play Master Mind in the passive role, i.e., the computer is the code maker and the human is the code-breaker. This is relatively trivial; the challenge is writing a program that can also play actively as a code-breaker.

Actually, the task of getting the computer to deduce the correct combination is not at all difficult. Imagine, for instance, that you made a list of all possible codes. To begin, you select a guess from your list at random. Then, as you receive clues, you cross off from the list those combinations which you know are impossible. For example if your guess is Red Red Green Green and you receive no pegs, then you know that any combination containing either a red or a green peg is impossible and may be crossed of the list. The process is continued until the correct solution is reached or there are no more combinations left on the list (in which case you know that the code-maker made a mistake in giving you the clues somewhere).

Note that in this particular implementation, we never actually create a list of the combinations, but merely keep track of which ones (in sequential order) may be correct. Using this system, we can easily say that the 523rd combination may be correct, but to actually produce the 523rd combination we have to count all the way from the first combination (or the previous one, if it was lower than 523). Actually, this problem could be simplified to a conversion from base 10 to base (number of colors) and then adjusting the values used in the MID$ function so as not to take a zeroth character from a string if you want to experiment. We did try a version that kept an actual list of all possible combinations (as a string array), which was significantly faster than this version, but which ate tremendous amounts of memory.

At the beginning of this game, you input the number of colors and number of positions you wish to use (which will directly affect the number of combinations) and the number of rounds you wish to play. While you are playing as the code-breaker, you may type BOARD at any time to get a list of your previous guesses and clues, and QUIT to end the game. Note that this version uses string arrays, but this is merely for convenience and can easily be converted for a BASIC that has no string arrays as long as it has a MID$ function. This is because the string arrays are one-dimensional, never exceed a length greater than the number of positions and the elements never contain more than one character.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=110)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=125)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

### How the computer deduces your guess.

The computer takes the number of black pegs and white pegs that the user reports
and uses that information as a target. It then assumes its guess is the answer
and proceeds to compare the black and white pegs against all remaining possible
answers. For each set of black and white pegs it gets in these comparisons, if 
they don't match what the user reported, then they can not be part of the solution.
This can be a non-intuitive assumption, so we'll walk through it with a three color,
three position example (27 possible solutions.)

Let's just suppose our secret code we're hiding from the computer is `BWB`

First let's point out the commutative property of comparing two codes for their
black and white pegs. A black peg meaning correct color and correct position, and
a white peg meaning correct color and wrong position.  If the computer guesses
`RBW` then the black/white peg report is 0 black, 2 white.  But if `RBW` is the 
secret code and the computer guesses `BWB` the reporting for `BWB` is going to be
the same, 0 black, 2 white. 

Now lets look at a table with the reporting for every possible guess the computer 
can make while our secret code is `BWB`.
                                                         
| Guess | Black | White |     | Guess | Black | White |     | Guess | Black | White |
|-------|-------|-------|-----|-------|-------|-------|-----|-------|-------|-------|
| BBB   | 2     | 0     |     | WBB   | 1     | 2     |     | RBB   | 1     | 1     |   
| BBW   | 1     | 2     |     | WBW   | 0     | 2     |     | RBW   | 0     | 2     |   
| BBR   | 1     | 1     |     | WBR   | 0     | 2     |     | RBR   | 0     | 1     |    
| BWB   | 3     | 0     |     | WWB   | 2     | 0     |     | RWB   | 2     | 0     |    
| BWW   | 2     | 0     |     | WWW   | 1     | 0     |     | RWW   | 1     | 0     |    
| BWR   | 2     | 0     |     | WWR   | 1     | 0     |     | RWR   | 1     | 0     |    
| BRB   | 2     | 0     |     | WRB   | 1     | 1     |     | RRB   | 1     | 0     |    
| BRW   | 1     | 1     |     | WRW   | 0     | 1     |     | RRW   | 0     | 1     |    
| BRR   | 1     | 0     |     | WRR   | 0     | 1     |     | RRR   | 0     | 0     | 

The computer has guessed `RBW` and the report on it is 0 black, 2 white. The code
used to eliminate other solutions looks like this:

`1060 IF B1<>B OR W1<>W THEN I(X)=0`

which says set `RBW` as the secret and compare it to all remaining solutions and 
get rid of any that don't match the same black and white report, 0 black and 2 white. 
So let's do that.

Remember, `RBW` is pretending to be the secret code here. These are the remaining
solutions reporting their black and white pegs against `RBW`.

| Guess | Black | White |     | Guess | Black | White |     | Guess | Black | White |
|-------|-------|-------|-----|-------|-------|-------|-----|-------|-------|-------|
| BBB   | 1     | 0     |     | WBB   | 1     | 1     |     | RBB   | 2     | 0     |   
| BBW   | 2     | 0     |     | WBW   | 2     | 0     |     | RBW   | 3     | 0     |   
| BBR   | 1     | 1     |     | WBR   | 1     | 2     |     | RBR   | 2     | 0     |    
| BWB   | 0     | 2     |     | WWB   | 0     | 2     |     | RWB   | 1     | 2     |    
| BWW   | 1     | 1     |     | WWW   | 1     | 0     |     | RWW   | 2     | 0     |    
| BWR   | 0     | 3     |     | WWR   | 1     | 1     |     | RWR   | 1     | 1     |    
| BRB   | 0     | 2     |     | WRB   | 0     | 3     |     | RRB   | 1     | 1     |    
| BRW   | 1     | 2     |     | WRW   | 1     | 1     |     | RRW   | 2     | 0     |    
| BRR   | 0     | 2     |     | WRR   | 0     | 2     |     | RRR   | 1     | 0     | 

Now we are going to eliminate every solution that **DOESN'T** match 0 black and 2 white.

| Guess    | Black | White |     | Guess    | Black | White |     | Guess    | Black | White |
|----------|-------|-------|-----|----------|-------|-------|-----|----------|-------|-------|
| ~~~BBB~~ | 1     | 0     |     | ~~~WBB~~ | 1     | 1     |     | ~~~RBB~~ | 2     | 0     |   
| ~~~BBW~~ | 2     | 0     |     | ~~~WBW~~ | 2     | 0     |     | ~~~RBW~~ | 3     | 0     |   
| ~~~BBR~~ | 1     | 1     |     | ~~~WBR~~ | 1     | 2     |     | ~~~RBR~~ | 2     | 0     |    
| BWB      | 0     | 2     |     | WWB      | 0     | 2     |     | ~~~RWB~~ | 1     | 2     |    
| ~~~BWW~~ | 1     | 1     |     | ~~~WWW~~ | 1     | 0     |     | ~~~RWW~~ | 2     | 0     |    
| ~~~BWR~~ | 0     | 3     |     | ~~~WWR~~ | 1     | 1     |     | ~~~RWR~~ | 1     | 1     |    
| BRB      | 0     | 2     |     | ~~~WRB~~ | 0     | 3     |     | ~~~RRB~~ | 1     | 1     |    
| ~~~BRW~~ | 1     | 2     |     | ~~~WRW~~ | 1     | 1     |     | ~~~RRW~~ | 2     | 0     |    
| BRR      | 0     | 2     |     | WRR      | 0     | 2     |     | ~~~RRR~~ | 1     | 0     |          
                                   
 That wipes out all but five solutions. Notice how the entire right column of solutions 
 is eliminated, including our original guess of `RBW`, therefore eliminating any 
 special case to specifically eliminate this guess from the solution set when we first find out
 its not the answer.
 
 Continuing on, we have the following solutions left of which our secret code, `BWB` 
 is one of them. Remember our commutative property explained previously. 

| Guess | Black | White |
|-------|-------|-------|
| BWB   | 0     | 2     |
| BRB   | 0     | 2     |
| BRR   | 0     | 2     |
| WWB   | 0     | 2     |
| WRR   | 0     | 2     |

So for its second pick, the computer will randomly pick one of these remaining solutions. Let's pick
the middle one, `BRR`, and perform the same ritual. Our user reports to the computer 
that it now has 1 black, 0 whites when comparing to our secret code `BWB`. Let's 
now compare `BRR` to the remaining five solutions and eliminate any that **DON'T**
report 1 black and 0 whites.

| Guess    | Black | White |
|----------|-------|-------|
| BWB      | 1     | 0     |
| ~~~BRB~~ | 2     | 0     |
| ~~~BRR~~ | 3     | 0     |
| ~~~WWB~~ | 0     | 1     |
| ~~~WRR~~ | 2     | 0     | 

Only one solution matches and it's our secret code! The computer will guess this
one next as it's the only choice left, for a total of three moves. 
Coincidentally, I believe the expected maximum number of moves the computer will 
make is the number of positions plus one for the initial guess with no information.
This is because it is winnowing down the solutions 
logarithmically on average. You noticed on the first pass, it wiped out 22 
solutions. If it was doing this logarithmically the worst case guess would 
still eliminate 18 of the solutions leaving 9 (3<sup>2</sup>).  So we have as
a guideline:

 Log<sub>(# of Colors)</sub>TotalPossibilities
 
but TotalPossibilities = (# of Colors)<sup># of Positions</sup>

so you end up with the number of positions as a guess limit. If you consider the
simplest non-trivial puzzle, two colors with two positions, and you guess BW or 
WB first, the most you can logically deduce if you get 1 black and 1 white is 
that it is either WW, or BB which could bring your total guesses up to three 
which is the number of positions plus one.  So if your computer's turn is taking
longer than the number of positions plus one to find the answer then something 
is wrong with your code. 

#### Known Bugs

- Line 622 is unreachable, as the previous line ends in a GOTO and that line number is not referenced anywhere.  It appears that the intent was to tell the user the correct combination after they fail to guess it in 10 tries, which would be a very nice feature, but does not actually work.  (In the MiniScript port, I have made this feature work.)


# `60_Mastermind/csharp/Code.cs`

This code appears to be a CompareTo method for a class that represents a Basic疗程序（可能是某个游戏的脚本）。

其实现中包含两个函数，一个是`Compare`，另一个是`ToString`。

`Compare`函数的实现包含以下内容：

首先，从实现中可以得知`Compare`函数只接受一个`Code`类型的参数，而且这个`Code`对象中包含一个包含颜色的数组，长度与`m_colors.Length`相等。

接着，定义了两个变量`blacks`和`whites`，用于跟踪比较中黑色和白色位置的数量。

然后，使用两层循环来遍历`m_colors`数组，与`other.m_colors`数组相比较。

在循环内部，判断当前颜色是否与`other.m_colors`数组中的元素相同。如果是，则将`blacks`自增，并将当前位置标记为已匹配。

如果不是，则在循环内部遍历`m_colors`数组，查找与`other.m_colors`数组中的元素相同的位置，并将`whites`自增，并将该位置标记为已匹配。

最后，返回比较结果（`blacks`和`whites`），并使用字符串格式化将比较结果返回。

`ToString`函数的实现类似于`toString`方法，只是返回一个字符串形式的`Code`对象，其中每个元素都是一个颜色名称（如`Colors.Red`、`Colors.Green`等）。

总之，这个实现中包含比较两个`Code`对象中颜色位置是否匹配的功能。


```
﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace Game
{
    /// <summary>
    /// Represents a secret code in the game.
    /// </summary>
    public class Code
    {
        private readonly int[] m_colors;

        /// <summary>
        /// Initializes a new instance of the Code class from the given set
        /// of positions.
        /// </summary>
        /// <param name="colors">
        /// Contains the color for each position.
        /// </param>
        public Code(IEnumerable<int> colors)
        {
            m_colors = colors.ToArray();
            if (m_colors.Length == 0)
                throw new ArgumentException("A code must contain at least one position");
        }

        /// <summary>
        /// Compares this code with the given code.
        /// </summary>
        /// <param name="other">
        /// The code to compare.
        /// </param>
        /// <returns>
        /// A number of black pegs and a number of white pegs.  The number
        /// of black pegs is the number of positions that contain the same
        /// color in both codes.  The number of white pegs is the number of
        /// colors that appear in both codes, but in the wrong positions.
        /// </returns>
        public (int blacks, int whites) Compare(Code other)
        {
            // What follows is the O(N^2) from the original BASIC program
            // (where N is the number of positions in the code).  Note that
            // there is an O(N) algorithm.  (Finding it is left as an
            // exercise for the reader.)
            if (other.m_colors.Length != m_colors.Length)
                throw new ArgumentException("Only codes of the same length can be compared");

            // Keeps track of which positions in the other code have already
            // been marked as exact or close matches.
            var consumed = new bool[m_colors.Length];

            var blacks = 0;
            var whites = 0;

            for (var i = 0; i < m_colors.Length; ++i)
            {
                if (m_colors[i] == other.m_colors[i])
                {
                    ++blacks;
                    consumed[i] = true;
                }
                else
                {
                    // Check if the current color appears elsewhere in the
                    // other code.  We must be careful not to consider
                    // positions that are also exact matches.
                    for (var j = 0; j < m_colors.Length; ++j)
                    {
                        if (!consumed[j] &&
                            m_colors[i] == other.m_colors[j] &&
                            m_colors[j] != other.m_colors[j])
                        {
                            ++whites;
                            consumed[j] = true;
                            break;
                        }
                    }
                }
            }

            return (blacks, whites);
        }

        /// <summary>
        /// Gets a string representation of the code.
        /// </summary>
        public override string ToString() =>
            new (m_colors.Select(index => Colors.List[index].ShortName).ToArray());
    }
}

```

# `60_Mastermind/csharp/CodeFactory.cs`

This is a class that represents a code factory that can create codes with specified positions and colors.

The `Create` method creates a code with the specified position and color count. If the given position is negative or the given colors数量 are less than one, an exception will be thrown.

The `EnumerateCodes` method generates a collection of all codes that the factory can create exactly once. It uses a loop to enumerate through the possible positions and colors, and in each iteration, it generates a code and updates the current position.

It is important to note that the code factory has some limitations, such as the maximum number of positions and colors that can be used, and it is not guaranteed to have all possible codes.


```
﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace Game
{
    /// <summary>
    /// Provides methods for generating codes with a given number of positions
    /// and colors.
    /// </summary>
    public class CodeFactory
    {
        /// <summary>
        /// Gets the number of colors in codes generated by this factory.
        /// </summary>
        public int Colors { get; }

        /// <summary>
        /// Gets the number of positions in codes generated by this factory.
        /// </summary>
        public int Positions { get; }

        /// <summary>
        /// Gets the number of distinct codes that this factory can
        /// generate.
        /// </summary>
        public int Possibilities { get; }

        /// <summary>
        /// Initializes a new instance of the CodeFactory class.
        /// </summary>
        /// <param name="positions">
        /// The number of positions.
        /// </param>
        /// <param name="colors">
        /// The number of colors.
        /// </param>
        public CodeFactory(int positions, int colors)
        {
            if (positions < 1)
                throw new ArgumentException("A code must contain at least one position");

            if (colors < 1)
                throw new ArgumentException("A code must contain at least one color");

            if (colors > Game.Colors.List.Length)
                throw new ArgumentException($"A code can contain no more than {Game.Colors.List.Length} colors");

            Positions     = positions;
            Colors        = colors;
            Possibilities = (int)Math.Pow(colors, positions);
        }

        /// <summary>
        /// Creates a specified code.
        /// </summary>
        /// <param name="number">
        /// The number of the code to create from 0 to Possibilities - 1.
        /// </param>
        public Code Create(int number) =>
            EnumerateCodes().Skip(number).First();

        /// <summary>
        /// Creates a random code using the provided random number generator.
        /// </summary>
        /// <param name="random">
        /// The random number generator.
        /// </param>
        public Code Create(Random random) =>
            Create(random.Next(Possibilities));

        /// <summary>
        /// Generates a collection of codes containing every code that this
        /// factory can create exactly once.
        /// </summary>
        public IEnumerable<Code> EnumerateCodes()
        {
            var current = new int[Positions];
            var position = default(int);

            do
            {
                yield return new Code(current);

                position = 0;
                while (position < Positions && ++current[position] == Colors)
                    current[position++] = 0;
            }
            while (position < Positions);
        }
    }
}

```

# `60_Mastermind/csharp/ColorInfo.cs`

这段代码定义了一个名为ColorInfo的枚举类型，用于表示颜色信息。

ColorInfo包含两个成员变量，分别是短名和长名。短名是一个字符类型的变量，通过它可以获取颜色的简称，长名是一个字符串类型的变量，存储颜色的全名(包括颜色名称和颜色调色板)。

这两个成员变量都使用了字符串初始化方式，即将它们都初始化为空字符串。


```
﻿using System;

namespace Game
{
    /// <summary>
    /// Stores information about a color.
    /// </summary>
    public record ColorInfo
    {
        /// <summary>
        /// Gets a single character that represents the color.
        /// </summary>
        public char ShortName { get; init; }

        /// <summary>
        /// Gets the color's full name.
        /// </summary>
        public string LongName { get; init; } = String.Empty;
    }
}

```

# `60_Mastermind/csharp/Colors.cs`

这段代码是一个名为 "Game.Colors" 的命名空间，其中包含一个名为 "Colors" 的静态类，该类提供关于可以使用哪些颜色编码的信息。

该类中包含一个名为 "List" 的常量数组，该数组包含一个名为 "ColorInfo" 的类，该类具有 "ColorInfo" 类的一个实例，并从 "Colors" 命名空间中获取了 "ColorInfo" 类的列表。

"ColorInfo" 类是一个接口，定义了每种颜色编码的颜色名称，这些名称类似于 CSS 颜色名称。

这里，该代码定义了一系列颜色信息的列表，以便在代码中使用，这样就可以在需要使用特定的颜色时，通过类的方式访问它们。


```
﻿namespace Game
{
    /// <summary>
    /// Provides information about the colors that can be used in codes.
    /// </summary>
    public static class Colors
    {
        public static readonly ColorInfo[] List = new[]
        {
            new ColorInfo { ShortName = 'B', LongName = "BLACK"  },
            new ColorInfo { ShortName = 'W', LongName = "WHITE"  },
            new ColorInfo { ShortName = 'R', LongName = "RED"    },
            new ColorInfo { ShortName = 'G', LongName = "GREEN"  },
            new ColorInfo { ShortName = 'O', LongName = "ORANGE" },
            new ColorInfo { ShortName = 'Y', LongName = "YELLOW" },
            new ColorInfo { ShortName = 'P', LongName = "PURPLE" },
            new ColorInfo { ShortName = 'T', LongName = "TAN"    }
        };
    }
}

```

# `60_Mastermind/csharp/Command.cs`

这段代码定义了一个名为“Command”的枚举类型，包含了三个枚举值，分别是个名为“MakeGuess”的“Make a guess”、个名为“ShowBoard”的“Show the board”和个名为“Quit”的“Quit the game”。枚举类型定义了枚举值可以继承的关系，使得你可以使用如下代码来创建枚举实例：
```
namespace Game
{
   public enum Command
   {
       MakeGuess,
       ShowBoard,
       Quit
   }
}
```
然后，在需要使用这些枚举值的时候，可以通过以下方式来创建一个枚举实例：
```
namespace Game
{
   public class Game
   {
       public enum Command
       {
           MakeGuess,
           ShowBoard,
           Quit
       }
   }
}
```
这段代码创建了一个名为“Game”的命名空间，其中包含了一个名为“Command”的枚举类型。通过创建一个名为“Game”的类，可以创建多个“Command”枚举实例，如上所述。


```
﻿namespace Game
{
    /// <summary>
    /// Enumerates the different commands that the user can issue during
    /// the game.
    /// </summary>
    public enum Command
    {
        MakeGuess,
        ShowBoard,
        Quit
    }
}

```

# `60_Mastermind/csharp/Controller.cs`

This is a class that provides a simple way for the user to interact with the console.
The class contains several methods, including `WaitUntilReady()`, `GetBlacksWhites()`, and `TranslateColor()`.

`WaitUntilReady()`方法waits until the user indicates that they are ready to continue by pressing `Enter`.

`GetBlacksWhites()` method asks the user to provide the number of black and white pixels for a given code from the user. It reads the input until the user provides it. It then returns the number of black and white pixels.

`TranslateColor()` method takes a character from the user and returns the corresponding color code from the color codes defined in the `ColorsByKey` class. If the color code is not found, it returns `null`.


```
﻿using System;
using System.Collections.Immutable;
using System.Linq;

namespace Game
{
    /// <summary>
    /// Contains functions for getting input from the end user.
    /// </summary>
    public static class Controller
    {
        /// <summary>
        /// Maps the letters for each color to the integer value representing
        /// that color.
        /// </summary>
        /// <remarks>
        /// We derive this map from the Colors list rather than defining the
        /// entries directly in order to keep all color related information
        /// in one place.  (This makes it easier to change the color options
        /// later.)
        /// </remarks>
        private static ImmutableDictionary<char, int> ColorsByKey = Colors.List
            .Select((info, index) => (key: info.ShortName, index))
            .ToImmutableDictionary(entry => entry.key, entry => entry.index);

        /// <summary>
        /// Gets the number of colors to use in the secret code.
        /// </summary>
        public static int GetNumberOfColors()
        {
            var maximumColors = Colors.List.Length;
            var colors = 0;

            while (colors < 1 || colors > maximumColors)
            {
                colors = GetInteger(View.PromptNumberOfColors);
                if (colors > maximumColors)
                    View.NotifyTooManyColors(maximumColors);
            }

            return colors;
        }

        /// <summary>
        /// Gets the number of positions in the secret code.
        /// </summary>
        /// <returns></returns>
        public static int GetNumberOfPositions()
        {
            // Note: We should probably ensure that the user enters a sane
            //  number of positions here.  (Things go south pretty quickly
            //  with a large number of positions.)  But since the original
            //  program did not, neither will we.
            return GetInteger(View.PromptNumberOfPositions);
        }

        /// <summary>
        /// Gets the number of rounds to play.
        /// </summary>
        public static int GetNumberOfRounds()
        {
            // Note: Silly numbers of rounds (like 0, or a negative number)
            //  are harmless, but it would still make sense to validate.
            return GetInteger(View.PromptNumberOfRounds);
        }

        /// <summary>
        /// Gets a command from the user.
        /// </summary>
        /// <param name="moveNumber">
        /// The current move number.
        /// </param>
        /// <param name="positions">
        /// The number of code positions.
        /// </param>
        /// <param name="colors">
        /// The maximum number of code colors.
        /// </param>
        /// <returns>
        /// The entered command and guess (if applicable).
        /// </returns>
        public static (Command command, Code? guess) GetCommand(int moveNumber, int positions, int colors)
        {
            while (true)
            {
                View.PromptGuess (moveNumber);

                var input = Console.ReadLine();
                if (input is null)
                    Environment.Exit(0);

                switch (input.ToUpperInvariant())
                {
                    case "BOARD":
                        return (Command.ShowBoard, null);
                    case "QUIT":
                        return (Command.Quit, null);
                    default:
                        if (input.Length != positions)
                            View.NotifyBadNumberOfPositions();
                        else
                        if (input.FindFirstIndex(c => !TranslateColor(c).HasValue) is int invalidPosition)
                            View.NotifyInvalidColor(input[invalidPosition]);
                        else
                            return (Command.MakeGuess, new Code(input.Select(c => TranslateColor(c)!.Value)));

                        break;
                }
            }
        }

        /// <summary>
        /// Waits until the user indicates that he or she is ready to continue.
        /// </summary>
        public static void WaitUntilReady()
        {
            View.PromptReady();
            var input = Console.ReadLine();
            if (input is null)
                Environment.Exit(0);
        }

        /// <summary>
        /// Gets the number of blacks and whites for the given code from the
        /// user.
        /// </summary>
        public static (int blacks, int whites) GetBlacksWhites(Code code)
        {
            while (true)
            {
                View.PromptBlacksWhites(code);

                var input = Console.ReadLine();
                if (input is null)
                    Environment.Exit(0);

                var parts = input.Split(',');

                if (parts.Length != 2)
                    View.PromptTwoValues();
                else
                if (!Int32.TryParse(parts[0], out var blacks) || !Int32.TryParse(parts[1], out var whites))
                    View.PromptValidInteger();
                else
                    return (blacks, whites);
            }
        }

        /// <summary>
        /// Gets an integer value from the user.
        /// </summary>
        private static int GetInteger(Action prompt)
        {
            while (true)
            {
                prompt();

                var input = Console.ReadLine();
                if (input is null)
                    Environment.Exit(0);

                if (Int32.TryParse(input, out var result))
                    return result;
                else
                    View.PromptValidInteger();
            }
        }

        /// <summary>
        /// Translates the given character into the corresponding color.
        /// </summary>
        private static int? TranslateColor(char c) =>
            ColorsByKey.TryGetValue(c, out var index) ? index : null;
    }
}

```

# `60_Mastermind/csharp/EnumerableExtensions.cs`

This is a class that provides a method to find the first element in an input sequence that satisfies a given predicate function. The class uses the `Select` and `FirstOrDefault` methods from the `Enumerable` class to provide a way to filter and return the first element that matches the predicate. The `FindFirstIndex` method takes an `IEnumerable<T>` and a predicate function of type `T`, and returns the index of the first element in the source sequence that satisfies the predicate, or a default value if no elements match the predicate. The `FirstOrDefault` method takes an `IEnumerable<T>`, a predicate function of type `T`, and a default value of type `T`, and returns the first item in the source sequence that matches the predicate, or the default value if no elements match the predicate.

To use the `FindFirstIndex` method, you would call it like this:
```
int? result = FindFirstIndex<int>(myList, x => x > 10);
```
This would find the first element in the `myList` sequence that is greater than 10, and return the index if it exists, otherwise return a default value.

To use the `FirstOrDefault` method, you would call it like this:
```
T result = FirstOrDefault<T>(myList, x => x.Id > 10);
```
This would find the first element in the `myList` sequence that has an `Id` value greater than 10, and return the element. If no elements match the predicate, the default value of the `T` type would be returned.

Note that these classes are part of the `System.Linq` namespace, and are intended to be used in functional programming style applications.


```
﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace Game
{
    /// <summary>
    /// Provides additional methods for the <see cref="IEnumerable{T}"/>
    /// interface.
    /// </summary>
    public static class EnumerableExtensions
    {
        /// <summary>
        /// Cycles through the integer values in the range [0, count).
        /// </summary>
        /// <param name="start">
        /// The first value to return.
        /// </param>
        /// <param name="count">
        /// The number of values to return.
        /// </param>
        public static IEnumerable<int> Cycle(int start, int count)
        {
            if (count < 1)
                throw new ArgumentException("count must be at least 1");

            if (start < 0 || start >= count)
                throw new ArgumentException("start must be in the range [0, count)");

            for (var i = start; i < count; ++i)
                yield return i;

            for (var i = 0; i < start; ++i)
                yield return i;
        }

        /// <summary>
        /// Finds the index of the first item in the given sequence that
        /// satisfies the given predicate.
        /// </summary>
        /// <typeparam name="T">
        /// The type of elements in the sequence.
        /// </typeparam>
        /// <param name="source">
        /// The source sequence.
        /// </param>
        /// <param name="predicate">
        /// The predicate function.
        /// </param>
        /// <returns>
        /// The index of the first element in the source sequence for which
        /// predicate(element) is true.  If there is no such element, return
        /// is null.
        /// </returns>
        public static int? FindFirstIndex<T>(this IEnumerable<T> source, Func<T, bool> predicate) =>
            source.Select((element, index) => predicate(element) ? index : default(int?))
                .FirstOrDefault(index => index.HasValue);

        /// <summary>
        /// Returns the first item in the given sequence that matches the
        /// given predicate.
        /// </summary>
        /// <typeparam name="T">
        /// The type of elements in the sequence.
        /// </typeparam>
        /// <param name="source">
        /// The source sequence.
        /// </param>
        /// <param name="predicate">
        /// The predicate to check against each element.
        /// </param>
        /// <param name="defaultValue">
        /// The value to return if no elements match the predicate.
        /// </param>
        /// <returns>
        /// The first item in the source sequence that matches the given
        /// predicate, or the provided default value if none do.
        /// </returns>
        public static T FirstOrDefault<T>(this IEnumerable<T> source, Func<T, bool> predicate, T defaultValue)
        {
            foreach (var element in source)
                if (predicate(element))
                    return element;

            return defaultValue;
        }
    }
}

```

# `60_Mastermind/csharp/Program.cs`

The code you provided is a Python implementation of a game where the user is given a series of code snippets and is asked to guess the solution. The game has two main parts: the code that the user is asked to guess and the scoring system.

The `codeFactory` class seems to be responsible for generating all the possible code snippets. It uses an Enumerable extension method to cycle through the possible codes and uses a FirstOrDefault method to return the first code snippet that is a valid candidate solution.

The `Guessing` class seems to be the main class for the game. It uses a while loop to repeatedly ask the user to guess a code snippet. Inside the loop, it generates a random code snippet using the `codeFactory` class. It then compares the generated code snippet to the known candidate solutions and marks any code snippets that are no longer potential solutions. Finally, it returns `true` if the user was able to guess a code snippet, otherwise it returns `false`.

The `View` class seems to be responsible for displaying the results of the game. It displays the computed code snippet if the user was able to guess a code snippet, or it displays a message if the user was not able to guess a code snippet. It also displays the scores for the user and the computer.

Overall, it looks like the game is a simple game where the user is asked to guess a code snippet. The game uses a while loop to repeatedly generate random code snippets and compare them to the known candidate solutions. If the user is able to guess a code snippet, the game displays the computed code snippet and updates its score.


```
﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace Game
{
    // MASTERMIND II
    // STEVE NORTH
    // CREATIVE COMPUTING
    // PO BOX 789-M MORRISTOWN NEW JERSEY 07960
    class Program
    {
        public const int MaximumGuesses = 10;

        static void Main()
        {
            var (codeFactory, rounds) = StartGame();

            var random        = new Random();
            var humanScore    = 0;
            var computerScore = 0;

            for (var round = 1; round <= rounds; ++round)
            {
                View.ShowStartOfRound(round);

                if (!HumanTakesTurn())
                    return;

                while (!ComputerTakesTurn())
                    View.ShowInconsistentInformation();
            }

            View.ShowScores(humanScore, computerScore, isFinal: true);

            /// <summary>
            /// Gets the game start parameters from the user.
            /// </summary>
            (CodeFactory codeFactory, int rounds) StartGame()
            {
                View.ShowBanner();

                var colors    = Controller.GetNumberOfColors();
                var positions = Controller.GetNumberOfPositions();
                var rounds    = Controller.GetNumberOfRounds();

                var codeFactory = new CodeFactory(positions, colors);

                View.ShowTotalPossibilities(codeFactory.Possibilities);
                View.ShowColorTable(codeFactory.Colors);

                return (codeFactory, rounds);
            }

            /// <summary>
            /// Executes the human's turn.
            /// </summary>
            /// <returns>
            /// True if thue human completed his or her turn and false if
            /// he or she quit the game.
            /// </returns>
            bool HumanTakesTurn()
            {
                // Store a history of the human's guesses (used for the show
                // board command below).
                var history     = new List<TurnResult>();
                var code        = codeFactory.Create(random);
                var guessNumber = default(int);

                for (guessNumber = 1; guessNumber <= MaximumGuesses; ++guessNumber)
                {
                    var guess = default(Code);

                    while (guess is null)
                    {
                        switch (Controller.GetCommand(guessNumber, codeFactory.Positions, codeFactory.Colors))
                        {
                            case (Command.MakeGuess, Code input):
                                guess = input;
                                break;
                            case (Command.ShowBoard, _):
                                View.ShowBoard(history);
                                break;
                            case (Command.Quit, _):
                                View.ShowQuitGame(code);
                                return false;
                        }
                    }

                    var (blacks, whites) = code.Compare(guess);
                    if (blacks == codeFactory.Positions)
                        break;

                    View.ShowResults(blacks, whites);

                    history.Add(new TurnResult(guess, blacks, whites));
                }

                if (guessNumber <= MaximumGuesses)
                    View.ShowHumanGuessedCode(guessNumber);
                else
                    View.ShowHumanFailedToGuessCode(code);

                humanScore += guessNumber;

                View.ShowScores(humanScore, computerScore, isFinal: false);
                return true;
            }

            /// <summary>
            /// Executes the computers turn.
            /// </summary>
            /// <returns>
            /// True if the computer completes its turn successfully and false
            /// if it does not (due to human error).
            /// </returns>
            bool ComputerTakesTurn()
            {
                var isCandidate = new bool[codeFactory.Possibilities];
                var guessNumber = default(int);

                Array.Fill(isCandidate, true);

                View.ShowComputerStartTurn();
                Controller.WaitUntilReady();

                for (guessNumber = 1; guessNumber <= MaximumGuesses; ++guessNumber)
                {
                    // Starting with a random code, cycle through codes until
                    // we find one that is still a candidate solution.  If
                    // there are no remaining candidates, then it implies that
                    // the user made an error in one or more responses.
                    var codeNumber = EnumerableExtensions.Cycle(random.Next(codeFactory.Possibilities), codeFactory.Possibilities)
                        .FirstOrDefault(i => isCandidate[i], -1);

                    if (codeNumber < 0)
                        return false;

                    var guess = codeFactory.Create(codeNumber);

                    var (blacks, whites) = Controller.GetBlacksWhites(guess);
                    if (blacks == codeFactory.Positions)
                        break;

                    // Mark codes which are no longer potential solutions.  We
                    // know that the current guess yields the above number of
                    // blacks and whites when compared to the solution, so any
                    // code that yields a different number of blacks or whites
                    // can't be the answer.
                    foreach (var (candidate, index) in codeFactory.EnumerateCodes().Select((candidate, index) => (candidate, index)))
                    {
                        if (isCandidate[index])
                        {
                            var (candidateBlacks, candidateWhites) = guess.Compare(candidate);
                            if (blacks != candidateBlacks || whites != candidateWhites)
                                isCandidate[index] = false;
                        }
                    }
                }

                if (guessNumber <= MaximumGuesses)
                    View.ShowComputerGuessedCode(guessNumber);
                else
                    View.ShowComputerFailedToGuessCode();

                computerScore += guessNumber;
                View.ShowScores(humanScore, computerScore, isFinal: false);

                return true;
            }
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `60_Mastermind/csharp/TurnResult.cs`

This code defines a namespace called "Game" that contains a record called "TurnResult" that stores the result of a player's turn.

The TurnResult record has three fields:

1. Guess: This is a field of type Code that represents the code guessed by the player.
2. Blacks: This is a field of type int that represents the number of black pegs resulting from the guess.
3. Whites: This is a field of type int that represents the number of white pegs resulting from the guess.

The TurnResult record also has a constructor that takes three parameters: guess, blacks, and whites. The constructor initializes a new instance of the TurnResult record with the values of these parameters.

The purpose of this code is to define a TurnResult record that can be used to store the result of a player's turn. The TurnResult record can be used to keep track of the player's guess, the number of black and white pegs, and so on.


```
﻿namespace Game
{
    /// <summary>
    /// Stores the result of a player's turn.
    /// </summary>
    public record TurnResult
    {
        /// <summary>
        /// Gets the code guessed by the player.
        /// </summary>
        public Code Guess { get; }

        /// <summary>
        /// Gets the number of black pegs resulting from the guess.
        /// </summary>
        public int Blacks { get; }

        /// <summary>
        /// Gets the number of white pegs resulting from the guess.
        /// </summary>
        public int Whites { get; }

        /// <summary>
        /// Initializes a new instance of the TurnResult record.
        /// </summary>
        /// <param name="guess">
        /// The player's guess.
        /// </param>
        /// <param name="blacks">
        /// The number of black pegs.
        /// </param>
        /// <param name="whites">
        /// The number of white pegs.
        /// </param>
        public TurnResult(Code guess, int blacks, int whites) =>
            (Guess, Blacks, Whites) = (guess, blacks, whites);
    }
}

```

# `60_Mastermind/csharp/View.cs`

This is a class that contains functions forPromptNumberOfColors, PromptNumberOfPositions, PromptNumberOfRounds, PromptGuess, PromptReady, PromptBlacksWhites, and PromptTwoValues. These functions are used to interact with the user to ask for information, such as the number of colors, positions, rounds, and the guess for each round.

The functions are of the form:
```sql
public static void PromptGuess(int moveNumber)
{
   // Code for asking the user to guess the number
}
```

```vbnet
public static void PromptReady()
{
   // Code for asking the user if they are ready to start the game
}
```

```sql
public static void PromptBlacksWhites(Code code)
{
   // Code for asking the user if they want to play as black or white
}
```

```sql
public static void PromptTwoValues()
{
   // Code for asking the user to enter two values separated by a comma
}
```

```sql
public static void PromptValidInteger()
{
   // Code for asking the user to enter an integer value
}
```

```sql
public static void PromptNumberOfColors()
{
   // Code for asking the user if they want to specify the number of colors
}
```

```sql
public static void PromptNumberOfPositions()
{
   // Code for asking the user if they want to specify the number of positions
}
```

```sql
public static void PromptNumberOfRounds()
{
   // Code for asking the user if they want to specify the number of rounds
}
```

```sql
public static void PromptGuess(int moveNumber)
{
   // Code for asking the user if they can guess the number
}
```

```sql
public static void PromptReady()
{
   // Code for asking the user if they are ready to start the game
}
```

```sql
public static void PromptBlacksWhites(Code code)
{
   // Code for asking the user if they want to play as black or white
}
```

```sql
public static void PromptTwoValues()
{
   // Code for asking the user if they want to enter two values separated by a comma
}
```

```sql
public static void PromptValidInteger()
{
   // Code for asking the user if they can enter an integer value
}
```

```sql
public static void PromptNumberOfColors()
{
   // Code for asking the user if they want to specify the number of colors
}
```

```sql
public static void PromptNumberOfPositions()
{
   // Code for asking the user if they want to specify the number of positions
}
```

```sql
public static void PromptNumberOfRounds()
{
   // Code for asking the user if they want to specify the number of rounds
}
```

```sql
public static void PromptGuess(int moveNumber)
{
   // Code for asking the user if they can guess the number
}
```

```sql
public static void PromptReady()
{
   // Code for asking the user if they are ready to start the game
}
```

```sql
public static void PromptBlacksWhites(Code code)
{
   // Code for asking the user if they want to play as black or white
}
```

```sql
public static void PromptTwoValues()
{
   // Code for asking the user if they want to enter two values separated by a comma
}
```

```sql
public static void PromptValidInteger()
{
   // Code for asking the user if they can enter an integer value
}
```


```
﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace Game
{
    /// <summary>
    /// Contains functions for displaying information to the end user.
    /// </summary>
    public static class View
    {
        public static void ShowBanner()
        {
            Console.WriteLine("                              MASTERMIND");
            Console.WriteLine("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
        }

        public static void ShowTotalPossibilities(int possibilities)
        {
            Console.WriteLine($"TOTAL POSSIBILITIES = {possibilities}");
            Console.WriteLine();
        }

        public static void ShowColorTable(int numberOfColors)
        {
            Console.WriteLine();
            Console.WriteLine("COLOR     LETTER");
            Console.WriteLine("=====     ======");

            foreach (var color in Colors.List.Take(numberOfColors))
                Console.WriteLine($"{color.LongName,-13}{color.ShortName}");

            Console.WriteLine();
        }

        public static void ShowStartOfRound(int roundNumber)
        {
            Console.WriteLine();
            Console.WriteLine($"ROUND NUMBER {roundNumber} ----");
            Console.WriteLine();
            Console.WriteLine("GUESS MY COMBINATION.");
            Console.WriteLine();
        }

        public static void ShowBoard(IEnumerable<TurnResult> history)
        {
            Console.WriteLine();
            Console.WriteLine("BOARD");
            Console.WriteLine("MOVE     GUESS          BLACK     WHITE");

            var moveNumber = 0;
            foreach (var result in history)
                Console.WriteLine($"{++moveNumber,-9}{result.Guess,-16}{result.Blacks,-10}{result.Whites}");

            Console.WriteLine();
        }

        public static void ShowQuitGame(Code code)
        {
            Console.WriteLine($"QUITTER!  MY COMBINATION WAS: {code}");
            Console.WriteLine("GOOD BYE");
        }

        public static void ShowResults(int blacks, int whites)
        {
            Console.WriteLine($"YOU HAVE  {blacks}  BLACKS AND  {whites}  WHITES.");
        }

        public static void ShowHumanGuessedCode(int guessNumber)
        {
            Console.WriteLine($"YOU GUESSED IT IN  {guessNumber}  MOVES!");
        }

        public static void ShowHumanFailedToGuessCode(Code code)
        {
            // Note: The original code did not print out the combination, but
            // this appears to be a bug.
            Console.WriteLine("YOU RAN OUT OF MOVES!  THAT'S ALL YOU GET!");
            Console.WriteLine($"THE ACTUAL COMBINATION WAS: {code}");
        }

        public static void ShowScores(int humanScore, int computerScore, bool isFinal)
        {
            if (isFinal)
            {
                Console.WriteLine("GAME OVER");
                Console.WriteLine("FINAL SCORE:");
            }
            else
                Console.WriteLine("SCORE:");

            Console.WriteLine($"     COMPUTER  {computerScore}");
            Console.WriteLine($"     HUMAN     {humanScore}");
            Console.WriteLine();
        }

        public static void ShowComputerStartTurn()
        {
            Console.WriteLine("NOW I GUESS.  THINK OF A COMBINATION.");
        }

        public static void ShowInconsistentInformation()
        {
            Console.WriteLine("YOU HAVE GIVEN ME INCONSISTENT INFORMATION.");
            Console.WriteLine("TRY AGAIN, AND THIS TIME PLEASE BE MORE CAREFUL.");
        }

        public static void ShowComputerGuessedCode(int guessNumber)
        {
            Console.WriteLine($"I GOT IT IN  {guessNumber}  MOVES!");
        }

        public static void ShowComputerFailedToGuessCode()
        {
            Console.WriteLine("I USED UP ALL MY MOVES!");
            Console.WriteLine("I GUESS MY CPU IS JUST HAVING AN OFF DAY.");
        }

        public static void PromptNumberOfColors()
        {
            Console.Write("NUMBER OF COLORS? ");
        }

        public static void PromptNumberOfPositions()
        {
            Console.Write("NUMBER OF POSITIONS? ");
        }

        public static void PromptNumberOfRounds()
        {
            Console.Write("NUMBER OF ROUNDS? ");
        }

        public static void PromptGuess(int moveNumber)
        {
            Console.Write($"MOVE #  {moveNumber}  GUESS ? ");
        }

        public static void PromptReady()
        {
            Console.Write("HIT RETURN WHEN READY ? ");
        }

        public static void PromptBlacksWhites(Code code)
        {
            Console.Write($"MY GUESS IS: {code}");
            Console.Write("  BLACKS, WHITES ? ");
        }

        public static void PromptTwoValues()
        {
            Console.WriteLine("PLEASE ENTER TWO VALUES, SEPARATED BY A COMMA");
        }

        public static void PromptValidInteger()
        {
            Console.WriteLine("PLEASE ENTER AN INTEGER VALUE");
        }

        public static void NotifyBadNumberOfPositions()
        {
            Console.WriteLine("BAD NUMBER OF POSITIONS");
        }

        public static void NotifyInvalidColor(char colorKey)
        {
            Console.WriteLine($"'{colorKey}' IS UNRECOGNIZED.");
        }

        public static void NotifyTooManyColors(int maxColors)
        {
            Console.WriteLine($"NO MORE THAN {maxColors}, PLEASE!");
        }
    }
}

```

# `60_Mastermind/java/Mastermind.java`

这段代码是一个Java port implementation，基于BASIC游戏规则，并实现了一些扩展功能。其目的是在Java环境中提供一个类似于BASIC游戏的秘密编码游戏。这个版本使用了一种基于数字基转换的方法来将解决方案ID转换为颜色代码字符串。与原始BASIC不同的是，这个版本使用了一个数组列表来存储解决方案ID，而不是使用一个位图。这个数组列表可以用来检查解决方案是否超过了BASIC中定义的最大解决方案数。此外，这个版本还实现了一个 ceiling 检查，以防止在计算颜色代码字符串时使用过多的内存。这个版本还增加了一些额外的信息，以帮助玩家了解游戏规则和如何退出游戏。


```
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.Scanner;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * A port of the BASIC Mastermind game in java.
 *
 * Differences between this and the original BASIC:
 *    Uses a number base conversion approach to converting solution ids to
 *    color code strings. The original performs an inefficient add by 1
 *    with carry technique where every carry increments the next positions
 *    color id.
 *
 *    Implements a ceiling check on the number of positions in a secret code
 *    to not run out of memory. Because of the algorithm that the computer
 *    uses to deduce the players secret code, it searches through the entire
 *    possible spectrum of solutions. This can be a very large number because
 *    it's (number of colors) ^ (number of positions). The original will
 *    happily try to allocate all the memory on the system if this number is
 *    too large. If it did successfully allocate the memory on a large solution
 *    set then it would also take too long to compute code strings via its
 *    technique mentioned in the previous note.
 *
 *    An extra message is given at the start to alert the player to the
 *    BOARD and QUIT commands.
 */
```

This is a Java program that reads input from the user. It uses a scanner class to read lines from the user, which is similar to the java.util.Scanner class.

The program reads two types of input: numbers and words. For numbers, the user is prompted to enter a number, and then the number is extracted and stored in the first element of the nums array. For words, the user is prompted to enter a word, and then the word is stored in the word variable.

The program also has a catch block that handles any exceptions that may occur during the input process. If the input is not formatted as expected, the program will print an error message and stop.

Overall, this program is designed to read input from the user and extract the necessary information from them.


```
public class Mastermind {
  final Random random = new Random();
  // some less verbose printing methods
  static private void pf(String s, Object... o){ System.out.printf(s, o);}
  static private void pl(String s){System.out.println(s);}
  static private void pl(){System.out.println();}

  public static void main(String[] args) {
    title();
    Mastermind game = setup();
    game.play();
  }

  /**
   * The eight possible color codes.
   */
  private enum Color {
    B("BLACK"), W("WHITE"), R("RED"), G("GREEN"),
    O("ORANGE"), Y("YELLOW"), P("PURPLE"), T("TAN");
    public final String name;

    Color(String name) {
      this.name = name;
    }
  }

  /**
   * Represents a guess and the subsequent number of colors in the correct
   * position (blacks), and the number of colors present but not in the correct
   * position (whites.)
   */
  private record Guess(int guessNum, String guess, int blacks, int whites){}


  private void play() {
    IntStream.rangeClosed(1,rounds).forEach(this::playRound);
    pl("GAME OVER");
    pl("FINAL SCORE: ");
    pl(getScore());
  }

  /**
   * Builder-ish pattern for creating Mastermind game
   * @return Mastermind game object
   */
  private static Mastermind setup() {
    int numOfColors;
    pf("NUMBER OF COLORS? > ");
    numOfColors = getPositiveNumberUpTo(Color.values().length);
    int maxPositions = getMaxPositions(numOfColors);
    pf("NUMBER OF POSITIONS (MAX %d)? > ", maxPositions);
    int positions = getPositiveNumberUpTo(maxPositions);
    pf("NUMBER OF ROUNDS? > ");
    int rounds = getPositiveNumber();
    pl("ON YOUR TURN YOU CAN ENTER 'BOARD' TO DISPLAY YOUR PREVIOUS GUESSES,");
    pl("OR 'QUIT' TO GIVE UP.");
    return new Mastermind(numOfColors, positions, rounds, 10);
  }

  /**
   * Computes the number of allowable positions to prevent the total possible
   * solution set that the computer has to check to a reasonable number, and
   * to prevent out of memory errors.
   *
   * The computer guessing algorithm uses a BitSet which has a limit of 2^31
   * bits (Integer.MAX_VALUE bits). Since the number of possible solutions to
   * any mastermind game is (numColors) ^ (numPositions) we need find the
   * maximum number of positions by finding the Log|base-NumOfColors|(2^31)
   *
   * @param numOfColors  number of different colors
   * @return             max number of positions in the secret code.
   */
  private static int getMaxPositions(int numOfColors){
    return (int)(Math.log(Integer.MAX_VALUE)/Math.log(numOfColors));
  }

  final int numOfColors, positions, rounds, possibilities;
  int humanMoves, computerMoves;
  final BitSet solutionSet;
  final Color[] colors;
  final int maxTries;

  // A recording of human guesses made during the round for the BOARD command.
  final List<Guess> guesses = new ArrayList<>();

  // A regular expression to validate user guess strings
  final String guessValidatorRegex;

  public Mastermind(int numOfColors, int positions, int rounds, int maxTries) {
    this.numOfColors = numOfColors;
    this.positions = positions;
    this.rounds = rounds;
    this.maxTries = maxTries;
    this.humanMoves = 0;
    this.computerMoves = 0;
    String colorCodes = Arrays.stream(Color.values())
                              .limit(numOfColors)
                              .map(Color::toString)
                              .collect(Collectors.joining());
    // regex that limits the number of color codes and quantity for a guess.
    this.guessValidatorRegex = "^[" + colorCodes + "]{" + positions + "}$";
    this.colors = Color.values();
    this.possibilities = (int) Math.round(Math.pow(numOfColors, positions));
    pf("TOTAL POSSIBILITIES =% d%n", possibilities);
    this.solutionSet = new BitSet(possibilities);
    displayColorCodes(numOfColors);
  }

  private void playRound(int round) {
    pf("ROUND NUMBER % d ----%n%n",round);
    humanTurn();
    computerTurn();
    pl(getScore());
  }

  private void humanTurn() {
    guesses.clear();
    String secretCode = generateColorCode();
    pl("GUESS MY COMBINATION. \n");
    int guessNumber = 1;
    while (true) {   // User input loop
      pf("MOVE #%d GUESS ?", guessNumber);
      final String guess = getWord();
      if (guess.equals(secretCode)) {
        guesses.add(new Guess(guessNumber, guess, positions, 0));
        pf("YOU GUESSED IT IN %d MOVES!%n", guessNumber);
        humanMoves++;
        pl(getScore());
        return;
      } else if ("BOARD".equals(guess)) {  displayBoard();
      } else if ("QUIT".equals(guess))  {  quit(secretCode);
      } else if (!validateGuess(guess)) {  pl(guess + " IS UNRECOGNIZED.");
      } else {
        Guess g = evaluateGuess(guessNumber, guess, secretCode);
        pf("YOU HAVE %d BLACKS AND %d WHITES.%n", g.blacks(), g.whites());
        guesses.add(g);
        humanMoves++;
        guessNumber++;
      }
      if (guessNumber > maxTries) {
        pl("YOU RAN OUT OF MOVES!  THAT'S ALL YOU GET!");
        pl("THE ACTUAL COMBINATION WAS: " + secretCode);
        return;
      }
    }
  }

  private void computerTurn(){
    while (true) {
      pl("NOW I GUESS.  THINK OF A COMBINATION.");
      pl("HIT RETURN WHEN READY:");
      solutionSet.set(0, possibilities);  // set all bits to true
      getInput("RETURN KEY", Scanner::nextLine, Objects::nonNull);
      int guessNumber = 1;
      while(true){
        if (solutionSet.cardinality() == 0) {
          // user has given wrong information, thus we have cancelled out
          // any remaining possible valid solution.
          pl("YOU HAVE GIVEN ME INCONSISTENT INFORMATION.");
          pl("TRY AGAIN, AND THIS TIME PLEASE BE MORE CAREFUL.");
          break;
        }
        // Randomly pick an untried solution.
        int solution = solutionSet.nextSetBit(generateSolutionID());
        if (solution == -1) {
          solution = solutionSet.nextSetBit(0);
        }
        String guess = solutionIdToColorCode(solution);
        pf("MY GUESS IS: %s  BLACKS, WHITES ? ",guess);
        int[] bAndWPegs = getPegCount(positions);
        if (bAndWPegs[0] == positions) {
          pf("I GOT IT IN % d MOVES!%n", guessNumber);
          computerMoves+=guessNumber;
          return;
        }
        // wrong guess, first remove this guess from solution set
        solutionSet.clear(solution);
        int index = 0;
        // Cycle through remaining solution set, marking any solutions as invalid
        // that don't exactly match what the user said about our guess.
        while ((index = solutionSet.nextSetBit(index)) != -1) {
          String solutionStr = solutionIdToColorCode(index);
          Guess possibleSolution = evaluateGuess(0, solutionStr, guess);
          if (possibleSolution.blacks() != bAndWPegs[0] ||
              possibleSolution.whites() != bAndWPegs[1]) {
            solutionSet.clear(index);
          }
          index++;
        }
        guessNumber++;
      }
    }
  }

  // tally black and white pegs
  private Guess evaluateGuess(int guessNum, String guess, String secretCode) {
    int blacks = 0, whites = 0;
    char[] g = guess.toCharArray();
    char[] sc = secretCode.toCharArray();
    // An incremented number that marks this position as having been counted
    // as a black or white peg already.
    char visited = 0x8000;
    // Cycle through guess letters and check for color and position match
    // with the secretCode. If both match, mark it black.
    // Else cycle through remaining secretCode letters and check if color
    // matches. If this matches, a preventative check must be made against
    // the guess letter matching the secretCode letter at this position in
    // case it would be counted as a black in one of the next passes.
    for (int j = 0; j < positions; j++) {
      if (g[j] == sc[j]) {
        blacks++;
        g[j] = visited++;
        sc[j] = visited++;
      }
      for (int k = 0; k < positions; k++) {
        if (g[j] == sc[k] && g[k] != sc[k]) {
          whites++;
          g[j] = visited++;
          sc[k] = visited++;
        }
      }
    }
    return new Guess(guessNum, guess, blacks, whites);
  }

  private boolean validateGuess(String guess) {
    return guess.length() == positions && guess.matches(guessValidatorRegex);
  }

  private String getScore() {
    return "SCORE:%n\tCOMPUTER \t%d%n\tHUMAN \t%d%n"
        .formatted(computerMoves, humanMoves);
  }

  private void printGuess(Guess g){
    pf("% 3d%9s% 15d% 10d%n",g.guessNum(),g.guess(),g.blacks(),g.whites());
  }
  
  private void displayBoard() {
    pl();
    pl("BOARD");
    pl("MOVE     GUESS          BLACK     WHITE");
    guesses.forEach(this::printGuess);
    pl();
  }

  private void quit(String secretCode) {
    pl("QUITTER!  MY COMBINATION WAS: " + secretCode);
    pl("GOOD BYE");
    System.exit(0);
  }

  /**
   * Generates a set of color codes randomly.
   */
  private String generateColorCode() {
    int solution = generateSolutionID();
    return solutionIdToColorCode(solution);
  }

  /**
   * From the total possible number of solutions created at construction, choose
   * one randomly.
   *
   * @return one of many possible solutions
   */
  private int generateSolutionID() {
    return random.nextInt(0, this.possibilities);
  }

  /**
   * Given the number of colors and positions in a secret code, decode one of
   * those permutations, a solution number, into a string of letters
   * representing colored pegs.
   *
   * The pattern can be decoded easily as a number with base `numOfColors` and
   * `positions` representing the digits. For example if numOfColors is 5 and
   * positions is 3 then the pattern is converted to a number that is base 5
   * with three digits. Each digit then maps to a particular color.
   *
   * @param solution one of many possible solutions
   * @return String representing this solution's color combination.
   */
  private String solutionIdToColorCode(final int solution) {
    StringBuilder secretCode = new StringBuilder();
    int pos = possibilities;
    int remainder = solution;
    for (int i = positions - 1; i > 0; i--) {
      pos = pos / numOfColors;
      secretCode.append(colors[remainder / pos].toString());
      remainder = remainder % pos;
    }
    secretCode.append(colors[remainder].toString());
    return secretCode.toString();
  }

  private static void displayColorCodes(int numOfColors) {
    pl("\n\nCOLOR     LETTER\n=====     ======");
    Arrays.stream(Color.values())
          .limit(numOfColors)
          .map(c -> c.name + " ".repeat(13 - c.name.length()) + c)
          .forEach(Mastermind::pl);
    pl();pl();
  }

  private static void title() {
    pl("""    
                                  MASTERMIND
                   CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY%n%n%n
    """);
  }

  /////////////////////////////////////////////////////////
  // User input functions from here on

  /**
   * Base input function to be called from a specific input function.
   * Re-prompts upon unexpected or invalid user input.
   * Discards any remaining input in the line of user entered input once
   * it gets what it wants.
   * @param descriptor  Describes explicit type of input expected
   * @param extractor   The method performed against a Scanner object to parse
   *                    the type of input.
   * @param conditional A test that the input meets a minimum validation.
   * @param <T>         Input type returned.
   * @return            input type for this line of user input.
   */
  private static <T> T getInput(String descriptor,
                                Function<Scanner, T> extractor,
                                Predicate<T> conditional) {

    Scanner scanner = new Scanner(System.in);
    while (true) {
      try {
        T input = extractor.apply(scanner);
        if (conditional.test(input)) {
          return input;
        }
      } catch (Exception ex) {
        try {
          // If we are here then a call on the scanner was most likely unable to
          // parse the input. We need to flush whatever is leftover from this
          // line of interactive user input so that we can re-prompt for new input.
          scanner.nextLine();
        } catch (Exception ns_ex) {
          // if we are here then the input has been closed, or we received an
          // EOF (end of file) signal, usually in the form of a ctrl-d or
          // in the case of Windows, a ctrl-z.
          pl("END OF INPUT, STOPPING PROGRAM.");
          System.exit(1);
        }
      }
      pf("!%s EXPECTED - RETRY INPUT LINE%n? ", descriptor);
    }
  }

  private static int getPositiveNumber() {
    return getInput("NUMBER", Scanner::nextInt, num -> num > 0);
  }

  private static int getPositiveNumberUpTo(long to) {
    return getInput(
        "NUMBER FROM 1 TO " + to,
        Scanner::nextInt,
        num -> num > 0 && num <= to);
  }

  private static int[] getPegCount(int upperBound) {
    int[] nums = {Integer.MAX_VALUE, Integer.MAX_VALUE};
    while (true) {
      String input = getInput(
          "NUMBER, NUMBER",
          Scanner::nextLine,
          s -> s.matches("\\d+[\\s,]+\\d+$"));
      String[] numbers = input.split("[\\s,]+");
      nums[0] = Integer.parseInt(numbers[0].trim());
      nums[1] = Integer.parseInt(numbers[1].trim());
      if (nums[0] <= upperBound && nums[1] <= upperBound &&
          nums[0] >= 0 && nums[1] >= 0) {
        return nums;
      }
      pf("NUMBERS MUST BE FROM 0 TO %d.%n? ", upperBound);
    }
  }

  private static String getWord() {
    return getInput("WORD", Scanner::next, word -> !"".equals(word));
  }
}

```