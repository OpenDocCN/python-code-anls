# BasicComputerGames源码解析 26

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `16_Bug/python/bug.py`

这段代码定义了一个名为 State 的类，它是一个游戏中的状态对象。这个类包含了游戏中的玩家在游戏中的各种状态，如玩家是否已经死亡、玩家的身体各个部分的值、玩家的头朝哪个方向等。

import random 模块用于从游戏中获取随机数，import time 模块用于让程序暂停一段时间，from dataclasses import dataclass 说明这个 class 是dataclass 类的子类，from typing import Literal 说明 Literal 是这个 class 的一个 Literal 类型的成员。

在这个 class 中，我们定义了一个玩家状态的类，它继承自 dataclass 类。dataclass 类中定义了玩家状态的所有属性，而 State 类中则更加详细地定义了每个属性的具体实现。

State 类中定义了一个 is_finished 方法，它用于判断玩家是否已经死亡，如果玩家的 feelers 属性值已经变为 2,tail 属性值已经变为 1,legs 属性值已经变为 6,head 属性值已经变为 1,neck 属性值已经变为 1，那么玩家就可以被认为是死了。

State 类中还定义了一个 display 方法，它用于显示玩家的状态信息。这个方法会根据 playersetting 属性中的不同属性值来输出不同的信息。当 playersetting 中 feelers 属性值为 2 时，会输出 "You are dead!" 信息；当 playersetting 中 head 属性值为 1 时，会输出 "Your head is up" 信息；当 playersetting 中 neck 属性值为 1 时，会输出 "Your neck is up" 信息；当 playersetting 中 body 属性值为 1 时，会输出 "You have 1 hit points" 信息；当 playersetting 中 legs 属性值为 6 时，会输出 "You have 6 hit points" 信息；当 playersetting 中 tail 属性值为 1 时，会输出 "Your tail is up" 信息；当 playersetting 中 legs 属性值为 0 时，会输出 "You have 0 hit points" 信息。


```
import random
import time
from dataclasses import dataclass
from typing import Literal


@dataclass
class State:
    is_player: bool
    body: int = 0
    neck: int = 0
    head: int = 0
    feelers: int = 0
    tail: int = 0
    legs: int = 0

    def is_finished(self) -> bool:
        return (
            self.feelers == 2
            and self.tail == 1
            and self.legs == 6
            and self.head == 1
            and self.neck == 1
        )

    def display(self) -> None:
        if self.feelers != 0:
            print_feelers(self.feelers, is_player=self.is_player)
        if self.head != 0:
            print_head()
        if self.neck != 0:
            print_neck()
        if self.body != 0:
            print_body(True) if self.tail == 1 else print_body(False)
        if self.legs != 0:
            print_legs(self.legs)


```

这三段代码都是函数，用于打印特定的字符或单词。

第一段代码 `print_n_newlines(n)` 的作用是打印 `n` 个新行符。具体来说，它将循环 `n` 次，每次打印一个空行，然后在一个新行结束时打印一个空行。

第二段代码 `print_feelers(n_feelers, is_player=True)` 的作用是打印 `n_feelers` 个字符，其中 `is_player` 参数 `True` 时打印 "A"，否则打印 "F"。具体来说，它将循环 `4` 次，每次打印一个空格，然后循环 `n_feelers` 次，打印 `n_feelers` 个字符。

第三段代码 `print_head()` 的作用是在屏幕上打印 "HHHHHHH" 的字符串。具体来说，它将打印一个包含六个字符的单词头，其中每个字符都不同。


```
def print_n_newlines(n: int) -> None:
    for _ in range(n):
        print()


def print_feelers(n_feelers: int, is_player: bool = True) -> None:
    for _ in range(4):
        print(" " * 10, end="")
        for _ in range(n_feelers):
            print("A " if is_player else "F ", end="")
        print()


def print_head() -> None:
    print("        HHHHHHH")
    print("        H     H")
    print("        H O O H")
    print("        H     H")
    print("        H  V  H")
    print("        HHHHHHH")


```



这个代码定义了三个函数，每个函数都是回调函数(callback function)，也就是函数可以被作为实参传递给其他函数。 

第一个函数 `print_neck()` 没有返回值，但是无论如何它都会打印出两行字符，上面一行为 "          N N"，下面一行也是 "          N N"。

第二个函数 `print_body(has_tail: bool = False)` 打印出了一个带尾巴的矩形，当 `has_tail` 为 `True` 时打印 "TTTTTB" 四个字符，否则打印 "     BBBBBBBBBBBB"。

第三个函数 `print_legs(n_legs: int)` 打印出了 `n_legs` 行垂直的腿部，每行包含 `n_legs` 个 "L" 字符，每个 "L" 字符之间有一个空格，每行结尾打印一个空行。


```
def print_neck() -> None:
    print("          N N")
    print("          N N")


def print_body(has_tail: bool = False) -> None:
    print("     BBBBBBBBBBBB")
    print("     B          B")
    print("     B          B")
    print("TTTTTB          B") if has_tail else ""
    print("     BBBBBBBBBBBB")


def print_legs(n_legs: int) -> None:
    for _ in range(2):
        print(" " * 5, end="")
        for _ in range(n_legs):
            print(" L", end="")
        print()


```

This is a Python implementation of the game "20 Questions". The `dice` function has 6 possible outcomes (1 for Head, 2 for Eye, 3 for Nose, 4 for Ear, 5 for Mouth, 6 for Tail), each of which corresponds to one of the 6 body parts. The game logic uses a state machine to keep track of the current state of the game (e.g. Head, Body, Legs, etc.), the number of guesses remaining, and the round number. When the game is over and a winner has been determined, the game returns `True` to indicate that the game was completed.

The `return` statement in the last lines of the game logic function is used to return `True` if the game was completed and the player won, or `False` if it was not completed or the player lost.


```
def handle_roll(diceroll: Literal[1, 2, 3, 4, 5, 6], state: State) -> bool:
    who = "YOU" if state.is_player else "I"
    changed = False

    print(f"{who} ROLLED A", diceroll)
    if diceroll == 1:
        print("1=BODY")
        if state.body:
            print(f"{who} DO NOT NEED A BODY.")
        else:
            print(f"{who} NOW HAVE A BODY.")
            state.body = 1
            changed = True
    elif diceroll == 2:
        print("2=NECK")
        if state.neck:
            print(f"{who} DO NOT NEED A NECK.")
        elif state.body == 0:
            print(f"{who} DO NOT HAVE A BODY.")
        else:
            print(f"{who} NOW HAVE A NECK.")
            state.neck = 1
            changed = True
    elif diceroll == 3:
        print("3=HEAD")
        if state.neck == 0:
            print(f"{who} DO NOT HAVE A NECK.")
        elif state.head:
            print(f"{who} HAVE A HEAD.")
        else:
            print(f"{who} NEEDED A HEAD.")
            state.head = 1
            changed = True
    elif diceroll == 4:
        print("4=FEELERS")
        if state.head == 0:
            print(f"{who} DO NOT HAVE A HEAD.")
        elif state.feelers == 2:
            print(f"{who} HAVE TWO FEELERS ALREADY.")
        else:
            if state.is_player:
                print("I NOW GIVE YOU A FEELER.")
            else:
                print(f"{who} GET A FEELER.")
            state.feelers += 1
            changed = True
    elif diceroll == 5:
        print("5=TAIL")
        if state.body == 0:
            print(f"{who} DO NOT HAVE A BODY.")
        elif state.tail:
            print(f"{who} ALREADY HAVE A TAIL.")
        else:
            if state.is_player:
                print("I NOW GIVE YOU A TAIL.")
            else:
                print(f"{who} NOW HAVE A TAIL.")
            state.tail = 1
            changed = True
    elif diceroll == 6:
        print("6=LEG")
        if state.legs == 6:
            print(f"{who} HAVE 6 FEET ALREADY.")
        elif state.body == 0:
            print(f"{who} DO NOT HAVE A BODY.")
        else:
            state.legs += 1
            changed = True
            print(f"{who} NOW HAVE {state.legs} LEGS")
    return changed


```

This appears to be a Python script written in the Pygame module. It appears to simulate a game of cards where two players take turns playing cards and trying to guess the bot's hand. The script uses a combination of randomness and player input to determine which player wins the game.

The script defines two objects, `player` and `opponent`, which represent the two players in the game. The `State` class is used to keep track of the current state of the game, including the player's hand and the bot's hand.

The `handle_roll` function is used to simulate rolling a die and changing the game state accordingly. The `changed` variable is used to determine if the game state has changed since the last time the `handle_roll` function was called.

The `show_hand` function is used to display the player's hand to the user.

The main game loop runs until the `bugs_finished` variable reaches 2, indicating that one of the players has won the game.


```
def main() -> None:
    print(" " * 34 + "BUG")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print_n_newlines(3)

    print("THE GAME BUG")
    print("I HOPE YOU ENJOY THIS GAME.")
    print()
    want_instructions = input("DO YOU WANT INSTRUCTIONS? ")
    if want_instructions != "NO":
        print("THE OBJECT OF BUG IS TO FINISH YOUR BUG BEFORE I FINISH")
        print("MINE. EACH NUMBER STANDS FOR A PART OF THE BUG BODY.")
        print("I WILL ROLL THE DIE FOR YOU, TELL YOU WHAT I ROLLED FOR YOU")
        print("WHAT THE NUMBER STANDS FOR, AND IF YOU CAN GET THE PART.")
        print("IF YOU CAN GET THE PART I WILL GIVE IT TO YOU.")
        print("THE SAME WILL HAPPEN ON MY TURN.")
        print("IF THERE IS A CHANGE IN EITHER BUG I WILL GIVE YOU THE")
        print("OPTION OF SEEING THE PICTURES OF THE BUGS.")
        print("THE NUMBERS STAND FOR PARTS AS FOLLOWS:")
        table = [
            ["NUMBER", "PART", "NUMBER OF PART NEEDED"],
            ["1", "BODY", "1"],
            ["2", "NECK", "1"],
            ["3", "HEAD", "1"],
            ["4", "FEELERS", "2"],
            ["5", "TAIL", "1"],
            ["6", "LEGS", "6"],
        ]
        for row in table:
            print(f"{row[0]:<16}{row[1]:<16}{row[2]:<20}")
        print_n_newlines(2)

    player = State(is_player=True)
    opponent = State(is_player=False)
    bugs_finished = 0

    while bugs_finished <= 0:
        diceroll = random.randint(1, 6)
        print()
        changed = handle_roll(diceroll, player)  # type: ignore

        diceroll = random.randint(1, 6)
        print()
        time.sleep(2)

        changed_op = handle_roll(diceroll, opponent)  # type: ignore

        changed = changed or changed_op

        if player.is_finished():
            print("YOUR BUG IS FINISHED.")
            bugs_finished += 1
        if opponent.is_finished():
            print("MY BUG IS FINISHED.")
            bugs_finished += 1
        if not changed:
            continue
        want_pictures = input("DO YOU WANT THE PICTURES? ")
        if want_pictures != "NO":
            print("*****YOUR BUG*****")
            print_n_newlines(2)
            player.display()
            print_n_newlines(4)
            print("*****MY BUG*****")
            print_n_newlines(3)
            opponent.display()

            if bugs_finished != 0:
                break

    print("I HOPE YOU ENJOYED THE GAME, PLAY IT AGAIN SOON!!")


```

这段代码是一个Python程序中的一个if语句。if语句是Python中的一种条件语句，用于判断一个表达式的值是否为真，如果是真，则执行if语句内部的代码，否则跳过if语句并继续执行if语句之后的代码。

在这段if语句中，表达式为`__name__ == "__main__"`。这里使用了Python中的自解释能力，即`__name__`是一个保留字，它会在程序运行时自动获取当前脚本的全名，而`__main__`则是在程序运行时获取当前脚本的可执行名。因此，这个表达式的值为`True`,if语句内部的代码将会被执行。

if语句的作用是让程序在运行时先执行if语句内部的代码，如果在运行时可以获取到`__name__`的全名，则自动执行该脚本。这个代码片段的作用是定义了一个函数`main()`，因此if语句内部的代码将会在程序运行时执行该函数体内的内容。


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


### Bullfight

In this simulated bullfight, you are the matador — i.e., the one with the principle role and the one who must kill the bull or be killed (or run from the ring).

On each pass of the bull, you may try:
- 0: Veronica (dangerous inside move of the cape)
- 1: Less dangerous outside move of the cape
- 2: Ordinary swirl of the cape

Or you may try to kill the bull:
- 4: Over the horns
- 5: In the chest

The crowd will determine what award you deserve, posthumously if necessary. The braver you are, the better the reward you receive. It’s nice to stay alive too. The better the job the picadores and toreadores do, the better your chances.

David Sweet of Dartmouth wrote the original version of this program. It was then modified by students at Lexington High School and finally by Steve North of Creative Computing.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=32)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=47)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

- There is a fundamental assumption in the pre-fight subroutine at line 1610, that the Picadores and Toreadores are more likely to do a bad job (and possibly get killed) with a low-quality bull. This appears to be a mistake in the original code, but should be retained.

- Lines 1800-1820 (part of the pre-fight subroutine) can never be reached.


# `17_Bullfight/csharp/Action.cs`

这段代码定义了一个名为"Action"的枚举类型，它列举了玩家在每个回合玩家可以采取的动作。枚举类型包含四个成员变量：Dodge、Kill、Freeze和Panic。

在枚举类型的定义之后，接下来没有任何其他代码活动，因此，只要这段代码没有被调用或者定义为类或函数，它就无法对程序产生任何实际影响。


```
﻿namespace Game
{
    /// <summary>
    /// Enumerates the different actions that the player can take on each round
    /// of the fight.
    /// </summary>
    public enum Action
    {
        /// <summary>
        /// Dodge the bull.
        /// </summary>
        Dodge,

        /// <summary>
        /// Kill the bull.
        /// </summary>
        Kill,

        /// <summary>
        /// Freeze in place and don't do anything.
        /// </summary>
        Panic
    }
}

```

# `17_Bullfight/csharp/ActionResult.cs`

这段代码定义了一个名为ActionResult的枚举类型，用于表示玩家在不同情况下的行动结果。

该程序的主要目的是在游戏过程中记录玩家在不同情况下的行动结果，以便在游戏结束时可以统计和统计。

具体来说，这个程序可以记录以下一些行动结果：

- FightContinues：战斗继续。
- PlayerFlees：玩家逃跑。
- BullGoresPlayer：牛股死了玩家。
- BullKillsPlayer：牛杀死了玩家。
- PlayerKillsBull：玩家宰了牛。
- Draw：打斗结束，没有任何结果。

玩家在每个行动中都可以选择其中一个行动结果，这将会导致游戏继续或结束。


```
﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Game
{
    /// <summary>
    /// Enumerates the different possible outcomes of the player's action.
    /// </summary>
    public enum ActionResult
    {
        /// <summary>
        /// The fight continues.
        /// </summary>
        FightContinues,

        /// <summary>
        /// The player fled from the ring.
        /// </summary>
        PlayerFlees,

        /// <summary>
        /// The bull has gored the player.
        /// </summary>
        BullGoresPlayer,

        /// <summary>
        /// The bull killed the player.
        /// </summary>
        BullKillsPlayer,

        /// <summary>
        /// The player killed the bull.
        /// </summary>
        PlayerKillsBull,

        /// <summary>
        /// The player attempted to kill the bull and both survived.
        /// </summary>
        Draw
    }
}

```

# `17_Bullfight/csharp/BullFight.cs`

This is a programming scenario written in C#. It appears to be a game ofSurvival where players must avoid or overcome a variety of challenges to survive until the end.

The player must make strategic decisions, such as whether to逃跑， fight, or surrender, in order to increase their chances of survival. The player can also receive rewards, such as extra lives or items, which can help them in their quest.

The game includes several different levels of difficulty, as well as a wide range of random events that can affect the player's survival. The player must use their survival skills and make the best decisions they can in order to come out on top.

This game is played by a AI ( artificial intelligence ) program and it does not have a graphical interface, it is all text-based.


```
﻿using System;
using System.Collections.Generic;

namespace Game
{
    /// <summary>
    /// Provides a method for simulating a bull fight.
    /// </summary>
    public static class BullFight
    {
        /// <summary>
        /// Begins a new fight.
        /// </summary>
        /// <param name="mediator">
        /// Object used to communicate with the player.
        /// </param>
        /// <returns>
        /// The sequence of events that take place during the fight.
        /// </returns>
        /// <remarks>
        /// After receiving each event, the caller must invoke the appropriate
        /// mediator method to inform this coroutine what to do next.  Failure
        /// to do so will result in an exception.
        /// </remarks>
        public static IEnumerable<Events.Event> Begin(Mediator mediator)
        {
            var random = new Random();
            var result = ActionResult.FightContinues;

            var bullQuality          = GetBullQuality();
            var toreadorePerformance = GetHelpQuality(bullQuality);
            var picadorePerformance  = GetHelpQuality(bullQuality);

            var bullStrength    = 6 - (int)bullQuality;
            var assistanceLevel = (12 - (int)toreadorePerformance - (int)picadorePerformance) * 0.1;
            var bravery         = 1.0;
            var style           = 1.0;
            var passNumber      = 0;

            yield return new Events.MatchStarted(
                bullQuality,
                toreadorePerformance,
                picadorePerformance,
                GetHumanCasualties(toreadorePerformance),
                GetHumanCasualties(picadorePerformance),
                GetHorseCasualties(picadorePerformance));

            while (result == ActionResult.FightContinues)
            {
                yield return new Events.BullCharging(++passNumber);

                var (action, riskLevel) = mediator.GetInput<(Action, RiskLevel)>();
                result = action switch
                {
                    Action.Dodge => TryDodge(riskLevel),
                    Action.Kill  => TryKill(riskLevel),
                    _            => Panic()
                };

                var first = true;
                while (result == ActionResult.BullGoresPlayer)
                {
                    yield return new Events.PlayerGored(action == Action.Panic, first);
                    first = false;

                    result = TrySurvive();
                    if (result == ActionResult.FightContinues)
                    {
                        yield return new Events.PlayerSurvived();

                        var runFromRing = mediator.GetInput<bool>();
                        if (runFromRing)
                            result = Flee();
                        else
                            result = IgnoreInjury(action);
                    }
                }
            }

            yield return new Events.MatchCompleted(
                result,
                bravery == 2,
                GetReward());

            Quality GetBullQuality() =>
                (Quality)random.Next(1, 6);

            Quality GetHelpQuality(Quality bullQuality) =>
                ((3.0 / (int)bullQuality) * random.NextDouble()) switch
                {
                    < 0.37 => Quality.Superb,
                    < 0.50 => Quality.Good,
                    < 0.63 => Quality.Fair,
                    < 0.87 => Quality.Poor,
                    _      => Quality.Awful
                };

            int GetHumanCasualties(Quality performance) =>
                performance switch
                {
                    Quality.Poor  => random.Next(0, 2),
                    Quality.Awful => random.Next(1, 3),
                    _             => 0
                };

            int GetHorseCasualties(Quality performance) =>
                performance switch
                {
                    // NOTE: The code for displaying a single horse casuality
                    //  following a poor picadore peformance was unreachable
                    //  in the original BASIC version.  I've assumed this was
                    //  a bug.
                    Quality.Poor  => 1,
                    Quality.Awful => random.Next(1, 3),
                    _             => 0
                };

            ActionResult TryDodge(RiskLevel riskLevel)
            {
                var difficultyModifier = riskLevel switch
                {
                    RiskLevel.High   => 3.0,
                    RiskLevel.Medium => 2.0,
                    _                => 0.5
                };

                var outcome = (bullStrength + (difficultyModifier / 10)) * random.NextDouble() /
                    ((assistanceLevel + (passNumber / 10.0)) * 5);

                if (outcome < 0.51)
                {
                    style += difficultyModifier;
                    return ActionResult.FightContinues;
                }
                else
                    return ActionResult.BullGoresPlayer;
            }

            ActionResult TryKill(RiskLevel riskLevel)
            {
                var luck = bullStrength * 10 * random.NextDouble() / (assistanceLevel * 5 * passNumber);

                return ((riskLevel == RiskLevel.High && luck > 0.2) || luck > 0.8) ?
                    ActionResult.BullGoresPlayer : ActionResult.PlayerKillsBull;
            }

            ActionResult Panic() =>
                ActionResult.BullGoresPlayer;

            ActionResult TrySurvive()
            {
                if (random.Next(2) == 0)
                {
                    bravery = 1.5;
                    return ActionResult.BullKillsPlayer;
                }
                else
                    return ActionResult.FightContinues;
            }

            ActionResult Flee()
            {
                bravery = 0.0;
                return ActionResult.PlayerFlees;
            }

            ActionResult IgnoreInjury(Action action)
            {
                if (random.Next(2) == 0)
                {
                    bravery = 2.0;
                    return action == Action.Dodge ? ActionResult.FightContinues : ActionResult.Draw;
                }
                else
                    return ActionResult.BullGoresPlayer;
            }

            Reward GetReward()
            {
                var score = CalculateScore();

                if (score * random.NextDouble() < 2.4)
                    return Reward.Nothing;
                else
                if (score * random.NextDouble() < 4.9)
                    return Reward.OneEar;
                else
                if (score * random.NextDouble() < 7.4)
                    return Reward.TwoEars;
                else
                    return Reward.CarriedFromRing;
            }

            double CalculateScore()
            {
                var score = 4.5;

                // Style
                score += style / 6;

                // Assisstance
                score -= assistanceLevel * 2.5;

                // Courage
                score += 4 * bravery;

                // Kill bonus
                score += (result == ActionResult.PlayerKillsBull) ? 4 : 2;

                // Match length
                score -= Math.Pow(passNumber, 2) / 120;

                // Difficulty
                score -= (int)bullQuality;

                return score;
            }
        }
    }
}

```

# `17_Bullfight/csharp/Controller.cs`

This is a class written in C# that defines a `View` class for a game. This class provides methods for checking the player's intention to flee (or not) and getting a yes or no response from the player.

The `GetPlayerRunsFromRing` method checks whether the player has intentions to flee. If the player does not intend to flee, the method will display a message to the player.

The `GetYesOrNo` method asks the player if they intend to flee or not. The method reads the player's input from the console and converts it to uppercase. If the input is "YES", the method returns `true`, otherwise it returns `false`.


```
﻿using System;

namespace Game
{
    /// <summary>
    /// Contains functions for getting input from the user.
    /// </summary>
    public static class Controller
    {
        /// <summary>
        /// Handles the initial interaction with the player.
        /// </summary>
        public static void StartGame()
        {
            View.ShowBanner();
            View.PromptShowInstructions();

            var input = Console.ReadLine();
            if (input is null)
                Environment.Exit(0);

            if (input.ToUpperInvariant() != "NO")
                View.ShowInstructions();

            View.ShowSeparator();
        }

        /// <summary>
        /// Gets the player's action for the current round.
        /// </summary>
        /// <param name="passNumber">
        /// The current pass number.
        /// </param>
        public static (Action action, RiskLevel riskLevel) GetPlayerIntention(int passNumber)
        {
            if (passNumber < 3)
                View.PromptKillBull();
            else
                View.PromptKillBullBrief();

            var attemptToKill = GetYesOrNo();

            if (attemptToKill)
            {
                View.PromptKillMethod();

                var input = Console.ReadLine();
                if (input is null)
                    Environment.Exit(0);

                return input switch
                {
                    "4" => (Action.Kill,  RiskLevel.High),
                    "5" => (Action.Kill,  RiskLevel.Low),
                    _   => (Action.Panic, default(RiskLevel))
                };
            }
            else
            {
                if (passNumber < 2)
                    View.PromptCapeMove();
                else
                    View.PromptCapeMoveBrief();

                var action = Action.Panic;
                var riskLevel = default(RiskLevel);

                while (action == Action.Panic)
                {
                    var input = Console.ReadLine();
                    if (input is null)
                        Environment.Exit(0);

                    (action, riskLevel) = input switch
                    {
                        "0" => (Action.Dodge, RiskLevel.High),
                        "1" => (Action.Dodge, RiskLevel.Medium),
                        "2" => (Action.Dodge, RiskLevel.Low),
                        _   => (Action.Panic, default(RiskLevel))
                    };

                    if (action == Action.Panic)
                        View.PromptDontPanic();
                }

                return (action, riskLevel);
            }
        }

        /// <summary>
        /// Gets the player's intention to flee (or not).
        /// </summary>
        /// <returns>
        /// True if the player flees; otherwise, false.
        /// </returns>
        public static bool GetPlayerRunsFromRing()
        {
            View.PromptRunFromRing();

            var playerFlees = GetYesOrNo();
            if (!playerFlees)
                View.ShowPlayerFoolhardy();

            return playerFlees;
        }

        /// <summary>
        /// Gets a yes or no response from the player.
        /// </summary>
        /// <returns>
        /// True if the user answered yes; otherwise, false.
        /// </returns>
        public static bool GetYesOrNo()
        {
            while (true)
            {
                var input = Console.ReadLine();
                if (input is null)
                    Environment.Exit(0);

                switch (input.ToUpperInvariant())
                {
                    case "YES":
                        return true;
                    case "NO":
                        return false;
                    default:
                        Console.WriteLine("INCORRECT ANSWER - - PLEASE TYPE 'YES' OR 'NO'.");
                        break;
                }
            }
        }
    }
}

```

# `17_Bullfight/csharp/Mediator.cs`



这段代码定义了一个名为 `Mediator` 的类，用于在两个游戏循环之间发送消息。类的实例上定义了以下方法：

- `Dodge(RiskLevel riskLevel)`：方法用于发出避开风险的信号，并指定风险级别。
- `Kill(RiskLevel riskLevel)`：方法用于发出杀戮的信号，并指定风险级别。
- `Panic()`：方法用于发出恐慌的信号，并指定默认的风险级别。
- `RunFromRing()`：方法用于从环中运行代码。
- `ContinueFighting()`：方法用于继续进行战斗。
- `GetInput<T>()`：方法用于从用户接收输入，其中 `T` 是输入的类型。

通过这些方法， Mediator 类可以在两个游戏循环之间发送消息，使游戏更加灵活和可扩展。例如，在游戏中，玩家可以使用 `GetInput<Kill>()` 方法来请求杀戮，游戏开发者可以使用 `Dodge(RiskLevel riskLevel)` 方法来回避风险。


```
﻿using System.Diagnostics;

namespace Game
{
    /// <summary>
    /// Facilitates sending messages between the two game loops.
    /// </summary>
    /// <remarks>
    /// This class serves as a little piece of glue in between the main program
    /// loop and the bull fight coroutine.  When the main program calls one of
    /// its methods, the mediator creates the appropriate input data that the
    /// bull fight coroutine later retrieves with <see cref="GetInput{T}"/>.
    /// </remarks>
    public class Mediator
    {
        private object? m_input;

        public void Dodge(RiskLevel riskLevel) =>
            m_input = (Action.Dodge, riskLevel);

        public void Kill(RiskLevel riskLevel) =>
            m_input = (Action.Kill, riskLevel);

        public void Panic() =>
            m_input = (Action.Panic, default(RiskLevel));

        public void RunFromRing() =>
            m_input = true;

        public void ContinueFighting() =>
            m_input = false;

        /// <summary>
        /// Gets the next input from the user.
        /// </summary>
        /// <typeparam name="T">
        /// The type of input to receive.
        /// </typeparam>
        public T GetInput<T>()
        {
            Debug.Assert(m_input is not null, "No input received");
            Debug.Assert(m_input.GetType() == typeof(T), "Invalid input received");
            var result = (T)m_input;
            m_input = null;
            return result;
        }
    }
}

```

# `17_Bullfight/csharp/Program.cs`

这段代码是一个C#程序，定义了一个名为Game的命名空间，其中包含一个名为Program的类。这个Program类包含一个名为Main的静态方法。

Main方法中的static void Main()方法是程序的入口点。在这个方法中，首先创建一个名为Controller的类的一个新的实例，然后调用其StartGame方法来启动游戏。

然后，定义一个名为Mediator的类，继承自System.Collections.Generic类。这个Mediator类用于管理游戏中的所有事件。

接着，定义一个名为BullFight的类，继承自Events.Event类。这个BullFight类包含一个名为Begin的静态方法，用于启动一个新的事件循环，并且包含一个名为evt的参数，用于指定要处理的事件类型。

在Main方法的程序段中，使用一个for循环来遍历BullFight类中所有的事件类型。对于每个事件类型，调用其相应的方法来处理。在这些方法中，switch语句用于根据事件类型来决定要执行的操作。

具体来说，如果事件类型为Events.MatchStarted，将显示开始游戏的条件。如果事件类型为Events.BullCharging，将显示玩家开始攻击的界面，并获取攻击的目标。在攻击面板上，根据玩家的行动，调用不同的方法来处理。如果玩家决定躲避，将调用名为Dodge的方法，如果玩家决定杀死敌人，将调用名为Kill的方法，如果玩家决定恐慌，将调用名为Panic的方法。

如果事件类型为Events.PlayerGored，将显示玩家是否受伤的界面，并获取玩家是否心跳的值。如果玩家受伤，将调用名为RunFromRing的方法，让玩家从 ring 中出来。如果玩家没有心跳，将继续进行游戏。如果事件类型为Events.MatchCompleted，将显示最终结果，包括比赛的结果，玩家是否可以从环中逃跑，以及奖励。


```
﻿namespace Game
{
    class Program
    {
        static void Main()
        {
            Controller.StartGame();

            var mediator = new Mediator();
            foreach (var evt in BullFight.Begin(mediator))
            {
                switch (evt)
                {
                    case Events.MatchStarted matchStarted:
                        View.ShowStartingConditions(matchStarted);
                        break;

                    case Events.BullCharging bullCharging:
                        View.ShowStartOfPass(bullCharging.PassNumber);
                        var (action, riskLevel) = Controller.GetPlayerIntention(bullCharging.PassNumber);
                        switch (action)
                        {
                            case Action.Dodge:
                                mediator.Dodge(riskLevel);
                                break;
                            case Action.Kill:
                                mediator.Kill(riskLevel);
                                break;
                            case Action.Panic:
                                mediator.Panic();
                                break;
                        }
                        break;

                    case Events.PlayerGored playerGored:
                        View.ShowPlayerGored(playerGored.Panicked, playerGored.FirstGoring);
                        break;

                    case Events.PlayerSurvived:
                        View.ShowPlayerSurvives();
                        if (Controller.GetPlayerRunsFromRing())
                            mediator.RunFromRing();
                        else
                            mediator.ContinueFighting();
                        break;

                    case Events.MatchCompleted matchCompleted:
                        View.ShowFinalResult(matchCompleted.Result, matchCompleted.ExtremeBravery, matchCompleted.Reward);
                        break;
                }
            }
        }
    }
}

```

# `17_Bullfight/csharp/Quality.cs`

这段代码定义了一个名为"Quality"的枚举类型，用于表示游戏中的不同品质水平。枚举类型包括6个成员变量，分别表示"Superb"、"Good"、"Fair"、"Poor"和"Awful"等级。

枚举类型定义了一种序列化机制，可以使程序在需要时将对象序列化为特定的Quality等级。例如，在游戏加载时，玩家可以通过选择游戏难度来加载不同的游戏品质，游戏开发者可以使用SerializedObject来序列化玩家选择的难度，然后在游戏运行时将其转换为Quality等级并应用给游戏对象。

枚举类型还可以用于比较和排序，例如在查找最佳的游戏难度之前，程序可以先将所有游戏对象按Quality等级排序，然后从中查找最佳等级的游戏对象。


```
﻿namespace Game
{
    /// <summary>
    /// Enumerates the different levels of quality in the game.
    /// </summary>
    /// <remarks>
    /// Quality applies both to the bull and to the help received from the
    /// toreadores and picadores.  Note that the ordinal values are significant
    /// (these are used in various calculations).
    /// </remarks>
    public enum Quality
    {
        Superb  = 1,
        Good    = 2,
        Fair    = 3,
        Poor    = 4,
        Awful   = 5
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `17_Bullfight/csharp/Reward.cs`

这段代码定义了一个名为 "Reward" 的枚举类型，用于描述玩家可以获得的奖励类型。枚举类型包含四种奖励类型：Nothing、OneEar、TwoEars 和 CarriedFromRing。在大多数情况下，玩家可以通过完成游戏任务或达成特定的目标来获得这些奖励。


```
﻿namespace Game
{
    /// <summary>
    /// Enumerates the different things the player can be awarded.
    /// </summary>
    public enum Reward
    {
        Nothing,
        OneEar,
        TwoEars,
        CarriedFromRing
    }
}

```

# `17_Bullfight/csharp/RiskLevel.cs`

这段代码定义了一个名为RiskLevel的枚举类型，用于描述游戏中移动操作的风险水平。该枚举类型包含三个成员变量，分别命名为Low、Medium和High，分别表示低风险、中等风险和高风险。

这个枚举类型的目的是在代码中为不同的移动操作风险水平提供一种简化的表示方式。通过枚举类型的定义，游戏开发者可以使用这种类型安全的语法来表示风险水平，而不是使用具体的数字或字符串。例如，在游戏设计中，移动操作的风险水平可以使用数字值来表示，这样可以更加清楚地表达每个操作的风险水平。而使用枚举类型，则可以更加方便地在不同情况下使用不同的风险水平。


```
﻿namespace Game
{
    /// <summary>
    /// Enumerates the different levels of risk for manoeuvres in the game.
    /// </summary>
    public enum RiskLevel
    {
        Low,
        Medium,
        High
    }
}

```

# `17_Bullfight/csharp/View.cs`

This is a class that contains several methods for giving the player instructions on how to interact with the game world. These methods include "PromptShowInstructions", "PromptKillBull", "PromptKillBullBrief", "PromptKillMethod", "PromptCapeMove", and "PromptCapeMoveBrief". Each method displays a message on the screen and prompts the player to enter a response. The responses include text, number or ole, and some other options.

It's important to note that some of the prompts may be filtered out by the developer depending on the game design or story.


```
﻿using System;

namespace Game
{
    /// <summary>
    /// Contains functions for displaying information to the user.
    /// </summary>
    public static class View
    {
        private static readonly string[] QualityString = { "SUPERB", "GOOD", "FAIR", "POOR", "AWFUL" };

        public static void ShowBanner()
        {
            Console.WriteLine("                                  BULL");
            Console.WriteLine("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
        }

        public static void ShowInstructions()
        {
            Console.WriteLine("HELLO, ALL YOU BLOODLOVERS AND AFICIONADOS.");
            Console.WriteLine("HERE IS YOUR BIG CHANCE TO KILL A BULL.");
            Console.WriteLine();
            Console.WriteLine("ON EACH PASS OF THE BULL, YOU MAY TRY");
            Console.WriteLine("0 - VERONICA (DANGEROUS INSIDE MOVE OF THE CAPE)");
            Console.WriteLine("1 - LESS DANGEROUS OUTSIDE MOVE OF THE CAPE");
            Console.WriteLine("2 - ORDINARY SWIRL OF THE CAPE.");
            Console.WriteLine();
            Console.WriteLine("INSTEAD OF THE ABOVE, YOU MAY TRY TO KILL THE BULL");
            Console.WriteLine("ON ANY TURN: 4 (OVER THE HORNS), 5 (IN THE CHEST).");
            Console.WriteLine("BUT IF I WERE YOU,");
            Console.WriteLine("I WOULDN'T TRY IT BEFORE THE SEVENTH PASS.");
            Console.WriteLine();
            Console.WriteLine("THE CROWD WILL DETERMINE WHAT AWARD YOU DESERVE");
            Console.WriteLine("(POSTHUMOUSLY IF NECESSARY).");
            Console.WriteLine("THE BRAVER YOU ARE, THE BETTER THE AWARD YOU RECEIVE.");
            Console.WriteLine();
            Console.WriteLine("THE BETTER THE JOB THE PICADORES AND TOREADORES DO,");
            Console.WriteLine("THE BETTER YOUR CHANCES ARE.");
        }

        public static void ShowSeparator()
        {
            Console.WriteLine();
            Console.WriteLine();
        }

        public static void ShowStartingConditions(Events.MatchStarted matchStarted)
        {
            ShowBullQuality();
            ShowHelpQuality("TOREADORES", matchStarted.ToreadorePerformance, matchStarted.ToreadoresKilled, 0);
            ShowHelpQuality("PICADORES", matchStarted.PicadorePerformance, matchStarted.PicadoresKilled, matchStarted.HorsesKilled);

            void ShowBullQuality()
            {
                Console.WriteLine($"YOU HAVE DRAWN A {QualityString[(int)matchStarted.BullQuality - 1]} BULL.");

                if (matchStarted.BullQuality > Quality.Poor)
                {
                    Console.WriteLine("YOU'RE LUCKY");
                }
                else
                if (matchStarted.BullQuality < Quality.Good)
                {
                    Console.WriteLine("GOOD LUCK.  YOU'LL NEED IT.");
                    Console.WriteLine();
                }

                Console.WriteLine();
            }

            static void ShowHelpQuality(string helperName, Quality helpQuality, int helpersKilled, int horsesKilled)
            {
                Console.WriteLine($"THE {helperName} DID A {QualityString[(int)helpQuality - 1]} JOB.");

                // NOTE: The code below makes some *strong* assumptions about
                //  how the casualty numbers were generated.  It is written
                //  this way to preserve the behaviour of the original BASIC
                //  version, but it would make more sense ignore the helpQuality
                //  parameter and just use the provided numbers to decide what
                //  to display.
                switch (helpQuality)
                {
                    case Quality.Poor:
                        if (horsesKilled > 0)
                            Console.WriteLine($"ONE OF THE HORSES OF THE {helperName} WAS KILLED.");

                        if (helpersKilled > 0)
                            Console.WriteLine($"ONE OF THE {helperName} WAS KILLED.");
                        else
                            Console.WriteLine($"NO {helperName} WERE KILLED.");
                        break;

                    case Quality.Awful:
                        if (horsesKilled > 0)
                            Console.WriteLine($" {horsesKilled} OF THE HORSES OF THE {helperName} KILLED.");

                        Console.WriteLine($" {helpersKilled} OF THE {helperName} KILLED.");
                        break;
                }
            }
        }

        public static void ShowStartOfPass(int passNumber)
        {
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine($"PASS NUMBER {passNumber}");
        }

        public static void ShowPlayerGored(bool playerPanicked, bool firstGoring)
        {
            Console.WriteLine((playerPanicked, firstGoring) switch
            {
                (true,  true) => "YOU PANICKED.  THE BULL GORED YOU.",
                (false, true) => "THE BULL HAS GORED YOU!",
                (_, false)    => "YOU ARE GORED AGAIN!"
            });
        }

        public static void ShowPlayerSurvives()
        {
            Console.WriteLine("YOU ARE STILL ALIVE.");
            Console.WriteLine();
        }

        public static void ShowPlayerFoolhardy()
        {
            Console.WriteLine("YOU ARE BRAVE.  STUPID, BUT BRAVE.");
        }

        public static void ShowFinalResult(ActionResult result, bool extremeBravery, Reward reward)
        {
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();

            switch (result)
            {
                case ActionResult.PlayerFlees:
                    Console.WriteLine("COWARD");
                    break;
                case ActionResult.BullKillsPlayer:
                    Console.WriteLine("YOU ARE DEAD.");
                    break;
                case ActionResult.PlayerKillsBull:
                    Console.WriteLine("YOU KILLED THE BULL!");
                    break;
            }

            if (result == ActionResult.PlayerFlees)
            {
                Console.WriteLine("THE CROWD BOOS FOR TEN MINUTES.  IF YOU EVER DARE TO SHOW");
                Console.WriteLine("YOUR FACE IN A RING AGAIN, THEY SWEAR THEY WILL KILL YOU--");
                Console.WriteLine("UNLESS THE BULL DOES FIRST.");
            }
            else
            {
                if (extremeBravery)
                    Console.WriteLine("THE CROWD CHEERS WILDLY!");
                else
                if (result == ActionResult.PlayerKillsBull)
                {
                    Console.WriteLine("THE CROWD CHEERS!");
                    Console.WriteLine();
                }

                Console.WriteLine("THE CROWD AWARDS YOU");
                switch (reward)
                {
                    case Reward.Nothing:
                        Console.WriteLine("NOTHING AT ALL.");
                        break;
                    case Reward.OneEar:
                        Console.WriteLine("ONE EAR OF THE BULL.");
                        break;
                    case Reward.TwoEars:
                        Console.WriteLine("BOTH EARS OF THE BULL!");
                        Console.WriteLine("OLE!");
                        break;
                    default:
                        Console.WriteLine("OLE!  YOU ARE 'MUY HOMBRE'!! OLE!  OLE!");
                        break;
                }
            }

            Console.WriteLine();
            Console.WriteLine("ADIOS");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
        }

        public static void PromptShowInstructions()
        {
            Console.Write("DO YOU WANT INSTRUCTIONS? ");
        }

        public static void PromptKillBull()
        {
            Console.WriteLine("THE BULL IS CHARGING AT YOU!  YOU ARE THE MATADOR--");
            Console.Write("DO YOU WANT TO KILL THE BULL? ");
        }

        public static void PromptKillBullBrief()
        {
            Console.Write("HERE COMES THE BULL.  TRY FOR A KILL? ");
        }

        public static void PromptKillMethod()
        {
            Console.WriteLine();
            Console.WriteLine("IT IS THE MOMENT OF TRUTH.");
            Console.WriteLine();

            Console.Write("HOW DO YOU TRY TO KILL THE BULL? ");
        }

        public static void PromptCapeMove()
        {
            Console.Write("WHAT MOVE DO YOU MAKE WITH THE CAPE? ");
        }

        public static void PromptCapeMoveBrief()
        {
            Console.Write("CAPE MOVE? ");
        }

        public static void PromptDontPanic()
        {
            Console.WriteLine("DON'T PANIC, YOU IDIOT!  PUT DOWN A CORRECT NUMBER");
            Console.Write("? ");
        }

        public static void PromptRunFromRing()
        {
            Console.Write("DO YOU RUN FROM THE RING? ");
        }
    }
}

```

# `17_Bullfight/csharp/Events/BullCharging.cs`

这段代码定义了一个名为 "Game.Events" 的命名空间，其中定义了一个名为 "BullCharging" 的记录类型，其概述为 "Indicates that the bull is charing the player。"。

具体来说，这个记录类型包含一个名为 "PassNumber" 的整数成员，它用于表示 bull(或者称为牛)的充电次数。当玩家被 bull 攻击时，如果此记录的 "PassNumber" 成员的值等于 1，那么这个记录将被创建并记录下来，以便在游戏进程中跟踪和记录这种事件。

这个记录类型还定义了一个名为 "Event" 的枚举类型，其值为 "BullCharging"，用于表示这个记录类型的所有可能值。

最后，在代码文件的顶部，使用了一个导出声明，允许在使用这个命名空间的游戏组件中访问它。


```
﻿namespace Game.Events
{
    /// <summary>
    /// Indicates that the bull is charing the player.
    /// </summary>
    public sealed record BullCharging(int PassNumber) : Event;
}

```

# `17_Bullfight/csharp/Events/Event.cs`

这段代码定义了一个名为 "Game.Events" 的命名空间，其中包含一个名为 "Event" 的抽象类。

这个 "Event" 抽象类表示所有游戏中事件的基本模板。在任何游戏事件中，都可以使用这个模板来定义一个事件以及定义事件执行时的相关参数。

因为所有事件都必须从该抽象类派生，所以任何派生自 "Event" 的具体事件类都必须实现这个抽象类的所有成员属性和方法。

举个例子，如果你想要在游戏中实现一个玩家行动的事件，你可以创建一个名为 "MyActionEvent" 的具体事件类，并使用 "Event" 抽象类来定义该事件。然后，你可以实现 "MyActionEvent" 类中的所有成员属性和方法，以实现该事件的标准行为。


```
﻿namespace Game.Events
{
    /// <summary>
    /// Common base class for all events in the game.
    /// </summary>
    public abstract record Event();
}

```

# `17_Bullfight/csharp/Events/MatchCompleted.cs`

这段代码定义了一个名为“MatchCompleted”的事件记录类型，表示战斗已经完成。这个类型包含三个属性：
1. 获取战斗结果的泛型类型：ActionResult<bool>。
2. 表示战斗是否使用了极端勇气的布尔类型：bool。
3. 表示战斗中获得的奖励的奖励类型：Reward<int>。

这个事件记录类型可以被用于在游戏中的战斗事件中，当战斗完成时触发。例如，游戏进程可以记录一个“MatchCompleted”事件，当战斗完成时，可以调用这个事件，然后通过这个事件获取战斗结果、判断是否使用了极端勇气、以及获取获得的奖励。


```
﻿namespace Game.Events
{
    /// <summary>
    /// Indicates that the fight has completed.
    /// </summary>
    public sealed record MatchCompleted(ActionResult Result, bool ExtremeBravery, Reward Reward) : Event;
}

```

# `17_Bullfight/csharp/Events/MatchStarted.cs`

这段代码定义了一个名为“MatchStarted”的记录类型，用于表示一场新的匹配已经启动。这个记录类型包含以下字段：

- Quality: 匹配的质量 (低质量、中等质量和高质量)
- ToreadorePerformance: 选手Toreadore的表演质量 (0表示未评分，1表示表现优秀，2表示表现一般，3表示表现差)
- PicadorePerformance: 选手Picadore的表演质量 (0表示未评分，1表示表现优秀，2表示表现一般，3表示表现差)
- ToreatorsKilled: 已知Toreador选手杀死的使用机数量
- PicadoresKilled: 已知Picadore选手杀死的使用机数量
- HorsesKilled: 已知Horse选手杀死的使用机数量

MatchStarted记录类型由事件组成，当一个MatchStarted实例被创建时，它表示一场新的比赛已经开始了。


```
﻿namespace Game.Events
{
    /// <summary>
    /// Indicates that a new match has started.
    /// </summary>
    public sealed record MatchStarted(
        Quality BullQuality,
        Quality ToreadorePerformance,
        Quality PicadorePerformance,
        int ToreadoresKilled,
        int PicadoresKilled,
        int HorsesKilled) : Event;
}

```

# `17_Bullfight/csharp/Events/PlayerGored.cs`

这段代码定义了一个名为“PlayerGored”的事件记录类，该类表示玩家被公牛扎扎实盖了。这个类包含两个事件：玩家被公牛扎扎实盖了（event type: true）以及表示玩家是否恐慌了（event type: false）。

换句话说，这个类描述了一个游戏中的事件，当玩家被公牛扎扎实盖时，会触发这个事件。虽然这个事件看起来很重要，但实际游戏中可能还有其他事情需要发生，所以游戏的逻辑代码可能还需要进一步处理这个事件。


```
﻿namespace Game.Events
{
    /// <summary>
    /// Indicates that the player has been gored by the bull.
    /// </summary>
    public sealed record PlayerGored(bool Panicked, bool FirstGoring) : Event;
}

```

# `17_Bullfight/csharp/Events/PlayerSurvived.cs`

这是一个发生在游戏中的事件记录namespace，其中定义了一个名为PlayerSurvived的公共记录类型。这个记录类型表示玩家已经成功地在愤怒的 bull上存活下来，这是一种非常鼓舞人心的事件。

PlayerSurvived记录类型中包含一个事件，当玩家在游戏中的某个场景中，被 bull攻击并且没有立即死亡的情况下，系统将记录这个事件。这样，玩家就可以在游戏中获得一个buff(增益状态)，以表示他们已经成功地在 bull上存活下来。

这个事件记录可以用于在游戏进行中的UI元素，比如在玩家成功地在 bull上存活下来时，显示一个标题“你存活了下来！”。这个记录也可以在游戏统计中使用，以表示游戏中的玩家存活率。


```
﻿namespace Game.Events
{
    /// <summary>
    /// Indicates that the player has survived being gored by the bull.
    /// </summary>
    public sealed record PlayerSurvived() : Event;
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `17_Bullfight/javascript/bullfight.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

`print` 函数的作用是在文档中的一个元素（具体是哪个元素没有在代码中明确说明）中添加一个新的文本节点，并将传入的参数作为文本内容。这里 `print` 函数会在调用时添加一个新的文本节点，并将其添加到指定的元素中，然后将其合并文档。

`input` 函数的作用是从用户那里获取一个字符串，并在客户端代码中将其存储在变量 `input_str` 中。它通过创建一个 `INPUT` 元素，并且在元素的 `type` 属性中设置为 `text`，设置元素的 `length` 属性为 `50`，这样就可以从用户那里获取一个字符串，并将其存储在 `input_str` 变量中。该函数通过监听 `keydown` 事件来捕获用户按键，当用户按下了 `CTRL+C`（或者 `Cmd+C`）时，它会在终端窗口中输出获取到的字符串，并将其存储在 `input_str` 变量中。


```
// BULLFIGHT
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

这段代码定义了一个名为 `tab` 的函数，接受一个参数 `space`。该函数的作用是在空白字符串中插入指定的字符或字符串，直到达到指定的空白字符数为止。

具体来说，代码中定义了一个名为 `str` 的变量，用于存储插入空白字符后的空白字符串。然后，代码使用一个 `while` 循环来遍历空白字符数 `space`。在循环中，代码使用字符 `" "`(双引号)来存储每个空白字符。

每次循环，代码都会将 `str` 中的左移一位，并将空白字符数 `space` 减去。当循环结束后，`str` 应该包含从左到右的四个空白字符，即在空白字符串中插入了一个字符或字符串。

此外，代码还定义了五个变量 `a`、`b`、`c`、`l` 和 `t`，分别用于存储输入的用户名和密码。


```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

var a;
var b;
var c;
var l;
var t;
var as;
var bs;
```

该代码定义了一个名为"af"的函数和一个名为"cf"的函数。接着定义了一个名为"d"的数组，数组中有六个元素，这些元素都是字符串类型的随机数。

然后，该代码实现了一个af函数和一个cf函数。这两个函数分别从两个不同的数组中获取随机元素，并将它们与一个变量q相乘，然后将得到的结果乘以另外一个变量Math.random()生成的随机数，最后对结果进行一些计算得到一个数值。

af函数的实现主要是：通过数学计算得到一个0到1之间的随机数，再乘以2，加上1，最后取整得到一个0到1之间的随机数。

cf函数的实现主要是在af函数的基础上，对输入参数q进行一些计算，得到一个数值，然后将这个数值乘以Math.random()生成的随机数，最后再对结果进行一些计算得到一个数值。

d数组中的元素可能是通过一些随机数生成器生成的，但生成的随机数可能不是0到1之间的随机数，因此af和cf函数中可能需要对随机数进行一些额外的处理来确保结果是0到1之间的随机数。


```
var d = [];
var ls = [, "SUPERB", "GOOD", "FAIR", "POOR", "AWFUL"];

function af(k)
{
    return Math.floor(Math.random() * 2 + 1);
}

function cf(q)
{
    return df(q) * Math.random();
}

function df(q)
{
    return (4.5 + l / 6 - (d[1] + d[2]) * 2.5 + 4 * d[4] + 2 * d[5] - Math.pow(d[3], 2) / 120 - a);
}

```

该函数的作用是设置一些辅助函数，以方便后续程序的操作。具体解释如下：

1. `Math.random()` 函数用于生成一个 0 到 1 之间的随机数。
2. `b / a * Math.random()` 用于生成一个介于 0 和 1 之间的随机数，然后将其除以 `a` 并乘以该随机数，最后将结果赋值给 `b`。
3. `if (b < 0.37)` 用于判断生成的随机数是否小于 0.37。如果是，则执行以下语句：
   a. `c = 0.5`
   b. `if (as != "TOREAD")`
     c. `print(af(0) + " OF THE HORSES OF THE " + as + bs + " KILLED.\n")`
   c. `print(af(0) + " OF THE " + as + bs + " KILLED.\n")`
   d. `print("\n")`
   e. 如果 `as == "TOREAD"`，则跳过第三条语句，否则第三条语句为前两条语句。
4. `else if (b < 0.5)` 如果生成的不随机数大于 0.37 且不是 `TOREOAD`，则执行以下语句：
   a. `c = 0.4`
   b. `if (as != "TOREOAD")`
     c. `print(af(0) + " OF THE HORSES OF THE " + as + bs + " KILLED.\n")`
   c. `print(af(0) + " OF THE " + as + bs + " KILLED.\n")`
   d. `print("\n")`
   e. 如果 `as == "TOREOAD"`，则跳过第三条语句，否则第三条语句为前两条语句。
5. `else if (b < 0.63)` 如果生成的不随机数大于 0.5 且不是 `TOREOAD`，则执行以下语句：
   a. `c = 0.3`
   b. `if (as != "TOREOAD")`
     c. `print(af(0) + " OF THE HORSES OF THE " + as + bs + " KILLED.\n")`
   c. `print(af(0) + " OF THE " + as + bs + " KILLED.\n")`
   d. `print("\n")`
   e. 如果 `as == "TOREOAD"`，则跳过第三条语句，否则第三条语句为前两条语句。
6. `else` 如果前五次循环没有生成 0.37 以下的随机数，则执行以下语句：
   a. `t = 10 * c + 0.2`
   b. `print("THE " + as + bs + " DID A " + ls[t] + " JOB.\n")`
   c. `if (4 <= t)`
     d. `if (5 != t)`
       e. `switch (af(0))`
         f. `case 1:`
           g. `print("ONE OF THE " + as + bs + " WAS KILLED.\n");`
           `break;`
         f. `case 2:`
           g. `print("NO " + as + b + " WERE KILLED.\n");`
           `break;`
         f. `case 3:`
           g. `print(af(0) + " OF THE HORSES OF THE " + as + bs + " KILLED.\n");`
           `break;`
         f. `case 4:`
           g. `print(af(0) + " OF THE " + as + bs + " KILLED.\n");`
           `break;`
         f. `case 5:`
           g. `print("\n");`
           `break;`
       `e. `endswitch`
       f. `print("\n")`
   f. `print("\\n")`
   g. `break;`
   h. `else`


```
function setup_helpers()
{
    b = 3 / a * Math.random();
    if (b < 0.37)
        c = 0.5;
    else if (b < 0.5)
        c = 0.4;
    else if (b < 0.63)
        c = 0.3;
    else if (b < 0.87)
        c = 0.2;
    else
        c = 0.1;
    t = Math.floor(10 * c + 0.2);
    print("THE " + as + bs + " DID A " + ls[t] + " JOB.\n");
    if (4 <= t) {
        if (5 != t) {
            // Lines 1800 and 1810 of original program are unreachable
            switch (af(0)) {
                case 1:
                    print("ONE OF THE " + as + bs + " WAS KILLED.\n");
                    break;
                case 2:
                    print("NO " + as + b + " WERE KILLED.\n");
                    break;
            }
        } else {
            if (as != "TOREAD")
                print(af(0) + " OF THE HORSES OF THE " + as + bs + " KILLED.\n");
            print(af(0) + " OF THE " + as + bs + " KILLED.\n");
        }
    }
    print("\n");
}

```

It looks like you\'re trying to write a story about a coward who is forced to face a group of cows. Is there anything specific you would like me to add or change?



```
// Main program
async function main()
{
    print(tab(34) + "BULL\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    l = 1;
    print("DO YOU WANT INSTRUCTIONS");
    str = await input();
    if (str != "NO") {
        print("HELLO, ALL YOU BLOODLOVERS AND AFICIONADOS.\n");
        print("HERE IS YOUR BIG CHANCE TO KILL A BULL.\n");
        print("\n");
        print("ON EACH PASS OF THE BULL, YOU MAY TRY\n");
        print("0 - VERONICA (DANGEROUS INSIDE MOVE OF THE CAPE)\n");
        print("1 - LESS DANGEROUS OUTSIDE MOVE OF THE CAPE\n");
        print("2 - ORDINARY SWIRL OF THE CAPE.\n");
        print("\n");
        print("INSTEAD OF THE ABOVE, YOU MAY TRY TO KILL THE BULL\n");
        print("ON ANY TURN: 4 (OVER THE HORNS), 5 (IN THE CHEST).\n");
        print("BUT IF I WERE YOU,\n");
        print("I WOULDN'T TRY IT BEFORE THE SEVENTH PASS.\n");
        print("\n");
        print("THE CROWD WILL DETERMINE WHAT AWARD YOU DESERVE\n");
        print("(POSTHUMOUSLY IF NECESSARY).\n");
        print("THE BRAVER YOU ARE, THE BETTER THE AWARD YOU RECEIVE.\n");
        print("\n");
        print("THE BETTER THE JOB THE PICADORES AND TOREADORES DO,\n");
        print("THE BETTER YOUR CHANCES ARE.\n");
    }
    print("\n");
    print("\n");
    d[5] = 1;
    d[4] = 1;
    d[3] = 0;
    a = Math.floor(Math.random() * 5 + 1);
    print("YOU HAVE DRAWN A " + ls[a] + " BULL.\n");
    if (a > 4) {
        print("YOU'RE LUCKY.\n");
    } else if (a < 2) {
        print("GOOD LUCK.  YOU'LL NEED IT.\n");
        print("\n");
    }
    print("\n");
    as = "PICADO";
    bs = "RES";
    setup_helpers();
    d[1] = c;
    as = "TOREAD";
    bs = "ORES";
    setup_helpers();
    d[2] = c;
    print("\n");
    print("\n");
    z = 0;
    while (z == 0) {
        d[3]++;
        print("PASS NUMBER " + d[3] + "\n");
        if (d[3] >= 3) {
            print("HERE COMES THE BULL.  TRY FOR A KILL");
            while (1) {
                str = await input();
                if (str != "YES" && str != "NO")
                    print("INCORRECT ANSWER - - PLEASE TYPE 'YES' OR 'NO'.\n");
                else
                    break;
            }
            z1 = (str == "YES") ? 1 : 2;
            if (z1 != 1) {
                print("CAPE MOVE");
            }
        } else {
            print("THE BULL IS CHARGING AT YOU!  YOU ARE THE MATADOR--\n");
            print("DO YOU WANT TO KILL THE BULL");
            while (1) {
                str = await input();
                if (str != "YES" && str != "NO")
                    print("INCORRECT ANSWER - - PLEASE TYPE 'YES' OR 'NO'.\n");
                else
                    break;
            }
            z1 = (str == "YES") ? 1 : 2;
            if (z1 != 1) {
                print("WHAT MOVE DO YOU MAKE WITH THE CAPE");
            }
        }
        gore = 0;
        if (z1 != 1) {
            while (1) {
                e = parseInt(await input());
                if (e >= 3) {
                    print("DON'T PANIC, YOU IDIOT!  PUT DOWN A CORRECT NUMBER\n");
                } else {
                    break;
                }
            }
            if (e == 0)
                m = 3;
            else if (e == 1)
                m = 2;
            else
                m = 0.5;
            l += m;
            f = (6 - a + m / 10) * Math.random() / ((d[1] + d[2] + d[3] / 10) * 5);
            if (f < 0.51)
                continue;
            gore = 1;
        } else {
            z = 1;
            print("\n");
            print("IT IS THE MOMENT OF THE TRUTH.\n");
            print("\n");
            print("HOW DO YOU TRY TO KILL THE BULL");
            h = parseInt(await input());
            if (h != 4 && h != 5) {
                print("YOU PANICKED.  THE BULL GORED YOU.\n");
                gore = 2;
            } else {
                k = (6 - a) * 10 * Math.random() / ((d[1] + d[2]) * 5 * d[3]);
                if (h != 4) {   // Bug in original game, it says J instead of H
                    if (k > 0.2)
                        gore = 1;
                } else {
                    if (k > 0.8)
                        gore = 1;
                }
                if (gore == 0) {
                    print("YOU KILLED THE BULL!\n");
                    d[5] = 2;
                    break;
                }
            }
        }
        if (gore) {
            if (gore == 1)
                print("THE BULL HAS GORED YOU!\n");
            kill = false;
            while (1) {
                if (af(0) == 1) {
                    print("YOU ARE DEAD.\n");
                    d[4] = 1.5;
                    kill = true;
                    break;
                }
                print("YOU ARE STILL ALIVE.\n");
                print("\n");
                print("DO YOU RUN FROM THE RING");
                while (1) {
                    str = await input();
                    if (str != "YES" && str != "NO")
                        print("INCORRECT ANSWER - - PLEASE TYPE 'YES' OR 'NO'.\n");
                    else
                        break;
                }
                z1 = (str == "YES") ? 1 : 2;
                if (z1 != 2) {
                    print("COWARD\n");
                    d[4] = 0;
                    kill = true;
                    break;
                }
                print("YOU ARE BRAVE.  STUPID, BUT BRAVE.\n");
                if (af(0) == 1) {
                    d[4] = 2;
                    kill = false;
                    break;
                }
                print("YOU ARE GORED AGAIN!\n");
            }
            if (kill)
                break;
            continue;
        }
    }
    print("\n");
    print("\n");
    print("\n");
    if (d[4] == 0) {
        print("THE CROWD BOOS FOR TEN MINUTES.  IF YOU EVER DARE TO SHOW\n");
        print("YOUR FACE IN A RING AGAIN, THEY SWEAR THEY WILL KILL YOU--\n");
        print("UNLESS THE BULL DOES FIRST.\n");
    } else {
        if (d[4] == 2) {
            print("THE CROWD CHEERS WILDLY!\n");
        } else if (d[5] == 2) {
            print("THE CROWD CHEERS!\n");
            print("\n");
        }
        print("THE CROWD AWARDS YOU\n");
        if (cf(0) < 2.4) {
            print("NOTHING AT ALL.\n");
        } else if (cf(0) < 4.9) {
            print("ONE EAR OF THE BULL.\n");
        } else if (cf(0) < 7.4) {
            print("BOTH EARS OF THE BULL!\n");
            print("OLE!\n");
        } else {
            print("OLE!  YOU ARE 'MUY HOMBRE'!! OLE!  OLE!\n");
        }
        print("\n");
        print("ADIOS\n");
        print("\n");
        print("\n");
        print("\n");
    }
}


```

这是 C 语言中的一个程序，名为 "main()"。程序的作用是启动 C 语言程序并开始执行。在实际运行时，程序会首先读取一个整数（可能是浮点数或整数），然后将其累加到变量 "out" 中。代码的最后部分定义了一个变量 "a"，但没有对它做出任何初始化。



```
main();

```