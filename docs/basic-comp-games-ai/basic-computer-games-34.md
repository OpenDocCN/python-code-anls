# BasicComputerGames源码解析 34

# `28_Combat/csharp/Controller.cs`

This is a class that simulates a military simulation game. It includes functions for getting the military branch for the user's next attack, getting a valid attack size, and getting an integer value from the user.

The military branches include "2" for the Navy, "3" for the Air Force, and "Army" for the Army. The attack size can be chosen from the user, and the attack direction can also be set.


```
﻿using System;

namespace Game
{
    /// <summary>
    /// Contains functions for interacting with the user.
    /// </summary>
    public class Controller
    {
        /// <summary>
        /// Gets the player's initial armed forces distribution.
        /// </summary>
        /// <param name="computerForces">
        /// The computer's initial armed forces.
        /// </param>
        public static ArmedForces GetInitialForces(ArmedForces computerForces)
        {
            var playerForces = default(ArmedForces);

            // BUG: This loop allows the player to assign negative values to
            //  some branches, leading to strange results.
            do
            {
                View.ShowDistributeForces();

                View.PromptArmySize(computerForces.Army);
                var army = InputInteger();

                View.PromptNavySize(computerForces.Navy);
                var navy = InputInteger();

                View.PromptAirForceSize(computerForces.AirForce);
                var airForce = InputInteger();

                playerForces = new ArmedForces
                {
                    Army     = army,
                    Navy     = navy,
                    AirForce = airForce
                };
            }
            while (playerForces.TotalTroops > computerForces.TotalTroops);

            return playerForces;
        }

        /// <summary>
        /// Gets the military branch for the user's next attack.
        /// </summary>
        public static MilitaryBranch GetAttackBranch(WarState state, bool isFirstTurn)
        {
            if (isFirstTurn)
                View.PromptFirstAttackBranch();
            else
                View.PromptNextAttackBranch(state.ComputerForces, state.PlayerForces);

            // If the user entered an invalid branch number in the original
            // game, the code fell through to the army case.  We'll preserve
            // that behaviour here.
            return Console.ReadLine() switch
            {
                "2" => MilitaryBranch.Navy,
                "3" => MilitaryBranch.AirForce,
                _   => MilitaryBranch.Army
            };
        }

        /// <summary>
        /// Gets a valid attack size from the player for the given branch
        /// of the armed forces.
        /// </summary>
        /// <param name="troopsAvailable">
        /// The number of troops available.
        /// </param>
        public static int GetAttackSize(int troopsAvailable)
        {
            var attackSize = 0;

            do
            {
                View.PromptAttackSize();
                attackSize = InputInteger();
            }
            while (attackSize < 0 || attackSize > troopsAvailable);

            return attackSize;
        }

        /// <summary>
        /// Gets an integer value from the user.
        /// </summary>
        public static int InputInteger()
        {
            var value = default(int);

            while (!Int32.TryParse(Console.ReadLine(), out value))
                View.PromptValidInteger();

            return value;
        }
    }
}

```

# `28_Combat/csharp/FinalCampaign.cs`

This appears to be a script written in a language similar to人行道德语。 It appears to be a game script, and the dialogue and text are meant to be趣味ful and almost comical in nature. The script appears to be telling the player that they have won the battle and sunk two of their battleships, but theplayer is also left with a sense of victory, as their country has been left in ruins.


```
﻿namespace Game
{
    /// <summary>
    /// Represents the state of the game during the final campaign of the war.
    /// </summary>
    public sealed class FinalCampaign : WarState
    {
        /// <summary>
        /// Initializes a new instance of the FinalCampaign class.
        /// </summary>
        /// <param name="computerForces">
        /// The computer's forces.
        /// </param>
        /// <param name="playerForces">
        /// The player's forces.
        /// </param>
        public FinalCampaign(ArmedForces computerForces, ArmedForces playerForces)
            : base(computerForces, playerForces)
        {
        }

        protected override (WarState nextState, string message) AttackWithArmy(int attackSize)
        {
            if (attackSize < ComputerForces.Army / 2)
            {
                return
                (
                    new Ceasefire(
                        ComputerForces,
                        PlayerForces with
                        {
                            Army = PlayerForces.Army - attackSize
                        }),
                    "I WIPED OUT YOUR ATTACK!"
                );
            }
            else
            {
                return
                (
                    new Ceasefire(
                        ComputerForces with
                        {
                            Army = 0
                        },
                        PlayerForces),
                    "YOU DESTROYED MY ARMY!"
                );
            }
        }

        protected override (WarState nextState, string message) AttackWithNavy(int attackSize)
        {
            if (attackSize < ComputerForces.Navy / 2)
            {
                return
                (
                    new Ceasefire(
                        ComputerForces,
                        PlayerForces with
                        {
                            Army = PlayerForces.Army / 4,
                            Navy = PlayerForces.Navy / 2
                        }),
                    "I SUNK TWO OF YOUR BATTLESHIPS, AND MY AIR FORCE\n" +
                    "WIPED OUT YOUR UNGAURDED CAPITOL."
                );
            }
            else
            {
                return
                (
                    new Ceasefire(
                        ComputerForces with
                        {
                            AirForce = 2 * ComputerForces.AirForce / 3,
                            Navy     = ComputerForces.Navy / 2
                        },
                        PlayerForces),
                    "YOUR NAVY SHOT DOWN THREE OF MY XIII PLANES,\n" +
                    "AND SUNK THREE BATTLESHIPS."
                );
            }
        }

        protected override (WarState nextState, string message) AttackWithAirForce(int attackSize)
        {
            // BUG? Usually, larger attacks lead to better outcomes.
            //  It seems odd that the logic is suddenly reversed here,
            //  but this could be intentional.
            if (attackSize > ComputerForces.AirForce / 2)
            {
                return
                (
                    new Ceasefire(
                        ComputerForces,
                        PlayerForces with
                        {
                            Army     = PlayerForces.Army  / 3,
                            Navy     = PlayerForces.Navy / 3,
                            AirForce = PlayerForces.AirForce / 3
                        }),
                    "MY NAVY AND AIR FORCE IN A COMBINED ATTACK LEFT\n" +
                    "YOUR COUNTRY IN SHAMBLES."
                );
            }
            else
            {
                return
                (
                    new Ceasefire(
                        ComputerForces,
                        PlayerForces,
                        absoluteVictory: true),
                    "ONE OF YOUR PLANES CRASHED INTO MY HOUSE. I AM DEAD.\n" +
                    "MY COUNTRY FELL APART."
                );
            }
        }
    }
}

```

# `28_Combat/csharp/InitialCampaign.cs`

This is a text-based augmented reality game. The game is using a state machine to keep track of the different states the game is in. The game has two main states, "GAME" and "CHECK-IN", and a few additional states like "ATTACK", "WISCONGRATULATIONS", and "FINAL-JUDGMENT". The game is played by a player who is part of an virtual army, and the player is given the task of completing a mission to destroy enemy army bases.

As the player progresses through the game, they will encounter various enemies and obstacles that they will have to overcome in order to complete their mission. The player can use various tactics, such as attacking with their army or using airforce, to defeat their enemies. The player will also have to manage their resources, such as their food and air, as well as their morale, which can be affected by the weather and other factors.

As the player progresses through the game, they will be given various choices to make, such as whether to help a certain character or to continue their mission. The player's choices will have a significant impact on the outcome of the game, and will determine the final outcome of the story.

The game is using a random number generator to determine the outcome of the player's choices, as well as the enemy's actions. The player will have to make strategic decisions to complete their mission and destroy all of the enemy army bases.

The game also has a feature where the player can be抹了好几回还没成功，it seems like the game is looping.

Overall, this game is quite unique and interesting, and players will have to put their skills and strategies to the test in order to complete it.



```
﻿namespace Game
{
    /// <summary>
    /// Represents the state of the game during the initial campaign of the war.
    /// </summary>
    public sealed class InitialCampaign : WarState
    {
        /// <summary>
        /// Initializes a new instance of the InitialCampaign class.
        /// </summary>
        /// <param name="computerForces">
        /// The computer's forces.
        /// </param>
        /// <param name="playerForces">
        /// The player's forces.
        /// </param>
        public InitialCampaign(ArmedForces computerForces, ArmedForces playerForces)
            : base(computerForces, playerForces)
        {
        }

        protected override (WarState nextState, string message) AttackWithArmy(int attackSize)
        {
            // BUG: Why are we comparing attack size to the size of our own
            //   military?  This leads to some truly absurd results if our
            //   army is tiny.
            if (attackSize < PlayerForces.Army / 3)
            {
                return
                (
                    new FinalCampaign(
                        ComputerForces,
                        PlayerForces with
                        {
                            Army = PlayerForces.Army - attackSize
                        }),
                    $"YOU LOST {attackSize} MEN FROM YOUR ARMY."
                );
            }
            else
            if (attackSize < 2 * PlayerForces.Army / 3)
            {
                return
                (
                    new FinalCampaign(
                        ComputerForces with
                        {
                            // BUG: Clearly not what we claim below...
                            Army = 0
                        },
                        PlayerForces with
                        {
                            Army = PlayerForces.Army - attackSize / 3
                        }),
                    $"YOU LOST {attackSize / 3} MEN, BUT I LOST {2 * ComputerForces.Army / 3}"
                );
            }
            else
            {
                // BUG? This is identical to the third outcome when attacking
                //  with the navy.  It seems unlikely that this was the
                //  intent.  Probably line 115 in the original source was
                //  supposed to say "GOTO 170" instead of "GOTO 270".
                //  (Line 170 is conspicuously absent.)
                return
                (
                    new FinalCampaign(
                        ComputerForces with
                        {
                            Navy = 2 * ComputerForces.Navy / 3
                        },
                        PlayerForces with
                        {
                            Army     = PlayerForces.Army / 3,
                            AirForce = PlayerForces.AirForce / 3
                        }),
                    "YOU SUNK ONE OF MY PATROL BOATS, BUT I WIPED OUT TWO\n" +
                    "OF YOUR AIR FORCE BASES AND 3 ARMY BASES."
                );
            }
        }

        protected override (WarState nextState, string message) AttackWithNavy(int attackSize)
        {
            if (attackSize < ComputerForces.Navy / 3)
            {
                return
                (
                    new FinalCampaign(
                        ComputerForces,
                        PlayerForces with
                        {
                            Navy = PlayerForces.Navy - attackSize
                        }),
                    "YOUR ATTACK WAS STOPPED!"
                );
            }
            else
            if (attackSize < 2 * ComputerForces.Navy / 3)
            {
                return
                (
                    new FinalCampaign(
                        ComputerForces with
                        {
                            Navy = ComputerForces.Navy / 3
                        },
                        PlayerForces),
                    $"YOU DESTROYED {2 * ComputerForces.Navy / 3} OF MY ARMY."
                );
            }
            else
            {
                return
                (
                    new FinalCampaign(
                        ComputerForces with
                        {
                            Navy = 2 * ComputerForces.Navy / 3
                        },
                        PlayerForces with
                        {
                            Army     = PlayerForces.Army / 3,
                            AirForce = PlayerForces.AirForce / 3
                        }),
                    "YOU SUNK ONE OF MY PATROL BOATS, BUT I WIPED OUT TWO\n" +
                    "OF YOUR AIR FORCE BASES AND 3 ARMY BASES."
                );
            }
        }

        protected override (WarState nextState, string message) AttackWithAirForce(int attackSize)
        {
            // BUG: Why are we comparing the attack size to the size of
            //  our own air force?  Surely we meant to compare to the
            //  computer's air force.
            if (attackSize < PlayerForces.AirForce / 3)
            {
                return
                (
                    new FinalCampaign(
                        ComputerForces,
                        PlayerForces with
                        {
                            AirForce = PlayerForces.AirForce - attackSize
                        }),
                    "YOUR ATTACK WAS WIPED OUT."
                );
            }
            else
            if (attackSize < 2 * PlayerForces.AirForce / 3)
            {
                return
                (
                    new FinalCampaign(
                        ComputerForces with
                        {
                            Army     = 2 * ComputerForces.Army / 3,
                            Navy     = ComputerForces.Navy / 3,
                            AirForce = ComputerForces.AirForce / 3
                        },
                        PlayerForces),
                    "WE HAD A DOGFIGHT. YOU WON - AND FINISHED YOUR MISSION."
                );
            }
            else
            {

                return
                (
                    new FinalCampaign(
                        ComputerForces with
                        {
                            Army = 2 * ComputerForces.Army / 3
                        },
                        PlayerForces with
                        {
                            Army = PlayerForces.Army / 4,
                            Navy = PlayerForces.Navy / 3
                        }),
                    "YOU WIPED OUT ONE OF MY ARMY PATROLS, BUT I DESTROYED" +
                    "TWO NAVY BASES AND BOMBED THREE ARMY BASES."
                );
            }
        }
    }
}

```

# `28_Combat/csharp/MilitaryBranch.cs`

这段代码定义了一个名为"MilitaryBranch"的枚举类型，包含了三个成员变量，分别代表陆军、海军和空军。这个枚举类型可以用来定义一个整数类型的变量army,navy或airforce，并可以使用using指令继承自它。例如，可以在游戏项目中创建一个名为" MilitaryBranch"的类，并定义一个名为"Army"、"Navy"和"AirForce"的成员变量。


```
﻿namespace Game
{
    /// <summary>
    /// Enumerates the different branches of the military.
    /// </summary>
    public enum MilitaryBranch
    {
        Army,
        Navy,
        AirForce
    }
}

```

# `28_Combat/csharp/Program.cs`

这段代码是一个名为 "Game.Program" 的 C# 应用程序类，其名为 "Main" 的方法。该程序的主要目的是在游戏开始时显示游戏界面上的标语和说明，然后模拟一次有约束条件的攻击，并显示攻击结果。

具体来说，代码首先创建了一个名为 "ArmedForces" 的类，它包含了一个军队的实例，军队包括陆军、海军和空军。然后，代码创建了一个名为 "Controller" 的类，它使用 "GetInitialForces" 方法来获取计算机军队的初始实力，并使用 "GetAttackBranch" 方法来获取攻击分支。

接下来，代码创建了一个名为 "InitialCampaign" 的类，它包含了一个 "WarState" 的实例，以及一个名为 "FinalOutcome" 的属性，用于存储游戏的最终结果。然后，代码创建了一个名为 "Program" 的类，它的 "Main" 方法包含游戏的主要逻辑。在 "Main" 方法中，代码首先创建了一个 "ArmedForces" 的实例，然后使用 "GetInitialForces" 方法获取计算机军队的初始实力，并使用 "Controller" 的 "GetAttackSize" 方法获取攻击的大小。

接下来，代码创建了一个循环，该循环在每次攻击结束后，显示攻击的结果，并从下一状态重新开始循环。在循环中，代码使用 "GetAttackBranch" 方法获取攻击分支，并使用 "state.LaunchAttack" 方法来攻击分支。在攻击结束后，代码使用 "View.ShowMessage" 方法显示攻击结果，并使用 "isFirstTurn" 属性来控制是否是第一次攻击。最后，代码在循环结束后，使用 "View.ShowResult" 方法来显示游戏的最终结果。


```
﻿namespace Game
{
    class Program
    {
        static void Main()
        {
            View.ShowBanner();
            View.ShowInstructions();

            var computerForces = new ArmedForces { Army = 30000, Navy = 20000, AirForce = 22000 };
            var playerForces   = Controller.GetInitialForces(computerForces);

            var state = (WarState) new InitialCampaign(computerForces, playerForces);
            var isFirstTurn = true;

            while (!state.FinalOutcome.HasValue)
            {
                var branch = Controller.GetAttackBranch(state, isFirstTurn);
                var attackSize = Controller.GetAttackSize(state.PlayerForces[branch]);

                var (nextState, message) = state.LaunchAttack(branch, attackSize);
                View.ShowMessage(message);

                state = nextState;
                isFirstTurn = false;
            }

            View.ShowResult(state);
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)

The original BASIC code has a surprising number of bugs for such a small program.
For the sake of preserving the original behaviour, I've left them in place and
commented the ones I noticed.


# `28_Combat/csharp/View.cs`

This is a class that contains functions for various military branches. These functions are used to prompt the user for input on how many men they have available for each branch of military.

The `PromptAirForceSize` function is an example of a military function, while the other functions are examples of functions for army and navy.

The `PromptAttackSize` function asks the user to enter the number of men for each branch of military. The other functions also ask the user to enter a valid integer value, and some of them also ask for the next branch of military to attack.

It's worth noting that the output of the function may not be what the programmer intended, and it's always the user's responsibility to enter the correct input.


```
﻿using System;

namespace Game
{
    /// <summary>
    /// Contains functions for displaying information to the user.
    /// </summary>
    public static class View
    {
        public static void ShowBanner()
        {
            Console.WriteLine("                                 COMBAT");
            Console.WriteLine("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        }

        public static void ShowInstructions()
        {
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("I AM AT WAR WITH YOU.");
            Console.WriteLine("WE HAVE 72000 SOLDIERS APIECE.");
        }

        public static void ShowDistributeForces()
        {
            Console.WriteLine();
            Console.WriteLine("DISTRIBUTE YOUR FORCES.");
            Console.WriteLine("\tME\t  YOU");
        }

        public static void ShowMessage(string message)
        {
            Console.WriteLine(message);
        }

        public static void ShowResult(WarState finalState)
        {
            if (!finalState.IsAbsoluteVictory)
            {
                Console.WriteLine();
                Console.WriteLine("FROM THE RESULTS OF BOTH OF YOUR ATTACKS,");
            }

            switch (finalState.FinalOutcome)
            {
            case WarResult.ComputerVictory:
                Console.WriteLine("YOU LOST-I CONQUERED YOUR COUNTRY.  IT SERVES YOU");
                Console.WriteLine("RIGHT FOR PLAYING THIS STUPID GAME!!!");
                break;
            case WarResult.PlayerVictory:
                Console.WriteLine("YOU WON, OH! SHUCKS!!!!");
                break;
            case WarResult.PeaceTreaty:
                Console.WriteLine("THE TREATY OF PARIS CONCLUDED THAT WE TAKE OUR");
                Console.WriteLine("RESPECTIVE COUNTRIES AND LIVE IN PEACE.");
                break;
            }
        }

        public static void PromptArmySize(int computerArmySize)
        {
            Console.Write($"ARMY\t{computerArmySize}\t? ");
        }

        public static void PromptNavySize(int computerNavySize)
        {
            Console.Write($"NAVY\t{computerNavySize}\t? ");
        }

        public static void PromptAirForceSize(int computerAirForceSize)
        {
            Console.Write($"A. F.\t{computerAirForceSize}\t? ");
        }

        public static void PromptFirstAttackBranch()
        {
            Console.WriteLine("YOU ATTACK FIRST. TYPE (1) FOR ARMY; (2) FOR NAVY;");
            Console.WriteLine("AND (3) FOR AIR FORCE.");
            Console.Write("? ");
        }

        public static void PromptNextAttackBranch(ArmedForces computerForces, ArmedForces playerForces)
        {
            // BUG: More of a nit-pick really, but the order of columns in the
            //  table is reversed from what we showed when distributing troops.
            //  The tables should be consistent.
            Console.WriteLine();
            Console.WriteLine("\tYOU\tME");
            Console.WriteLine($"ARMY\t{playerForces.Army}\t{computerForces.Army}");
            Console.WriteLine($"NAVY\t{playerForces.Navy}\t{computerForces.Navy}");
            Console.WriteLine($"A. F.\t{playerForces.AirForce}\t{computerForces.AirForce}");

            Console.WriteLine("WHAT IS YOUR NEXT MOVE?");
            Console.WriteLine("ARMY=1  NAVY=2  AIR FORCE=3");
            Console.Write("? ");
        }

        public static void PromptAttackSize()
        {
            Console.WriteLine("HOW MANY MEN");
            Console.Write("? ");
        }

        public static void PromptValidInteger()
        {
            Console.WriteLine("ENTER A VALID INTEGER VALUE");
        }
    }
}

```

# `28_Combat/csharp/WarResult.cs`

这段代码定义了一个名为 `WarResult` 的枚举类型，用于表示战争的可能结果。枚举类型包含三种可能的值：`ComputerVictory`,`PlayerVictory`，和 `PeaceTreaty`。

枚举类型的定义对于程序中的任何涉及战争结果的代码都可以使用，例如在选择游戏结束时选择获胜方或判断两个玩家是否在某种游戏中胜出等。

在程序中，可以使用以下方式来枚举类型中的值：

```
var result = GameResult.ComputerVictory;
```

这将把 `GameResult.ComputerVictory` 常量赋值给 `result` 变量，该常量的值为 `WarResult.ComputerVictory`。


```
﻿namespace Game
{
    /// <summary>
    /// Enumerates the possible outcomes of the war.
    /// </summary>
    public enum WarResult
    {
        ComputerVictory,

        PlayerVictory,

        PeaceTreaty
    }
}

```

# `28_Combat/csharp/WarState.cs`

This is a class in a game framework that simulates a military attack. The class has three parameters: the attacker's military branch, the number of men and women in the attack, and the number of men and women in the target military branch.

The class has three methods: `AttackWithArmy`, `AttackWithNavy`, and `AttackWithAirForce`. These methods, depending on the military branch, will conduct an attack using the given number of men and women, and return the new game state and a message describing the result.

If the military branch is not found, an ArgumentException is thrown.


```
﻿using System;

namespace Game
{
    /// <summary>
    /// Represents the current state of the war.
    /// </summary>
    public abstract class WarState
    {
        /// <summary>
        /// Gets the computer's armed forces.
        /// </summary>
        public ArmedForces ComputerForces { get; }

        /// <summary>
        /// Gets the player's armed forces.
        /// </summary>
        public ArmedForces PlayerForces { get; }

        /// <summary>
        /// Gets a flag indicating whether this state represents absolute
        /// victory for the player.
        /// </summary>
        public virtual bool IsAbsoluteVictory => false;

        /// <summary>
        /// Gets the final outcome of the war.
        /// </summary>
        /// <remarks>
        /// If the war is ongoing, this property will be null.
        /// </remarks>
        public virtual WarResult? FinalOutcome => null;

        /// <summary>
        /// Initializes a new instance of the state class.
        /// </summary>
        /// <param name="computerForces">
        /// The computer's forces.
        /// </param>
        /// <param name="playerForces">
        /// The player's forces.
        /// </param>
        public WarState(ArmedForces computerForces, ArmedForces playerForces) =>
            (ComputerForces, PlayerForces) = (computerForces, playerForces);

        /// <summary>
        /// Launches an attack.
        /// </summary>
        /// <param name="branch">
        /// The branch of the military to use for the attack.
        /// </param>
        /// <param name="attackSize">
        /// The number of men and women to use for the attack.
        /// </param>
        /// <returns>
        /// The new state of the game resulting from the attack and a message
        /// describing the result.
        /// </returns>
        public (WarState nextState, string message) LaunchAttack(MilitaryBranch branch, int attackSize) =>
            branch switch
            {
                MilitaryBranch.Army     => AttackWithArmy(attackSize),
                MilitaryBranch.Navy     => AttackWithNavy(attackSize),
                MilitaryBranch.AirForce => AttackWithAirForce(attackSize),
                _               => throw new ArgumentException("INVALID BRANCH")
            };

        /// <summary>
        /// Conducts an attack with the player's army.
        /// </summary>
        /// <param name="attackSize">
        /// The number of men and women used in the attack.
        /// </param>
        /// <returns>
        /// The new game state and a message describing the result.
        /// </returns>
        protected abstract (WarState nextState, string message) AttackWithArmy(int attackSize);

        /// <summary>
        /// Conducts an attack with the player's navy.
        /// </summary>
        /// <param name="attackSize">
        /// The number of men and women used in the attack.
        /// </param>
        /// <returns>
        /// The new game state and a message describing the result.
        /// </returns>
        protected abstract (WarState nextState, string message) AttackWithNavy(int attackSize);

        /// <summary>
        /// Conducts an attack with the player's air force.
        /// </summary>
        /// <param name="attackSize">
        /// The number of men and women used in the attack.
        /// </param>
        /// <returns>
        /// The new game state and a message describing the result.
        /// </returns>
        protected abstract (WarState nextState, string message) AttackWithAirForce(int attackSize);
    }
}

```

# `28_Combat/java/Combat.java`

这段代码是一个Java程序，它包括以下功能：

1. 导入Math库，它是Java标准库中的数学库。
2. 从Scanner类中导入Scanner对象，以便从用户那里读取输入。
3. 定义了一个名为"Game of Combat"的类。
4. 在类中定义了一个名为"Battle"的静态方法，它接受一个int类型的参数x，一个int类型的参数y，以及一个int类型的参数z。
5. 在方法中，使用Math库中的Math.random()函数生成一个0到1之间的随机整数作为战斗结果。
6. 在方法中，使用Scanner对象的nextInt()方法从用户那里获取x，y，z的值。
7. 最后，使用Scanner对象的close()方法关闭与用户的交互。


```
import java.lang.Math;
import java.util.Scanner;

/**
 * Game of Combat
 * <p>
 * Based on the BASIC game of Combat here
 * https://github.com/coding-horror/basic-computer-games/blob/main/28%20Combat/combat.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

```

3.0)
public class Combat {

   //空中战斗类型枚举类型
   private static final int COMBAT_TYPE = 0;
   private static final int RACE_TYPE = 1;
   private static final int ARRAY_SIZE = 3;

   // 各个字段
   private int usrArmy;
   private int usrNavy;
   private int usrAir;
   private int cpuArmy;
   private int cpuNavy;
   private int cpuAir;
   private int planeCrashWin;

   // 处理各种异常情况
   private void handleException(int result) {
       if (result < 0) {
           System.out.println("WAR IS NOT OVER YET.");
           System.out.println("NORTH OR SOUTH, OR😉");
           System.out.println("THIS IS NOT AN APPRENTIENCE.");
           System.out.println("PLANE CRASH.");
           System.out.println("WE DID NOT GET TOO LIKE。" + result);
       }
       else {
           // 飞机坠毁，显示文本
           System.out.println("ONE OF YOUR PLANES CRASHED INTO MY HOUSE. I AM DEAD.");
           System.out.println("MY COUNTRY FELL APART.");
           planeCrashWin = true;
       }
   }

   // 根据飞机部署类型，判断结果
   private void calculateUsrAir(int deployType, int usrArmy, int cpuArmy, int cpuNavy) {
       int arr = 0;
       int num = 0;

       // 空战，结果
       if (deployType == COMBAT_TYPE) {
           // 计算敌军空中单位数量
           int enemyAvg = (int) Math.floor((3.0 / 2.0 * (cpuArmy + cpuNavy + cpuAir)));
           num = enemyAvg;
           // 如果我方部署了空中单位，就将它们带回家，自己不够用时再用敌方数量计算
           if (usrAir > usrArmy) {
               num = (int) Math.ceil(num / 3.0);
               usrAir = (int) Math.floor(usrAir / 3.0);
           }
           // 统计结果
           int result = (int) Math.ceil(usrAir / (int) Math.sqrt(Math.pow(usrAir, 3) / (Math.pow(num, 3) / 4)));
           arr = (int) Math.ceil(usrAir / (int) Math.sqrt(Math.pow(enemyAvg, 3) / (Math.pow(Math.floor((3.0 / 2.0 * (cpuArmy + cpuNavy + cpuAir)), 3) / (Math.pow(usrAir, 3) / (Math.pow(enemyAvg, 3), 3))));
       }

       // 空地战，结果
       else {
           // 计算我方空中单位数量
           int myAvg = (int) Math.ceil((3.0 / 2.0 * (cpuArmy + cpuNavy + cpuAir)));
           num = myAvg;
           // 如果我方部署了空中单位，就将它们带回家，自己不够用时再用敌方数量计算
           if (usrAir > usrArmy) {
               num = (int) Math.ceil(num / 3.0);
               usrAir = (int) Math.floor(usrAir / 3.0);
           }
           // 统计结果
           int result = (int) Math.ceil((int) Math.sqrt(Math.pow(usrAir, 3) / (Math.pow(num, 3) / (Math.ceil((3.0 / 2.0 * (cpuArmy + cpuNavy + cpuAir)), 3))));
           arr = (int) Math.ceil(usrAir / (int) Math.sqrt(Math.pow(myAvg, 3) / (Math.pow(Math.floor((3.0 / 2.0 * (cpuArmy + cpuNavy + cpuAir)), 3) / (Math.pow(usrAir, 3) / (Math.pow(enemyAvg, 3), 3))));
       }

       // 如果 Arr 大于 0，则显示计算出来的平均值
       if (arr


```


```
public class Combat {

  private static final int MAX_UNITS = 72000;  // Maximum number of total units per player

  private final Scanner scan;  // For user input

  private boolean planeCrashWin = false;

  private int usrArmy = 0;      // Number of user Army units
  private int usrNavy = 0;      // Number of user Navy units
  private int usrAir = 0;       // Number of user Air Force units
  private int cpuArmy = 30000;  // Number of cpu Army units
  private int cpuNavy = 20000;  // Number of cpu Navy units
  private int cpuAir = 22000;   // Number of cpu Air Force units

  public Combat() {

    scan = new Scanner(System.in);

  }  // End of constructor Combat

  public void play() {

    showIntro();
    getForces();
    attackFirst();
    attackSecond();

  }  // End of method play

  private static void showIntro() {

    System.out.println(" ".repeat(32) + "COMBAT");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");
    System.out.println("I AM AT WAR WITH YOU.");
    System.out.println("WE HAVE " + MAX_UNITS + " SOLDIERS APIECE.\n");

  }  // End of method showIntro

  private void getForces() {

    do {
      System.out.println("DISTRIBUTE YOUR FORCES.");
      System.out.println("              ME              YOU");
      System.out.print("ARMY           " + cpuArmy + "        ? ");
      usrArmy = scan.nextInt();
      System.out.print("NAVY           " + cpuNavy + "        ? ");
      usrNavy = scan.nextInt();
      System.out.print("A. F.          " + cpuAir + "        ? ");
      usrAir = scan.nextInt();

    } while ((usrArmy + usrNavy + usrAir) > MAX_UNITS);  // Avoid exceeding the maximum number of total units

  }  // End of method getForces

  private void attackFirst() {

    int numUnits = 0;
    int unitType = 0;

    do {
      System.out.println("YOU ATTACK FIRST. TYPE (1) FOR ARMY; (2) FOR NAVY;");
      System.out.println("AND (3) FOR AIR FORCE.");
      System.out.print("? ");
      unitType = scan.nextInt();
    } while ((unitType < 1) || (unitType > 3));  // Avoid out-of-range values

    do {
      System.out.println("HOW MANY MEN");
      System.out.print("? ");
      numUnits = scan.nextInt();
    } while ((numUnits < 0) ||                // Avoid negative values
             ((unitType == 1) && (numUnits > usrArmy)) ||  // Avoid exceeding the number of available Army units
             ((unitType == 2) && (numUnits > usrNavy)) ||  // Avoid exceeding the number of available Navy units
             ((unitType == 3) && (numUnits > usrAir)));    // Avoid exceeding the number of available Air Force units

    // Begin handling deployment type
    switch (unitType) {
      case 1:  // Army deployed

        if (numUnits < (usrArmy / 3.0)) {  // User deployed less than one-third of their Army units
          System.out.println("YOU LOST " + numUnits + " MEN FROM YOUR ARMY.");
          usrArmy = usrArmy - numUnits;
        }
        else if (numUnits < (2.0 * usrArmy / 3.0)) {  // User deployed less than two-thirds of their Army units
          System.out.println("YOU LOST " + (int) Math.floor(numUnits / 3.0) + " MEN, BUT I LOST " + (int) Math.floor(2.0 * cpuArmy / 3.0));
          usrArmy = (int) Math.floor(usrArmy - numUnits / 3.0);
          cpuArmy = 0;
        }
        else {  // User deployed two-thirds or more of their Army units
          System.out.println("YOU SUNK ONE OF MY PATROL BOATS, BUT I WIPED OUT TWO");
          System.out.println("OF YOUR AIR FORCE BASES AND 3 ARMY BASES.");
          usrArmy = (int) Math.floor(usrArmy / 3.0);
          usrAir = (int) Math.floor(usrAir / 3.0);
          cpuNavy = (int) Math.floor(2.0 * cpuNavy / 3.0);
        }
        break;

      case 2:  // Navy deployed

        if (numUnits < (cpuNavy / 3.0)) {  // User deployed less than one-third relative to cpu Navy units
          System.out.println("YOUR ATTACK WAS STOPPED!");
          usrNavy = usrNavy - numUnits;
        }
        else if (numUnits < (2.0 * cpuNavy / 3.0)) {  // User deployed less than two-thirds relative to cpu Navy units
          System.out.println("YOU DESTROYED " + (int) Math.floor(2.0 * cpuNavy / 3.0) + " OF MY ARMY.");
          cpuNavy = (int) Math.floor(cpuNavy / 3.0);
        }
        else {  // User deployed two-thirds or more relative to cpu Navy units
          System.out.println("YOU SUNK ONE OF MY PATROL BOATS, BUT I WIPED OUT TWO");
          System.out.println("OF YOUR AIR FORCE BASES AND 3 ARMY BASES.");
          usrArmy = (int) Math.floor(usrArmy / 3.0);
          usrAir = (int) Math.floor(usrAir / 3.0);
          cpuNavy = (int) Math.floor(2.0 * cpuNavy / 3.0);
        }
        break;

      case 3:  // Air Force deployed

        if (numUnits < (usrAir / 3.0)) {  // User deployed less than one-third of their Air Force units
          System.out.println("YOUR ATTACK WAS WIPED OUT.");
          usrAir = usrAir - numUnits;
        }
        else if (numUnits < (2.0 * usrAir / 3.0)) {  // User deployed less than two-thirds of their Air Force units
          System.out.println("WE HAD A DOGFIGHT. YOU WON - AND FINISHED YOUR MISSION.");
          cpuArmy = (int) Math.floor(2.0 * cpuArmy / 3.0);
          cpuNavy = (int) Math.floor(cpuNavy / 3.0);
          cpuAir = (int) Math.floor(cpuAir / 3.0);
        }
        else {  // User deployed two-thirds or more of their Air Force units
          System.out.println("YOU WIPED OUT ONE OF MY ARMY PATROLS, BUT I DESTROYED");
          System.out.println("TWO NAVY BASES AND BOMBED THREE ARMY BASES.");
          usrArmy = (int) Math.floor(usrArmy / 4.0);
          usrNavy = (int) Math.floor(usrNavy / 3.0);
          cpuArmy = (int) Math.floor(2.0 * cpuArmy / 3.0);
        }
        break;

    }  // End handling deployment type

  }  // End of method attackFirst

  private void attackSecond() {

    int numUnits = 0;
    int unitType = 0;

    System.out.println("");
    System.out.println("              YOU           ME");
    System.out.print("ARMY           ");
    System.out.format("%-14s%s\n", usrArmy, cpuArmy);
    System.out.print("NAVY           ");
    System.out.format("%-14s%s\n", usrNavy, cpuNavy);
    System.out.print("A. F.          ");
    System.out.format("%-14s%s\n", usrAir, cpuAir);

    do {
      System.out.println("WHAT IS YOUR NEXT MOVE?");
      System.out.println("ARMY=1  NAVY=2  AIR FORCE=3");
      System.out.print("? ");
      unitType = scan.nextInt();
    } while ((unitType < 1) || (unitType > 3));  // Avoid out-of-range values

    do {
      System.out.println("HOW MANY MEN");
      System.out.print("? ");
      numUnits = scan.nextInt();
    } while ((numUnits < 0) ||                // Avoid negative values
             ((unitType == 1) && (numUnits > usrArmy)) ||  // Avoid exceeding the number of available Army units
             ((unitType == 2) && (numUnits > usrNavy)) ||  // Avoid exceeding the number of available Navy units
             ((unitType == 3) && (numUnits > usrAir)));    // Avoid exceeding the number of available Air Force units

    // Begin handling deployment type
    switch (unitType) {
      case 1:  // Army deployed

        if (numUnits < (cpuArmy / 2.0)) {  // User deployed less than half relative to cpu Army units
          System.out.println("I WIPED OUT YOUR ATTACK!");
          usrArmy = usrArmy - numUnits;
        }
        else {  // User deployed half or more relative to cpu Army units
          System.out.println("YOU DESTROYED MY ARMY!");
          cpuArmy = 0;
        }
        break;

      case 2:  // Navy deployed

        if (numUnits < (cpuNavy / 2.0)) {  // User deployed less than half relative to cpu Navy units
          System.out.println("I SUNK TWO OF YOUR BATTLESHIPS, AND MY AIR FORCE");
          System.out.println("WIPED OUT YOUR UNGUARDED CAPITOL.");
          usrArmy = (int) Math.floor(usrArmy / 4.0);
          usrNavy = (int) Math.floor(usrNavy / 2.0);
        }
        else { // User deployed half or more relative to cpu Navy units
          System.out.println("YOUR NAVY SHOT DOWN THREE OF MY XIII PLANES,");
          System.out.println("AND SUNK THREE BATTLESHIPS.");
          cpuAir = (int) Math.floor(2.0 * cpuAir / 3.0);
          cpuNavy = (int) Math.floor(cpuNavy / 2.0);
        }
        break;

      case 3:  // Air Force deployed

        if (numUnits > (cpuAir / 2.0)) {  // User deployed more than half relative to cpu Air Force units
          System.out.println("MY NAVY AND AIR FORCE IN A COMBINED ATTACK LEFT");
          System.out.println("YOUR COUNTRY IN SHAMBLES.");
          usrArmy = (int) Math.floor(usrArmy / 3.0);
          usrNavy = (int) Math.floor(usrNavy / 3.0);
          usrAir = (int) Math.floor(usrAir / 3.0);
        }
        else {  // User deployed half or less relative to cpu Air Force units
          System.out.println("ONE OF YOUR PLANES CRASHED INTO MY HOUSE. I AM DEAD.");
          System.out.println("MY COUNTRY FELL APART.");
          planeCrashWin = true;
        }
        break;

    }  // End handling deployment type

    // Suppress message for plane crashes
    if (planeCrashWin == false) {
      System.out.println("");
      System.out.println("FROM THE RESULTS OF BOTH OF YOUR ATTACKS,");
    }

    // User wins
    if ((planeCrashWin == true) ||
        ((usrArmy + usrNavy + usrAir) > ((int) Math.floor((3.0 / 2.0 * (cpuArmy + cpuNavy + cpuAir)))))) {
      System.out.println("YOU WON, OH! SHUCKS!!!!");
    }
    // User loses
    else if ((usrArmy + usrNavy + usrAir) < ((int) Math.floor((2.0 / 3.0 * (cpuArmy + cpuNavy + cpuAir))))) {  // User loss
      System.out.println("YOU LOST-I CONQUERED YOUR COUNTRY.  IT SERVES YOU");
      System.out.println("RIGHT FOR PLAYING THIS STUPID GAME!!!");
    }
    // Peaceful outcome
    else {
      System.out.println("THE TREATY OF PARIS CONCLUDED THAT WE TAKE OUR");
      System.out.println("RESPECTIVE COUNTRIES AND LIVE IN PEACE.");
    }

  }  // End of method attackSecond

  public static void main(String[] args) {

    Combat combat = new Combat();
    combat.play();

  }  // End of method main

}  // End of class Combat

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `28_Combat/javascript/combat.js`

这段代码定义了两个函数，分别是`print()`和`input()`。

`print()`函数的作用是在页面上打印一段字符串，将该字符串插入到页面上由`document.getElementById("output")`引用的一棵`document.createTextNode`元素中。

`input()`函数的作用是接收用户输入的字符串，并返回该字符串的JavaScript转义形式。该函数使用了Promise对象，并且在函数内部使用了事件监听器来捕获用户按下的键盘上的按键。当用户按下了键盘上的数字13时，该函数会将用户输入的字符串存储到变量`input_str`中，并将其打印到页面上。函数还会在打印前将输入的字符串中的换行符替换成两个空格，以便在打印时更好地显示。

总之，这两个函数一起工作，使得用户可以在输入框中输入字符串，并自动将其转换为JavaScript转义形式，然后在页面上打印出来。


```
// COMBAT
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

It looks like you're implementing a simple game where one player tries to attack the other's country, and the other player tries to defend. The game keeps track of which country is under attack and which country has won.

The game starts with a random attacker and a defending country. The defending country has a fixed number of planes and a random number of attack planes. The attacker starts with a fixed number of planes and a random number of attack planes.

The game then enters a loop where the two countries attack each other. The attacker can choose to attack any of their own attack planes, or attack the defending country's country. The defending country can choose to defend their country or attack the attacker's country.

If the attacker chooses to attack, they have a 1/2 chance of hitting their country, a 1/2 chance of hitting the defending country, or a 0/2 chance of hitting an attack plane. If the attacker chooses to defend, they have a 1/2 chance of hitting their country, a 1/2 chance of hitting the defending country, or a 0/2 chance of hitting an attack plane.

The game keeps track of which country is under attack and which country has won. If the attacking country has won, the game ends and the defending country is defeated. If the defending country has won, the game ends and the attacking country is defeated. If both countries are still tied, the game ends without a winner.

It looks like you have a basic understanding of game logic and have implemented some game mechanics, such as the random attacker and the fixed number of attack planes for the defending country. There are also some areas of the game that could be improved upon, such as the text output that describes the game state.


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
    print(tab(33) + "COMBAT\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("I AM AT WAR WITH YOU.\n");
    print("WE HAVE 72000 SOLDIERS APIECE.\n");
    do {
        print("\n");
        print("DISTRIBUTE YOUR FORCES.\n");
        print("\tME\t  YOU\n");
        print("ARMY\t30000\t");
        a = parseInt(await input());
        print("NAVY\t20000\t");
        b = parseInt(await input());
        print("A. F.\t22000\t");
        c = parseInt(await input());
    } while (a + b + c > 72000) ;
    d = 30000;
    e = 20000;
    f = 22000;
    print("YOU ATTACK FIRST. TYPE (1) FOR ARMY; (2) FOR NAVY;\n");
    print("AND (3) FOR AIR FORCE.\n");
    y = parseInt(await input());
    do {
        print("HOW MANY MEN\n");
        x = parseInt(await input());
    } while ((y == 1 && x > a) || (y == 2 && x > b) || (y == 3 && x > c)) ;
    switch (y) {
        case 1:
            if (x < a / 3.0) {
                print("YOU LOST " + x + " MEN FROM YOUR ARMY.\n");
                a -= x;
                break;
            }
            if (x < 2 * a / 3) {
                print("YOU LOST " + Math.floor(x / 3.0) + " MEN, BUT I LOST " + Math.floor(2 * d / 3.0) + "\n");
                a = Math.floor(a - x / 3.0);
                d = 0;
                break;
            }
            print("YOU SUNK ONE OF MY PATROL BOATS, BUT I WIPED OUT TWO\n");
            print("OF YOUR AIR FORCE BASES AND 3 ARMY BASES.\n");
            a = Math.floor(a / 3.0);
            c = Math.floor(c / 3.0);
            e = Math.floor(2 * e / 3.0);
            break;
        case 2:
            if (x < e / 3) {
                print("YOUR ATTACK WAS STOPPED!\n");
                b -= x;
                break;
            }
            if (x < 2 * e / 3) {
                print("YOU DESTROYED " + Math.floor(2 * e / 3.0) + " OF MY ARMY.\n");
                e = Math.floor(e / 3.0);
                break;
            }
            print("YOU SUNK ONE OF MY PATROL BOATS, BUT I WIPED OUT TWO\n");
            print("OF YOUR AIR FORCE BASES AND 3 ARMY BASES.\n");
            a = Math.floor(a / 3.0);
            c = Math.floor(c / 3.0);
            e = Math.floor(2 * e / 3.0);
            break;
        case 3:
            if (x < c / 3.0) {
                print("YOUR ATTACK WAS WIPED OUT.\n");
                c -= x;
                break;
            }
            if (x < 2 * c / 3) {
                print("WE HAD A DOGFIGHT. YOU WON - AND FINISHED YOUR MISSION.\n");
                d = Math.floor(2 * d / 3.0);
                e = Math.floor(e / 3.0);
                f = Math.floor(f / 3.0);
                break;
            }
            print("YOU WIPED OUT ONE OF MY ARMY PATROLS, BUT I DESTROYED\n");
            print("TWO NAVY BASES AND BOMBED THREE ARMY BASES.\n");
            a = Math.floor(a / 4);
            b = Math.floor(b / 3.0);
            d = Math.floor(2 * d / 3.0);
            break;
    }
    print("\n");
    print("\tYOU\tME\n");
    print("ARMY\t" + a + "\t" + d + "\n");
    print("NAVY\t" + b + "\t" + e + "\n");
    print("A. F.\t" + c + "\t" + f + "\n");
    print("WHAT IS YOUR NEXT MOVE?\n");
    print("ARMY=1  NAVY=2  AIR FORCE=3\n");
    g = parseInt(await input());
    do {
        print("HOW MANY MEN\n");
        t = parseInt(await input());
    } while (t < 0 || (g == 1 && t > a) || (g == 2 && t > b) || (g == 3 && t > c)) ;
    crashed = false;
    switch (g) {
        case 1:
            if (t < d / 2) {
                print("I WIPED OUT YOUR ATTACK!\n");
                a -= t;
            } else {
                print("YOU DESTROYED MY ARMY!\n");
                d = 0;
            }
            break;
        case 2:
            if (t < e / 2) {
                print("I SUNK TWO OF YOUR BATTLESHIPS, AND MY AIR FORCE\n");
                print("WIPED OUT YOUR UNGUARDED CAPITOL.\n");
                a /= 4.0;
                b /= 2.0;
                break;
            }
            print("YOUR NAVY SHOT DOWN THREE OF MY XIII PLANES.\n");
            print("AND SUNK THREE BATTLESHIPS.\n");
            f = 2 * f / 3;
            e /= 2;
            break;
        case 3:
            if (t > f / 2) {
                print("MY NAVY AND AIR FORCE IN A COMBINED ATTACK LEFT\n");
                print("YOUR COUNTRY IN SHAMBLES.\n");
                a /= 3.0;
                b /= 3.0;
                c /= 3.0;
                break;
            }
            print("ONE OF YOUR PLANES CRASHED INTO MY HOUSE. I AM DEAD.\n");
            print("MY COUNTRY FELL APART.\n");
            crashed = true;
            won = 1;
            break;
    }
    if (!crashed) {
        won = 0;
        print("\n");
        print("FROM THE RESULTS OF BOTH OF YOUR ATTACKS,\n");
        if (a + b + c > 3.0 / 2.0 * (d + e + f))
            won = 1;
        if (a + b + c < 2.0 / 3.0 * (d + e + f))
            won = 2;
    }
    if (won == 0) {
        print("THE TREATY OF PARIS CONCLUDED THAT WE TAKE OUR\n");
        print("RESPECTIVE COUNTRIES AND LIVE IN PEACE.\n");
    } else if (won == 1) {
        print("YOU WON, OH! SHUCKS!!!!\n");
    } else if (won == 2) {
        print("YOU LOST-I CONQUERED YOUR COUNTRY.  IT SERVES YOU\n");
        print("RIGHT FOR PLAYING THIS STUPID GAME!!!\n");
    }
}

```

这道题的代码是一个C语言程序，包含了两个主要部分：`main()`函数和`printf()`函数。我们需要分析这两个部分的作用，以便解释整个程序的作用。

1. `main()`函数：

`main()`函数是程序的入口点，程序从这里开始执行。在这个函数中，可能会做一些初始化操作，然后执行其他函数。但是，在这个例子中，我们只有一个`main()`函数，而且它的作用非常简单，就是没有其他操作。所以，这个程序的`main()`函数可能只是一个占位符，用来告诉编译器程序从这里开始执行。

2. `printf()`函数：

`printf()`函数是一个标准库函数，用于将一个格式化的字符串输出到屏幕。在这个程序中，我们使用`printf()`函数来输出字符串"Hello World"。`printf()`函数的第一个参数是一个字符串，第二个参数是一个格式控制符，用于告诉函数如何格式化这个字符串。在这个程序中，我们将字符串"Hello World"作为第一个参数，`%s`作为格式控制符，所以`printf()`函数会输出这个字符串。


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


# `28_Combat/python/combat.py`

这段代码是一个用于游戏中的脚本，主要作用是设置游戏中的参数和变量，以及输出介绍游戏背景信息。

MAX_UNITS = 72000: 设置游戏中的最大单位数量为72000，即游戏中的士兵数量最大为72000人。

plane_crash_win = False: 设置飞机坠毁赢的值为False，即飞机坠毁时玩家可以获胜。

usr_army = 0: 设置usr_army的值为0，即usr_army表示的游戏中的士兵数量为0。

usr_navy = 0: 设置usr_navy的值为0，即usr_navy表示的游戏中的士兵数量为0。

usr_air = 0: 设置usr_air的值为0，即usr_air表示的游戏中的士兵数量为0。

cpu_army = 30000: 设置cpu_army的值为30000，即表示计算机武装部队的数量为30000。

cpu_navy = 20000: 设置cpu_navy的值为20000，即表示计算机海军的数量为20000。

cpu_air = 22000: 设置cpu_air的值为22000，即表示计算机空军的数量为22000。

show_intro()函数：输出游戏介绍信息，包括游戏中的最大单位数量、飞机坠毁赢的值等。

show_intro()函数的作用是在游戏开始时被调用，可以设置游戏中的参数和变量，以及输出介绍游戏背景信息。


```
MAX_UNITS = 72000
plane_crash_win = False
usr_army = 0
usr_navy = 0
usr_air = 0
cpu_army = 30000
cpu_navy = 20000
cpu_air = 22000


def show_intro() -> None:
    global MAX_UNITS

    print(" " * 32 + "COMBAT")
    print(" " * 14 + "CREATIVE COMPUTING MORRISTOWN, NEW JERSEY")
    print("\n\n")
    print("I AM AT WAR WITH YOU.")
    print("WE HAVE " + str(MAX_UNITS) + " SOLDIERS APIECE.")


```

这段代码的作用是请求玩家分配他们的部队，并保持一个游戏循环，直到玩家停止。在每次循环中，代码会提示玩家他们自己的部队数量，并提示他们如何分配这些部队。然后，代码会从玩家输入中获取他们的部队数量，并检查分配的部队数量是否超过 MAX_UNITS 变量。如果部队数量没有超过 MAX_UNITS，则循环会从结结束。


```
def get_forces() -> None:
    global usr_army, usr_navy, usr_air

    while True:
        print("DISTRIBUTE YOUR FORCES.")
        print("              ME              YOU")
        print("ARMY           " + str(cpu_army) + "        ? ", end="")
        usr_army = int(input())
        print("NAVY           " + str(cpu_navy) + "        ? ", end="")
        usr_navy = int(input())
        print("A. F.          " + str(cpu_air) + "        ? ", end="")
        usr_air = int(input())
        if (usr_army + usr_navy + usr_air) <= MAX_UNITS:
            break


```

It seems like you're describing a game with different attack units, and the objective is to sink certain units or destroy enemy units. Based on the information you've provided, I assume you're trying to simulate a wargame or a similar game.

The game appears to have different attack units, including some surface ships, submarines, and aircraft. Each unit has its own set of abilities, such as the number of units it can attack and the number of units it can sink or destroy. The player must use these abilities strategically to achieve their objectives, such as sinking all the enemy's submarines or destroying all the enemy's aircrafts.

It's important to note that this game is based on your imagination, and the outcome may vary depending on the player's decisions and the game's rules.


```
def attack_first() -> None:
    global usr_army, usr_navy, usr_air
    global cpu_army, cpu_navy, cpu_air

    num_units = 0
    unit_type = 0

    while True:
        print("YOU ATTACK FIRST. TYPE (1) FOR ARMY; (2) FOR NAVY;")
        print("AND (3) FOR AIR FORCE.")
        print("?", end=" ")
        unit_type = int(input())
        if not (unit_type < 1 or unit_type > 3):
            break

    while True:
        print("HOW MANY MEN")
        print("?", end=" ")
        num_units = int(input())
        if not (
            (num_units < 0)
            or ((unit_type == 1) and (num_units > usr_army))
            or ((unit_type == 2) and (num_units > usr_navy))
            or ((unit_type == 3) and (num_units > usr_air))
        ):
            break

    if unit_type == 1:
        if num_units < (usr_army / 3):
            print("YOU LOST " + str(num_units) + " MEN FROM YOUR ARMY.")
            usr_army = usr_army - num_units
        elif num_units < (2 * usr_army / 3):
            print(
                "YOU LOST "
                + str(int(num_units / 3))
                + " MEN, BUT I LOST "
                + str(int(2 * cpu_army / 3))
            )
            usr_army = int(usr_army - (num_units / 3))
            cpu_army = 0
        else:
            print("YOU SUNK ONE OF MY PATROL BOATS, BUT I WIPED OUT TWO")
            print("OF YOUR AIR FORCE BASES AND 3 ARMY BASES.")
            usr_army = int(usr_army / 3)
            usr_air = int(usr_air / 3)
            cpu_navy = int(2 * cpu_navy / 3)
    elif unit_type == 2:
        if num_units < cpu_navy / 3:
            print("YOUR ATTACK WAS STOPPED!")
            usr_navy = usr_navy - num_units
        elif num_units < 2 * cpu_navy / 3:
            print("YOU DESTROYED " + str(int(2 * cpu_navy / 3)) + " OF MY ARMY.")
            cpu_navy = int(cpu_navy / 3)
        else:
            print("YOU SUNK ONE OF MY PATROL BOATS, BUT I WIPED OUT TWO")
            print("OF YOUR AIR FORCE BASES AND 3 ARMY BASES.")
            usr_army = int(usr_army / 3)
            usr_air = int(usr_air / 3)
            cpu_navy = int(2 * cpu_navy / 3)
    elif unit_type == 3:
        if num_units < usr_air / 3:
            print("YOUR ATTACK WAS WIPED OUT.")
            usr_air = usr_air - num_units
        elif num_units < 2 * usr_air / 3:
            print("WE HAD A DOGFIGHT. YOU WON - AND FINISHED YOUR MISSION.")
            cpu_army = int(2 * cpu_army / 3)
            cpu_navy = int(cpu_navy / 3)
            cpu_air = int(cpu_air / 3)
        else:
            print("YOU WIPED OUT ONE OF MY ARMY PATROLS, BUT I DESTROYED")
            print("TWO NAVY BASES AND BOMBED THREE ARMY BASES.")
            usr_army = int(usr_army / 4)
            usr_navy = int(usr_navy / 3)
            cpu_army = int(2 * cpu_army / 3)


```

It looks like you're trying to compare the output of a game where you are an AI character and two human players. The AI character is trying to attack the players' planes, and the output is supposed to indicate the outcome of the attack.

If the AI character is successful in attacking all the planes of the players, it will print "YOUR NAVY AND AIR FORCE IN A COMBINED ATTACK LEFT" and "YOUR COUNTRY IN SHAMBLES." If it fails, it will print "ONE OF YOUR PLANES CRASHED INTO MY HOUSE. I AM DEAD."

If the AI character manages to destroy one of the players' planes, it will print "MY COUNTRY FELL APART." and "PLANE_CRASH_WIN=True"

It's important to note that this is a game with a very strange plot and it's not based on any real world events or situations.


```
def attack_second() -> None:
    global usr_army, usr_navy, usr_air, cpu_army, cpu_navy, cpu_air
    global plane_crash_win
    num_units = 0
    unit_type = 0

    print()
    print("              YOU           ME")
    print("ARMY           ", end="")
    print("%-14s%s\n" % (usr_army, cpu_army), end="")
    print("NAVY           ", end="")
    print("%-14s%s\n" % (usr_navy, cpu_navy), end="")
    print("A. F.          ", end="")
    print("%-14s%s\n" % (usr_air, cpu_air), end="")

    while True:
        print("WHAT IS YOUR NEXT MOVE?")
        print("ARMY=1  NAVY=2  AIR FORCE=3")
        print("? ", end="")
        unit_type = int(input())
        if not ((unit_type < 1) or (unit_type > 3)):
            break

    while True:
        print("HOW MANY MEN")
        print("? ", end="")
        num_units = int(input())
        if not (
            (num_units < 0)
            or ((unit_type == 1) and (num_units > usr_army))
            or ((unit_type == 2) and (num_units > usr_navy))
            or ((unit_type == 3) and (num_units > usr_air))
        ):
            break

    if unit_type == 1:
        if num_units < (cpu_army / 2):
            print("I WIPED OUT YOUR ATTACK!")
            usr_army = usr_army - num_units
        else:
            print("YOU DESTROYED MY ARMY!")
            cpu_army = 0
    elif unit_type == 2:
        if num_units < (cpu_navy / 2):
            print("I SUNK TWO OF YOUR BATTLESHIPS, AND MY AIR FORCE")
            print("WIPED OUT YOUR UNGUARDED CAPITOL.")
            usr_army = int(usr_army / 4)
            usr_navy = int(usr_navy / 2)
        else:
            print("YOUR NAVY SHOT DOWN THREE OF MY XIII PLANES,")
            print("AND SUNK THREE BATTLESHIPS.")
            cpu_air = int(2 * cpu_air / 3)
            cpu_navy = int(cpu_navy / 2)
    elif unit_type == 3:
        if num_units > (cpu_air / 2):
            print("MY NAVY AND AIR FORCE IN A COMBINED ATTACK LEFT")
            print("YOUR COUNTRY IN SHAMBLES.")
            usr_army = int(usr_army / 3)
            usr_navy = int(usr_navy / 3)
            usr_air = int(usr_air / 3)
        else:
            print("ONE OF YOUR PLANES CRASHED INTO MY HOUSE. I AM DEAD.")
            print("MY COUNTRY FELL APART.")
            plane_crash_win = True

    if not plane_crash_win:
        print()
        print("FROM THE RESULTS OF BOTH OF YOUR ATTACKS,")

    if plane_crash_win or (
        (usr_army + usr_navy + usr_air) > (int(3 / 2 * (cpu_army + cpu_navy + cpu_air)))
    ):
        print("YOU WON, OH! SHUCKS!!!!")
    elif (usr_army + usr_navy + usr_air) < int(2 / 3 * (cpu_army + cpu_navy + cpu_air)):
        print("YOU LOST-I CONQUERED YOUR COUNTRY.  IT SERVES YOU")
        print("RIGHT FOR PLAYING THIS STUPID GAME!!!")
    else:
        print("THE TREATY OF PARIS CONCLUDED THAT WE TAKE OUR")
        print("RESPECTIVE COUNTRIES AND LIVE IN PEACE.")


```

这段代码是一个Python程序，定义了一个名为`main`的函数，该函数返回一个`None`类型的值。接下来会定义一系列函数，分别命名为`show_intro`、`get_forces`、`attack_first`和`attack_second`。这些函数的具体实现将在下面分别解释。

首先，程序会调用一个名为`show_intro`的函数。由于该函数没有定义返回值，因此它的作用在程序运行时会被略过。

接下来，程序会调用一个名为`get_forces`的函数。这个函数的具体实现超出了程序的描述，因此它的作用在程序运行时会自动执行。具体来说，程序会读取用户输入的一行或多行数据，并将其存储在一个名为`forces`的变量中。

然后，程序会调用一个名为`attack_first`的函数。这个函数的具体实现超出了程序的描述，因此它的作用在程序运行时会自动执行。具体来说，程序会模拟攻击者的行为，首先尝试攻击所有防御力较低的目标，然后攻击所有防御力较高的目标。

最后，程序会调用一个名为`attack_second`的函数。这个函数的具体实现超出了程序的描述，因此它的作用在程序运行时会自动执行。具体来说，程序会模拟攻击者的行为，首先尝试攻击所有防御力较低的目标，然后攻击所有防御力较高的目标。

程序的最终目的是让用户在一个攻击和防御的世界中进行战斗。在这个过程中，用户需要不断输入指令，来控制主角的行动。


```
def main() -> None:
    show_intro()
    get_forces()
    attack_first()
    attack_second()


if __name__ == "__main__":
    main()

```