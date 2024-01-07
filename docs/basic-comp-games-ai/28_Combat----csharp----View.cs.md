# `basic-computer-games\28_Combat\csharp\View.cs`

```

// 命名空间 Game 包含了用于向用户显示信息的函数
namespace Game
{
    /// <summary>
    /// 包含了用于向用户显示信息的函数
    /// </summary>
    public static class View
    {
        // 显示游戏横幅
        public static void ShowBanner()
        {
            Console.WriteLine("                                 COMBAT");
            Console.WriteLine("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        }

        // 显示游戏说明
        public static void ShowInstructions()
        {
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("I AM AT WAR WITH YOU.");
            Console.WriteLine("WE HAVE 72000 SOLDIERS APIECE.");
        }

        // 显示分配兵力的提示
        public static void ShowDistributeForces()
        {
            Console.WriteLine();
            Console.WriteLine("DISTRIBUTE YOUR FORCES.");
            Console.WriteLine("\tME\t  YOU");
        }

        // 显示消息
        public static void ShowMessage(string message)
        {
            Console.WriteLine(message);
        }

        // 显示战争结果
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

        // 提示输入计算机的陆军规模
        public static void PromptArmySize(int computerArmySize)
        {
            Console.Write($"ARMY\t{computerArmySize}\t? ");
        }

        // 提示输入计算机的海军规模
        public static void PromptNavySize(int computerNavySize)
        {
            Console.Write($"NAVY\t{computerNavySize}\t? ");
        }

        // 提示输入计算机的空军规模
        public static void PromptAirForceSize(int computerAirForceSize)
        {
            Console.Write($"A. F.\t{computerAirForceSize}\t? ");
        }

        // 提示选择第一次攻击的兵种
        public static void PromptFirstAttackBranch()
        {
            Console.WriteLine("YOU ATTACK FIRST. TYPE (1) FOR ARMY; (2) FOR NAVY;");
            Console.WriteLine("AND (3) FOR AIR FORCE.");
            Console.Write("? ");
        }

        // 提示选择下一次攻击的兵种
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

        // 提示输入攻击规模
        public static void PromptAttackSize()
        {
            Console.WriteLine("HOW MANY MEN");
            Console.Write("? ");
        }

        // 提示输入有效的整数值
        public static void PromptValidInteger()
        {
            Console.WriteLine("ENTER A VALID INTEGER VALUE");
        }
    }
}

```