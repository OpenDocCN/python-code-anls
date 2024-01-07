# `basic-computer-games\27_Civil_War\csharp\GameOptions.cs`

```

// 引入 System 命名空间
using System;
// 引入 CivilWar.ConsoleUtils 命名空间
using static CivilWar.ConsoleUtils;

// 定义 GameOptions 记录类型，包含两个布尔类型的属性：TwoPlayers 和 ShowDescriptions
namespace CivilWar
{
    public record GameOptions(bool TwoPlayers, bool ShowDescriptions)
    {
        // 静态方法，用于从用户输入中创建 GameOptions 对象
        public static GameOptions Input()
        {
            // 输出游戏说明
            Console.WriteLine(
@"                          Civil War
               Creative Computing, Morristown, New Jersey


Do you want instructions?");
            
            // 游戏说明内容
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
You may surrender by typing a '5' for your strategy.";

            // 如果用户选择需要游戏说明，则输出游戏说明内容
            if (InputYesOrNo())
                WriteWordWrap(instructions);

            // 输出是否有两位将军在场
            Console.WriteLine("\n\nAre there two generals present?");
            // 获取用户输入，判断是否有两位将军在场
            bool twoPlayers = InputYesOrNo();
            // 如果没有两位将军在场，则输出提示信息
            if (!twoPlayers)
                Console.WriteLine("\nYou are the confederacy.  Good luck!\n");

            // 输出是否需要战斗描述
            WriteWordWrap(
            @"Select a battle by typing a number from 1 to 14 on request.  Type any other number to end the simulation. But '0' brings back exact previous battle situation allowing you to replay it.

Note: a negative Food$ entry causes the program to use the entries from the previous battle

After requesting a battle, do you wish battle descriptions (answer yes or no)");
            // 获取用户输入，判断是否需要战斗描述
            bool showDescriptions = InputYesOrNo();

            // 返回 GameOptions 对象，包含用户输入的两个布尔值
            return new GameOptions(twoPlayers, showDescriptions);
        }
    }
}

```