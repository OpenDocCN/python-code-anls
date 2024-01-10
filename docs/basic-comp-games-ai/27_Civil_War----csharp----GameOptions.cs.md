# `basic-computer-games\27_Civil_War\csharp\GameOptions.cs`

```
// 引入命名空间 System，用于访问系统类库
using System;
// 引入 CivilWar.ConsoleUtils 命名空间，用于访问控制台工具类
using static CivilWar.ConsoleUtils;

// 定义 CivilWar 命名空间
namespace CivilWar
{
    // 定义 GameOptions 记录类型，包含两个布尔类型属性 TwoPlayers 和 ShowDescriptions
    public record GameOptions(bool TwoPlayers, bool ShowDescriptions)
    {
        // 定义静态方法 Input，用于获取游戏选项
        public static GameOptions Input()
        {
            // 输出游戏说明和提示信息
            Console.WriteLine(
@"                          Civil War
               Creative Computing, Morristown, New Jersey


Do you want instructions?");
            
            // 定义游戏说明内容
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

            // 输出提示信息，询问是否有两位将军参与游戏
            Console.WriteLine("\n\nAre there two generals present?");
            // 获取用户输入的是否有两位将军参与游戏的布尔值
            bool twoPlayers = InputYesOrNo();
            // 如果没有两位将军参与游戏，则输出提示信息
            if (!twoPlayers)
                Console.WriteLine("\nYou are the confederacy.  Good luck!\n");

            // 输出提示信息，询问用户选择战斗
            WriteWordWrap(
            @"Select a battle by typing a number from 1 to 14 on request.  Type any other number to end the simulation. But '0' brings back exact previous battle situation allowing you to replay it.

Note: a negative Food$ entry causes the program to use the entries from the previous battle
# 请求战斗后，询问是否显示战斗描述，接受用户输入并将结果存储在showDescriptions变量中
bool showDescriptions = InputYesOrNo();

# 返回一个新的GameOptions对象，其中包含两个玩家信息和是否显示战斗描述的选项
return new GameOptions(twoPlayers, showDescriptions);
```