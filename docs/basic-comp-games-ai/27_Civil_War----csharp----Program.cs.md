# `basic-computer-games\27_Civil_War\csharp\Program.cs`

```py
// 引入必要的命名空间
using System;
using System.Collections.Generic;
using System.Linq;
using CivilWar;

// 获取游戏选项
var options = GameOptions.Input();
// 创建军队列表，根据选项决定玩家数量和对手类型
var armies = new List<Army> { new Army(Side.Confederate), options.TwoPlayers ? new Army(Side.Union) : new ComputerArmy(Side.Union) };

// 初始化战斗对象
Battle? battle = null;
// 进行一系列战斗，直到有一方获胜
while (OneBattle(ref battle)) { }
// 显示战斗结果
DisplayResult();

// 进行一场战斗
bool OneBattle(ref Battle? previous)
{
    // 选择战斗
    var (option, selected) = Battle.SelectBattle();
    // 根据选择的选项进行相应的操作
    var (battle, isReplay, quit) = option switch
    {
        Option.Battle => (selected!, false, false),
        Option.Replay when previous != null => (previous, true, false), // 如果没有之前的战斗，则无法重播
        _ => (null!, false, true),
    };
    // 如果选择退出，则返回 false
    if (quit)
        return false;

    // 如果不是重播，则准备战斗
    if (!isReplay)
    {
        Console.WriteLine($"This is the battle of {battle.Name}.");
        // 如果选项中显示描述，则输出战斗描述
        if (options.ShowDescriptions)
            ConsoleUtils.WriteWordWrap(battle.Description);
        // 为每支军队准备战斗
        armies.ForEach(a => a.PrepareBattle(battle.Men[(int)a.Side], battle.Casualties[(int)a.Side]));
    }

    // 输出军队信息表格
    ConsoleUtils.WriteTable(armies, new()
    {
        new("", a => a.Side),
        new("Men", a => a.Men),
        new("Money", a => a.Money, Before: "$"),
        new("Inflation", a => a.InflationDisplay, After: "%")
    });

    // 为每支军队分配资源
    armies.ForEach(a => a.AllocateResources());
    // 显示每支军队的士气
    armies.ForEach(a => a.DisplayMorale());

    // 根据战斗的进攻方确定输出信息
    string offensive = battle.Offensive switch
    {
        Side.Confederate => "You are on the offensive",
        Side.Union => "You are on the defensive",
        _ => "Both sides are on the offensive"
    };
    Console.WriteLine($"Confederate general---{offensive}");

    // 如果有一方选择投降，则返回 false
    if (armies.Any(a => a.ChooseStrategy(isReplay)))
    {
        return false; // someone surrendered
    }
    // 计算每支军队的损失
    armies[0].CalculateLosses(armies[1]);
    armies[1].CalculateLosses(armies[0]);

    // 输出军队损失信息表格
    ConsoleUtils.WriteTable(armies, new()
    {
        new("", a => a.Side),
        new("Casualties", a => a.Casualties),
        new("Desertions", a => a.Desertions),
    });
}
    # 如果选项中有两名玩家
    if (options.TwoPlayers)
    {
        # 创建一个包含数字1的数组
        var oneDataCol = new[] { 1 };
        # 打印输出比较实际伤亡人数的信息
        Console.WriteLine($"Compared to the actual casualties at {battle.Name}");
        # 使用ConsoleUtils.WriteTable方法输出表格
        ConsoleUtils.WriteTable(oneDataCol, armies.Select(a => new ConsoleUtils.TableRow<int>(
            a.Side.ToString(),
            _ => $"{(double)a.Casualties / battle.Casualties[(int)a.Side]}", After: "% of the original")
        ).ToList());
    }

    # 定义变量winner
    Side winner;
    # 根据不同的条件进行判断
    switch (armies[0].AllLost, armies[1].AllLost, armies[0].MenLost - armies[1].MenLost)
    {
        # 如果两方都全军覆没或者两方伤亡人数相等
        case (true, true, _) or (false, false, 0):
            # 打印输出战斗结果未决
            Console.WriteLine("Battle outcome unresolved");
            # 将winner设为Both，表示平局
            winner = Side.Both; // Draw
            break;
        # 如果联邦军全军覆没或者联邦军伤亡人数少于南方联盟军
        case (false, true, _) or (false, false, < 0):
            # 打印输出南方联盟军获胜的信息
            Console.WriteLine($"The Confederacy wins {battle.Name}");
            # 将winner设为Confederate，表示南方联盟军获胜
            winner = Side.Confederate;
            break;
        # 如果南方联盟军全军覆没或者南方联盟军伤亡人数多于联邦军
        case (true, false, _) or (false, false, > 0):
            # 打印输出联邦军获胜的信息
            Console.WriteLine($"The Union wins {battle.Name}");
            # 将winner设为Union，表示联邦军获胜
            winner = Side.Union;
            break;
    }
    # 如果不是重播模式
    if (!isReplay)
    {
        # 遍历armies列表，记录战斗结果
        armies.ForEach(a => a.RecordResult(winner));
    }
    # 打印输出分隔线
    Console.WriteLine("---------------");
    # 将当前战斗battle赋值给previous
    previous = battle;
    # 返回true
    return true;
# 显示战争结果
void DisplayResult()
{
    # 显示第一个军队与第二个军队的战争结果
    armies[0].DisplayWarResult(armies[1]);

    # 获取战斗次数
    int battles = armies[0].BattlesFought;
    # 如果有战斗发生
    if (battles > 0)
    {
        # 打印战斗次数（不包括重演）
        Console.WriteLine($"For the {battles} battles fought (excluding reruns)");

        # 使用ConsoleUtils.WriteTable方法打印军队的历史损失、模拟损失和损失百分比
        ConsoleUtils.WriteTable(armies, new()
        {
            new("", a => a.Side),
            new("Historical Losses", a => a.CumulativeHistoricCasualties),
            new("Simulated Losses", a => a.CumulativeSimulatedCasualties),
            new("  % of original", a => ((double)a.CumulativeSimulatedCasualties / a.CumulativeHistoricCasualties).ToString("p2"))
        }, transpose: true);

        # 显示第二个军队的战略
        armies[1].DisplayStrategies();
    }
}
```