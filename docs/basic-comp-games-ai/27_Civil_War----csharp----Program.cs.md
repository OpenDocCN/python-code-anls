# `basic-computer-games\27_Civil_War\csharp\Program.cs`

```

// 引入必要的命名空间
using System;
using System.Collections.Generic;
using System.Linq;
using CivilWar;

// 获取游戏选项
var options = GameOptions.Input();
// 创建军队列表，根据选项确定玩家数量和对手类型
var armies = new List<Army> { new Army(Side.Confederate), options.TwoPlayers ? new Army(Side.Union) : new ComputerArmy(Side.Union) };

// 初始化战斗对象
Battle? battle = null;
// 进行一系列战斗，直到返回 false
while (OneBattle(ref battle)) { }
// 显示战斗结果
DisplayResult();

// 进行一场战斗
bool OneBattle(ref Battle? previous)
{
    // 选择战斗
    var (option, selected) = Battle.SelectBattle();
    // 根据选择进行相应操作
    var (battle, isReplay, quit) = option switch
    {
        Option.Battle => (selected!, false, false),
        Option.Replay when previous != null => (previous, true, false), // 如果没有之前的战斗则无法重播
        _ => (null!, false, true),
    };
    // 如果选择退出，则返回 false
    if (quit)
        return false;

    // 如果不是重播，则准备战斗
    if (!isReplay)
    {
        Console.WriteLine($"This is the battle of {battle.Name}.");
        if (options.ShowDescriptions)
            ConsoleUtils.WriteWordWrap(battle.Description);
        armies.ForEach(a => a.PrepareBattle(battle.Men[(int)a.Side], battle.Casualties[(int)a.Side]));
    }

    // 显示军队信息
    ConsoleUtils.WriteTable(armies, new()
    {
        new("", a => a.Side),
        new("Men", a => a.Men),
        new("Money", a => a.Money, Before: "$"),
        new("Inflation", a => a.InflationDisplay, After: "%")
    });

    // 分配资源
    armies.ForEach(a => a.AllocateResources());
    // 显示士气
    armies.ForEach(a => a.DisplayMorale());

    // 根据进攻方确定进攻方信息
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
    // 计算损失
    armies[0].CalculateLosses(armies[1]);
    armies[1].CalculateLosses(armies[0]);

    // 显示损失情况
    ConsoleUtils.WriteTable(armies, new()
    {
        new("", a => a.Side),
        new("Casualties", a => a.Casualties),
        new("Desertions", a => a.Desertions),
    });
    // 如果是双人游戏，则显示实际损失情况
    if (options.TwoPlayers)
    {
        var oneDataCol = new[] { 1 };
        Console.WriteLine($"Compared to the actual casualties at {battle.Name}");
        ConsoleUtils.WriteTable(oneDataCol, armies.Select(a => new ConsoleUtils.TableRow<int>(
            a.Side.ToString(),
            _ => $"{(double)a.Casualties / battle.Casualties[(int)a.Side]}", After: "% of the original")
        ).ToList());
    }

    // 根据损失情况确定胜利方
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
    // 如果不是重播，则记录结果
    if (!isReplay)
    {
        armies.ForEach(a => a.RecordResult(winner));
    }
    Console.WriteLine("---------------");
    previous = battle;
    return true;
}

// 显示战斗结果
void DisplayResult()
{
    // 显示战争结果
    armies[0].DisplayWarResult(armies[1]);

    // 显示战斗数量和损失情况
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