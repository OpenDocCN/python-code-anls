# `27_Civil_War\csharp\Program.cs`

```
var options = GameOptions.Input(); // 从用户输入获取游戏选项
var armies = new List<Army> { new Army(Side.Confederate), options.TwoPlayers ? new Army(Side.Union) : new ComputerArmy(Side.Union) }; // 创建军队列表，根据选项确定玩家数量和对应的军队类型

Battle? battle = null; // 初始化战斗对象为null
while (OneBattle(ref battle)) { } // 循环进行战斗，直到OneBattle函数返回false
DisplayResult(); // 显示战斗结果

bool OneBattle(ref Battle? previous) // 定义一个函数，进行一场战斗
{
    var (option, selected) = Battle.SelectBattle(); // 从Battle类中选择一场战斗
    var (battle, isReplay, quit) = option switch // 根据选项进行不同的操作
    {
        Option.Battle => (selected!, false, false), // 如果选项是Battle，则进行选定的战斗
        Option.Replay when previous != null => (previous, true, false), // 如果选项是Replay且之前有战斗记录，则进行重播
        _ => (null!, false, true), // 其他情况设定战斗对象为null，退出循环
    };
    // 如果退出标志为真，则返回假
    if (quit)
        return false;

    // 如果不是重播模式
    if (!isReplay)
    {
        // 打印战斗的名称
        Console.WriteLine($"This is the battle of {battle.Name}.");
        // 如果选项中显示描述，则打印战斗描述
        if (options.ShowDescriptions)
            ConsoleUtils.WriteWordWrap(battle.Description);
        // 对每个军队进行战斗准备
        armies.ForEach(a => a.PrepareBattle(battle.Men[(int)a.Side], battle.Casualties[(int)a.Side]));
    }

    // 打印军队信息表格
    ConsoleUtils.WriteTable(armies, new()
    {
        new("", a => a.Side), // 军队方向
        new("Men", a => a.Men), // 军队人数
        new("Money", a => a.Money, Before: "$"), // 军队金钱
        new("Inflation", a => a.InflationDisplay, After: "%") // 通货膨胀率
    });
    # 对 armies 列表中的每个元素调用 AllocateResources() 方法，分配资源
    armies.ForEach(a => a.AllocateResources());
    # 对 armies 列表中的每个元素调用 DisplayMorale() 方法，显示士气
    armies.ForEach(a => a.DisplayMorale());

    # 根据 battle 的 Offensive 属性的值，选择不同的字符串赋值给 offensive 变量
    string offensive = battle.Offensive switch
    {
        Side.Confederate => "You are on the offensive",
        Side.Union => "You are on the defensive",
        _ => "Both sides are on the offensive"
    };
    # 在控制台输出包含 offensive 变量的字符串
    Console.WriteLine($"Confederate general---{offensive}");

    # 如果 armies 列表中的任何元素调用 ChooseStrategy(isReplay) 方法返回 True，则返回 False，表示有人投降
    if (armies.Any(a => a.ChooseStrategy(isReplay)))
    {
        return false; # someone surrendered
    }
    # 计算第一个军队的损失
    armies[0].CalculateLosses(armies[1]);
    # 计算第二个军队的损失
    armies[1].CalculateLosses(armies[0]);

    # 使用 ConsoleUtils.WriteTable 方法在控制台输出 armies 列表的内容
    ConsoleUtils.WriteTable(armies, new()
    {
        new("", a => a.Side), // 创建一个新的对象，属性名为空，值为a的Side属性
        new("Casualties", a => a.Casualties), // 创建一个新的对象，属性名为"Casualties"，值为a的Casualties属性
        new("Desertions", a => a.Desertions), // 创建一个新的对象，属性名为"Desertions"，值为a的Desertions属性
    });
    if (options.TwoPlayers) // 如果options中的TwoPlayers属性为true
    {
        var oneDataCol = new[] { 1 }; // 创建一个包含1的数组
        Console.WriteLine($"Compared to the actual casualties at {battle.Name}"); // 在控制台输出字符串
        ConsoleUtils.WriteTable(oneDataCol, armies.Select(a => new ConsoleUtils.TableRow<int>( // 使用ConsoleUtils类的WriteTable方法输出表格
            a.Side.ToString(), // 将a的Side属性转换为字符串
            _ => $"{(double)a.Casualties / battle.Casualties[(int)a.Side]}", After: "% of the original") // 计算并输出a的Casualties属性占battle.Casualties[(int)a.Side]的百分比
        ).ToList());
    }

    Side winner; // 声明一个变量winner，类型为Side枚举
    switch (armies[0].AllLost, armies[1].AllLost, armies[0].MenLost - armies[1].MenLost) // 根据armies[0].AllLost、armies[1].AllLost和armies[0].MenLost - armies[1].MenLost进行切换
    {
        case (true, true, _) or (false, false, 0): // 如果armies[0].AllLost和armies[1].AllLost都为true，或者都为false且armies[0].MenLost - armies[1].MenLost为0
            Console.WriteLine("Battle outcome unresolved"); // 在控制台输出字符串"Battle outcome unresolved"
            winner = Side.Both; // 将winner赋值为Side枚举的Both成员，表示平局
            break;  # 结束当前的 case 分支
        case (false, true, _) or (false, false, < 0):  # 如果条件为 (false, true, _) 或者 (false, false, < 0)
            Console.WriteLine($"The Confederacy wins {battle.Name}");  # 打印输出南方联盟获胜的信息
            winner = Side.Confederate;  # 将获胜方设为南方联盟
            break;  # 结束当前的 case 分支
        case (true, false, _) or (false, false, > 0):  # 如果条件为 (true, false, _) 或者 (false, false, > 0)
            Console.WriteLine($"The Union wins {battle.Name}");  # 打印输出联邦获胜的信息
            winner = Side.Union;  # 将获胜方设为联邦
            break;  # 结束当前的 case 分支
    }
    if (!isReplay)  # 如果不是回放模式
    {
        armies.ForEach(a => a.RecordResult(winner));  # 对每个军队记录结果
    }
    Console.WriteLine("---------------");  # 打印分隔线
    previous = battle;  # 将当前战斗设为上一场战斗
    return true;  # 返回 true
}

void DisplayResult()  # 定义 DisplayResult 函数
# 显示第一个军队与第二个军队之间的战争结果
armies[0].DisplayWarResult(armies[1]);

# 获取第一个军队已经进行的战斗次数
int battles = armies[0].BattlesFought;

# 如果已经进行过战斗
if (battles > 0)
{
    # 打印已经进行的战斗次数（不包括重新进行的战斗）
    Console.WriteLine($"For the {battles} battles fought (excluding reruns)");

    # 使用ConsoleUtils中的WriteTable方法打印军队的历史损失、模拟损失和模拟损失占历史损失的百分比
    ConsoleUtils.WriteTable(armies, new()
    {
        new("", a => a.Side),  # 军队的阵营
        new("Historical Losses", a => a.CumulativeHistoricCasualties),  # 历史损失
        new("Simulated Losses", a => a.CumulativeSimulatedCasualties),  # 模拟损失
        new("  % of original", a => ((double)a.CumulativeSimulatedCasualties / a.CumulativeHistoricCasualties).ToString("p2"))  # 模拟损失占历史损失的百分比
    }, transpose: true);  # 转置表格

    # 显示第二个军队的战略
    armies[1].DisplayStrategies();
}
```