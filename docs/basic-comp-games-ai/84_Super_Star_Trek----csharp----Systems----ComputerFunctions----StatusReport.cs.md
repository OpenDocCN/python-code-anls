# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\ComputerFunctions\StatusReport.cs`

```

// 引入所需的命名空间
using Games.Common.IO;
using SuperStarTrek.Commands;
using SuperStarTrek.Objects;
using SuperStarTrek.Space;

// 定义一个名为 StatusReport 的内部类，继承自 ComputerFunction 类
internal class StatusReport : ComputerFunction
{
    // 声明私有变量
    private readonly Game _game;
    private readonly Galaxy _galaxy;
    private readonly Enterprise _enterprise;

    // 构造函数，接受 Game、Galaxy、Enterprise 和 IReadWrite 对象作为参数
    internal StatusReport(Game game, Galaxy galaxy, Enterprise enterprise, IReadWrite io)
        : base("Status report", io) // 调用基类的构造函数
    {
        // 初始化私有变量
        _game = game;
        _galaxy = galaxy;
        _enterprise = enterprise;
    }

    // 重写基类的 Execute 方法
    internal override void Execute(Quadrant quadrant)
    {
        // 输出状态报告标题
        IO.WriteLine("   Status report:");
        // 输出克林贡人数
        IO.Write("Klingon".Pluralize(_galaxy.KlingonCount));
        IO.WriteLine($" left:  {_galaxy.KlingonCount}");
        // 输出任务剩余时间
        IO.WriteLine($"Mission must be completed in {_game.StardatesRemaining:0.#} stardates.");

        // 如果星舰基地数量大于 0
        if (_galaxy.StarbaseCount > 0)
        {
            // 输出星舰基地数量
            IO.Write($"The Federation is maintaining {_galaxy.StarbaseCount} ");
            // 输出星舰基地的复数形式
            IO.Write("starbase".Pluralize(_galaxy.StarbaseCount));
            IO.WriteLine(" in the galaxy.");
        }
        else
        {
            // 输出无星舰基地的提示
            IO.WriteLine("Your stupidity has left you on your own in");
            IO.WriteLine("  the galaxy -- you have no starbases left!");
        }

        // 执行 DAM 命令
        _enterprise.Execute(Command.DAM);
    }
}

```