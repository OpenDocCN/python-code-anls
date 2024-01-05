# `d:/src/tocomm/basic-computer-games\84_Super_Star_Trek\csharp\Systems\ComputerFunctions\StatusReport.cs`

```
using Games.Common.IO;  # 导入 Games.Common.IO 模块
using SuperStarTrek.Commands;  # 导入 SuperStarTrek.Commands 模块
using SuperStarTrek.Objects;  # 导入 SuperStarTrek.Objects 模块
using SuperStarTrek.Space;  # 导入 SuperStarTrek.Space 模块

namespace SuperStarTrek.Systems.ComputerFunctions;  # 定义 SuperStarTrek.Systems.ComputerFunctions 命名空间

internal class StatusReport : ComputerFunction  # 定义 StatusReport 类，继承自 ComputerFunction 类
{
    private readonly Game _game;  # 声明私有变量 _game，类型为 Game
    private readonly Galaxy _galaxy;  # 声明私有变量 _galaxy，类型为 Galaxy
    private readonly Enterprise _enterprise;  # 声明私有变量 _enterprise，类型为 Enterprise

    internal StatusReport(Game game, Galaxy galaxy, Enterprise enterprise, IReadWrite io)  # 定义 StatusReport 类的构造函数，接受 game, galaxy, enterprise 和 io 四个参数
        : base("Status report", io)  # 调用父类的构造函数，传入字符串 "Status report" 和 io 参数
    {
        _game = game;  # 将传入的 game 参数赋值给 _game 变量
        _galaxy = galaxy;  # 将传入的 galaxy 参数赋值给 _galaxy 变量
        _enterprise = enterprise;  # 将传入的 enterprise 参数赋值给 _enterprise 变量
    }
}
    # 重写 Execute 方法，接受一个 Quadrant 参数
    internal override void Execute(Quadrant quadrant)
    {
        # 输出状态报告
        IO.WriteLine("   Status report:");
        # 输出 Klingon 的数量和剩余数量
        IO.Write("Klingon".Pluralize(_galaxy.KlingonCount));
        IO.WriteLine($" left:  {_galaxy.KlingonCount}");
        # 输出任务剩余的星际日期
        IO.WriteLine($"Mission must be completed in {_game.StardatesRemaining:0.#} stardates.");

        # 如果星球基地数量大于 0
        if (_galaxy.StarbaseCount > 0)
        {
            # 输出星际联邦维护的星球基地数量
            IO.Write($"The Federation is maintaining {_galaxy.StarbaseCount} ");
            # 输出星球基地的复数形式
            IO.Write("starbase".Pluralize(_galaxy.StarbaseCount));
            IO.WriteLine(" in the galaxy.");
        }
        # 如果星球基地数量等于 0
        else
        {
            # 输出留下的星球基地数量为 0 的信息
            IO.WriteLine("Your stupidity has left you on your own in");
            IO.WriteLine("  the galaxy -- you have no starbases left!");
        }
# 调用 _enterprise 对象的 Execute 方法，参数为 Command.DAM
_enterprise.Execute(Command.DAM);
# 结束类定义
}
# 结束方法定义
}
```