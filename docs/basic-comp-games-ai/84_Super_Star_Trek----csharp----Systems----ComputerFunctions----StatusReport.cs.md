# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\ComputerFunctions\StatusReport.cs`

```py
using Games.Common.IO;  // 导入 Games.Common.IO 命名空间
using SuperStarTrek.Commands;  // 导入 SuperStarTrek.Commands 命名空间
using SuperStarTrek.Objects;  // 导入 SuperStarTrek.Objects 命名空间
using SuperStarTrek.Space;  // 导入 SuperStarTrek.Space 命名空间

namespace SuperStarTrek.Systems.ComputerFunctions;  // 定义 SuperStarTrek.Systems.ComputerFunctions 命名空间

internal class StatusReport : ComputerFunction  // 定义 StatusReport 类，继承自 ComputerFunction 类
{
    private readonly Game _game;  // 声明私有只读字段 _game，类型为 Game
    private readonly Galaxy _galaxy;  // 声明私有只读字段 _galaxy，类型为 Galaxy
    private readonly Enterprise _enterprise;  // 声明私有只读字段 _enterprise，类型为 Enterprise

    internal StatusReport(Game game, Galaxy galaxy, Enterprise enterprise, IReadWrite io)  // 定义 StatusReport 类的构造函数，接受 Game、Galaxy、Enterprise 和 IReadWrite 参数
        : base("Status report", io)  // 调用基类的构造函数，传入字符串 "Status report" 和 io 参数
    {
        _game = game;  // 将传入的 game 参数赋值给 _game 字段
        _galaxy = galaxy;  // 将传入的 galaxy 参数赋值给 _galaxy 字段
        _enterprise = enterprise;  // 将传入的 enterprise 参数赋值给 _enterprise 字段
    }

    internal override void Execute(Quadrant quadrant)  // 重写基类的 Execute 方法，接受 Quadrant 参数
    {
        IO.WriteLine("   Status report:");  // 输出 "   Status report:" 到控制台
        IO.Write("Klingon".Pluralize(_galaxy.KlingonCount));  // 输出根据 _galaxy.KlingonCount 数量变化的 "Klingon" 到控制台
        IO.WriteLine($" left:  {_galaxy.KlingonCount}");  // 输出 " left:  " 后接 _galaxy.KlingonCount 到控制台
        IO.WriteLine($"Mission must be completed in {_game.StardatesRemaining:0.#} stardates.");  // 输出 "Mission must be completed in " 后接 _game.StardatesRemaining 格式化后的值到控制台

        if (_galaxy.StarbaseCount > 0)  // 如果 _galaxy.StarbaseCount 大于 0
        {
            IO.Write($"The Federation is maintaining {_galaxy.StarbaseCount} ");  // 输出 "The Federation is maintaining " 后接 _galaxy.StarbaseCount 到控制台
            IO.Write("starbase".Pluralize(_galaxy.StarbaseCount));  // 输出根据 _galaxy.StarbaseCount 数量变化的 "starbase" 到控制台
            IO.WriteLine(" in the galaxy.");  // 输出 " in the galaxy." 到控制台
        }
        else  // 否则
        {
            IO.WriteLine("Your stupidity has left you on your own in");  // 输出 "Your stupidity has left you on your own in" 到控制台
            IO.WriteLine("  the galaxy -- you have no starbases left!");  // 输出 "  the galaxy -- you have no starbases left!" 到控制台
        }

        _enterprise.Execute(Command.DAM);  // 调用 _enterprise 对象的 Execute 方法，传入 Command.DAM 参数
    }
}
```