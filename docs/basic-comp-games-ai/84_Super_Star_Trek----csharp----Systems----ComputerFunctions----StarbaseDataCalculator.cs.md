# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\ComputerFunctions\StarbaseDataCalculator.cs`

```

// 使用 Games.Common.IO 命名空间
// 使用 SuperStarTrek.Objects 命名空间
// 使用 SuperStarTrek.Resources 命名空间
// 使用 SuperStarTrek.Space 命名空间
namespace SuperStarTrek.Systems.ComputerFunctions;

// StarbaseDataCalculator 类继承自 NavigationCalculator 类
internal class StarbaseDataCalculator : NavigationCalculator
{
    // 声明私有变量 _enterprise，类型为 Enterprise
    private readonly Enterprise _enterprise;

    // StarbaseDataCalculator 类的构造函数，接受 Enterprise 和 IReadWrite 作为参数
    // 调用基类 NavigationCalculator 的构造函数
    internal StarbaseDataCalculator(Enterprise enterprise, IReadWrite io)
        : base("Starbase nav data", io)
    {
        // 初始化 _enterprise 变量
        _enterprise = enterprise;
    }

    // 重写基类的 Execute 方法
    internal override void Execute(Quadrant quadrant)
    {
        // 如果象限中没有星舰基地
        if (!quadrant.HasStarbase)
        {
            // 输出提示信息
            IO.WriteLine(Strings.NoStarbase);
            // 退出方法
            return;
        }

        // 输出提示信息
        IO.WriteLine("From Enterprise to Starbase:");

        // 调用 WriteDirectionAndDistance 方法，传入企业的区坐标和象限中星舰基地的区坐标
        WriteDirectionAndDistance(_enterprise.SectorCoordinates, quadrant.Starbase.Sector);
    }
}

```