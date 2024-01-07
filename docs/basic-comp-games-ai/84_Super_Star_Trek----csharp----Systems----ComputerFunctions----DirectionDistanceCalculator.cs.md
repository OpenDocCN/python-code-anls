# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\ComputerFunctions\DirectionDistanceCalculator.cs`

```

// 引入所需的命名空间
using Games.Common.IO;
using SuperStarTrek.Objects;
using SuperStarTrek.Space;

// 定义一个名为 DirectionDistanceCalculator 的内部类，继承自 NavigationCalculator
internal class DirectionDistanceCalculator : NavigationCalculator
{
    // 声明私有变量 _enterprise 和 _io
    private readonly Enterprise _enterprise;
    private readonly IReadWrite _io;

    // 构造函数，接受 Enterprise 和 IReadWrite 对象作为参数
    internal DirectionDistanceCalculator(Enterprise enterprise, IReadWrite io)
        : base("Direction/distance calculator", io)
    {
        // 初始化 _enterprise 和 _io
        _enterprise = enterprise;
        _io = io;
    }

    // 重写父类的 Execute 方法
    internal override void Execute(Quadrant quadrant)
    {
        // 输出提示信息
        IO.WriteLine("Direction/distance calculator:");
        IO.Write($"You are at quadrant {_enterprise.QuadrantCoordinates}");
        IO.WriteLine($" sector {_enterprise.SectorCoordinates}");
        IO.WriteLine("Please enter");

        // 调用 WriteDirectionAndDistance 方法，传入初始坐标和最终坐标
        WriteDirectionAndDistance(
            _io.GetCoordinates("  Initial coordinates"),
            _io.GetCoordinates("  Final coordinates"));
    }
}

```