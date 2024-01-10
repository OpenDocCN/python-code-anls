# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\ComputerFunctions\DirectionDistanceCalculator.cs`

```
// 引入命名空间 Games.Common.IO，SuperStarTrek.Objects，SuperStarTrek.Space
using Games.Common.IO;
using SuperStarTrek.Objects;
using SuperStarTrek.Space;

// 定义名为 DirectionDistanceCalculator 的内部类，继承自 NavigationCalculator
internal class DirectionDistanceCalculator : NavigationCalculator
{
    // 声明私有字段 _enterprise，类型为 Enterprise
    private readonly Enterprise _enterprise;
    // 声明私有字段 _io，类型为 IReadWrite
    private readonly IReadWrite _io;

    // 定义 DirectionDistanceCalculator 的构造函数，接受 Enterprise 和 IReadWrite 作为参数
    internal DirectionDistanceCalculator(Enterprise enterprise, IReadWrite io)
        // 调用基类 NavigationCalculator 的构造函数，传入字符串 "Direction/distance calculator" 和 io
        : base("Direction/distance calculator", io)
    {
        // 将参数 enterprise 赋值给私有字段 _enterprise
        _enterprise = enterprise;
        // 将参数 io 赋值给私有字段 _io
        _io = io;
    }

    // 重写基类的 Execute 方法，接受 Quadrant 作为参数
    internal override void Execute(Quadrant quadrant)
    {
        // 输出 "Direction/distance calculator:"
        IO.WriteLine("Direction/distance calculator:");
        // 输出 "You are at quadrant" 和 _enterprise.QuadrantCoordinates
        IO.Write($"You are at quadrant {_enterprise.QuadrantCoordinates}");
        // 输出 " sector " 和 _enterprise.SectorCoordinates
        IO.WriteLine($" sector {_enterprise.SectorCoordinates");
        // 输出 "Please enter"
        IO.WriteLine("Please enter");

        // 调用 WriteDirectionAndDistance 方法，传入 _io.GetCoordinates("  Initial coordinates") 和 _io.GetCoordinates("  Final coordinates") 作为参数
        WriteDirectionAndDistance(
            _io.GetCoordinates("  Initial coordinates"),
            _io.GetCoordinates("  Final coordinates"));
    }
}
```