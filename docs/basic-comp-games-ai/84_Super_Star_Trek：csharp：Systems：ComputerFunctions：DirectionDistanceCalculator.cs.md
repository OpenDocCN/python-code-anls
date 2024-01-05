# `84_Super_Star_Trek\csharp\Systems\ComputerFunctions\DirectionDistanceCalculator.cs`

```
using Games.Common.IO;  # 导入 Games.Common.IO 模块
using SuperStarTrek.Objects;  # 导入 SuperStarTrek.Objects 模块
using SuperStarTrek.Space;  # 导入 SuperStarTrek.Space 模块

namespace SuperStarTrek.Systems.ComputerFunctions;  # 定义 SuperStarTrek.Systems.ComputerFunctions 命名空间

internal class DirectionDistanceCalculator : NavigationCalculator  # 定义 DirectionDistanceCalculator 类，继承自 NavigationCalculator 类
{
    private readonly Enterprise _enterprise;  # 声明私有属性 _enterprise，类型为 Enterprise
    private readonly IReadWrite _io;  # 声明私有属性 _io，类型为 IReadWrite

    internal DirectionDistanceCalculator(Enterprise enterprise, IReadWrite io)  # 定义 DirectionDistanceCalculator 类的构造函数，接受 Enterprise 和 IReadWrite 类型的参数
        : base("Direction/distance calculator", io)  # 调用父类 NavigationCalculator 的构造函数，传入字符串 "Direction/distance calculator" 和 io 参数
    {
        _enterprise = enterprise;  # 将传入的 enterprise 参数赋值给 _enterprise 属性
        _io = io;  # 将传入的 io 参数赋值给 _io 属性
    }

    internal override void Execute(Quadrant quadrant)  # 定义 Execute 方法，接受 Quadrant 类型的参数 quadrant
    {
# 输出提示信息，指示方向/距离计算器
IO.WriteLine("Direction/distance calculator:");

# 输出当前位置的象限坐标和扇区坐标
IO.Write($"You are at quadrant {_enterprise.QuadrantCoordinates}");
IO.WriteLine($" sector {_enterprise.SectorCoordinates}");

# 输出提示信息，提示用户输入初始和最终坐标
IO.WriteLine("Please enter");

# 调用WriteDirectionAndDistance函数，传入初始坐标和最终坐标作为参数
WriteDirectionAndDistance(
    _io.GetCoordinates("  Initial coordinates"),
    _io.GetCoordinates("  Final coordinates"));
```