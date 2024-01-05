# `84_Super_Star_Trek\csharp\Systems\ComputerFunctions\TorpedoDataCalculator.cs`

```
using Games.Common.IO;  # 导入 Games.Common.IO 模块
using SuperStarTrek.Objects;  # 导入 SuperStarTrek.Objects 模块
using SuperStarTrek.Resources;  # 导入 SuperStarTrek.Resources 模块
using SuperStarTrek.Space;  # 导入 SuperStarTrek.Space 模块

namespace SuperStarTrek.Systems.ComputerFunctions;  # 定义 SuperStarTrek.Systems.ComputerFunctions 命名空间

internal class TorpedoDataCalculator : NavigationCalculator  # 定义 TorpedoDataCalculator 类，继承自 NavigationCalculator 类
{
    private readonly Enterprise _enterprise;  # 声明私有属性 _enterprise，类型为 Enterprise

    internal TorpedoDataCalculator(Enterprise enterprise, IReadWrite io)  # 定义 TorpedoDataCalculator 类的构造函数，接受 Enterprise 和 IReadWrite 类型的参数
        : base("Photon torpedo data", io)  # 调用父类 NavigationCalculator 的构造函数，传入字符串 "Photon torpedo data" 和 io 参数
    {
        _enterprise = enterprise;  # 将传入的 enterprise 参数赋值给 _enterprise 属性
    }

    internal override void Execute(Quadrant quadrant)  # 定义 Execute 方法，接受 Quadrant 类型的参数
    {
        if (!quadrant.HasKlingons)  # 如果 quadrant 对象的 HasKlingons 属性为假
        {
            IO.WriteLine(Strings.NoEnemyShips);  # 输出消息：没有敌方飞船
            return;  # 返回
        }

        IO.WriteLine("From Enterprise to Klingon battle cruiser".Pluralize(quadrant.KlingonCount));  # 输出消息：从企业到克林贡战舰，根据克林贡数量进行复数化处理

        foreach (var klingon in quadrant.Klingons)  # 遍历象限内的克林贡飞船
        {
            WriteDirectionAndDistance(_enterprise.SectorCoordinates, klingon.Sector);  # 调用函数，输出企业到克林贡飞船的方向和距离
        }
    }
}
```