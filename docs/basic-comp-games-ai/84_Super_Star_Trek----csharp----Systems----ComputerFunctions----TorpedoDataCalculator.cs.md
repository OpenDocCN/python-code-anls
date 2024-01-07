# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\ComputerFunctions\TorpedoDataCalculator.cs`

```

// 引入所需的命名空间
using Games.Common.IO;
using SuperStarTrek.Objects;
using SuperStarTrek.Resources;
using SuperStarTrek.Space;

// 定义 TorpedoDataCalculator 类，继承自 NavigationCalculator 类
namespace SuperStarTrek.Systems.ComputerFunctions;

internal class TorpedoDataCalculator : NavigationCalculator
{
    private readonly Enterprise _enterprise; // 声明私有属性 _enterprise，表示企业号

    // TorpedoDataCalculator 类的构造函数，接受 Enterprise 对象和 IReadWrite 对象作为参数
    internal TorpedoDataCalculator(Enterprise enterprise, IReadWrite io)
        : base("Photon torpedo data", io) // 调用父类的构造函数
    {
        _enterprise = enterprise; // 初始化 _enterprise 属性
    }

    // 重写父类的 Execute 方法
    internal override void Execute(Quadrant quadrant)
    {
        // 如果象限中没有克林贡战舰，则输出提示信息并返回
        if (!quadrant.HasKlingons)
        {
            IO.WriteLine(Strings.NoEnemyShips);
            return;
        }

        // 输出从企业号到克林贡战舰的信息
        IO.WriteLine("From Enterprise to Klingon battle cruiser".Pluralize(quadrant.KlingonCount));

        // 遍历象限中的克林贡战舰，输出企业号到克林贡战舰的方向和距离
        foreach (var klingon in quadrant.Klingons)
        {
            WriteDirectionAndDistance(_enterprise.SectorCoordinates, klingon.Sector);
        }
    }
}

```