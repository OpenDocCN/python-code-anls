# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\ComputerFunctions\TorpedoDataCalculator.cs`

```py
// 引入所需的命名空间
using Games.Common.IO;
using SuperStarTrek.Objects;
using SuperStarTrek.Resources;
using SuperStarTrek.Space;

// 定义 TorpedoDataCalculator 类，继承自 NavigationCalculator 类
namespace SuperStarTrek.Systems.ComputerFunctions
{
    // 声明 TorpedoDataCalculator 类
    internal class TorpedoDataCalculator : NavigationCalculator
    {
        // 声明私有字段 _enterprise，用于存储 Enterprise 对象
        private readonly Enterprise _enterprise;

        // TorpedoDataCalculator 类的构造函数，接受 Enterprise 对象和 IReadWrite 对象作为参数
        internal TorpedoDataCalculator(Enterprise enterprise, IReadWrite io)
            // 调用父类 NavigationCalculator 的构造函数，传入字符串 "Photon torpedo data" 和 io 对象
            : base("Photon torpedo data", io)
        {
            // 将传入的 Enterprise 对象赋值给 _enterprise 字段
            _enterprise = enterprise;
        }

        // 重写父类的 Execute 方法，接受 Quadrant 对象作为参数
        internal override void Execute(Quadrant quadrant)
        {
            // 如果象限中没有克林贡战舰
            if (!quadrant.HasKlingons)
            {
                // 输出字符串 "No enemy ships"
                IO.WriteLine(Strings.NoEnemyShips);
                // 退出方法
                return;
            }

            // 输出字符串 "From Enterprise to Klingon battle cruiser"，并使用 Pluralize 方法对 KlingonCount 进行复数化处理
            IO.WriteLine("From Enterprise to Klingon battle cruiser".Pluralize(quadrant.KlingonCount));

            // 遍历象限中的克林贡战舰
            foreach (var klingon in quadrant.Klingons)
            {
                // 调用 WriteDirectionAndDistance 方法，传入企业的区坐标和克林贡战舰的区坐标
                WriteDirectionAndDistance(_enterprise.SectorCoordinates, klingon.Sector);
            }
        }
    }
}
```