# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\ComputerFunctions\StarbaseDataCalculator.cs`

```
// 引入所需的命名空间
using Games.Common.IO;
using SuperStarTrek.Objects;
using SuperStarTrek.Resources;
using SuperStarTrek.Space;

// 定义 StarbaseDataCalculator 类，继承自 NavigationCalculator 类
namespace SuperStarTrek.Systems.ComputerFunctions
{
    // 定义 StarbaseDataCalculator 类
    internal class StarbaseDataCalculator : NavigationCalculator
    {
        // 声明私有字段 _enterprise，类型为 Enterprise
        private readonly Enterprise _enterprise;

        // 定义 StarbaseDataCalculator 类的构造函数，接受 Enterprise 和 IReadWrite 类型的参数
        internal StarbaseDataCalculator(Enterprise enterprise, IReadWrite io)
            // 调用基类 NavigationCalculator 的构造函数，传入字符串 "Starbase nav data" 和 io 参数
            : base("Starbase nav data", io)
        {
            // 将传入的 enterprise 参数赋值给 _enterprise 字段
            _enterprise = enterprise;
        }

        // 重写基类的 Execute 方法，接受 Quadrant 类型的参数
        internal override void Execute(Quadrant quadrant)
        {
            // 如果当前象限没有星舰基地
            if (!quadrant.HasStarbase)
            {
                // 输出 "No starbase" 字符串
                IO.WriteLine(Strings.NoStarbase);
                // 退出方法
                return;
            }

            // 输出 "From Enterprise to Starbase:" 字符串
            IO.WriteLine("From Enterprise to Starbase:");

            // 调用 WriteDirectionAndDistance 方法，传入企业的扇区坐标和当前象限的星舰基地扇区坐标
            WriteDirectionAndDistance(_enterprise.SectorCoordinates, quadrant.Starbase.Sector);
        }
    }
}
```