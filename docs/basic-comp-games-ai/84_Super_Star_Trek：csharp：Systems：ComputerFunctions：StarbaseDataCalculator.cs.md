# `84_Super_Star_Trek\csharp\Systems\ComputerFunctions\StarbaseDataCalculator.cs`

```
using Games.Common.IO;  # 导入 Games.Common.IO 模块
using SuperStarTrek.Objects;  # 导入 SuperStarTrek.Objects 模块
using SuperStarTrek.Resources;  # 导入 SuperStarTrek.Resources 模块
using SuperStarTrek.Space;  # 导入 SuperStarTrek.Space 模块

namespace SuperStarTrek.Systems.ComputerFunctions;  # 定义 SuperStarTrek.Systems.ComputerFunctions 命名空间

internal class StarbaseDataCalculator : NavigationCalculator  # 定义 StarbaseDataCalculator 类，继承自 NavigationCalculator 类
{
    private readonly Enterprise _enterprise;  # 声明私有属性 _enterprise，类型为 Enterprise

    internal StarbaseDataCalculator(Enterprise enterprise, IReadWrite io)  # 定义 StarbaseDataCalculator 类的构造函数，参数为 enterprise 和 io
        : base("Starbase nav data", io)  # 调用父类 NavigationCalculator 的构造函数，传入参数 "Starbase nav data" 和 io
    {
        _enterprise = enterprise;  # 初始化 _enterprise 属性为传入的 enterprise 参数
    }

    internal override void Execute(Quadrant quadrant)  # 定义 Execute 方法，参数为 quadrant
    {
        if (!quadrant.HasStarbase)  # 如果 quadrant 没有星舰基地
        {
            IO.WriteLine(Strings.NoStarbase);  # 输出字符串 "NoStarbase"，表示星舰没有星空站
            return;  # 返回，结束函数
        }

        IO.WriteLine("From Enterprise to Starbase:");  # 输出字符串 "From Enterprise to Starbase:"，表示从企业号到星空站的信息

        WriteDirectionAndDistance(_enterprise.SectorCoordinates, quadrant.Starbase.Sector);  # 调用函数 WriteDirectionAndDistance，传入企业号的扇区坐标和星空站的扇区坐标
    }
}
```