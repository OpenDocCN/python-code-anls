# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\ComputerFunctions\CumulativeGalacticRecord.cs`

```
using System.Collections.Generic; // 导入 System.Collections.Generic 命名空间，用于使用泛型集合类
using System.Linq; // 导入 System.Linq 命名空间，用于使用 LINQ 查询
using Games.Common.IO; // 导入 Games.Common.IO 命名空间，用于输入输出操作
using SuperStarTrek.Space; // 导入 SuperStarTrek.Space 命名空间，用于访问太空相关的类

namespace SuperStarTrek.Systems.ComputerFunctions // 定义 SuperStarTrek.Systems.ComputerFunctions 命名空间
{
    internal class CumulativeGalacticRecord : GalacticReport // 定义 CumulativeGalacticRecord 类，继承自 GalacticReport 类
    {
        internal CumulativeGalacticRecord(IReadWrite io, Galaxy galaxy) // 定义 CumulativeGalacticRecord 类的构造函数，接受 IReadWrite 和 Galaxy 类型的参数
            : base("Cumulative galactic record", io, galaxy) // 调用基类 GalacticReport 的构造函数，传入指定的参数
        {
        }

        protected override void WriteHeader(Quadrant quadrant) // 重写基类 GalacticReport 的 WriteHeader 方法，接受 Quadrant 类型的参数
        {
            IO.WriteLine(); // 输出空行
            IO.WriteLine($"Computer record of galaxy for quadrant {quadrant.Coordinates}"); // 输出指定格式的字符串
            IO.WriteLine(); // 输出空行
        }

        protected override IEnumerable<string> GetRowData() => // 重写基类 GalacticReport 的 GetRowData 方法，返回 IEnumerable<string> 类型的数据
            Galaxy.Quadrants.Select(row => " " + string.Join("   ", row)); // 使用 LINQ 查询获取每个象限的数据，并返回结果
    }
}
```