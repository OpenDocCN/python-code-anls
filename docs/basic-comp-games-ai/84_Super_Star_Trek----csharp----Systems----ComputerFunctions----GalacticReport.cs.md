# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\ComputerFunctions\GalacticReport.cs`

```
// 引入命名空间
using System.Collections.Generic;
using System.Linq;
using Games.Common.IO;
using SuperStarTrek.Space;

// 声明一个内部抽象类 GalacticReport，继承自 ComputerFunction
namespace SuperStarTrek.Systems.ComputerFunctions
{
    internal abstract class GalacticReport : ComputerFunction
    {
        // 声明一个受保护的 Galaxy 属性
        internal GalacticReport(string description, IReadWrite io, Galaxy galaxy)
            : base(description, io)
        {
            Galaxy = galaxy;
        }

        // 受保护的 Galaxy 属性
        protected Galaxy Galaxy { get; }

        // 受保护的抽象方法，用于写入报告头部信息
        protected abstract void WriteHeader(Quadrant quadrant);

        // 受保护的抽象方法，用于获取报告行数据
        protected abstract IEnumerable<string> GetRowData();

        // 重写父类的 Execute 方法
        internal sealed override void Execute(Quadrant quadrant)
        {
            // 调用 WriteHeader 方法写入报告头部信息
            WriteHeader(quadrant);
            // 输出表头信息
            IO.WriteLine("       1     2     3     4     5     6     7     8");
            IO.WriteLine("     ----- ----- ----- ----- ----- ----- ----- -----");

            // 遍历获取的行数据，输出每一行的信息
            foreach (var (row, index) in GetRowData().Select((r, i) => (r, i)))
            {
                IO.WriteLine($" {index+1}   {row}");
                IO.WriteLine("     ----- ----- ----- ----- ----- ----- ----- -----");
            }
        }
    }
}
```