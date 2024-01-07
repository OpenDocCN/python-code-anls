# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\ComputerFunctions\GalacticReport.cs`

```

// 引入命名空间
using System.Collections.Generic;
using System.Linq;
using Games.Common.IO;
using SuperStarTrek.Space;

// 定义一个抽象类 GalacticReport，继承自 ComputerFunction
namespace SuperStarTrek.Systems.ComputerFunctions;
internal abstract class GalacticReport : ComputerFunction
{
    // 构造函数，接受描述、I/O 接口和星系作为参数
    internal GalacticReport(string description, IReadWrite io, Galaxy galaxy)
        : base(description, io)
    {
        Galaxy = galaxy;
    }

    // 受保护的属性，表示星系
    protected Galaxy Galaxy { get; }

    // 受保护的抽象方法，用于写入报告头部信息
    protected abstract void WriteHeader(Quadrant quadrant);

    // 受保护的抽象方法，用于获取报告的行数据
    protected abstract IEnumerable<string> GetRowData();

    // 重写父类的 Execute 方法，执行报告生成
    internal sealed override void Execute(Quadrant quadrant)
    {
        // 调用 WriteHeader 方法，写入报告头部信息
        WriteHeader(quadrant);
        // 输出表头
        IO.WriteLine("       1     2     3     4     5     6     7     8");
        IO.WriteLine("     ----- ----- ----- ----- ----- ----- ----- -----");

        // 遍历行数据，输出每行的内容
        foreach (var (row, index) in GetRowData().Select((r, i) => (r, i)))
        {
            IO.WriteLine($" {index+1}   {row}");
            IO.WriteLine("     ----- ----- ----- ----- ----- ----- ----- -----");
        }
    }
}

```