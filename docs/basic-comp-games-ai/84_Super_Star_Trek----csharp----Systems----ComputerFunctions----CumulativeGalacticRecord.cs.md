# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\ComputerFunctions\CumulativeGalacticRecord.cs`

```

// 引入命名空间
using System.Collections.Generic;
using System.Linq;
using Games.Common.IO;
using SuperStarTrek.Space;

// 定义 CumulativeGalacticRecord 类，继承自 GalacticReport 类
namespace SuperStarTrek.Systems.ComputerFunctions;

internal class CumulativeGalacticRecord : GalacticReport
{
    // 构造函数，接受 IReadWrite 和 Galaxy 对象作为参数
    internal CumulativeGalacticRecord(IReadWrite io, Galaxy galaxy)
        : base("Cumulative galactic record", io, galaxy)
    {
    }

    // 重写 WriteHeader 方法
    protected override void WriteHeader(Quadrant quadrant)
    {
        // 输出空行
        IO.WriteLine();
        // 输出当前象限的计算机记录
        IO.WriteLine($"Computer record of galaxy for quadrant {quadrant.Coordinates}");
        // 输出空行
        IO.WriteLine();
    }

    // 重写 GetRowData 方法，返回一个字符串列表
    protected override IEnumerable<string> GetRowData() =>
        // 获取 Galaxy 对象中每个象限的数据，并以空格分隔连接成字符串
        Galaxy.Quadrants.Select(row => " " + string.Join("   ", row));
}

```