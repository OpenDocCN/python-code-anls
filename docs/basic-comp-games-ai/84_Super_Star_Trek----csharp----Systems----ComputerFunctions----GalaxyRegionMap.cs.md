# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\ComputerFunctions\GalaxyRegionMap.cs`

```

// 引入必要的命名空间
using System.Collections.Generic;
using System.Linq;
using Games.Common.IO;
using SuperStarTrek.Resources;
using SuperStarTrek.Space;

// 定义 GalaxyRegionMap 类，继承自 GalacticReport 类
internal class GalaxyRegionMap : GalacticReport
{
    // 构造函数，接受 IReadWrite 对象和 Galaxy 对象作为参数
    internal GalaxyRegionMap(IReadWrite io, Galaxy galaxy)
        : base("Galaxy 'region name' map", io, galaxy)
    {
    }

    // 重写 WriteHeader 方法，输出标题
    protected override void WriteHeader(Quadrant quadrant) =>
        IO.WriteLine("                        The Galaxy");

    // 重写 GetRowData 方法，返回行数据
    protected override IEnumerable<string> GetRowData() =>
        Strings.RegionNames.Split('\n').Select(n => n.TrimEnd('\r'));
}

```