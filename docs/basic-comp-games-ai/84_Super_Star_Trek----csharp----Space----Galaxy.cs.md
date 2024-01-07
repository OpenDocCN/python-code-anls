# `basic-computer-games\84_Super_Star_Trek\csharp\Space\Galaxy.cs`

```

// 引入必要的命名空间
using System.Collections.Generic;
using System.Linq;
using Games.Common.Randomness;
using SuperStarTrek.Resources;

// 使用静态导入简化代码
using static System.StringSplitOptions;

// 定义 SuperStarTrek.Space 命名空间下的 Galaxy 类
namespace SuperStarTrek.Space;

// Galaxy 类的内部实现
internal class Galaxy
{
    // 静态只读字段，存储区域名称和子区域标识符
    private static readonly string[] _regionNames;
    private static readonly string[] _subRegionIdentifiers;
    // 存储象限信息的二维数组
    private readonly QuadrantInfo[][] _quadrants;

    // 静态构造函数，初始化区域名称和子区域标识符
    static Galaxy()
    {
        _regionNames = Strings.RegionNames.Split(new[] { ' ', '\n' }, RemoveEmptyEntries | TrimEntries);
        _subRegionIdentifiers = new[] { "I", "II", "III", "IV" };
    }

    // Galaxy 类的构造函数，初始化象限信息
    internal Galaxy(IRandom random)
    {
        // 使用 LINQ 初始化象限信息的二维数组
        _quadrants = Enumerable
            .Range(0, 8)
            .Select(x => Enumerable
                .Range(0, 8)
                .Select(y => new Coordinates(x, y))
                .Select(c => QuadrantInfo.Create(c, GetQuadrantName(c), random))
                .ToArray())
            .ToArray();

        // 如果星舰基地数量为 0，则随机选择一个象限添加星舰基地和克林贡战舰
        if (StarbaseCount == 0)
        {
            var randomQuadrant = this[random.NextCoordinate()];
            randomQuadrant.AddStarbase();

            if (randomQuadrant.KlingonCount < 2)
            {
                randomQuadrant.AddKlingon();
            }
        }
    }

    // 索引器，根据坐标获取象限信息
    internal QuadrantInfo this[Coordinates coordinate] => _quadrants[coordinate.X][coordinate.Y];

    // 获取克林贡战舰数量
    internal int KlingonCount => _quadrants.SelectMany(q => q).Sum(q => q.KlingonCount);

    // 获取星舰基地数量
    internal int StarbaseCount => _quadrants.SelectMany(q => q).Count(q => q.HasStarbase);

    // 获取所有象限信息的集合
    internal IEnumerable<IEnumerable<QuadrantInfo>> Quadrants => _quadrants;

    // 根据坐标获取象限名称
    private static string GetQuadrantName(Coordinates coordinates) =>
        $"{_regionNames[coordinates.RegionIndex]} {_subRegionIdentifiers[coordinates.SubRegionIndex]}";

    // 获取邻近象限信息的集合
    internal IEnumerable<IEnumerable<QuadrantInfo>> GetNeighborhood(Quadrant quadrant) =>
        Enumerable.Range(-1, 3)
            .Select(dx => dx + quadrant.Coordinates.X)
            .Select(x => GetNeighborhoodRow(quadrant, x));
    private IEnumerable<QuadrantInfo> GetNeighborhoodRow(Quadrant quadrant, int x) =>
        Enumerable.Range(-1, 3)
            .Select(dy => dy + quadrant.Coordinates.Y)
            .Select(y => y < 0 || y > 7 || x < 0 || x > 7 ? null : _quadrants[x][y]);
}

```