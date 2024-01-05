# `d:/src/tocomm/basic-computer-games\84_Super_Star_Trek\csharp\Space\Galaxy.cs`

```
using System.Collections.Generic; // 导入 System.Collections.Generic 命名空间，用于使用泛型集合类
using System.Linq; // 导入 System.Linq 命名空间，用于使用 LINQ 查询
using Games.Common.Randomness; // 导入 Games.Common.Randomness 命名空间
using SuperStarTrek.Resources; // 导入 SuperStarTrek.Resources 命名空间

using static System.StringSplitOptions; // 使用 static 关键字导入 System.StringSplitOptions 枚举的所有成员，方便直接使用

namespace SuperStarTrek.Space; // 声明 SuperStarTrek.Space 命名空间

internal class Galaxy // 声明 Galaxy 类
{
    private static readonly string[] _regionNames; // 声明并初始化只读的字符串数组 _regionNames
    private static readonly string[] _subRegionIdentifiers; // 声明并初始化只读的字符串数组 _subRegionIdentifiers
    private readonly QuadrantInfo[][] _quadrants; // 声明只读的二维数组 _quadrants，元素类型为 QuadrantInfo

    static Galaxy() // Galaxy 类的静态构造函数
    {
        _regionNames = Strings.RegionNames.Split(new[] { ' ', '\n' }, RemoveEmptyEntries | TrimEntries); // 使用字符串的 Split 方法将 Strings.RegionNames 按空格和换行符分割成字符串数组，并去除空白项和空格
        _subRegionIdentifiers = new[] { "I", "II", "III", "IV" }; // 初始化 _subRegionIdentifiers 数组为包含 "I", "II", "III", "IV" 的字符串数组
    }
}
    internal Galaxy(IRandom random)  # 创建一个名为Galaxy的内部构造函数，接受一个IRandom类型的参数random
    {
        _quadrants = Enumerable  # 使用Enumerable类创建一个名为_quadrants的属性
            .Range(0, 8)  # 生成一个从0到7的整数序列
            .Select(x => Enumerable  # 对整数序列进行映射操作
                .Range(0, 8)  # 生成一个从0到7的整数序列
                .Select(y => new Coordinates(x, y))  # 对y坐标进行映射操作，创建一个Coordinates对象
                .Select(c => QuadrantInfo.Create(c, GetQuadrantName(c), random))  # 对Coordinates对象进行映射操作，创建一个QuadrantInfo对象
                .ToArray())  # 将映射后的结果转换为数组
            .ToArray();  # 将映射后的结果转换为数组

        if (StarbaseCount == 0)  # 如果星舰基地数量为0
        {
            var randomQuadrant = this[random.NextCoordinate()];  # 创建一个名为randomQuadrant的变量，赋值为随机坐标对应的象限
            randomQuadrant.AddStarbase();  # 在randomQuadrant中添加一个星舰基地

            if (randomQuadrant.KlingonCount < 2)  # 如果randomQuadrant中克林贡数量小于2
            {
                randomQuadrant.AddKlingon();  # 在randomQuadrant中添加一个克林贡
    }
```
这是一个代码块的结束。

```
        }
```
这是一个代码块的结束。

```
    }
```
这是一个代码块的结束。

```
    internal QuadrantInfo this[Coordinates coordinate] => _quadrants[coordinate.X][coordinate.Y];
```
这是一个索引器，根据坐标返回对应的象限信息。

```
    internal int KlingonCount => _quadrants.SelectMany(q => q).Sum(q => q.KlingonCount);
```
这是一个属性，返回象限中克林贡的数量总和。

```
    internal int StarbaseCount => _quadrants.SelectMany(q => q).Count(q => q.HasStarbase);
```
这是一个属性，返回象限中星舰基地的数量。

```
    internal IEnumerable<IEnumerable<QuadrantInfo>> Quadrants => _quadrants;
```
这是一个属性，返回象限的集合。

```
    private static string GetQuadrantName(Coordinates coordinates) =>
        $"{_regionNames[coordinates.RegionIndex]} {_subRegionIdentifiers[coordinates.SubRegionIndex]}";
```
这是一个静态方法，根据坐标返回象限的名称。

```
    internal IEnumerable<IEnumerable<QuadrantInfo>> GetNeighborhood(Quadrant quadrant) =>
        Enumerable.Range(-1, 3)
            .Select(dx => dx + quadrant.Coordinates.X)
            .Select(x => GetNeighborhoodRow(quadrant, x));
```
这是一个方法，返回给定象限的邻居象限集合。

```
    private IEnumerable<QuadrantInfo> GetNeighborhoodRow(Quadrant quadrant, int x) =>
```
这是一个私有方法的声明。
        # 创建一个范围从-1到3的可枚举集合
        Enumerable.Range(-1, 3)
            # 对集合中的每个元素执行操作，将其加上象限坐标的Y值
            .Select(dy => dy + quadrant.Coordinates.Y)
            # 对集合中的每个元素执行操作，判断是否超出边界，如果是则返回null，否则返回_quadrants[x][y]
            .Select(y => y < 0 || y > 7 || x < 0 || x > 7 ? null : _quadrants[x][y]);
}
```