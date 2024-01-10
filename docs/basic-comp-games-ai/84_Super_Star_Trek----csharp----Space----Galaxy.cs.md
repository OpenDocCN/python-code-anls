# `basic-computer-games\84_Super_Star_Trek\csharp\Space\Galaxy.cs`

```
using System.Collections.Generic;  // 导入 System.Collections.Generic 命名空间
using System.Linq;  // 导入 System.Linq 命名空间
using Games.Common.Randomness;  // 导入 Games.Common.Randomness 命名空间
using SuperStarTrek.Resources;  // 导入 SuperStarTrek.Resources 命名空间

using static System.StringSplitOptions;  // 导入 System.StringSplitOptions 命名空间中的所有静态成员

namespace SuperStarTrek.Space  // 定义 SuperStarTrek.Space 命名空间
{
    internal class Galaxy  // 定义 Galaxy 类
    {
        private static readonly string[] _regionNames;  // 声明并初始化只读字符串数组 _regionNames
        private static readonly string[] _subRegionIdentifiers;  // 声明并初始化只读字符串数组 _subRegionIdentifiers
        private readonly QuadrantInfo[][] _quadrants;  // 声明只读二维数组 _quadrants，元素类型为 QuadrantInfo

        static Galaxy()  // 静态构造函数
        {
            _regionNames = Strings.RegionNames.Split(new[] { ' ', '\n' }, RemoveEmptyEntries | TrimEntries);  // 使用空格和换行符对字符串 RegionNames 进行分割，并去除空白项和修剪条目，然后赋值给 _regionNames
            _subRegionIdentifiers = new[] { "I", "II", "III", "IV" };  // 初始化 _subRegionIdentifiers 数组
        }

        internal Galaxy(IRandom random)  // Galaxy 类的构造函数，接受一个 IRandom 类型的参数 random
        {
            _quadrants = Enumerable  // 使用 Enumerable 类的方法创建二维数组 _quadrants
                .Range(0, 8)  // 生成一个从 0 到 7 的序列
                .Select(x => Enumerable  // 对每个 x 执行以下操作
                    .Range(0, 8)  // 生成一个从 0 到 7 的序列
                    .Select(y => new Coordinates(x, y))  // 对每个 y 创建一个 Coordinates 对象
                    .Select(c => QuadrantInfo.Create(c, GetQuadrantName(c), random))  // 对每个 Coordinates 对象创建一个 QuadrantInfo 对象
                    .ToArray())  // 将结果转换为数组
                .ToArray();  // 将结果转换为数组

            if (StarbaseCount == 0)  // 如果 StarbaseCount 为 0
            {
                var randomQuadrant = this[random.NextCoordinate()];  // 获取随机坐标对应的 QuadrantInfo 对象
                randomQuadrant.AddStarbase();  // 给随机 QuadrantInfo 对象添加星舰基地

                if (randomQuadrant.KlingonCount < 2)  // 如果随机 QuadrantInfo 对象的 KlingonCount 小于 2
                {
                    randomQuadrant.AddKlingon();  // 给随机 QuadrantInfo 对象添加克林贡
                }
            }
        }

        internal QuadrantInfo this[Coordinates coordinate] => _quadrants[coordinate.X][coordinate.Y];  // 索引器，根据坐标获取对应的 QuadrantInfo 对象

        internal int KlingonCount => _quadrants.SelectMany(q => q).Sum(q => q.KlingonCount);  // 获取所有 QuadrantInfo 对象的 KlingonCount 总和

        internal int StarbaseCount => _quadrants.SelectMany(q => q).Count(q => q.HasStarbase);  // 获取所有 QuadrantInfo 对象中包含星舰基地的数量

        internal IEnumerable<IEnumerable<QuadrantInfo>> Quadrants => _quadrants;  // 获取 QuadrantInfo 对象的嵌套集合

        private static string GetQuadrantName(Coordinates coordinates) =>  // 定义私有静态方法 GetQuadrantName，接受 Coordinates 类型的参数，返回字符串
            $"{_regionNames[coordinates.RegionIndex]} {_subRegionIdentifiers[coordinates.SubRegionIndex]}";  // 根据坐标的 RegionIndex 和 SubRegionIndex 获取象限名称

        internal IEnumerable<IEnumerable<QuadrantInfo>> GetNeighborhood(Quadrant quadrant) =>  // 定义内部方法 GetNeighborhood，接受 Quadrant 类型的参数，返回嵌套集合
            Enumerable.Range(-1, 3)  // 生成一个从 -1 到 2 的序列
                .Select(dx => dx + quadrant.Coordinates.X)  // 对每个 dx 执行操作，获取与 quadrant 的 X 坐标相加后的值
                .Select(x => GetNeighborhoodRow(quadrant, x));  // 对每个 x 执行操作，获取与 quadrant 相邻的 QuadrantInfo 对象的集合
    }
}
    # 获取指定象限的邻近行的信息
    private IEnumerable<QuadrantInfo> GetNeighborhoodRow(Quadrant quadrant, int x) =>
        # 生成一个范围为-1到3的整数序列
        Enumerable.Range(-1, 3)
            # 对每个整数进行操作，将其加上象限的Y坐标，得到新的Y坐标
            .Select(dy => dy + quadrant.Coordinates.Y)
            # 对每个新的Y坐标进行操作，判断是否超出边界，如果超出则返回null，否则返回对应象限的信息
            .Select(y => y < 0 || y > 7 || x < 0 || x > 7 ? null : _quadrants[x][y]);
# 闭合前面的函数定义
```