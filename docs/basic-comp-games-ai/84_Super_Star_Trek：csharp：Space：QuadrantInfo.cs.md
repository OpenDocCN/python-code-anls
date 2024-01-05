# `d:/src/tocomm/basic-computer-games\84_Super_Star_Trek\csharp\Space\QuadrantInfo.cs`

```
using Games.Common.Randomness;  // 导入 Games.Common.Randomness 命名空间

namespace SuperStarTrek.Space;  // 定义 SuperStarTrek.Space 命名空间

internal class QuadrantInfo  // 定义 QuadrantInfo 类
{
    private bool _isKnown;  // 声明私有布尔变量 _isKnown

    private QuadrantInfo(Coordinates coordinates, string name, int klingonCount, int starCount, bool hasStarbase)  // 定义私有构造函数，接受坐标、名称、克林贡数量、星球数量和是否有星舰基地作为参数
    {
        Coordinates = coordinates;  // 初始化 Coordinates 属性
        Name = name;  // 初始化 Name 属性
        KlingonCount = klingonCount;  // 初始化 KlingonCount 属性
        StarCount = starCount;  // 初始化 StarCount 属性
        HasStarbase = hasStarbase;  // 初始化 HasStarbase 属性
    }

    internal Coordinates Coordinates { get; }  // 定义只读的 Coordinates 属性

    internal string Name { get; }  // 定义只读的 Name 属性
```
    internal int KlingonCount { get; private set; }  // 定义一个私有的属性 KlingonCount，用于存储克林贡的数量

    internal bool HasStarbase { get; private set; }  // 定义一个私有的属性 HasStarbase，用于存储星舰基地的存在情况

    internal int StarCount { get; }  // 定义一个只读属性 StarCount，用于存储星星的数量

    internal static QuadrantInfo Create(Coordinates coordinates, string name, IRandom random)  // 创建一个静态方法 Create，用于生成 QuadrantInfo 对象
    {
        var klingonCount = random.NextFloat() switch  // 根据随机数生成克林贡的数量
        {
            > 0.98f => 3,
            > 0.95f => 2,
            > 0.80f => 1,
            _ => 0
        };
        var hasStarbase = random.NextFloat() > 0.96f;  // 根据随机数确定是否存在星舰基地
        var starCount = random.Next1To8Inclusive();  // 生成1到8之间的随机数，表示星星的数量

        return new QuadrantInfo(coordinates, name, klingonCount, starCount, hasStarbase);  // 返回一个新的 QuadrantInfo 对象
    } // 结束方法或代码块

    internal void AddKlingon() => KlingonCount += 1; // 增加克林贡数量的方法

    internal void AddStarbase() => HasStarbase = true; // 添加星舰基地的方法

    internal void MarkAsKnown() => _isKnown = true; // 将对象标记为已知的方法

    internal string Scan() // 扫描方法
    {
        _isKnown = true; // 将对象标记为已知
        return ToString(); // 返回对象的字符串表示形式
    }

    public override string ToString() => _isKnown ? $"{KlingonCount}{(HasStarbase ? 1 : 0)}{StarCount}" : "***"; // 返回对象的字符串表示形式，根据是否已知来决定返回值

    internal void RemoveKlingon() // 移除克林贡的方法
    {
        if (KlingonCount > 0) // 如果克林贡数量大于0
        {
# 减少克林贡计数
KlingonCount -= 1;
# 结束 if 语句块
}
# 结束 RemoveStarbase 方法
}

# 移除星舰基地
internal void RemoveStarbase() => HasStarbase = false;
# 结束代码块
}
```