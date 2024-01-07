# `basic-computer-games\84_Super_Star_Trek\csharp\Space\QuadrantInfo.cs`

```

// 使用 Games.Common.Randomness 命名空间
namespace SuperStarTrek.Space;

// 定义 QuadrantInfo 类
internal class QuadrantInfo
{
    // 是否已知信息
    private bool _isKnown;

    // 私有构造函数，初始化坐标、名称、克林贡数量、星球数量、是否有星舰基地
    private QuadrantInfo(Coordinates coordinates, string name, int klingonCount, int starCount, bool hasStarbase)
    {
        Coordinates = coordinates;
        Name = name;
        KlingonCount = klingonCount;
        StarCount = starCount;
        HasStarbase = hasStarbase;
    }

    // 坐标属性
    internal Coordinates Coordinates { get; }

    // 名称属性
    internal string Name { get; }

    // 克林贡数量属性
    internal int KlingonCount { get; private set; }

    // 是否有星舰基地属性
    internal bool HasStarbase { get; private set; }

    // 星球数量属性
    internal int StarCount { get; }

    // 创建 QuadrantInfo 实例的静态方法
    internal static QuadrantInfo Create(Coordinates coordinates, string name, IRandom random)
    {
        // 根据随机数确定克林贡数量
        var klingonCount = random.NextFloat() switch
        {
            > 0.98f => 3,
            > 0.95f => 2,
            > 0.80f => 1,
            _ => 0
        };
        // 根据随机数确定是否有星舰基地
        var hasStarbase = random.NextFloat() > 0.96f;
        // 根据随机数确定星球数量
        var starCount = random.Next1To8Inclusive();

        // 返回新的 QuadrantInfo 实例
        return new QuadrantInfo(coordinates, name, klingonCount, starCount, hasStarbase);
    }

    // 增加克林贡数量
    internal void AddKlingon() => KlingonCount += 1;

    // 增加星舰基地
    internal void AddStarbase() => HasStarbase = true;

    // 标记为已知
    internal void MarkAsKnown() => _isKnown = true;

    // 扫描方法
    internal string Scan()
    {
        _isKnown = true;
        return ToString();
    }

    // 覆盖 ToString 方法
    public override string ToString() => _isKnown ? $"{KlingonCount}{(HasStarbase ? 1 : 0)}{StarCount}" : "***";

    // 减少克林贡数量
    internal void RemoveKlingon()
    {
        if (KlingonCount > 0)
        {
            KlingonCount -= 1;
        }
    }

    // 移除星舰基地
    internal void RemoveStarbase() => HasStarbase = false;
}

```