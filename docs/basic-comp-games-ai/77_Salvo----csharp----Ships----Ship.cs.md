# `basic-computer-games\77_Salvo\csharp\Ships\Ship.cs`

```

// 命名空间 Salvo.Ships，表示这个类属于 Salvo.Ships 命名空间
internal abstract class Ship
{
    // 私有字段，用于存储船的位置信息
    private readonly List<Position> _positions = new();

    // 受保护的构造函数，接受 IReadWrite 接口和可选的后缀参数
    protected Ship(IReadWrite io, string? nameSuffix = null)
    {
        // 设置船的名称为类的名称加上可选的后缀
        Name = GetType().Name + nameSuffix;
        // 从 IReadWrite 接口中读取船的位置信息，并转换为列表存储在 _positions 中
        _positions = io.ReadPositions(Name, Size).ToList();
    }

    // 受保护的构造函数，接受 IRandom 接口和可选的后缀参数
    protected Ship(IRandom random, string? nameSuffix = null)
    {
        // 设置船的名称为类的名称加上可选的后缀
        Name = GetType().Name + nameSuffix;

        // 从 IRandom 接口中获取随机的船的位置信息，并存储在 _positions 中
        var (start, delta) = random.GetRandomShipPositionInRange(Size);
        for (var i = 0; i < Size; i++)
        {
            _positions.Add(start + delta * i);
        }
    }

    // 公共属性，表示船的名称
    internal string Name { get; }
    // 抽象属性，表示船的射击次数
    internal abstract int Shots { get; }
    // 抽象属性，表示船的大小
    internal abstract int Size { get; }
    // 表示船是否受损，如果位置信息数量大于 0 且小于大小，则表示受损
    internal bool IsDamaged => _positions.Count > 0 && _positions.Count < Size;
    // 表示船是否被摧毁，如果位置信息数量为 0，则表示被摧毁
    internal bool IsDestroyed => _positions.Count == 0;

    // 判断船是否被击中，如果被击中则移除相应的位置信息
    internal bool IsHit(Position position) => _positions.Remove(position);

    // 计算船与另一艘船的最短距离
    internal float DistanceTo(Ship other)
        => _positions.SelectMany(a => other._positions.Select(b => a.DistanceTo(b))).Min();

    // 重写 ToString 方法，返回船的名称和位置信息
    public override string ToString() 
        => string.Join(Environment.NewLine, _positions.Select(p => p.ToString()).Prepend(Name));
}

```