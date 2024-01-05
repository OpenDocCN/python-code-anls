# `77_Salvo\csharp\Ships\Ship.cs`

```
    // 声明一个私有的位置列表，用于存储船的位置信息
    private readonly List<Position> _positions = new();

    // 构造函数，接受一个 IReadWrite 接口和一个可选的名称后缀参数
    protected Ship(IReadWrite io, string? nameSuffix = null)
    {
        // 设置船的名称为当前类的名称加上可选的名称后缀
        Name = GetType().Name + nameSuffix;
        // 使用 IReadWrite 接口读取船的位置信息，并转换为列表存储在 _positions 中
        _positions = io.ReadPositions(Name, Size).ToList();
    }

    // 构造函数，接受一个 IRandom 接口和一个可选的名称后缀参数
    protected Ship(IRandom random, string? nameSuffix = null)
    {
        // 设置船的名称为当前类的名称加上可选的名称后缀
        Name = GetType().Name + nameSuffix;

        // 使用 IRandom 接口获取一个随机的船的起始位置和增量
        var (start, delta) = random.GetRandomShipPositionInRange(Size);
        // 循环将船的位置信息添加到 _positions 中
        for (var i = 0; i < Size; i++)
        {
            _positions.Add(start + delta * i);
        }
    }

    internal string Name { get; }  // 声明一个内部的只读属性，表示船的名称
    internal abstract int Shots { get; }  // 声明一个内部的抽象属性，表示船的射击次数
    internal abstract int Size { get; }  // 声明一个内部的抽象属性，表示船的大小
    internal bool IsDamaged => _positions.Count > 0 && _positions.Count < Size;  // 声明一个内部的只读属性，表示船是否受损
    internal bool IsDestroyed => _positions.Count == 0;  // 声明一个内部的只读属性，表示船是否被摧毁

    internal bool IsHit(Position position) => _positions.Remove(position);  // 声明一个内部的方法，表示船是否被击中，并在位置列表中移除被击中的位置

    internal float DistanceTo(Ship other)  // 声明一个内部的方法，表示船与另一艘船之间的距离
        => _positions.SelectMany(a => other._positions.Select(b => a.DistanceTo(b))).Min();  // 使用 LINQ 查询计算船与另一艘船之间的最小距离

    public override string ToString()  // 重写 ToString 方法
        => string.Join(Environment.NewLine, _positions.Select(p => p.ToString()).Prepend(Name));  // 返回船的名称和位置信息的字符串表示形式
}
```