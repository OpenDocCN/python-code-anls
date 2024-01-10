# `basic-computer-games\77_Salvo\csharp\Ships\Ship.cs`

```
// 命名空间 Salvo.Ships 下的抽象类 Ship
internal abstract class Ship
{
    // 私有只读字段，存储位置信息
    private readonly List<Position> _positions = new();

    // 有参构造函数，接受 IReadWrite 接口和可选的后缀参数
    protected Ship(IReadWrite io, string? nameSuffix = null)
    {
        // 设置 Name 属性为当前类型的名称加上后缀
        Name = GetType().Name + nameSuffix;
        // 从 io 中读取位置信息，并转换为列表存储在 _positions 中
        _positions = io.ReadPositions(Name, Size).ToList();
    }

    // 有参构造函数，接受 IRandom 接口和可选的后缀参数
    protected Ship(IRandom random, string? nameSuffix = null)
    {
        // 设置 Name 属性为当前类型的名称加上后缀
        Name = GetType().Name + nameSuffix;
        // 从 random 中获取随机的船只位置信息，并存储在 _positions 中
        var (start, delta) = random.GetRandomShipPositionInRange(Size);
        for (var i = 0; i < Size; i++)
        {
            _positions.Add(start + delta * i);
        }
    }

    // 公共属性，获取船只名称
    internal string Name { get; }
    // 抽象属性，获取船只的射击次数
    internal abstract int Shots { get; }
    // 抽象属性，获取船只的大小
    internal abstract int Size { get; }
    // 内部属性，判断船只是否受损
    internal bool IsDamaged => _positions.Count > 0 && _positions.Count < Size;
    // 内部属性，判断船只是否被摧毁
    internal bool IsDestroyed => _positions.Count == 0;

    // 内部方法，判断船只是否被击中
    internal bool IsHit(Position position) => _positions.Remove(position);

    // 内部方法，计算船只与其他船只的最小距离
    internal float DistanceTo(Ship other)
        => _positions.SelectMany(a => other._positions.Select(b => a.DistanceTo(b))).Min();

    // 重写 ToString 方法，返回船只的位置信息
    public override string ToString() 
        => string.Join(Environment.NewLine, _positions.Select(p => p.ToString()).Prepend(Name));
}
```