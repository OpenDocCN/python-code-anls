# `basic-computer-games\72_Queen\csharp\Computer.cs`

```

// 命名空间 Queen，表示该类在 Queen 命名空间下
namespace Queen;

// 内部类 Computer，表示该类只能在当前程序集内部访问
internal class Computer
{
    // 静态只读字段 _randomiseFrom，存储一组初始值为 41, 44, 73, 75, 126, 127 的 Position 对象
    private static readonly HashSet<Position> _randomiseFrom = new() { 41, 44, 73, 75, 126, 127 };
    // 静态只读字段 _desirable，存储一组初始值为 73, 75, 126, 127, 158 的 Position 对象
    private static readonly HashSet<Position> _desirable = new() { 73, 75, 126, 127, 158 };
    // 只读字段 _random，存储 IRandom 接口的实例
    private readonly IRandom _random;

    // 构造函数，接受一个 IRandom 接口的实例作为参数
    public Computer(IRandom random)
    {
        _random = random;
    }

    // 公共方法 GetMove，接受一个 Position 对象作为参数，返回一个 Position 对象
    public Position GetMove(Position from)
        => from + (_randomiseFrom.Contains(from) ? _random.NextMove() : FindMove(from));

    // 私有方法 FindMove，接受一个 Position 对象作为参数，返回一个 Move 对象
    private Move FindMove(Position from)
    {
        // 循环遍历 7 次
        for (int i = 7; i > 0; i--)
        {
            // 如果 Move.Left 是最佳移动方向，则返回该移动
            if (IsOptimal(Move.Left, out var move)) { return move; }
            // 如果 Move.Down 是最佳移动方向，则返回该移动
            if (IsOptimal(Move.Down, out move)) { return move; }
            // 如果 Move.DownLeft 是最佳移动方向，则返回该移动
            if (IsOptimal(Move.DownLeft, out move)) { return move; }

            // 判断是否是最佳移动方向
            bool IsOptimal(Move direction, out Move move)
            {
                move = direction * i;
                return _desirable.Contains(from + move);
            }
        }

        // 返回随机移动
        return _random.NextMove();
    }
}

```