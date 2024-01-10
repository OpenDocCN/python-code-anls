# `basic-computer-games\72_Queen\csharp\Computer.cs`

```
namespace Queen;

internal class Computer
{
    // 定义一个只读的集合，用于存储随机移动的起始位置
    private static readonly HashSet<Position> _randomiseFrom = new() { 41, 44, 73, 75, 126, 127 };
    // 定义一个只读的集合，用于存储期望移动的目标位置
    private static readonly HashSet<Position> _desirable = new() { 73, 75, 126, 127, 158 };
    // 定义一个私有的随机数生成器接口
    private readonly IRandom _random;

    // 构造函数，接受一个随机数生成器接口实例
    public Computer(IRandom random)
    {
        _random = random;
    }

    // 根据起始位置获取下一步移动的目标位置
    public Position GetMove(Position from)
        => from + (_randomiseFrom.Contains(from) ? _random.NextMove() : FindMove(from));

    // 根据起始位置找到最佳的移动目标位置
    private Move FindMove(Position from)
    {
        // 从最大步数开始逐步减小步数，寻找最佳移动方向
        for (int i = 7; i > 0; i--)
        {
            // 如果向左移动是最佳选择，则返回该移动
            if (IsOptimal(Move.Left, out var move)) { return move; }
            // 如果向下移动是最佳选择，则返回该移动
            if (IsOptimal(Move.Down, out move)) { return move; }
            // 如果向左下移动是最佳选择，则返回该移动
            if (IsOptimal(Move.DownLeft, out move)) { return move; }

            // 判断移动方向是否是最佳选择
            bool IsOptimal(Move direction, out Move move)
            {
                move = direction * i;
                return _desirable.Contains(from + move);
            }
        }

        // 如果没有找到最佳移动方向，则返回随机移动
        return _random.NextMove();
    }
}
```