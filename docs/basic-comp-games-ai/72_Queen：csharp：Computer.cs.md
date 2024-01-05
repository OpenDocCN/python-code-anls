# `d:/src/tocomm/basic-computer-games\72_Queen\csharp\Computer.cs`

```
namespace Queen;  # 命名空间声明

internal class Computer  # 声明一个内部类 Computer
{
    private static readonly HashSet<Position> _randomiseFrom = new() { 41, 44, 73, 75, 126, 127 };  # 声明一个静态只读的 HashSet 集合 _randomiseFrom，并初始化
    private static readonly HashSet<Position> _desirable = new() { 73, 75, 126, 127, 158 };  # 声明一个静态只读的 HashSet 集合 _desirable，并初始化
    private readonly IRandom _random;  # 声明一个只读的 IRandom 接口类型的字段 _random

    public Computer(IRandom random)  # 声明一个构造函数，接受一个 IRandom 类型的参数 random
    {
        _random = random;  # 将参数 random 赋值给字段 _random
    }

    public Position GetMove(Position from)  # 声明一个公共方法 GetMove，接受一个 Position 类型的参数 from，并返回一个 Position 类型的值
        => from + (_randomiseFrom.Contains(from) ? _random.NextMove() : FindMove(from));  # 返回一个表达式，根据条件选择返回 _random.NextMove() 或 FindMove(from) 的结果

    private Move FindMove(Position from)  # 声明一个私有方法 FindMove，接受一个 Position 类型的参数 from，并返回一个 Move 类型的值
    {
        for (int i = 7; i > 0; i--)  # 使用 for 循环，初始化 i 为 7，每次递减 1，循环条件为 i 大于 0
        {
            if (IsOptimal(Move.Left, out var move)) { return move; }  # 如果向左移动是最佳选择，则返回移动方向
            if (IsOptimal(Move.Down, out move)) { return move; }  # 如果向下移动是最佳选择，则返回移动方向
            if (IsOptimal(Move.DownLeft, out move)) { return move; }  # 如果向左下移动是最佳选择，则返回移动方向

            bool IsOptimal(Move direction, out Move move)  # 定义一个函数，判断给定方向是否是最佳选择
            {
                move = direction * i;  # 根据当前步数和给定方向计算移动距离
                return _desirable.Contains(from + move);  # 返回是否目标位置包含移动后的位置
            }
        }

        return _random.NextMove();  # 如果以上条件都不满足，则返回随机移动方向
    }
}
```