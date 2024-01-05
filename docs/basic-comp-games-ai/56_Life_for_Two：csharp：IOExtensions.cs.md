# `d:/src/tocomm/basic-computer-games\56_Life_for_Two\csharp\IOExtensions.cs`

```
internal static class IOExtensions
{
    // 从 IReadWrite 对象中读取玩家的坐标，并返回坐标对象
    internal static Coordinates ReadCoordinates(this IReadWrite io, int player, Board board)
    {
        // 将玩家编号写入 IReadWrite 对象
        io.Write(Formats.Player, player);
        // 调用重载方法，从 IReadWrite 对象中读取坐标并返回
        return io.ReadCoordinates(board);
    }

    // 从 IReadWrite 对象中读取坐标，并返回坐标对象
    internal static Coordinates ReadCoordinates(this IReadWrite io, Board board)
    {
        // 无限循环，直到满足条件才返回坐标对象
        while (true)
        {
            // 在 IReadWrite 对象中写入提示信息
            io.WriteLine("X,Y");
            // 从 IReadWrite 对象中读取两个数字，并存储在 values 变量中
            var values = io.Read2Numbers("&&&&&&\r");
            // 如果可以创建坐标对象并且在棋盘上该位置为空，则返回坐标对象
            if (Coordinates.TryCreate(values, out var coordinates) && board.IsEmptyAt(coordinates))
            {
                return coordinates;
            }
            // 在 IReadWrite 对象中写入非法坐标信息
            io.Write(Streams.IllegalCoords);
        }
这部分代码是一个函数的结束标志，表示函数的定义结束。在Python中，函数的定义使用关键字def开始，然后是函数的内容，最后使用冒号和缩进来表示函数的范围。在这个示例中，这部分代码是函数read_zip的结束标志。
```