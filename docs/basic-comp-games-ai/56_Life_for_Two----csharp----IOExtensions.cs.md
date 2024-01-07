# `basic-computer-games\56_Life_for_Two\csharp\IOExtensions.cs`

```

// 定义一个静态类，包含一些 IO 操作的扩展方法
internal static class IOExtensions
{
    // 从 IReadWrite 接口中读取玩家的坐标信息，并返回坐标对象
    internal static Coordinates ReadCoordinates(this IReadWrite io, int player, Board board)
    {
        // 将玩家信息写入 IO 流
        io.Write(Formats.Player, player);
        // 调用另一个重载的 ReadCoordinates 方法，传入棋盘对象
        return io.ReadCoordinates(board);
    }

    // 从 IReadWrite 接口中读取坐标信息，并返回坐标对象
    internal static Coordinates ReadCoordinates(this IReadWrite io, Board board)
    {
        // 循环读取坐标信息
        while (true)
        {
            // 在 IO 流中写入提示信息
            io.WriteLine("X,Y");
            // 从 IO 流中读取两个数字作为坐标值
            var values = io.Read2Numbers("&&&&&&\r");
            // 如果能够成功创建坐标对象，并且该坐标在棋盘上为空，则返回该坐标
            if (Coordinates.TryCreate(values, out var coordinates) && board.IsEmptyAt(coordinates))
            {
                return coordinates;
            }
            // 如果坐标非法，则在 IO 流中写入非法坐标信息
            io.Write(Streams.IllegalCoords);
        }
    }
}

```