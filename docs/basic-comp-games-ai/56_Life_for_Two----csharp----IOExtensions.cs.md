# `basic-computer-games\56_Life_for_Two\csharp\IOExtensions.cs`

```py
# 定义一个静态类，包含一些 IO 操作的扩展方法
internal static class IOExtensions
{
    # 从 IReadWrite 接口中读取玩家的坐标信息，并返回坐标对象
    internal static Coordinates ReadCoordinates(this IReadWrite io, int player, Board board)
    {
        # 将玩家信息写入 IO 流
        io.Write(Formats.Player, player);
        # 调用另一个重载的 ReadCoordinates 方法，传入棋盘对象
        return io.ReadCoordinates(board);
    }

    # 从 IReadWrite 接口中读取坐标信息，并返回坐标对象
    internal static Coordinates ReadCoordinates(this IReadWrite io, Board board)
    {
        # 无限循环，直到读取到合法的坐标信息
        while (true)
        {
            # 在 IO 流中写入提示信息
            io.WriteLine("X,Y");
            # 从 IO 流中读取两个数字，使用指定的格式进行匹配
            var values = io.Read2Numbers("&&&&&&\r");
            # 尝试根据读取到的值创建坐标对象，并检查是否在棋盘上为空
            if (Coordinates.TryCreate(values, out var coordinates) && board.IsEmptyAt(coordinates))
            {
                # 如果坐标合法且在棋盘上为空，则返回坐标对象
                return coordinates;
            }
            # 如果坐标非法或在棋盘上不为空，则在 IO 流中写入非法坐标信息
            io.Write(Streams.IllegalCoords);
        }
    }
}
```