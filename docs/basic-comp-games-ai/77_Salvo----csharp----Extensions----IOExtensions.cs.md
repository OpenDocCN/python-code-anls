# `basic-computer-games\77_Salvo\csharp\Extensions\IOExtensions.cs`

```

// 命名空间 Games.Common.IO
namespace Games.Common.IO;

// 内部静态类，包含 IO 扩展方法
internal static class IOExtensions
{
    // 从 IReadWrite 接口中读取位置信息
    internal static Position ReadPosition(this IReadWrite io) => Position.Create(io.Read2Numbers(""));

    // 从 IReadWrite 接口中读取有效的位置信息
    internal static Position ReadValidPosition(this IReadWrite io)
    {
        // 循环直到读取到有效的位置信息
        while (true)
        {
            // 尝试从 IReadWrite 接口中读取有效的位置信息
            if (Position.TryCreateValid(io.Read2Numbers(""), out var position)) 
            { 
                return position; 
            }
            // 如果位置信息非法，则写入 Streams.Illegal
            io.Write(Streams.Illegal);
        }
    }

    // 从 IReadWrite 接口中读取位置信息的集合
    internal static IEnumerable<Position> ReadPositions(this IReadWrite io, string shipName, int shipSize)
    {
        // 写入船只名称
        io.WriteLine(shipName);
        // 循环读取指定数量的位置信息
        for (var i = 0; i < shipSize; i++)
        {
             // 返回从 IReadWrite 接口中读取的位置信息
             yield return io.ReadPosition();
        }
    }
}

```