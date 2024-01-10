# `basic-computer-games\77_Salvo\csharp\Extensions\IOExtensions.cs`

```
# 命名空间 Games.Common.IO 下的内部静态类 IOExtensions
namespace Games.Common.IO;

internal static class IOExtensions
{
    # 从 IReadWrite 接口中读取两个数字，创建 Position 对象并返回
    internal static Position ReadPosition(this IReadWrite io) => Position.Create(io.Read2Numbers(""));

    # 从 IReadWrite 接口中读取有效的位置，如果位置无效则一直循环直到读取到有效位置
    internal static Position ReadValidPosition(this IReadWrite io)
    {
        while (true)
        {
            # 尝试从 IReadWrite 接口中读取两个数字，创建有效的 Position 对象并返回
            if (Position.TryCreateValid(io.Read2Numbers(""), out var position)) 
            { 
                return position; 
            }
            # 如果位置无效，则向接口写入 "Illegal" 字符串
            io.Write(Streams.Illegal);
        }
    }

    # 从 IReadWrite 接口中读取船只名称和大小，返回一个包含位置的可枚举集合
    internal static IEnumerable<Position> ReadPositions(this IReadWrite io, string shipName, int shipSize)
    {
        # 向接口写入船只名称
        io.WriteLine(shipName);
        # 遍历船只大小次数，每次从接口中读取位置并返回
        for (var i = 0; i < shipSize; i++)
        {
             yield return io.ReadPosition();
        }
    }
}
```