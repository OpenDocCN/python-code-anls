# `d:/src/tocomm/basic-computer-games\77_Salvo\csharp\Extensions\IOExtensions.cs`

```
namespace Games.Common.IO;  # 命名空间声明，指定代码所在的命名空间

internal static class IOExtensions  # 声明一个内部静态类 IOExtensions
{
    internal static Position ReadPosition(this IReadWrite io) => Position.Create(io.Read2Numbers(""));  # 声明一个内部静态方法 ReadPosition，接受一个 IReadWrite 参数，返回一个 Position 对象

    internal static Position ReadValidPosition(this IReadWrite io)  # 声明一个内部静态方法 ReadValidPosition，接受一个 IReadWrite 参数，返回一个 Position 对象
    {
        while (true)  # 进入一个无限循环
        {
            if (Position.TryCreateValid(io.Read2Numbers(""), out var position))  # 调用 Position 类的 TryCreateValid 方法，尝试创建一个有效的 Position 对象
            { 
                return position;  # 如果成功创建，则返回该 Position 对象
            }
            io.Write(Streams.Illegal);  # 如果创建失败，则调用 io 对象的 Write 方法，输出 Streams.Illegal
        }
    }

    internal static IEnumerable<Position> ReadPositions(this IReadWrite io, string shipName, int shipSize)  # 声明一个内部静态方法 ReadPositions，接受一个 IReadWrite 参数，一个字符串参数 shipName 和一个整数参数 shipSize，返回一个 Position 对象的集合
    {
        io.WriteLine(shipName);  # 输出船的名称
        for (var i = 0; i < shipSize; i++)  # 使用循环遍历船的大小
        {
             yield return io.ReadPosition();  # 返回每个位置的信息
        }
    }
}
```