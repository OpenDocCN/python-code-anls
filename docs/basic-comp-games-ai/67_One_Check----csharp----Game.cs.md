# `basic-computer-games\67_One_Check\csharp\Game.cs`

```

namespace OneCheck;

internal class Game
{
    private readonly IReadWrite _io; // 声明一个私有的 IReadWrite 接口类型的变量 _io

    public Game(IReadWrite io) // 构造函数，接受一个 IReadWrite 接口类型的参数 io
    {
        _io = io; // 将参数 io 赋值给私有变量 _io
    }

    public void Play() // Play 方法，用于游戏的进行
    {
        _io.Write(Streams.Introduction); // 使用 _io 对象的 Write 方法输出游戏介绍

        do
        {
            var board = new Board(); // 创建一个新的 Board 对象
            do
            {
                _io.WriteLine(board); // 使用 _io 对象的 WriteLine 方法输出当前棋盘状态
                _io.WriteLine(); // 使用 _io 对象的 WriteLine 方法输出空行
            } while (board.PlayMove(_io)); // 当玩家输入有效移动时继续循环

            _io.WriteLine(board.GetReport()); // 使用 _io 对象的 WriteLine 方法输出游戏报告
        } while (_io.ReadYesNo(Prompts.TryAgain) == "yes"); // 当玩家选择继续时继续循环

        _io.Write(Streams.Bye); // 使用 _io 对象的 Write 方法输出结束语
    }
}

internal static class IOExtensions
{
    internal static string ReadYesNo(this IReadWrite io, string prompt) // 为 IReadWrite 接口类型添加扩展方法 ReadYesNo
    {
        while (true) // 无限循环
        {
            var response = io.ReadString(prompt).ToLower(); // 使用 io 对象的 ReadString 方法获取用户输入并转换为小写

            if (response == "yes" || response == "no") { return response; } // 如果用户输入为 yes 或 no，则返回用户输入

            io.Write(Streams.YesOrNo); // 使用 io 对象的 Write 方法输出提示信息
        }
    }
}

```