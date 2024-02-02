# `basic-computer-games\67_One_Check\csharp\Game.cs`

```py
namespace OneCheck;

internal class Game
{
    private readonly IReadWrite _io;  // 声明私有只读字段 _io，用于输入输出操作

    public Game(IReadWrite io)  // 游戏类的构造函数，接受一个 IReadWrite 接口实例作为参数
    {
        _io = io;  // 将传入的 io 参数赋值给 _io 字段
    }

    public void Play()  // 游戏的主要逻辑方法
    {
        _io.Write(Streams.Introduction);  // 使用 _io 对象输出游戏介绍

        do
        {
            var board = new Board();  // 创建一个新的游戏棋盘对象
            do
            {
                _io.WriteLine(board);  // 使用 _io 对象输出游戏棋盘
                _io.WriteLine();  // 使用 _io 对象输出空行
            } while (board.PlayMove(_io));  // 在用户输入有效移动之前一直循环

            _io.WriteLine(board.GetReport());  // 使用 _io 对象输出游戏报告
        } while (_io.ReadYesNo(Prompts.TryAgain) == "yes");  // 当用户输入 yes 时继续游戏

        _io.Write(Streams.Bye);  // 使用 _io 对象输出结束语
    }
}

internal static class IOExtensions
{
    internal static string ReadYesNo(this IReadWrite io, string prompt)  // 为 IReadWrite 接口添加扩展方法 ReadYesNo
    {
        while (true)  // 无限循环，直到用户输入有效的 yes 或 no
        {
            var response = io.ReadString(prompt).ToLower();  // 使用 io 对象读取用户输入并转换为小写

            if (response == "yes" || response == "no") { return response; }  // 如果用户输入为 yes 或 no，则返回该值

            io.Write(Streams.YesOrNo);  // 如果用户输入无效，则使用 io 对象输出提示信息
        }
    }
}
```