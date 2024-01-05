# `d:/src/tocomm/basic-computer-games\67_One_Check\csharp\Game.cs`

```
namespace OneCheck;  # 命名空间声明

internal class Game  # 声明一个内部类 Game
{
    private readonly IReadWrite _io;  # 声明一个只读的 IReadWrite 接口类型的私有变量 _io

    public Game(IReadWrite io)  # Game 类的构造函数，接受一个 IReadWrite 类型的参数 io
    {
        _io = io;  # 将传入的 io 参数赋值给私有变量 _io
    }

    public void Play()  # 声明一个公共方法 Play
    {
        _io.Write(Streams.Introduction);  # 调用 _io 的 Write 方法，传入 Streams.Introduction 参数
        
        do  # 开始一个 do-while 循环
        {
            var board = new Board();  # 创建一个新的 Board 对象
            do  # 开始一个 do-while 循环
            {
                _io.WriteLine(board);  # 输出当前棋盘状态
                _io.WriteLine();  # 输出空行
            } while (board.PlayMove(_io));  # 当玩家继续下棋时，循环执行上述代码块

            _io.WriteLine(board.GetReport());  # 输出游戏报告
        } while (_io.ReadYesNo(Prompts.TryAgain) == "yes");  # 当玩家选择再试一次时，循环执行上述代码块

        _io.Write(Streams.Bye);  # 输出结束语
    }
}

internal static class IOExtensions  # 定义名为IOExtensions的内部静态类
{
    internal static string ReadYesNo(this IReadWrite io, string prompt)  # 定义名为ReadYesNo的内部静态方法，接受IReadWrite类型的参数io和字符串类型的参数prompt
    {
        while (true)  # 无限循环
        {
            var response = io.ReadString(prompt).ToLower();  # 从输入流中读取字符串并转换为小写

            if (response == "yes" || response == "no") { return response; }  # 如果响应是"yes"或"no"，则返回响应
# 创建一个字节流对象，用于写入数据
io = BytesIO()
# 将Streams.YesOrNo写入字节流对象
io.write(Streams.YesOrNo)
```