# `d:/src/tocomm/basic-computer-games\26_Chomp\csharp\Game.cs`

```
namespace Chomp;  # 命名空间声明

internal class Game  # 内部类 Game 声明
{
    private readonly IReadWrite _io;  # 声明私有只读字段 _io，类型为 IReadWrite 接口

    public Game(IReadWrite io)  # Game 类的构造函数，接受一个 IReadWrite 类型的参数 io
    {
        _io = io;  # 将传入的 io 参数赋值给 _io 字段
    }

    internal void Play()  # Play 方法声明
    {
        _io.Write(Resource.Streams.Introduction);  # 使用 _io 对象的 Write 方法输出 Resource.Streams.Introduction 的内容
        if (_io.ReadNumber("Do you want the rules (1=Yes, 0=No!)") != 0)  # 使用 _io 对象的 ReadNumber 方法读取用户输入，如果不等于 0
        {
            _io.Write(Resource.Streams.Rules);  # 输出 Resource.Streams.Rules 的内容
        }

        while (true)  # 进入无限循环
        {
            _io.Write(Resource.Streams.HereWeGo);  # 输出游戏开始的提示信息

            var (playerCount, rowCount, columnCount) = _io.ReadParameters();  # 从输入流中读取玩家数量、行数和列数

            var loser = Play(new Cookie(rowCount, columnCount), new PlayerNumber(playerCount));  # 调用Play方法进行游戏，并获取失败的玩家编号

            _io.WriteLine(string.Format(Resource.Formats.YouLose, loser));  # 输出失败玩家的信息

            if (_io.ReadNumber("Again (1=Yes, 0=No!)") != 1) { break; }  # 从输入流中读取是否再次进行游戏的选择，如果不是1则跳出循环
        }
    }

    private PlayerNumber Play(Cookie cookie, PlayerNumber player)
    {
        while (true)
        {
            _io.WriteLine(cookie);  # 输出当前饼干的状态

            var poisoned = Chomp(cookie, player);  # 调用Chomp方法进行游戏，获取是否有毒饼干被吃掉
            if (poisoned) { return player; }  # 如果被毒死了，就返回当前玩家
            player++;  # 玩家数加一
        }
    }

    private bool Chomp(Cookie cookie, PlayerNumber player)  # 定义一个名为Chomp的方法，接受Cookie和PlayerNumber作为参数
    {
        while (true)  # 进入一个无限循环
        {
            _io.WriteLine(string.Format(Resource.Formats.Player, player));  # 输出当前玩家的信息

            var (row, column) = _io.Read2Numbers(Resource.Prompts.Coordinates);  # 从输入中读取两个数字，分别赋值给row和column

            if (cookie.TryChomp((int)row, (int)column, out char chomped))  # 调用cookie对象的TryChomp方法，传入row和column作为参数，并将结果赋值给chomped
            {
                return chomped == 'P';  # 如果chomped等于'P'，返回true，否则返回false
            }
# 创建一个字节流对象，用于写入资源流
_io.Write(Resource.Streams.NoFair);
# 结束当前的命名空间
}
# 结束当前的类定义
}
# 结束当前的命名空间
}
```