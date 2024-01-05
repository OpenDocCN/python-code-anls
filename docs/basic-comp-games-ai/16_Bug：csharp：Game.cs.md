# `d:/src/tocomm/basic-computer-games\16_Bug\csharp\Game.cs`

```
using BugGame.Parts;  # 导入 BugGame.Parts 模块
using BugGame.Resources;  # 导入 BugGame.Resources 模块
using Games.Common.IO;  # 导入 Games.Common.IO 模块
using Games.Common.Randomness;  # 导入 Games.Common.Randomness 模块
using static System.StringComparison;  # 导入 System.StringComparison 模块的所有静态成员
namespace BugGame;  # 声明 BugGame 命名空间

internal class Game  # 声明一个内部类 Game
{
    private readonly IReadWrite _io;  # 声明一个只读字段 _io，类型为 IReadWrite 接口
    private readonly IRandom _random;  # 声明一个只读字段 _random，类型为 IRandom 接口

    public Game(IReadWrite io, IRandom random)  # Game 类的构造函数，接受 IReadWrite 和 IRandom 接口类型的参数
    {
        _io = io;  # 将传入的 io 参数赋值给 _io 字段
        _random = random;  # 将传入的 random 参数赋值给 _random 字段
    }

    public void Play()  # 声明一个公共方法 Play
    {
        _io.Write(Resource.Streams.Introduction);  # 输出介绍信息到控制台
        if (!_io.ReadString("Do you want instructions").Equals("no", InvariantCultureIgnoreCase))  # 从控制台读取用户输入，如果不是"no"则执行下面的代码
        {
            _io.Write(Resource.Streams.Instructions);  # 输出游戏说明到控制台
        }

        BuildBugs();  # 调用BuildBugs方法

        _io.Write(Resource.Streams.PlayAgain);  # 输出再玩一次的提示到控制台
    }

    private void BuildBugs()  # 定义BuildBugs方法
    {
        var yourBug = new Bug();  # 创建一个Bug对象并赋值给yourBug变量
        var myBug = new Bug();  # 创建一个Bug对象并赋值给myBug变量

        while (true)  # 进入无限循环
        {
            var partAdded = TryBuild(yourBug, m => m.You);  # 调用TryBuild方法，传入yourBug和一个lambda表达式，将返回值赋给partAdded变量
            Thread.Sleep(500);  # 线程休眠500毫秒
            _io.WriteLine(); // 输出空行

            // 尝试构建 myBug 对象的属性 m.I
            partAdded |= TryBuild(myBug, m => m.I);

            // 如果有部分属性被添加
            if (partAdded)
            {
                // 如果 yourBug 对象已经完成，则输出 "Your bug is finished."
                if (yourBug.IsComplete) { _io.WriteLine("Your bug is finished."); }
                // 如果 myBug 对象已经完成，则输出 "My bug is finished."
                if (myBug.IsComplete) { _io.WriteLine("My bug is finished."); }

                // 如果用户输入的字符串不是 "no"（不区分大小写），则执行以下代码块
                if (!_io.ReadString("Do you want the picture").Equals("no", InvariantCultureIgnoreCase))
                {
                    // 输出 yourBug 对象的字符串表示形式，带有 "Your" 前缀和 'A' 后缀
                    _io.Write(yourBug.ToString("Your", 'A'));
                    _io.WriteLine(); // 输出空行
                    _io.WriteLine(); // 输出空行
                    _io.WriteLine(); // 输出空行
                    _io.WriteLine(); // 输出空行
                    // 输出 myBug 对象的字符串表示形式，带有 "My" 前缀和 'F' 后缀
                    _io.Write(myBug.ToString("My", 'F'));
                }
            }

            // 如果 yourBug 对象已经完成或者 myBug 对象已经完成，则跳出循环
            if (yourBug.IsComplete || myBug.IsComplete) { break; }
    private bool TryBuild(Bug bug, Func<Message, string> messageTransform)
    {
        // 生成一个1到6的随机数
        var roll = _random.Next(6) + 1;
        // 输出掷骰子的信息
        _io.WriteLine(messageTransform(Message.Rolled.ForValue(roll)));

        // 根据掷骰子的结果选择不同的身体部位
        IPart part = roll switch
        {
            1 => new Body(),
            2 => new Neck(),
            3 => new Head(),
            4 => new Feeler(),
            5 => new Tail(),
            6 => new Leg(),
            _ => throw new Exception("Unexpected roll value")  // 如果掷骰子的结果不在1到6之间，抛出异常
        };
        // 输出掷骰子的结果和对应的身体部位名称
        _io.WriteLine($"{roll}={part.GetType().Name}");
        # 尝试将部件添加到 bug 中，并返回是否添加成功的布尔值
        var partAdded = bug.TryAdd(part, out var message);
        # 调用 messageTransform 函数对消息进行转换，并将转换后的消息输出到 _io
        _io.WriteLine(messageTransform.Invoke(message));

        # 返回部件是否成功添加的布尔值
        return partAdded;
    }
}
```