# `basic-computer-games\16_Bug\csharp\Game.cs`

```
# 导入 BugGame.Parts、BugGame.Resources、Games.Common.IO、Games.Common.Randomness 模块
using BugGame.Parts;
using BugGame.Resources;
using Games.Common.IO;
using Games.Common.Randomness;
# 导入 System.StringComparison 模块中的 StringComparison 类
using static System.StringComparison;

# 声明 BugGame 命名空间
namespace BugGame;

# 声明 Game 类
internal class Game
{
    # 声明私有的 IReadWrite 接口类型变量 _io
    private readonly IReadWrite _io;
    # 声明私有的 IRandom 接口类型变量 _random
    private readonly IRandom _random;

    # Game 类的构造函数，接受 IReadWrite 接口类型参数 io 和 IRandom 接口类型参数 random
    public Game(IReadWrite io, IRandom random)
    {
        # 将参数 io 赋值给成员变量 _io
        _io = io;
        # 将参数 random 赋值给成员变量 _random
        _random = random;
    }

    # Play 方法
    public void Play()
    {
        # 输出游戏介绍
        _io.Write(Resource.Streams.Introduction);
        # 如果用户输入不是 "no"，则输出游戏说明
        if (!_io.ReadString("Do you want instructions").Equals("no", InvariantCultureIgnoreCase))
        {
            _io.Write(Resource.Streams.Instructions);
        }

        # 调用 BuildBugs 方法
        BuildBugs();

        # 输出是否再玩一次的提示
        _io.Write(Resource.Streams.PlayAgain);
    }

    # BuildBugs 方法
    private void BuildBugs()
    {
        # 创建 Bug 对象 yourBug 和 myBug
        var yourBug = new Bug();
        var myBug = new Bug();

        # 循环直到用户完成自己的虫子或者程序完成虫子
        while (true)
        {
            # 尝试构建用户的虫子的部件
            var partAdded = TryBuild(yourBug, m => m.You);
            # 线程休眠 500 毫秒
            Thread.Sleep(500);
            # 输出空行
            _io.WriteLine();
            # 尝试构建程序的虫子的部件
            partAdded |= TryBuild(myBug, m => m.I);

            # 如果有部件被添加
            if (partAdded)
            {
                # 如果用户的虫子已经完成，输出提示信息
                if (yourBug.IsComplete) { _io.WriteLine("Your bug is finished."); }
                # 如果程序的虫子已经完成，输出提示信息
                if (myBug.IsComplete) { _io.WriteLine("My bug is finished."); }

                # 如果用户想要查看图片
                if (!_io.ReadString("Do you want the picture").Equals("no", InvariantCultureIgnoreCase))
                {
                    # 输出用户虫子的图片
                    _io.Write(yourBug.ToString("Your", 'A'));
                    _io.WriteLine();
                    _io.WriteLine();
                    _io.WriteLine();
                    _io.WriteLine();
                    # 输出程序虫子的图片
                    _io.Write(myBug.ToString("My", 'F'));
                }
            }

            # 如果用户的虫子或者程序的虫子已经完成，跳出循环
            if (yourBug.IsComplete || myBug.IsComplete) { break; }
        }
    }

    # TryBuild 方法，接受 Bug 类型参数 bug 和 Func<Message, string> 类型参数 messageTransform
    {
        // 生成一个1到6的随机数作为骰子的结果
        var roll = _random.Next(6) + 1;
        // 输出骰子的结果
        _io.WriteLine(messageTransform(Message.Rolled.ForValue(roll)));
    
        // 根据骰子的结果选择相应的身体部位
        IPart part = roll switch
        {
            1 => new Body(),
            2 => new Neck(),
            3 => new Head(),
            4 => new Feeler(),
            5 => new Tail(),
            6 => new Leg(),
            _ => throw new Exception("Unexpected roll value")
        };
        // 输出骰子的结果和对应的身体部位类型
        _io.WriteLine($"{roll}={part.GetType().Name}");
    
        // 尝试将选择的身体部位添加到 bug 对象中，并获取添加结果和消息
        var partAdded = bug.TryAdd(part, out var message);
        // 输出添加身体部位的消息
        _io.WriteLine(messageTransform.Invoke(message));
    
        // 返回身体部位是否成功添加的结果
        return partAdded;
    }
# 闭合了一个代码块
```