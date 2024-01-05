# `d:/src/tocomm/basic-computer-games\25_Chief\csharp\Game.cs`

```
using static Chief.Resources.Resource;  # 导入 Chief.Resources.Resource 命名空间中的所有静态成员

namespace Chief;  # 声明 Chief 命名空间

internal class Game  # 声明 Game 类，限定只能在当前程序集内访问
{
    private readonly IReadWrite _io;  # 声明私有只读字段 _io，类型为 IReadWrite 接口

    public Game(IReadWrite io)  # Game 类的构造函数，接受一个 IReadWrite 类型的参数 io
    {
        _io = io;  # 将参数 io 赋值给 _io 字段
    }

    internal void Play()  # 声明 Play 方法，限定只能在当前程序集内访问
    {
        DoIntroduction();  # 调用 DoIntroduction 方法

        var result = _io.ReadNumber(Prompts.Answer);  # 调用 _io 对象的 ReadNumber 方法，传入 Prompts.Answer 参数，并将返回值赋给 result 变量

        if (_io.ReadYes(Formats.Bet, Math.CalculateOriginal(result)))  # 调用 _io 对象的 ReadYes 方法，传入 Formats.Bet 和 Math.CalculateOriginal(result) 作为参数，并根据返回值执行相应的逻辑
        {
            _io.Write(Streams.Bye);  # 输出“再见”消息
            return;  # 结束函数执行
        }

        var original = _io.ReadNumber(Prompts.Original);  # 从用户输入中读取一个数字，并存储在变量original中

        _io.WriteLine(Math.ShowWorking(original));  # 调用Math类中的ShowWorking方法，将original作为参数传入，并将结果输出

        if (_io.ReadYes(Prompts.Believe))  # 从用户输入中读取是否为“是”的回答
        {
            _io.Write(Streams.Bye);  # 输出“再见”消息
            return;  # 结束函数执行
        }

        _io.Write(Streams.Lightning);  # 输出“闪电”消息
    }

    private void DoIntroduction()  # 定义一个私有函数DoIntroduction
        _io.Write(Streams.Title);  # 将标题内容写入输出流
        if (!_io.ReadYes(Prompts.Ready))  # 如果用户输入的内容不是肯定回答，则执行下面的代码
        {
            _io.Write(Streams.ShutUp);  # 将“ShutUp”内容写入输出流
        }

        _io.Write(Streams.Instructions);  # 将指令内容写入输出流
    }
}
```