# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\LibraryComputer.cs`

```
// 引入所需的命名空间
using Games.Common.IO;
using SuperStarTrek.Commands;
using SuperStarTrek.Space;
using SuperStarTrek.Systems.ComputerFunctions;

// 定义名为 LibraryComputer 的内部类，继承自 Subsystem
internal class LibraryComputer : Subsystem
{
    // 声明私有字段 _io 和 _functions
    private readonly IReadWrite _io;
    private readonly ComputerFunction[] _functions;

    // 定义构造函数，接受 IReadWrite 类型的参数 io 和可变数量的 ComputerFunction 类型的参数 functions
    internal LibraryComputer(IReadWrite io, params ComputerFunction[] functions)
        // 调用基类 Subsystem 的构造函数，传入"Library-Computer"、Command.COM 和 io 作为参数
        : base("Library-Computer", Command.COM, io)
    {
        // 将参数 io 赋值给字段 _io
        _io = io;
        // 将参数 functions 赋值给字段 _functions
        _functions = functions;
    }

    // 重写基类的 CanExecuteCommand 方法，判断计算机是否可执行命令
    protected override bool CanExecuteCommand() => IsOperational("Computer disabled");

    // 重写基类的 ExecuteCommandCore 方法，执行命令的核心逻辑
    protected override CommandResult ExecuteCommandCore(Quadrant quadrant)
    {
        // 获取要执行的功能的索引
        var index = GetFunctionIndex();
        // 写入空行
        _io.WriteLine();

        // 执行对应索引的功能
        _functions[index].Execute(quadrant);

        // 返回命令执行结果为成功
        return CommandResult.Ok;
    }

    // 定义私有方法 GetFunctionIndex，用于获取要执行的功能的索引
    private int GetFunctionIndex()
    {
        // 循环直到获取有效的功能索引
        while (true)
        {
            // 从输入中读取索引值
            var index = (int)_io.ReadNumber("Computer active and waiting command");
            // 如果索引值在 0 到 5 之间，则返回该索引值
            if (index >= 0 && index <= 5) { return index; }

            // 否则，循环遍历所有功能，输出功能索引和描述
            for (int i = 0; i < _functions.Length; i++)
            {
                _io.WriteLine($"   {i} = {_functions[i].Description}");
            }
        }
    }
}
```