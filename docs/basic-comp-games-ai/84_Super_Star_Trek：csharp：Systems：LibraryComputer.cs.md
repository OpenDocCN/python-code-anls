# `d:/src/tocomm/basic-computer-games\84_Super_Star_Trek\csharp\Systems\LibraryComputer.cs`

```
using Games.Common.IO;  // 导入 Games.Common.IO 包，用于输入输出操作
using SuperStarTrek.Commands;  // 导入 SuperStarTrek.Commands 包，用于命令操作
using SuperStarTrek.Space;  // 导入 SuperStarTrek.Space 包，用于空间操作
using SuperStarTrek.Systems.ComputerFunctions;  // 导入 SuperStarTrek.Systems.ComputerFunctions 包，用于计算机功能

namespace SuperStarTrek.Systems;  // 声明 SuperStarTrek.Systems 命名空间

internal class LibraryComputer : Subsystem  // 声明 LibraryComputer 类，继承自 Subsystem 类
{
    private readonly IReadWrite _io;  // 声明私有成员变量 _io，类型为 IReadWrite 接口
    private readonly ComputerFunction[] _functions;  // 声明私有成员变量 _functions，类型为 ComputerFunction 数组

    internal LibraryComputer(IReadWrite io, params ComputerFunction[] functions)  // 声明 LibraryComputer 类的构造函数，接受 IReadWrite 类型的参数 io 和可变长度的 ComputerFunction 类型参数 functions
        : base("Library-Computer", Command.COM, io)  // 调用父类 Subsystem 的构造函数，传入字符串 "Library-Computer"、Command.COM 和 io
    {
        _io = io;  // 将参数 io 赋值给成员变量 _io
        _functions = functions;  // 将参数 functions 赋值给成员变量 _functions
    }

    protected override bool CanExecuteCommand() => IsOperational("Computer disabled");  // 重写父类的 CanExecuteCommand 方法，判断计算机是否可执行命令，返回布尔值
}
    protected override CommandResult ExecuteCommandCore(Quadrant quadrant)
    {
        var index = GetFunctionIndex(); // 调用 GetFunctionIndex 方法获取函数索引
        _io.WriteLine(); // 在控制台输出空行

        _functions[index].Execute(quadrant); // 调用 _functions 数组中索引为 index 的函数的 Execute 方法，传入 quadrant 参数

        return CommandResult.Ok; // 返回 CommandResult.Ok
    }

    private int GetFunctionIndex()
    {
        while (true)
        {
            var index = (int)_io.ReadNumber("Computer active and waiting command"); // 从控制台读取一个整数作为函数索引
            if (index >= 0 && index <= 5) { return index; } // 如果索引在合法范围内，返回该索引值

            for (int i = 0; i < _functions.Length; i++) // 遍历 _functions 数组
            {
# 遍历 _functions 列表，将每个元素的索引和描述信息输出到控制台
for i in range(len(_functions)):
    # 输出索引和描述信息
    _io.WriteLine($"   {i} = {_functions[i].Description}");
# 结束 for 循环
```