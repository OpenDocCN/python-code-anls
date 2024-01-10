# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\Subsystem.cs`

```
// 引入命名空间 Games.Common.IO，SuperStarTrek.Commands，SuperStarTrek.Space
using Games.Common.IO;
using SuperStarTrek.Commands;
using SuperStarTrek.Space;

// 定义一个抽象类 Subsystem
internal abstract class Subsystem
{
    // 声明私有字段 _io，类型为 IReadWrite 接口
    private readonly IReadWrite _io;

    // 构造函数，接受 name、command、io 三个参数
    protected Subsystem(string name, Command command, IReadWrite io)
    {
        // 初始化 Name 属性为 name
        Name = name;
        // 初始化 Command 属性为 command
        Command = command;
        // 初始化 Condition 属性为 0
        Condition = 0;
        // 初始化 _io 字段为 io
        _io = io;
    }

    // 声明 Name 属性，只读
    internal string Name { get; }

    // 声明 Condition 属性，可读可写
    internal float Condition { get; private set; }

    // 声明 IsDamaged 属性，只读，判断 Condition 是否小于 0
    internal bool IsDamaged => Condition < 0;

    // 声明 Command 属性，只读
    internal Command Command { get; }

    // 声明 CanExecuteCommand 方法，返回布尔值，默认为 true
    protected virtual bool CanExecuteCommand() => true;

    // 声明 IsOperational 方法，接受 notOperationalMessage 参数
    protected bool IsOperational(string notOperationalMessage)
    {
        // 如果 IsDamaged 为真
        if (IsDamaged)
        {
            // 输出 notOperationalMessage 替换 {name} 为 Name
            _io.WriteLine(notOperationalMessage.Replace("{name}", Name));
            return false;
        }

        return true;
    }

    // 声明 ExecuteCommand 方法，接受 quadrant 参数，返回 CommandResult 类型
    internal CommandResult ExecuteCommand(Quadrant quadrant)
        // 如果 CanExecuteCommand 返回真，则调用 ExecuteCommandCore 方法，否则返回 CommandResult.Ok
        => CanExecuteCommand() ? ExecuteCommandCore(quadrant) : CommandResult.Ok;

    // 声明 ExecuteCommandCore 方法，接受 quadrant 参数，返回 CommandResult 类型，由子类实现
    protected abstract CommandResult ExecuteCommandCore(Quadrant quadrant);

    // 声明 Repair 方法，虚方法
    internal virtual void Repair()
    {
        // 如果 IsDamaged 为真，则将 Condition 置为 0
        if (IsDamaged)
        {
            Condition = 0;
        }
    }

    // 声明 Repair 方法，接受 repairWorkDone 参数，返回布尔值
    internal virtual bool Repair(float repairWorkDone)
    {
        // 如果 IsDamaged 为真
        if (IsDamaged)
        {
            // 将 Condition 增加 repairWorkDone
            Condition += repairWorkDone;
            // 如果 Condition 大于 -0.1f 且小于 0
            if (Condition > -0.1f && Condition < 0)
            {
                // 将 Condition 置为 -0.1f
                Condition = -0.1f;
            }
        }

        // 返回 Condition 是否大于等于 0
        return !IsDamaged;
    }

    // 声明 TakeDamage 方法，接受 damage 参数，将 Condition 减去 damage
    internal void TakeDamage(float damage) => Condition -= damage;
}
```