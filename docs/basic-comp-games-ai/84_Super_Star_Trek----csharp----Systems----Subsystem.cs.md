# `basic-computer-games\84_Super_Star_Trek\csharp\Systems\Subsystem.cs`

```

// 引入所需的命名空间
using Games.Common.IO;
using SuperStarTrek.Commands;
using SuperStarTrek.Space;

// 定义一个抽象的子系统类
namespace SuperStarTrek.Systems;
internal abstract class Subsystem
{
    // 声明私有的 IReadWrite 接口类型的变量 _io
    private readonly IReadWrite _io;

    // 子系统的构造函数，初始化名称、命令和 IReadWrite 接口
    protected Subsystem(string name, Command command, IReadWrite io)
    {
        Name = name;
        Command = command;
        Condition = 0;
        _io = io;
    }

    // 子系统的名称属性
    internal string Name { get; }

    // 子系统的状态属性
    internal float Condition { get; private set; }

    // 子系统是否受损的属性
    internal bool IsDamaged => Condition < 0;

    // 子系统的命令属性
    internal Command Command { get; }

    // 判断是否可以执行命令的方法
    protected virtual bool CanExecuteCommand() => true;

    // 判断子系统是否正常运行的方法
    protected bool IsOperational(string notOperationalMessage)
    {
        if (IsDamaged)
        {
            _io.WriteLine(notOperationalMessage.Replace("{name}", Name));
            return false;
        }
        return true;
    }

    // 执行命令的方法
    internal CommandResult ExecuteCommand(Quadrant quadrant)
        => CanExecuteCommand() ? ExecuteCommandCore(quadrant) : CommandResult.Ok;

    // 执行命令核心逻辑的抽象方法
    protected abstract CommandResult ExecuteCommandCore(Quadrant quadrant);

    // 修复子系统的方法
    internal virtual void Repair()
    {
        if (IsDamaged)
        {
            Condition = 0;
        }
    }

    // 进行修复工作的方法
    internal virtual bool Repair(float repairWorkDone)
    {
        if (IsDamaged)
        {
            Condition += repairWorkDone;
            if (Condition > -0.1f && Condition < 0)
            {
                Condition = -0.1f;
            }
        }
        return !IsDamaged;
    }

    // 受到伤害的方法
    internal void TakeDamage(float damage) => Condition -= damage;
}

```