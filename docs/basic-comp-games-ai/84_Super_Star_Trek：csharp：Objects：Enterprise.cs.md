# `84_Super_Star_Trek\csharp\Objects\Enterprise.cs`

```
# 导入所需的模块
import System
import Collections.Generic
import Linq
import Games.Common.IO
import Games.Common.Randomness
import SuperStarTrek.Commands
import SuperStarTrek.Resources
import SuperStarTrek.Space
import SuperStarTrek.Systems

# 定义 Enterprise 类
internal class Enterprise:
    # 初始化 Enterprise 类的属性
    def __init__(self, maxEnergy, io, systems, commandExecutors, random, quadrant):
        self._maxEnergy = maxEnergy  # 设置最大能量属性
        self._io = io  # 设置 I/O 属性
        self._systems = systems  # 设置系统列表属性
        self._commandExecutors = commandExecutors  # 设置命令执行器字典属性
        self._random = random  # 设置随机数生成器属性
        self._quadrant = quadrant  # 设置象限属性
    public Enterprise(int maxEnergy, Coordinates sector, IReadWrite io, IRandom random)
    {
        // 初始化飞船所在的区块坐标
        SectorCoordinates = sector;
        // 初始化总能量和最大能量
        TotalEnergy = _maxEnergy = maxEnergy;

        // 初始化系统列表
        _systems = new List<Subsystem>();
        // 初始化命令执行器字典
        _commandExecutors = new Dictionary<Command, Subsystem>();
        // 初始化输入输出接口
        _io = io;
        // 初始化随机数生成器
        _random = random;
    }

    // 获取当前所在的象限
    internal Quadrant Quadrant => _quadrant;

    // 获取当前所在象限的坐标
    internal Coordinates QuadrantCoordinates => _quadrant.Coordinates;

    // 获取当前所在的区块坐标
    internal Coordinates SectorCoordinates { get; private set; }

    // 获取当前飞船的状态
    internal string Condition => GetCondition();

    // 将 Command.COM 对应的命令执行器转换为 LibraryComputer 类型，并赋值给 Computer 属性
    internal LibraryComputer Computer => (LibraryComputer)_commandExecutors[Command.COM];

    // 将 Command.SHE 对应的命令执行器转换为 ShieldControl 类型，并赋值给 ShieldControl 属性
    internal ShieldControl ShieldControl => (ShieldControl)_commandExecutors[Command.SHE];

    // 计算并返回总能量减去护盾能量的值
    internal float Energy => TotalEnergy - ShieldControl.ShieldEnergy;

    // 总能量属性，只能在类内部设置
    internal float TotalEnergy { get; private set; }

    // 返回系统列表中受损的系统数量
    internal int DamagedSystemCount => _systems.Count(s => s.IsDamaged);

    // 返回系统列表
    internal IEnumerable<Subsystem> Systems => _systems;

    // 将 Command.TOR 对应的命令执行器转换为 PhotonTubes 类型，并赋值给 PhotonTubes 属性
    internal PhotonTubes PhotonTubes => (PhotonTubes)_commandExecutors[Command.TOR];

    // 判断企业是否停靠在星舰基地旁边
    internal bool IsDocked => _quadrant.EnterpriseIsNextToStarbase;

    // 判断企业是否搁浅，条件为总能量小于10或者能量小于10并且护盾受损
    internal bool IsStranded => TotalEnergy < 10 || Energy < 10 && ShieldControl.IsDamaged;

    // 向企业添加一个子系统，并返回企业对象
    internal Enterprise Add(Subsystem system)
        _systems.Add(system);  # 将系统添加到系统列表中
        _commandExecutors[system.Command] = system;  # 将系统命令和对应的系统对象存储在命令执行器字典中

        return this;  # 返回当前对象的引用
    }

    internal void StartIn(Quadrant quadrant)  # 在指定的象限开始游戏
    {
        _quadrant = quadrant;  # 将传入的象限赋值给私有变量_quadrant
        quadrant.Display(Strings.StartText);  # 在象限中显示游戏开始的文本
    }

    private string GetCondition() =>  # 获取当前状态的条件
        IsDocked switch  # 根据是否停靠来判断条件
        {
            true => "Docked",  # 如果停靠，则返回 "Docked"
            false when _quadrant.HasKlingons => "*Red*",  # 如果没有停靠且象限中有克林贡人，则返回 "*Red*"
            false when Energy / _maxEnergy < 0.1f => "Yellow",  # 如果没有停靠且能量低于最大能量的10%，则返回 "Yellow"
            false => "Green"  # 如果没有停靠且能量充足，则返回 "Green"
        };
    # 执行给定的命令，并返回结果
    internal CommandResult Execute(Command command)
    {
        # 如果命令是XXX，则返回游戏结束的结果
        if (command == Command.XXX) { return CommandResult.GameOver; }
        
        # 否则执行对应命令的执行器，并返回结果
        return _commandExecutors[command].ExecuteCommand(_quadrant);
    }

    # 对飞船进行加油，使其总能量恢复到最大值
    internal void Refuel() => TotalEnergy = _maxEnergy;

    # 重写 ToString 方法，返回特定的字符串表示
    public override string ToString() => "<*>";

    # 使用能量，减少总能量值
    internal void UseEnergy(float amountUsed)
    {
        TotalEnergy -= amountUsed;
    }

    # 飞船受到攻击，根据攻击力和攻击位置输出信息
    internal CommandResult TakeHit(Coordinates sector, int hitStrength)
    {
        _io.WriteLine($"{hitStrength} unit hit on Enterprise from sector {sector}");
    }
        # 调用ShieldControl对象的AbsorbHit方法，传入hitStrength参数，用于处理护盾吸收攻击
        ShieldControl.AbsorbHit(hitStrength);

        # 如果护盾能量小于等于0，输出销毁信息并返回游戏结束结果
        if (ShieldControl.ShieldEnergy <= 0)
        {
            _io.WriteLine(Strings.Destroyed);
            return CommandResult.GameOver;
        }

        # 输出护盾能量下降到特定单位的信息
        _io.WriteLine($"      <Shields down to {ShieldControl.ShieldEnergy} units>");

        # 如果攻击强度大于等于20，调用TakeDamage方法处理玩家受到的伤害
        if (hitStrength >= 20)
        {
            TakeDamage(hitStrength);
        }

        # 返回正常命令结果
        return CommandResult.Ok;
    }

    # 处理玩家受到的伤害
    private void TakeDamage(float hitStrength)
    {
        // 计算命中盾牌的比率
        var hitShieldRatio = hitStrength / ShieldControl.ShieldEnergy;
        // 如果随机数大于0.6或者命中盾牌的比率小于等于0.02，则返回
        if (_random.NextFloat() > 0.6 || hitShieldRatio <= 0.02f)
        {
            return;
        }

        // 从系统数组中随机选择一个系统
        var system = _systems[_random.Next1To8Inclusive() - 1];
        // 对选中的系统进行伤害处理
        system.TakeDamage(hitShieldRatio + 0.5f * _random.NextFloat());
        // 输出受损系统的报告
        _io.WriteLine($"Damage Control reports, '{system.Name} damaged by the hit.'");
    }

    internal void RepairSystems(float repairWorkDone)
    {
        // 存储已修复的系统名称的列表
        var repairedSystems = new List<string>();

        // 遍历所有受损的系统
        foreach (var system in _systems.Where(s => s.IsDamaged))
        {
            // 如果系统修复成功，则将其名称添加到已修复系统列表中
            if (system.Repair(repairWorkDone))
            {
                repairedSystems.Add(system.Name);
        }

        if (repairedSystems.Any())  # 检查修复系统列表是否有任何元素
        {
            _io.WriteLine("Damage Control report:");  # 输出修复系统报告的标题
            foreach (var systemName in repairedSystems)  # 遍历修复系统列表中的每个系统名
            {
                _io.WriteLine($"        {systemName} repair completed.");  # 输出每个修复系统的修复完成信息
            }
        }
    }

    internal void VaryConditionOfRandomSystem()  # 定义一个内部方法，用于随机改变系统的状态
    {
        if (_random.NextFloat() > 0.2f) { return; }  # 如果随机数大于0.2，则返回，不执行后续代码

        var system = _systems[_random.Next1To8Inclusive() - 1];  # 从系统列表中随机选择一个系统
        _io.Write($"Damage Control report:  {system.Name} ");  # 输出随机选择的系统的名称
        if (_random.NextFloat() >= 0.6)  # 如果随机数大于等于0.6
        {
            // 修复系统，根据随机数生成一个浮点数，乘以3再加1，作为修复的数值
            system.Repair(_random.NextFloat() * 3 + 1);
            // 输出修复状态信息
            _io.WriteLine("state of repair improved");
        }
        else
        {
            // 造成系统损坏，根据随机数生成一个浮点数，乘以5再加1，作为损坏的数值
            system.TakeDamage(_random.NextFloat() * 5 + 1);
            // 输出损坏状态信息
            _io.WriteLine("damaged");
        }
    }

    internal float Move(Course course, float warpFactor, int distance)
    {
        // 在星区内移动或者超出星区移动，返回移动后的象限和扇区
        var (quadrant, sector) = MoveWithinQuadrant(course, distance) ?? MoveBeyondQuadrant(course, distance);

        // 如果移动后的象限与当前象限不同，创建新的象限对象
        if (quadrant != _quadrant.Coordinates)
        {
            _quadrant = new Quadrant(_quadrant.Galaxy[quadrant], this, _random, _quadrant.Galaxy, _io);
        }
        // 设置企业号所在的扇区
        _quadrant.SetEnterpriseSector(sector);
        SectorCoordinates = sector;  # 将当前扇区坐标更新为循环中的扇区坐标

        TotalEnergy -= distance + 10;  # 减去移动距离和额外的10能量消耗
        if (Energy < 0)  # 如果能量小于0
        {
            _io.WriteLine("Shield Control supplies energy to complete the maneuver.");  # 输出信息到控制台
            ShieldControl.ShieldEnergy = Math.Max(0, TotalEnergy);  # 将护盾能量设置为0和总能量中的最大值
        }

        return GetTimeElapsed(quadrant, warpFactor);  # 返回通过GetTimeElapsed函数计算得到的时间
    }

    private (Coordinates, Coordinates)? MoveWithinQuadrant(Course course, int distance)  # 在象限内移动的私有函数，接受课程和距离作为参数
    {
        var currentSector = SectorCoordinates;  # 将当前扇区坐标保存到currentSector变量中
        foreach (var (sector, index) in course.GetSectorsFrom(SectorCoordinates).Select((s, i) => (s, i)))  # 遍历从当前扇区坐标开始的课程中的扇区
        {
            if (distance == 0) { break; }  # 如果距离为0，跳出循环

            if (_quadrant.HasObjectAt(sector))  # 如果象限中有对象在当前扇区
            {
                _io.WriteLine($"Warp engines shut down at sector {currentSector} dues to bad navigation");
                distance = 0;  // 将距离设为0，表示航行距离已经用尽
                break;  // 跳出循环
            }

            currentSector = sector;  // 更新当前扇区为目标扇区
            distance -= 1;  // 距离减1，表示航行一次
        }

        return distance == 0 ? (_quadrant.Coordinates, currentSector) : null;  // 如果距离为0，返回目的地坐标和当前扇区；否则返回null
    }

    private (Coordinates, Coordinates) MoveBeyondQuadrant(Course course, int distance)
    {
        var (complete, quadrant, sector) = course.GetDestination(QuadrantCoordinates, SectorCoordinates, distance);  // 获取目的地信息

        if (!complete)  // 如果未到达目的地
        {
            _io.Write(Strings.PermissionDenied, sector, quadrant);  // 输出无权限访问信息
        }

        return (quadrant, sector);
    }
```

这部分代码是一个方法的结束和返回语句。

```
    private float GetTimeElapsed(Coordinates finalQuadrant, float warpFactor) =>
        finalQuadrant == _quadrant.Coordinates
            ? Math.Min(1, (float)Math.Round(warpFactor, 1, MidpointRounding.ToZero))
            : 1;
```

这部分代码是一个私有方法 `GetTimeElapsed` 的定义，它接受两个参数 `finalQuadrant` 和 `warpFactor`，并返回一个浮点数。根据 `finalQuadrant` 是否等于 `_quadrant.Coordinates`，返回不同的值。
```