# `basic-computer-games\84_Super_Star_Trek\csharp\Objects\Enterprise.cs`

```py
using System;
using System.Collections.Generic;
using System.Linq;
using Games.Common.IO;
using Games.Common.Randomness;
using SuperStarTrek.Commands;
using SuperStarTrek.Resources;
using SuperStarTrek.Space;
using SuperStarTrek.Systems;

namespace SuperStarTrek.Objects
{
    internal class Enterprise
    {
        private readonly int _maxEnergy;  // 保存最大能量值
        private readonly IReadWrite _io;  // 保存输入输出接口
        private readonly List<Subsystem> _systems;  // 保存子系统列表
        private readonly Dictionary<Command, Subsystem> _commandExecutors;  // 保存命令执行器的字典
        private readonly IRandom _random;  // 保存随机数生成器
        private Quadrant _quadrant;  // 保存星区对象

        public Enterprise(int maxEnergy, Coordinates sector, IReadWrite io, IRandom random)
        {
            SectorCoordinates = sector;  // 设置初始扇区坐标
            TotalEnergy = _maxEnergy = maxEnergy;  // 设置总能量值

            _systems = new List<Subsystem>();  // 初始化子系统列表
            _commandExecutors = new Dictionary<Command, Subsystem>();  // 初始化命令执行器字典
            _io = io;  // 保存输入输出接口
            _random = random;  // 保存随机数生成器
        }

        internal Quadrant Quadrant => _quadrant;  // 获取当前星区

        internal Coordinates QuadrantCoordinates => _quadrant.Coordinates;  // 获取当前星区坐标

        internal Coordinates SectorCoordinates { get; private set; }  // 获取和设置当前扇区坐标

        internal string Condition => GetCondition();  // 获取企业状态

        internal LibraryComputer Computer => (LibraryComputer)_commandExecutors[Command.COM];  // 获取图书馆电脑对象

        internal ShieldControl ShieldControl => (ShieldControl)_commandExecutors[Command.SHE];  // 获取护盾控制对象

        internal float Energy => TotalEnergy - ShieldControl.ShieldEnergy;  // 获取当前能量值

        internal float TotalEnergy { get; private set; }  // 获取和设置总能量值

        internal int DamagedSystemCount => _systems.Count(s => s.IsDamaged);  // 获取受损的子系统数量

        internal IEnumerable<Subsystem> Systems => _systems;  // 获取所有子系统列表

        internal PhotonTubes PhotonTubes => (PhotonTubes)_commandExecutors[Command.TOR];  // 获取光子管对象

        internal bool IsDocked => _quadrant.EnterpriseIsNextToStarbase;  // 判断是否停靠在星舰基地旁边

        internal bool IsStranded => TotalEnergy < 10 || Energy < 10 && ShieldControl.IsDamaged;  // 判断是否陷入困境

        internal Enterprise Add(Subsystem system)  // 添加子系统
        {
            _systems.Add(system);  // 将子系统添加到列表中
            _commandExecutors[system.Command] = system;  // 将子系统的命令执行器添加到字典中

            return this;  // 返回企业对象
        }
    }
}
    // 在指定象限启动，并显示开始文本
    internal void StartIn(Quadrant quadrant)
    {
        _quadrant = quadrant;
        quadrant.Display(Strings.StartText);
    }

    // 获取条件状态的字符串表示
    private string GetCondition() =>
        IsDocked switch
        {
            true => "Docked",
            false when _quadrant.HasKlingons => "*Red*",
            false when Energy / _maxEnergy < 0.1f => "Yellow",
            false => "Green"
        };

    // 执行命令并返回结果
    internal CommandResult Execute(Command command)
    {
        if (command == Command.XXX) { return CommandResult.GameOver; }

        return _commandExecutors[command].ExecuteCommand(_quadrant);
    }

    // 加满能量
    internal void Refuel() => TotalEnergy = _maxEnergy;

    // 返回对象的字符串表示
    public override string ToString() => "<*>";

    // 使用能量
    internal void UseEnergy(float amountUsed)
    {
        TotalEnergy -= amountUsed;
    }

    // 处理受到攻击的情况
    internal CommandResult TakeHit(Coordinates sector, int hitStrength)
    {
        _io.WriteLine($"{hitStrength} unit hit on Enterprise from sector {sector}");
        ShieldControl.AbsorbHit(hitStrength);

        if (ShieldControl.ShieldEnergy <= 0)
        {
            _io.WriteLine(Strings.Destroyed);
            return CommandResult.GameOver;
        }

        _io.WriteLine($"      <Shields down to {ShieldControl.ShieldEnergy} units>");

        if (hitStrength >= 20)
        {
            TakeDamage(hitStrength);
        }

        return CommandResult.Ok;
    }

    // 处理受到伤害的情况
    private void TakeDamage(float hitStrength)
    {
        var hitShieldRatio = hitStrength / ShieldControl.ShieldEnergy;
        if (_random.NextFloat() > 0.6 || hitShieldRatio <= 0.02f)
        {
            return;
        }

        var system = _systems[_random.Next1To8Inclusive() - 1];
        system.TakeDamage(hitShieldRatio + 0.5f * _random.NextFloat());
        _io.WriteLine($"Damage Control reports, '{system.Name} damaged by the hit.'");
    }

    // 修复系统
    internal void RepairSystems(float repairWorkDone)
    {
        // 创建一个字符串列表，用于存储修复完成的系统名称
        var repairedSystems = new List<string>();
    
        // 遍历所有受损的系统，进行修复工作
        foreach (var system in _systems.Where(s => s.IsDamaged))
        {
            // 如果系统修复成功，则将系统名称添加到修复完成的系统列表中
            if (system.Repair(repairWorkDone))
            {
                repairedSystems.Add(system.Name);
            }
        }
    
        // 如果有系统修复完成，则输出修复报告
        if (repairedSystems.Any())
        {
            _io.WriteLine("Damage Control report:");
            // 遍历修复完成的系统列表，输出每个系统的修复完成信息
            foreach (var systemName in repairedSystems)
            {
                _io.WriteLine($"        {systemName} repair completed.");
            }
        }
    }
    
    // 随机改变一个系统的状态
    internal void VaryConditionOfRandomSystem()
    {
        // 如果随机数大于0.2，则不进行操作
        if (_random.NextFloat() > 0.2f) { return; }
    
        // 随机选择一个系统
        var system = _systems[_random.Next1To8Inclusive() - 1];
        _io.Write($"Damage Control report:  {system.Name} ");
        // 根据随机数决定系统是修复还是受损
        if (_random.NextFloat() >= 0.6)
        {
            system.Repair(_random.NextFloat() * 3 + 1);
            _io.WriteLine("state of repair improved");
        }
        else
        {
            system.TakeDamage(_random.NextFloat() * 5 + 1);
            _io.WriteLine("damaged");
        }
    }
    
    // 移动飞船
    internal float Move(Course course, float warpFactor, int distance)
    {
        // 在当前象限内移动或者超出当前象限移动
        var (quadrant, sector) = MoveWithinQuadrant(course, distance) ?? MoveBeyondQuadrant(course, distance);
    
        // 如果移动到了新的象限，则更新当前象限
        if (quadrant != _quadrant.Coordinates)
        {
            _quadrant = new Quadrant(_quadrant.Galaxy[quadrant], this, _random, _quadrant.Galaxy, _io);
        }
        // 设置飞船所在的扇区
        _quadrant.SetEnterpriseSector(sector);
        SectorCoordinates = sector;
    
        // 减去移动所消耗的能量
        TotalEnergy -= distance + 10;
        // 如果能量不足，则从护盾控制中获取能量
        if (Energy < 0)
        {
            _io.WriteLine("Shield Control supplies energy to complete the maneuver.");
            ShieldControl.ShieldEnergy = Math.Max(0, TotalEnergy);
        }
    
        // 返回移动所消耗的时间
        return GetTimeElapsed(quadrant, warpFactor);
    }
    
    // 在当前象限内移动
    private (Coordinates, Coordinates)? MoveWithinQuadrant(Course course, int distance)
    {
        // 获取当前的扇区坐标
        var currentSector = SectorCoordinates;
        // 遍历课程中从当前扇区开始的所有扇区
        foreach (var (sector, index) in course.GetSectorsFrom(SectorCoordinates).Select((s, i) => (s, i)))
        {
            // 如果距离为0，则跳出循环
            if (distance == 0) { break; }

            // 如果在当前扇区有物体存在
            if (_quadrant.HasObjectAt(sector))
            {
                // 输出错误信息并将距离设为0，然后跳出循环
                _io.WriteLine($"Warp engines shut down at sector {currentSector} dues to bad navigation");
                distance = 0;
                break;
            }

            // 更新当前扇区
            currentSector = sector;
            // 距离减1
            distance -= 1;
        }

        // 如果距离为0，则返回当前象限坐标和当前扇区坐标，否则返回null
        return distance == 0 ? (_quadrant.Coordinates, currentSector) : null;
    }

    // 移动超出象限
    private (Coordinates, Coordinates) MoveBeyondQuadrant(Course course, int distance)
    {
        // 获取目的地的象限和扇区
        var (complete, quadrant, sector) = course.GetDestination(QuadrantCoordinates, SectorCoordinates, distance);

        // 如果未完成移动，则输出错误信息
        if (!complete)
        {
            _io.Write(Strings.PermissionDenied, sector, quadrant);
        }

        // 返回目的地的象限和扇区
        return (quadrant, sector);
    }

    // 获取经过的时间
    private float GetTimeElapsed(Coordinates finalQuadrant, float warpFactor) =>
        // 如果最终象限与当前象限相同，则返回最小值为1和船的最大速度
        finalQuadrant == _quadrant.Coordinates
            ? Math.Min(1, (float)Math.Round(warpFactor, 1, MidpointRounding.ToZero))
            // 否则返回1
            : 1;
# 闭合前面的函数定义
```