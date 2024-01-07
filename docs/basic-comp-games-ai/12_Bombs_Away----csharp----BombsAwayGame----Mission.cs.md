# `basic-computer-games\12_Bombs_Away\csharp\BombsAwayGame\Mission.cs`

```

// 命名空间声明，表示该类所属的命名空间
namespace BombsAwayGame;

/// <summary>
/// 表示可以由 MissionSide 飞行的任务。
/// </summary>
/// <param name="Name">任务名称。</param>
/// <param name="Description">任务描述。</param>
// 内部记录类，用于表示任务
internal record class Mission(string Name, string Description);

```