# `basic-computer-games\12_Bombs_Away\csharp\BombsAwayGame\EnemyArtillery.cs`

```

// 声明名为 EnemyArtillery 的内部记录类，表示敌方炮火
/// <summary>
/// 表示敌方炮火。
/// </summary>
/// <param name="Name">炮火类型的名称。</param>
/// <param name="Accuracy">炮火的准确度。这是原始BASIC中的变量`T`。</param>
internal record class EnemyArtillery(string Name, int Accuracy);

```