# `basic-computer-games\15_Boxing\csharp\Program.cs`

```
// 输出标题
WriteLine(new string('\t', 33) + "BOXING");
// 输出创意计算的信息
WriteLine(new string('\t', 15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
// 输出游戏规则
WriteLine("{0}{0}{0}BOXING OLYMPIC STYLE (3 ROUNDS -- 2 OUT OF 3 WINS){0}", Environment.NewLine);

// 创建对手对象
var opponent = new Opponent();
// 设置对手的名字
opponent.SetName("WHAT IS YOUR OPPONENT'S NAME"); // J$
// 创建玩家对象
var player = new Boxer();
// 设置玩家的名字
player.SetName("INPUT YOUR MAN'S NAME"); // L$

// 输出拳击描述
PrintPunchDescription();
// 设置玩家的最佳拳
player.BestPunch = GetPunch("WHAT IS YOUR MANS BEST"); // B
// 设置玩家的脆弱性
player.Vulnerability = GetPunch("WHAT IS HIS VULNERABILITY"); // D
// 设置对手的随机拳
opponent.SetRandomPunches();
// 输出对手的优势和脆弱性
WriteLine($"{opponent}'S ADVANTAGE IS {opponent.BestPunch.ToFriendlyString()} AND VULNERABILITY IS SECRET.");

// 进行3轮比赛
for (var i = 1; i <= 3; i ++) // R
{
    // 创建新的回合对象
    var round = new Round(player, opponent, i);
    // 开始回合
    round.Start();
    // 检查对手是否获胜
    round.CheckOpponentWin();
    // 检查玩家是否获胜
    round.CheckPlayerWin();
    // 如果比赛结束则跳出循环
    if (round.GameEnded) break;
}
// 输出结束语
WriteLine("{0}{0}AND NOW GOODBYE FROM THE OLYMPIC ARENA.{0}", Environment.NewLine);
```