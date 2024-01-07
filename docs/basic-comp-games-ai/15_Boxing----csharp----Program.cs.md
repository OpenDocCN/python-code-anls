# `basic-computer-games\15_Boxing\csharp\Program.cs`

```

// 引入 Boxing 命名空间
using Boxing;
// 使用 Boxing.GameUtils 和 System.Console 的静态成员
using static Boxing.GameUtils;
using static System.Console;

// 打印标题
WriteLine(new string('\t', 33) + "BOXING");
// 打印创意计算公司信息
WriteLine(new string('\t', 15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
// 打印游戏规则
WriteLine("{0}{0}{0}BOXING OLYMPIC STYLE (3 ROUNDS -- 2 OUT OF 3 WINS){0}", Environment.NewLine);

// 创建对手对象
var opponent = new Opponent();
// 设置对手的名字
opponent.SetName("WHAT IS YOUR OPPONENT'S NAME"); // J$
// 创建玩家对象
var player = new Boxer();
// 设置玩家的名字
player.SetName("INPUT YOUR MAN'S NAME"); // L$

// 打印拳击描述
PrintPunchDescription();
// 设置玩家的最佳拳
player.BestPunch = GetPunch("WHAT IS YOUR MANS BEST"); // B
// 设置玩家的脆弱性
player.Vulnerability = GetPunch("WHAT IS HIS VULNERABILITY"); // D
// 对手随机设置拳
opponent.SetRandomPunches();
// 打印对手的优势和脆弱性
WriteLine($"{opponent}'S ADVANTAGE IS {opponent.BestPunch.ToFriendlyString()} AND VULNERABILITY IS SECRET.");

// 进行三轮比赛
for (var i = 1; i <= 3; i ++) // R
{
    // 创建新的比赛回合
    var round = new Round(player, opponent, i);
    // 开始比赛回合
    round.Start();
    // 检查对手是否获胜
    round.CheckOpponentWin();
    // 检查玩家是否获胜
    round.CheckPlayerWin();
    // 如果比赛结束，跳出循环
    if (round.GameEnded) break;
}
// 打印结束语
WriteLine("{0}{0}AND NOW GOODBYE FROM THE OLYMPIC ARENA.{0}", Environment.NewLine);

```