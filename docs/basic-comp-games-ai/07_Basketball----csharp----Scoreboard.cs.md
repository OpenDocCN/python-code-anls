# `basic-computer-games\07_Basketball\csharp\Scoreboard.cs`

```

// 引入所需的资源和命名空间
using Basketball.Resources;
using Games.Common.IO;

// 定义篮球比赛记分板类
namespace Basketball
{
    // 内部类，用于记录比分
    internal class Scoreboard
    {
        // 记录每支球队的得分
        private readonly Dictionary<Team, uint> _scores;
        // 用于输入输出操作的接口
        private readonly IReadWrite _io;

        // 构造函数，初始化比分板
        public Scoreboard(Team home, Team visitors, IReadWrite io)
        {
            // 初始化得分为0
            _scores = new() { [home] = 0, [visitors] = 0 };
            Home = home;
            Visitors = visitors;
            Offense = home;  // 暂时的值，直到第一次跳球
            _io = io;
        }

        // 判断两支球队的得分是否相等
        public bool ScoresAreEqual => _scores[Home] == _scores[Visitors];
        // 获取当前进攻的球队
        public Team Offense { get; set; }
        // 获取主队
        public Team Home { get; }
        // 获取客队
        public Team Visitors { get; }

        // 添加篮球得分
        public void AddBasket(string message) => AddScore(2, message);

        // 添加罚球得分
        public void AddFreeThrows(uint count, string message) => AddScore(count, message);

        // 添加得分的私有方法
        private void AddScore(uint score, string message)
        {
            // 如果进攻球队为空，则抛出异常
            if (Offense is null) { throw new InvalidOperationException("Offense must be set before adding to score."); }

            // 输出消息
            _io.WriteLine(message);
            // 更新进攻球队的得分
            _scores[Offense] += score;
            // 换攻
            Turnover();
            // 显示得分
            Display();
        }

        // 换攻
        public void Turnover(string? message = null)
        {
            // 如果有消息，则输出消息
            if (message is not null) { _io.WriteLine(message); }

            // 切换进攻球队
            Offense = Offense == Home ? Visitors : Home;
        }

        // 显示得分
        public void Display(string? format = null) =>
            _io.WriteLine(format ?? Resource.Formats.Score, Home, _scores[Home], Visitors, _scores[Visitors]);
    }
}

```