# `basic-computer-games\07_Basketball\csharp\Scoreboard.cs`

```
// 使用 Basketball.Resources 命名空间
// 使用 Games.Common.IO 命名空间
namespace Basketball
{
    // 定义 Scoreboard 类
    internal class Scoreboard
    {
        // 用于存储队伍得分的字典
        private readonly Dictionary<Team, uint> _scores;
        // 用于输入输出的接口
        private readonly IReadWrite _io;

        // Scoreboard 类的构造函数
        public Scoreboard(Team home, Team visitors, IReadWrite io)
        {
            // 初始化队伍得分字典
            _scores = new() { [home] = 0, [visitors] = 0 };
            // 设置主队和客队
            Home = home;
            Visitors = visitors;
            // 暂时设置进攻方为主队，直到第一次跳球
            Offense = home;
            // 设置输入输出接口
            _io = io;
        }

        // 判断主客队得分是否相等的属性
        public bool ScoresAreEqual => _scores[Home] == _scores[Visitors];
        // 获取或设置进攻方队伍
        public Team Offense { get; set; }
        // 获取主队
        public Team Home { get; }
        // 获取客队
        public Team Visitors { get; }

        // 添加篮球得分的方法
        public void AddBasket(string message) => AddScore(2, message);

        // 添加罚球得分的方法
        public void AddFreeThrows(uint count, string message) => AddScore(count, message);

        // 添加得分的私有方法
        private void AddScore(uint score, string message)
        {
            // 如果进攻方为空，则抛出异常
            if (Offense is null) { throw new InvalidOperationException("Offense must be set before adding to score."); }

            // 输出消息
            _io.WriteLine(message);
            // 更新队伍得分
            _scores[Offense] += score;
            // 转换进攻方
            Turnover();
            // 显示得分
            Display();
        }

        // 转换进攻方的方法
        public void Turnover(string? message = null)
        {
            // 如果有消息，则输出消息
            if (message is not null) { _io.WriteLine(message); }

            // 切换进攻方
            Offense = Offense == Home ? Visitors : Home;
        }

        // 显示得分的方法
        public void Display(string? format = null) =>
            // 输出格式化后的得分信息
            _io.WriteLine(format ?? Resource.Formats.Score, Home, _scores[Home], Visitors, _scores[Visitors]);
    }
}
```