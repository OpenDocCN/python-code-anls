# `d:/src/tocomm/basic-computer-games\07_Basketball\csharp\Scoreboard.cs`

```
using Basketball.Resources;  // 导入篮球资源
using Games.Common.IO;  // 导入通用输入输出库

namespace Basketball;  // 命名空间篮球

internal class Scoreboard  // 内部类记分牌
{
    private readonly Dictionary<Team, uint> _scores;  // 记录队伍得分的字典
    private readonly IReadWrite _io;  // 读写接口

    public Scoreboard(Team home, Team visitors, IReadWrite io)  // 记分牌构造函数，传入主队、客队和读写接口
    {
        _scores = new() { [home] = 0, [visitors] = 0 };  // 初始化队伍得分字典
        Home = home;  // 设置主队
        Visitors = visitors;  // 设置客队
        Offense = home;  // 临时值，直到第一次跳球确定
        _io = io;  // 设置读写接口
    }

    public bool ScoresAreEqual => _scores[Home] == _scores[Visitors];  // 判断主客队得分是否相等
    public Team Offense { get; set; }  # 定义一个公共属性 Offense，用于设置进攻方球队
    public Team Home { get; }  # 定义一个公共属性 Home，用于获取主场球队
    public Team Visitors { get; }  # 定义一个公共属性 Visitors，用于获取客场球队

    public void AddBasket(string message) => AddScore(2, message);  # 定义一个方法 AddBasket，用于添加篮球得分，调用 AddScore 方法

    public void AddFreeThrows(uint count, string message) => AddScore(count, message);  # 定义一个方法 AddFreeThrows，用于添加罚球得分，调用 AddScore 方法

    private void AddScore(uint score, string message)  # 定义一个私有方法 AddScore，用于添加得分
    {
        if (Offense is null) { throw new InvalidOperationException("Offense must be set before adding to score."); }  # 如果进攻方球队为空，则抛出异常

        _io.WriteLine(message);  # 输出消息
        _scores[Offense] += score;  # 更新进攻方球队的得分
        Turnover();  # 调用 Turnover 方法
        Display();  # 调用 Display 方法
    }

    public void Turnover(string? message = null)  # 定义一个方法 Turnover，用于发生失误
        if (message is not null) { _io.WriteLine(message); }  // 如果消息不为空，则将消息写入输出流

        Offense = Offense == Home ? Visitors : Home;  // 如果进攻方是主队，则将进攻方设置为客队，否则设置为主队
    }

    public void Display(string? format = null) =>
        _io.WriteLine(format ?? Resource.Formats.Score, Home, _scores[Home], Visitors, _scores[Visitors]);  // 显示比分，如果格式不为空则使用指定格式，否则使用默认格式
```