# `basic-computer-games\51_Hurkle\csharp\IHurkleView.cs`

```py
# 定义了一个名为hurkle的命名空间
namespace hurkle
{
    # 定义了一个名为IHurkleView的接口，用于表示Hurkle游戏的视图
    internal interface IHurkleView
    {
        # 定义了一个方法，用于获取玩家的猜测，并返回GamePoint对象
        GamePoint GetGuess(GuessViewModel guessViewModel);
        # 定义了一个方法，用于展示玩家获胜的情况
        void ShowVictory(VictoryViewModel victoryViewModel);
        # 定义了一个方法，用于展示玩家猜测的方向错误的情况
        void ShowDirection(FailedGuessViewModel failedGuessViewModel);
        # 定义了一个方法，用于展示玩家失败的情况
        void ShowLoss(LossViewModel lossViewModel);
    }
}
```