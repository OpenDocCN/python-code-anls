# `d:/src/tocomm/basic-computer-games\51_Hurkle\csharp\IHurkleView.cs`

```
# 定义命名空间 hurkle
namespace hurkle
{
    # 定义 IHurkleView 接口
    internal interface IHurkleView
    {
        # 定义 GetGuess 方法，接收 GuessViewModel 对象并返回 GamePoint 对象
        GamePoint GetGuess(GuessViewModel guessViewModel);
        # 定义 ShowVictory 方法，接收 VictoryViewModel 对象并无返回值
        void ShowVictory(VictoryViewModel victoryViewModel);
        # 定义 ShowDirection 方法，接收 FailedGuessViewModel 对象并无返回值
        void ShowDirection(FailedGuessViewModel failedGuessViewModel);
        # 定义 ShowLoss 方法，接收 LossViewModel 对象并无返回值
        void ShowLoss(LossViewModel lossViewModel);
    }
}
```