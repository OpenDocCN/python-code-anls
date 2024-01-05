# `51_Hurkle\csharp\LossViewModel.cs`

```
namespace hurkle
{
    internal class LossViewModel
    {
        // 定义一个属性，表示最大猜测次数
        public int MaxGuesses { get; init; }
        // 定义一个属性，表示隐藏的 Hurkle 的位置
        public GamePoint HurkleLocation { get; init; }
    }
}
```