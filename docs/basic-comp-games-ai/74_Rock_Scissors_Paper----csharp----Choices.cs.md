# `basic-computer-games\74_Rock_Scissors_Paper\csharp\Choices.cs`

```
using System;

namespace RockScissorsPaper
{
    public class Choices
    {
        // 创建 Rock 对象，表示石头
        public static readonly Choice Rock = new Choice("3", "Rock");
        // 创建 Scissors 对象，表示剪刀
        public static readonly Choice Scissors = new Choice("2", "Scissors");
        // 创建 Paper 对象，表示布
        public static readonly Choice Paper = new Choice("1", "Paper");

        // 创建私有静态 Choice 数组 _allChoices
        private static readonly Choice[] _allChoices;
        // 创建私有静态 Random 对象 _random
        private static readonly Random _random = new Random();

        // 静态构造函数，初始化对象之间的关系和所有选择
        static Choices()
        {
            // 设置石头可以战胜剪刀
            Rock.CanBeat = Scissors;
            // 设置剪刀可以战胜布
            Scissors.CanBeat = Paper;
            // 设置布可以战胜石头
            Paper.CanBeat = Rock;

            // 初始化所有选择的数组
            _allChoices = new[] { Rock, Scissors, Paper };
        }

        // 获取随机选择
        public static Choice GetRandom()
        {
            return _allChoices[_random.Next(_allChoices.GetLength(0))];
        }

        // 根据选择器获取选择，返回是否成功
        public static bool TryGetBySelector(string selector, out Choice choice)
        {
            foreach (var possibleChoice in _allChoices)
            {
                if (string.Equals(possibleChoice.Selector, selector))
                {
                    choice = possibleChoice;
                    return true;
                }
            }
            choice = null;
            return false;
        }
    }
}
```