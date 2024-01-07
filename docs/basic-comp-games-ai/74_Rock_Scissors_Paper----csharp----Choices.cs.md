# `basic-computer-games\74_Rock_Scissors_Paper\csharp\Choices.cs`

```

// 引入 System 命名空间
using System;

// 定义 Choices 类
namespace RockScissorsPaper
{
    public class Choices
    {
        // 定义 Rock、Scissors、Paper 三个选择，并初始化它们的值
        public static readonly Choice Rock = new Choice("3", "Rock");
        public static readonly Choice Scissors = new Choice("2", "Scissors");
        public static readonly Choice Paper = new Choice("1", "Paper");

        // 定义私有静态成员变量 _allChoices 和 _random
        private static readonly Choice[] _allChoices;
        private static readonly Random _random = new Random();

        // 静态构造函数，初始化 Rock、Scissors、Paper 之间的关系和 _allChoices 数组
        static Choices()
        {
            Rock.CanBeat = Scissors;
            Scissors.CanBeat = Paper;
            Paper.CanBeat = Rock;

            _allChoices = new[] { Rock, Scissors, Paper };
        }

        // 获取随机选择
        public static Choice GetRandom()
        {
            return _allChoices[_random.Next(_allChoices.GetLength(0))];
        }

        // 根据选择器获取选择，如果存在则返回 true，并将选择赋值给 choice，否则返回 false
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