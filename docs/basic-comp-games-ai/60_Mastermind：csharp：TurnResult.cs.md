# `60_Mastermind\csharp\TurnResult.cs`

```
        public int Whites { get; }

        /// <summary>
        /// Initializes a new instance of the TurnResult class.
        /// </summary>
        /// <param name="guess">The code guessed by the player.</param>
        /// <param name="blacks">The number of black pegs resulting from the guess.</param>
        /// <param name="whites">The number of white pegs resulting from the guess.</param>
        public TurnResult(Code guess, int blacks, int whites)
        {
            Guess = guess;
            Blacks = blacks;
            Whites = whites;
        }
    }
}
```

注释：

- `/// <summary>`：用于描述类、属性或方法的作用和功能
- `public record TurnResult`：定义了一个名为TurnResult的记录类型，用于存储玩家回合的结果
- `public Code Guess { get; }`：定义了一个名为Guess的属性，用于获取玩家猜测的代码
- `public int Blacks { get; }`：定义了一个名为Blacks的属性，用于获取猜测结果中黑色标记的数量
- `public int Whites { get; }`：定义了一个名为Whites的属性，用于获取猜测结果中白色标记的数量
- `public TurnResult(Code guess, int blacks, int whites)`：定义了一个构造函数，用于初始化TurnResult类的新实例，参数包括玩家猜测的代码、黑色标记数量和白色标记数量
- `Guess = guess;`：将传入的guess参数赋值给Guess属性
- `Blacks = blacks;`：将传入的blacks参数赋值给Blacks属性
- `Whites = whites;`：将传入的whites参数赋值给Whites属性
        public int Whites { get; }  // 定义一个公共属性，表示白色棋子的数量

        /// <summary>
        /// Initializes a new instance of the TurnResult record.
        /// </summary>
        /// <param name="guess">
        /// The player's guess.
        /// </param>
        /// <param name="blacks">
        /// The number of black pegs.
        /// </param>
        /// <param name="whites">
        /// The number of white pegs.
        /// </param>
        public TurnResult(Code guess, int blacks, int whites) =>
            (Guess, Blacks, Whites) = (guess, blacks, whites);  // 初始化一个新的 TurnResult 实例，传入玩家的猜测、黑色棋子的数量和白色棋子的数量，并将它们赋值给对应的属性
    }
}
```