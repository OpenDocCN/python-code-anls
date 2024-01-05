# `60_Mastermind\csharp\CodeFactory.cs`

```
        public int Positions { get; }

        /// <summary>
        /// Initializes a new instance of the CodeFactory class with the specified
        /// number of positions and colors.
        /// </summary>
        /// <param name="positions">The number of positions in the generated codes.</param>
        /// <param name="colors">The number of colors in the generated codes.</param>
        public CodeFactory(int positions, int colors)
        {
            // Set the number of positions and colors
            this.Positions = positions;
            this.Colors = colors;
        }

        /// <summary>
        /// Generates a random code based on the number of positions and colors
        /// specified in the constructor.
        /// </summary>
        /// <returns>A list of integers representing the code.</returns>
        public List<int> GenerateCode()
        {
            // Create a new random number generator
            Random rand = new Random();
            // Generate a random code based on the number of positions and colors
            List<int> code = Enumerable.Range(1, this.Positions)
                                       .Select(x => rand.Next(1, this.Colors + 1))
                                       .ToList();
            // Return the generated code
            return code;
        }
    }
}
        # 定义一个公共属性 Positions，表示生成的代码的位置数量
        public int Positions { get; }

        /// <summary>
        /// 获取此工厂可以生成的不同代码数量。
        /// </summary>
        public int Possibilities { get; }

        /// <summary>
        /// 初始化 CodeFactory 类的新实例。
        /// </summary>
        /// <param name="positions">
        /// 位置数量。
        /// </param>
        /// <param name="colors">
        /// 颜色数量。
        /// </param>
        public CodeFactory(int positions, int colors)
        {
            # 如果位置数量小于1，则抛出异常
            if (positions < 1)
                throw new ArgumentException("A code must contain at least one position");  // 如果位置数小于1，则抛出参数异常

            if (colors < 1)  // 如果颜色数小于1
                throw new ArgumentException("A code must contain at least one color");  // 则抛出参数异常

            if (colors > Game.Colors.List.Length)  // 如果颜色数大于游戏颜色列表的长度
                throw new ArgumentException($"A code can contain no more than {Game.Colors.List.Length} colors");  // 则抛出参数异常，提示代码不能包含超过指定颜色数

            Positions     = positions;  // 将传入的位置数赋值给类的位置属性
            Colors        = colors;  // 将传入的颜色数赋值给类的颜色属性
            Possibilities = (int)Math.Pow(colors, positions);  // 计算可能的组合数量，并赋值给类的可能性属性
        }

        /// <summary>
        /// Creates a specified code.
        /// </summary>
        /// <param name="number">
        /// The number of the code to create from 0 to Possibilities - 1.
        /// </param>
        public Code Create(int number) =>  // 创建一个指定的代码
            EnumerateCodes().Skip(number).First();  // 调用EnumerateCodes方法生成代码集合，并跳过指定数量的代码，返回第一个代码

        /// <summary>
        /// 使用提供的随机数生成器创建一个随机代码。
        /// </summary>
        /// <param name="random">
        /// 随机数生成器。
        /// </param>
        public Code Create(Random random) =>
            Create(random.Next(Possibilities));  // 使用提供的随机数生成器创建一个随机代码

        /// <summary>
        /// 生成一个包含此工厂可以精确创建一次的每个代码的代码集合。
        /// </summary>
        public IEnumerable<Code> EnumerateCodes()
        {
            var current = new int[Positions];  // 创建一个长度为Positions的整数数组current
            var position = default(int);  // 创建一个整数变量position并初始化为默认值

# 使用生成器函数来生成代码序列
def generate_code(Positions, Colors):
    # 初始化当前位置的颜色组合
    current = [0] * Positions
    # 使用 do-while 循环来生成所有可能的代码序列
    do
    {
        # 生成当前颜色组合对应的代码
        yield return new Code(current);
        # 重置位置为 0，然后逐个位置递增颜色值，直到达到最大颜色值
        position = 0;
        while (position < Positions && ++current[position] == Colors)
            current[position++] = 0;
    }
    while (position < Positions);  # 当所有位置的颜色组合都生成完毧，结束循环
}
```