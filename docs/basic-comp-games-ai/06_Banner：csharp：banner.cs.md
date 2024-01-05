# `06_Banner\csharp\banner.cs`

```
using System;  // 导入 System 命名空间
using System.Collections.Generic;  // 导入 System.Collections.Generic 命名空间
using System.Linq;  // 导入 System.Linq 命名空间

namespace banner  // 声明一个名为 banner 的命名空间
{
    class Banner  // 声明一个名为 Banner 的类
    {
        private int Horizontal { get; set; }  // 声明一个私有属性 Horizontal，用于存储横向值
        private int Vertical { get; set; }  // 声明一个私有属性 Vertical，用于存储纵向值
        private bool Centered { get; set; }  // 声明一个私有属性 Centered，用于存储是否居中
        private string Character { get; set; }  // 声明一个私有属性 Character，用于存储字符
        private string Statement { get; set; }  // 声明一个私有属性 Statement，用于存储语句

        // This provides a bit-ended representation of each symbol
        // that can be output.  Each symbol is defined by 7 parts -
        // where each part is an integer value that, when converted to
        // the binary representation, shows which section is filled in
        // with values and which are spaces.  i.e., the 'filled in'
        // parts represent the actual symbol on the paper.
        // 这提供了每个可以输出的符号的位结束表示。每个符号由 7 部分定义 -
        // 每个部分都是一个整数值，当转换为二进制表示时，显示哪些部分填充了值，哪些是空格。即，'填充' 部分代表纸上的实际符号。
    }
}
        # 创建一个字典，键为字符，值为整型数组
        Dictionary<char, int[]> letters = new Dictionary<char, int[]>()
        {
            # 初始化空格字符对应的整型数组
            {' ', new int[] { 0, 0, 0, 0, 0, 0, 0 } },
            # 初始化字符'A'对应的整型数组
            {'A', new int[] {505, 37, 35, 34, 35, 37, 505} },
            # 初始化字符'B'对应的整型数组
            {'B', new int[] {512, 274, 274, 274, 274, 274, 239} },
            # 初始化字符'C'对应的整型数组
            {'C', new int[] {125, 131, 258, 258, 258, 131, 69} },
            # 初始化字符'D'对应的整型数组
            {'D', new int[] {512, 258, 258, 258, 258, 131, 125} },
            # 初始化字符'E'对应的整型数组
            {'E', new int[] {512, 274, 274, 274, 274, 258, 258} },
            # 初始化字符'F'对应的整型数组
            {'F', new int[] {512, 18, 18, 18, 18, 2, 2} },
            # 初始化字符'G'对应的整型数组
            {'G', new int[] {125, 131, 258, 258, 290, 163, 101} },
            # 初始化字符'H'对应的整型数组
            {'H', new int[] {512, 17, 17, 17, 17, 17, 512} },
            # 初始化字符'I'对应的整型数组
            {'I', new int[] {258, 258, 258, 512, 258, 258, 258} },
            # 初始化字符'J'对应的整型数组
            {'J', new int[] {65, 129, 257, 257, 257, 129, 128} },
            # 初始化字符'K'对应的整型数组
            {'K', new int[] {512, 17, 17, 41, 69, 131, 258} },
            # 初始化字符'L'对应的整型数组
            {'L', new int[] {512, 257, 257, 257, 257, 257, 257} },
            # 初始化字符'M'对应的整型数组
            {'M', new int[] {512, 7, 13, 25, 13, 7, 512} },
            # 初始化字符'N'对应的整型数组
            {'N', new int[] {512, 7, 9, 17, 33, 193, 512} },
            # 初始化字符'O'对应的整型数组
            {'O', new int[] {125, 131, 258, 258, 258, 131, 125} },
            # 初始化字符'P'对应的整型数组
            {'P', new int[] {512, 18, 18, 18, 18, 18, 15} },
            # 初始化字符'Q'对应的整型数组
            {'Q', new int[] {125, 131, 258, 258, 322, 131, 381} },
# 创建一个字典，键为字符，值为整数数组
{
    'R', new int[] {512, 18, 18, 50, 82, 146, 271} 
    'S', new int[] {69, 139, 274, 274, 274, 163, 69} 
    'T', new int[] {2, 2, 2, 512, 2, 2, 2} 
    'U', new int[] {128, 129, 257, 257, 257, 129, 128} 
    'V', new int[] {64, 65, 129, 257, 129, 65, 64} 
    'W', new int[] {256, 257, 129, 65, 129, 257, 256} 
    'X', new int[] {388, 69, 41, 17, 41, 69, 388} 
    'Y', new int[] {8, 9, 17, 481, 17, 9, 8} 
    'Z', new int[] {386, 322, 290, 274, 266, 262, 260} 
    '0', new int[] {57, 69, 131, 258, 131, 69, 57} 
    '1', new int[] {0, 0, 261, 259, 512, 257, 257} 
    '2', new int[] {261, 387, 322, 290, 274, 267, 261} 
    '3', new int[] {66, 130, 258, 274, 266, 150, 100} 
    '4', new int[] {33, 49, 41, 37, 35, 512, 33} 
    '5', new int[] {160, 274, 274, 274, 274, 274, 226} 
    '6', new int[] {194, 291, 293, 297, 305, 289, 193} 
    '7', new int[] {258, 130, 66, 34, 18, 10, 8} 
    '8', new int[] {69, 171, 274, 274, 274, 171, 69} 
    '9', new int[] {263, 138, 74, 42, 26, 10, 7} 
    '?', new int[] {5, 3, 2, 354, 18, 11, 5} 
}
            {'*', new int[] {69, 41, 17, 512, 17, 41, 69} },  // 创建一个包含字符 '*' 和对应整数数组的字典
            {'=', new int[] {41, 41, 41, 41, 41, 41, 41} },  // 创建一个包含字符 '=' 和对应整数数组的字典
            {'!', new int[] {1, 1, 1, 384, 1, 1, 1} },  // 创建一个包含字符 '!' 和对应整数数组的字典
            {'.', new int[] {1, 1, 129, 449, 129, 1, 1} }  // 创建一个包含字符 '.' 和对应整数数组的字典
        };


        /// <summary>
        /// This displays the provided text on the screen and then waits for the user
        /// to enter a integer value greater than 0.
        /// </summary>
        /// <param name="DisplayText">Text to display on the screen asking for the input</param>
        /// <returns>The integer value entered by the user</returns>
        private int GetNumber(string DisplayText)
        {
            Console.Write(DisplayText);  // 在屏幕上显示提供的文本
            string TempStr = Console.ReadLine();  // 从控制台读取用户输入的文本

            Int32.TryParse(TempStr, out int TempInt);  // 将用户输入的文本转换为整数并存储在 TempInt 中
            if (TempInt <= 0)
            {
                throw new ArgumentException($"{DisplayText} must be greater than zero");
            }
            // 如果TempInt小于等于0，抛出参数异常，显示指定的文本
            return TempInt;
        }

        /// <summary>
        /// This displays the provided text on the screen and then waits for the user
        /// to enter a Y or N.  It cheats by just looking for a 'y' and returning that
        /// as true.  Anything else that the user enters is returned as false.
        /// </summary>
        /// <param name="DisplayText">Text to display on the screen asking for the input</param>
        /// <returns>Returns true or false</returns>
        private bool GetBool(string DisplayText)
        {
            Console.Write(DisplayText);
            // 在屏幕上显示提供的文本，然后等待用户输入Y或N。通过查找'y'并将其作为true返回来欺骗。用户输入的任何其他内容都将作为false返回。
            return (Console.ReadLine().StartsWith("y", StringComparison.InvariantCultureIgnoreCase));
        }
        /// <summary>
        /// This displays the provided text on the screen and then waits for the user
        /// to enter an arbitrary string.  That string is then returned 'as-is'.
        /// </summary>
        /// <param name="DisplayText">Text to display on the screen asking for the input</param>
        /// <returns>The string entered by the user.</returns>
        private string GetString(string DisplayText)
        {
            // 在屏幕上显示提供的文本，并等待用户输入任意字符串。然后将该字符串原样返回。
            Console.Write(DisplayText);
            return (Console.ReadLine().ToUpper()); // 将用户输入的字符串转换为大写并返回
        }

        /// <summary>
        /// This queries the user for the various inputs needed by the program.
        /// </summary>
        private void GetInput()
        {
            Horizontal = GetNumber("Horizontal "); // 询问用户水平方向的输入
            Vertical = GetNumber("Vertical "); // 询问用户垂直方向的输入
            Centered = GetBool("Centered ");  # 从用户输入获取布尔值，表示是否居中打印
            Character = GetString("Character (type 'ALL' if you want character being printed) ");  # 从用户输入获取字符，如果输入'ALL'则表示打印所有字符
            Statement = GetString("Statement ");  # 从用户输入获取语句
            // We don't care about what the user enters here.  This is just telling them
            // to set the page in the printer.
            _ = GetString("Set page ");  # 从用户输入获取设置页面信息，但在这里我们不关心用户输入的内容

        }

        /// <summary>
        /// This prints out a single character of the banner - adding
        /// a few blanks lines as a spacer between characters.
        /// </summary>
        private void PrintChar(char ch)
        {
            // In the trivial case (a space character), just print out the spaces
            if (ch.Equals(' '))
            {
                Console.WriteLine(new string('\n', 7 * Horizontal));  # 如果字符是空格，则打印出相应数量的换行符
                return;
            }
            // 如果用户提供了要打印的特定字符，则将其用作输出字符 - 否则取当前字符
            char outCh = Character == "ALL" ? ch : Character[0];
            // 创建一个长度为7的整型数组
            int[] letter = new int[7];
            try
            {
                // 将字母对应的数字模式复制到数组中
                letters[outCh].CopyTo(letter, 0);
            }
            catch (KeyNotFoundException)
            {
                // 如果提供的字母在字母列表中找不到，则抛出KeyNotFoundException
                throw new KeyNotFoundException($"The provided letter {outCh} was not found in the letters list");
            }

            // 这个循环遍历组成每个字母的部分。每个部分代表实际输出的1 *水平线。
            for (int idx = 0; idx < 7; idx++)
            {
                // 新建一个长度为7的整型数组，初始值为0
                // numSections决定了每个字符的每行需要打印多少个“部分”
                int[] numSections = new int[7];
                // fillInSection决定了每个字符的每个“部分”是填充字符还是空白
                int[] fillInSection = new int[9];

                // 这段代码根据每个部分的值来决定字符中哪些部分是空格，哪些部分是填充字符
                // 对于fillInSection中标记为1的部分，对应的垂直字符会被输出
                for (int exp = 8; exp >= 0; exp--)
                {
                    if (Math.Pow(2, exp) < letter[idx])
                    {
                        fillInSection[8 - exp] = 1;
                        letter[idx] -= (int)Math.Pow(2, exp);
                        if (letter[idx] == 1)
                        {
                            // 一旦我们用完了信件这部分定义的所有部分，
                            // 我们标记了那个数字并且跳出这个 for 循环。
                            numSections[idx] = 8 - exp;
                            break;
                        }
                    }
                }

                // 现在我们知道了这封信这部分的哪些部分是填充的或者是空格，
                // 我们可以实际创建要打印出的字符串。
                string lineStr = "";

                if (Centered)
                    lineStr += new string(' ', (int)(63 - 4.5 * Vertical) * 1 / 1 + 1);

                for (int idx2 = 0; idx2 <= numSections[idx]; idx2++)
                {
                    // 将 fillInSection 中的每个元素根据条件转换成空格或者指定字符，然后拼接成字符串
                    lineStr = lineStr + new string(fillInSection[idx2] == 0 ? ' ' : outCh, Vertical);
                }

                // 将上面拼接好的字符串打印 Horizontal 次
                for (int lineidx = 1; lineidx <= Horizontal; lineidx++)
                {
                    Console.WriteLine(lineStr);
                }
            }

            // 最后，为了可读性，在每个字符后面添加一些空格
            Console.WriteLine(new string('\n', 2 * Horizontal - 1));
        }

        /// <summary>
        /// 根据用户提供的参数打印整个横幅
        /// </summary>
        private void PrintBanner()
        {
            // Iterate through each character in the statement
            // 遍历语句中的每个字符
            foreach (char ch in Statement)
            {
                PrintChar(ch);
            }

            // In the original version, it would print an additional 75 blank
            // lines in order to feed the printer paper...don't really need this
            // since we're not actually printing.
            // 在原始版本中，为了给打印机提供纸张，会额外打印75行空白行...但实际上我们并不需要这个，因为我们并不真正打印。
            // Console.WriteLine(new string('\n', 75));
        }

        /// <summary>
        /// Main entry point into the banner class and handles the main loop.
        /// 主要进入横幅类并处理主循环的主要入口点。
        /// </summary>
        public void Play()
        {
            GetInput();
            PrintBanner();
        }
    }
    # 定义一个类Program
    class Program
    {
        # 定义一个静态方法Main，参数为字符串数组args
        static void Main(string[] args)
        {
            # 创建一个Banner对象并调用其Play方法
            new Banner().Play();
        }
    }
}
```