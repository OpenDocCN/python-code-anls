# `85_Synonym\csharp\Synonym.cs`

```
using System.Text;  // 导入 System.Text 命名空间

namespace Synonym  // 创建名为 Synonym 的命名空间
{
    class Synonym  // 创建名为 Synonym 的类
    {
        Random rand = new Random();  // 创建 Random 类的实例 rand

        // 初始化正确回答的列表
        private string[] Affirmations = { "Right", "Correct", "Fine", "Good!", "Check" };

        // 初始化单词及其同义词的列表
        private string[][] Words =  // 创建二维字符串数组 Words
        {
                new string[] {"first", "start", "beginning", "onset", "initial"},  // 第一个子数组
                new string[] {"similar", "alike", "same", "like", "resembling"},  // 第二个子数组
                new string[] {"model", "pattern", "prototype", "standard", "criterion"},  // 第三个子数组
                new string[] {"small", "insignificant", "little", "tiny", "minute"},  // 第四个子数组
                new string[] {"stop", "halt", "stay", "arrest", "check", "standstill"},  // 第五个子数组
                new string[] {"house", "dwelling", "residence", "domicile", "lodging", "habitation"},  // 第六个子数组
// 创建一个包含多个字符串数组的二维数组，每个数组包含同义词
new string[] {"pit", "hole", "hollow", "well", "gulf", "chasm", "abyss"},
new string[] {"push", "shove", "thrust", "prod", "poke", "butt", "press"},
new string[] {"red", "rouge", "scarlet", "crimson", "flame", "ruby"},
new string[] {"pain", "suffering", "hurt", "misery", "distress", "ache", "discomfort"}
};

// 定义一个方法用于显示游戏介绍
private void DisplayIntro()
{
    // 打印空行
    Console.WriteLine("");
    // 打印标题
    Console.WriteLine("SYNONYM".PadLeft(23));
    Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    Console.WriteLine("");
    // 打印游戏介绍
    Console.WriteLine("A synonym of a word means another word in the English");
    Console.WriteLine("language which has the same or very nearly the same meaning.");
    Console.WriteLine("I choose a word -- you type a synonym.");
    Console.WriteLine("If you can't think of a synonym, type the word 'help'");
    Console.WriteLine("and I will tell you a synonym.");
    Console.WriteLine("");
}
        private void DisplayOutro()
        {
            // 显示结束语
            Console.WriteLine("Synonym drill completed.");
        }

        private void RandomizeTheList()
        {
            // 随机化单词列表
            int[] Order = new int[Words.Length];
            foreach (int i in Order)
            {
                Order[i] = rand.Next(); // 为每个单词生成一个随机数
            }
            Array.Sort(Order, Words); // 根据生成的随机数对单词列表进行排序
        }

        private string GetAnAffirmation()
        {
            // 获取一条肯定的话
            return Affirmations[rand.Next(Affirmations.Length)]; // 从肯定话列表中随机选择一条返回
        }
        private bool CheckTheResponse(string WordName, int WordIndex, string LineInput, string[] WordList)
        {
            if (LineInput.Equals("help"))
            {
                // 如果输入为"help"，则选择一个不等于当前单词的随机正确的同义词响应
                int HelpIndex = rand.Next(WordList.Length);
                while (HelpIndex == WordIndex)
                {
                    HelpIndex = rand.Next(0, WordList.Length);
                }
                Console.WriteLine("**** A synonym of {0} is {1}.", WordName, WordList[HelpIndex]);

                return false;
            }
            else
            {
                // 检查响应是否是列出的同义词之一，且不是当前单词提示
                if (WordList.Contains(LineInput) && LineInput != WordName)
                {
                    // 随机显示五个正确答案中的一个感叹词
                    Console.WriteLine(GetAnAffirmation());

                    return true;
                }
                else
                {
                    // 回答不正确。再试一次。
                    Console.WriteLine("     再试一次。".PadLeft(5));

                    return false;
                }
            }
        }

        private string PromptForSynonym(string WordName)
        {
            // 提示用户输入与给定单词同义的词语
            Console.Write("     {0}的同义词是什么？ ", WordName);
            // 读取用户输入并去除首尾空格，转换为小写
            string LineInput = Console.ReadLine().Trim().ToLower();
            return LineInput;  // 返回用户输入的同义词
        }

        private void AskForSynonyms()
        {
            Random rand = new Random();  // 创建一个随机数生成器对象

            // 遍历经过随机化的单词列表，并显示每个列表中的一个随机单词，以提示用户输入其同义词
            foreach (string[] WordList in Words)
            {
                int WordIndex = rand.Next(WordList.Length);  // 获取当前单词列表中的随机位置
                string WordName = WordList[WordIndex];       // 获取随机位置上的实际单词
                bool Success = false;

                while (!Success)
                {
                    // 提示用户输入当前单词的同义词
                    string LineInput = PromptForSynonym(WordName);

                    // 检查用户输入的同义词
                    Success = CheckTheResponse(WordName, WordIndex, LineInput, WordList);
                    # 调用名为CheckTheResponse的函数，传入WordName, WordIndex, LineInput, WordList作为参数，并将返回值赋给Success变量

                    // Add extra line space for formatting
                    Console.WriteLine("");
                    # 在控制台输出一个空行，用于格式化输出
                }
            }
        }

        public void PlayTheGame()
        {
            RandomizeTheList();
            # 调用名为RandomizeTheList的函数，用于对列表进行随机化

            DisplayIntro();
            # 调用名为DisplayIntro的函数，用于显示游戏介绍

            AskForSynonyms();
            # 调用名为AskForSynonyms的函数，用于询问同义词

            DisplayOutro();
            # 调用名为DisplayOutro的函数，用于显示游戏结束语
        }
    }
    class Program
{
    # 定义一个静态方法，接受一个字符串数组作为参数
    static void Main(string[] args)
    {
        # 创建一个 Synonym 类的实例，并调用其 PlayTheGame 方法
        new Synonym().PlayTheGame();
    }
}
```
这段代码是C#语言的程序入口点，定义了一个静态方法 Main，该方法接受一个字符串数组作为参数。在方法内部，创建了一个 Synonym 类的实例，并调用了其 PlayTheGame 方法。
```