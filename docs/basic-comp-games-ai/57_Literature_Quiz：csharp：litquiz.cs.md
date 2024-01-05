# `d:/src/tocomm/basic-computer-games\57_Literature_Quiz\csharp\litquiz.cs`

```
# 导入系统模块
import System

# 定义类 litquiz
class litquiz:
    # 初始化静态变量 Score
    public static int Score = 0;

    # 定义主函数
    public static void Main(string[] args):
        # 打印标题和介绍
        Console.WriteLine("                         LITERATURE QUIZ")
        Console.WriteLine("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
        Console.WriteLine()
        Console.WriteLine()
        Console.WriteLine()
        Console.WriteLine("TEST YOUR KNOWLEDGE OF CHILDREN'S LITERATURE")
            Console.WriteLine();  // 输出空行
            Console.WriteLine("THIS IS A MULTIPLE-CHOICE QUIZ");  // 输出提示信息
            Console.WriteLine("TYPE A 1, 2, 3, OR 4 AFTER THE QUESTION MARK.");  // 输出提示信息
            Console.WriteLine();  // 输出空行
            Console.WriteLine("GOOD LUCK!");  // 输出祝福信息
            Console.WriteLine();  // 输出空行
            Console.WriteLine();  // 输出空行
            One();  // 调用函数 One()


        }

        public static void One() {
            Console.WriteLine("IN PINOCCHIO, WHAT WAS THE NAME OF THE CAT");  // 输出问题
            Console.WriteLine("1)TIGGER, 2)CICERO, 3)FIGARO, 4)GUIPETTO");  // 输出选项

            string answerOne;  // 声明字符串变量 answerOne
            answerOne = Console.ReadLine();  // 从控制台读取用户输入并赋值给 answerOne
            if(answerOne == "4")  # 如果答案是4
            {
                Console.WriteLine("VERY GOOD! HERE'S ANOTHER.");  # 输出“非常好！这是另一个。”
                Score = Score + 1;  # 分数加1
                Two();  # 调用Two()函数
            }
            else  # 否则
            {
                Console.WriteLine("SORRY...FIGARO WAS HIS NAME.");  # 输出“抱歉...菲加罗是他的名字。”
                Two();  # 调用Two()函数
            }

        }

        public static void Two()  # 定义名为Two()的函数
        {
            Console.WriteLine();  # 输出空行
            Console.WriteLine();  # 输出空行
            Console.WriteLine("FROM WHOSE GARDEN DID BUGS BUNNY STEAL THE CARROTS?");  # 输出“兔八哥从谁的花园偷了胡萝卜？”
            Console.WriteLine("1)MR. NIXON'S, 2)ELMER FUDD'S, 3)CLEM JUDD'S, 4)STROMBOLI'S");  # 输出选项
            string answerTwo;  # 声明一个字符串变量 answerTwo
            answerTwo = Console.ReadLine();  # 从控制台读取用户输入并赋值给 answerTwo

            if(answerTwo == "2"):  # 如果 answerTwo 的值等于 "2"
                Console.WriteLine("PRETTY GOOD!");  # 在控制台打印 "PRETTY GOOD!"
                Score = Score + 1;  # 分数加一
                Three();  # 调用函数 Three()
            else:  # 否则
                Console.WriteLine("TOO BAD...IT WAS ELMER FUDD'S GARDEN.");  # 在控制台打印 "TOO BAD...IT WAS ELMER FUDD'S GARDEN."
                Three();  # 调用函数 Three()
            }

        public static void Three():  # 定义一个名为 Three 的函数
            Console.WriteLine();  # 在控制台打印空行
# 输出空行
Console.WriteLine();
# 输出提示信息
Console.WriteLine("IN THE WIZARD OF OS, DOROTHY'S DOG WAS NAMED");
Console.WriteLine("1)CICERO, 2)TRIXIA, 3)KING, 4)TOTO");

# 读取用户输入的答案
string answerThree;
answerThree = Console.ReadLine();

# 判断用户输入的答案是否为 "4"
if(answerThree == "4"):
    # 输出回答正确的提示信息
    Console.WriteLine("YEA!  YOU'RE A REAL LITERATURE GIANT.");
    # 增加分数
    Score = Score + 1;
    # 调用函数 Four()
    Four();
else:
    # 输出回答错误的提示信息
    Console.WriteLine("BACK TO THE BOOKS,...TOTO WAS HIS NAME.");
    # 调用函数 Four()
    Four();
        }

        public static void Four()
        {
            // 输出空行
            Console.WriteLine();
            // 输出空行
            Console.WriteLine();
            // 输出题目
            Console.WriteLine("WHO WAS THE FAIR MAIDEN WHO ATE THE POISON APPLE");
            // 输出选项
            Console.WriteLine("1)SLEEPING BEAUTY, 2)CINDERELLA, 3)SNOW WHITE, 4)WENDY");

            // 声明并获取用户输入的答案
            string answerFour;
            answerFour = Console.ReadLine();

            // 判断用户输入的答案是否为3
            if(answerFour == "3")
            {
                // 输出回答正确的提示
                Console.WriteLine("GOOD MEMORY!");
                // 分数加1
                Score = Score + 1;
                // 调用结束函数
                End();
            }
```
            else
            {
                // 如果条件不满足，打印提示信息并调用 End() 函数
                Console.WriteLine("OH, COME ON NOW...IT WAS SNOW WHITE.");
                End();
            }

        }

        // 定义 End() 函数
        public static void End()
        {
            // 打印空行
            Console.WriteLine();
            Console.WriteLine();
            // 如果得分为 4，打印相应提示信息并返回
            if(Score == 4)
            {
                Console.WriteLine("WOW!  THAT'S SUPER!  YOU REALLY KNOW YOUR NURSERY");
                Console.WriteLine("YOUR NEXT QUIZ WILL BE ON 2ND CENTURY CHINESE");
                Console.WriteLine("LITERATURE (HA, HA, HA)");
                return;
            }
            // 如果得分小于 2
            else if(Score < 2)
# 如果条件成立，输出以下两行内容并返回
{
    Console.WriteLine("UGH.  THAT WAS DEFINITELY NOT TOO SWIFT.  BACK TO");
    Console.WriteLine("NURSERY SCHOOL FOR YOU, MY FRIEND.");
    return;
}
# 如果条件不成立，输出以下两行内容并返回
else
{
    Console.WriteLine("NOT BAD, BUT YOU MIGHT SPEND A LITTLE MORE TIME");
    Console.WriteLine("READING THE NURSERY GREATS.");
    return;
}
```