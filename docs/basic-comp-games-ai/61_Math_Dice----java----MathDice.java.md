# `basic-computer-games\61_Math_Dice\java\MathDice.java`

```py
// 导入 Scanner 类
import java.util.Scanner;

// 定义 MathDice 类
public class MathDice {

    // 主函数
    public static void main(String[] args) {
        // 创建 Scanner 对象
        Scanner in = new Scanner(System.in);
        // 创建两个骰子对象
        Die dieOne = new Die();
        Die dieTwo = new Die();
        // 初始化猜测值和答案
        int guess = 1;
        int answer;

        // 输出游戏标题和说明
        System.out.println("Math Dice");
        System.out.println("https://github.com/coding-horror/basic-computer-games");
        System.out.println();
        System.out.print("This program generates images of two dice.\n"
                + "When two dice and an equals sign followed by a question\n"
                + "mark have been printed, type your answer, and hit the ENTER\n" + "key.\n"
                + "To conclude the program, type 0.\n");

        // 游戏循环
        while (true) {
            // 打印第一个骰子
            dieOne.printDie();
            System.out.println("   +");
            // 打印第二个骰子
            dieTwo.printDie();
            System.out.println("   =");
            // 初始化尝试次数和答案
            int tries = 0;
            answer = dieOne.getFaceValue() + dieTwo.getFaceValue();

            // 猜测循环
            while (guess != answer && tries < 2) {
                if (tries == 1)
                    System.out.println("No, count the spots and give another answer.");
                try {
                    // 获取用户输入的猜测值
                    guess = in.nextInt();
                } catch (Exception e) {
                    System.out.println("Thats not a number!");
                    in.nextLine();
                }

                // 如果猜测值为 0，则退出程序
                if (guess == 0)
                    System.exit(0);

                tries++;
            }

            // 判断猜测结果并输出相应信息
            if (guess != answer) {
                System.out.println("No, the answer is " + answer + "!");
            } else {
                System.out.println("Correct");
            }
            System.out.println("The dice roll again....");
        }
    }

}
```