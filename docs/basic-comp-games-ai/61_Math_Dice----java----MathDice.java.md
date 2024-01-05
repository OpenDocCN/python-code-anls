# `61_Math_Dice\java\MathDice.java`

```
import java.util.Scanner;  // 导入 Scanner 类，用于从控制台读取输入

public class MathDice {

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);  // 创建 Scanner 对象，用于接收控制台输入
        Die dieOne = new Die();  // 创建一个骰子对象 dieOne
        Die dieTwo = new Die();  // 创建一个骰子对象 dieTwo
        int guess = 1;  // 初始化猜测值为 1
        int answer;  // 声明变量 answer，用于存储用户输入的答案

        System.out.println("Math Dice");  // 打印 "Math Dice"
        System.out.println("https://github.com/coding-horror/basic-computer-games");  // 打印 GitHub 仓库链接
        System.out.println();  // 打印空行
        System.out.print("This program generates images of two dice.\n"
                + "When two dice and an equals sign followed by a question\n"
                + "mark have been printed, type your answer, and hit the ENTER\n" + "key.\n"
                + "To conclude the program, type 0.\n");  // 打印程序说明

        while (true) {  // 进入无限循环
            dieOne.printDie(); // 打印第一个骰子的点数
            System.out.println("   +"); // 打印加号
            dieTwo.printDie(); // 打印第二个骰子的点数
            System.out.println("   ="); // 打印等号
            int tries = 0; // 初始化尝试次数为0
            answer = dieOne.getFaceValue() + dieTwo.getFaceValue(); // 计算两个骰子的点数之和

            while (guess!=answer && tries < 2) { // 当猜测的结果不等于正确答案且尝试次数小于2时循环
                if(tries == 1) // 如果尝试次数为1
                    System.out.println("No, count the spots and give another answer."); // 打印提示信息
                try{
                    guess = in.nextInt(); // 获取用户输入的猜测值
                } catch(Exception e) { // 捕获输入异常
                    System.out.println("Thats not a number!"); // 打印提示信息
                    in.nextLine(); // 清空输入缓冲区
                }

                if(guess == 0) // 如果猜测值为0
                    System.exit(0); // 退出程序
                tries++;  // 增加尝试次数的计数器

            }

            if(guess != answer){  // 如果猜测的数字不等于答案
                System.out.println("No, the answer is " + answer + "!");  // 输出猜测错误的提示，显示正确答案
            } else {  // 如果猜测的数字等于答案
                System.out.println("Correct");  // 输出猜测正确的提示
            }
            System.out.println("The dice roll again....");  // 输出骰子再次滚动的提示
        }
    }

}
```