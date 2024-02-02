# `basic-computer-games\66_Number\java\2\Number.java`

```py
// 导入 Scanner 类
import java.util.Scanner;

// 定义 Number 类
public class Number {

    // 主函数
    public static void main(String[] args) {
        // 调用 printIntro 函数
        printIntro();
        // 初始化用户的初始积分为 100
        int points = 100;

        // 创建 Scanner 对象
        Scanner scan = new Scanner(System.in);
        // 初始化循环标志
        boolean done = false;
        // 循环直到 done 为 true
        while (!done) {
            // 提示用户猜测一个 1 到 5 之间的数字
            System.out.print("GUESS A NUMBER FROM 1 TO 5? ");
            // 读取用户输入的整数
            int g = scan.nextInt();

            // 生成 5 个 1 到 5 之间的随机数
            var r = randomNumber(1);
            var s = randomNumber(1);
            var t = randomNumber(1);
            var u = randomNumber(1);
            var v = randomNumber(1);

            // 根据用户猜测的数字和随机数进行判断，更新积分
            if (r == g) {
                points -= 5;
            } else if (s == g) {
                points += 5;
            } else if (t == g) {
                points += points;
            } else if (u == g) {
                points += 1;
            } else if (v == g) {
                points -= points * 0.5;
            } else {
                continue; // 不匹配任何随机数，继续循环，要求用户再次猜测
            }

            // 如果积分超过 500，设置循环标志为 true
            if (points > 500) {
                done = true;
            } else {
                // 否则打印当前积分
                System.out.println("YOU HAVE " + points + " POINTS.");
            }
        }

        // 打印最终获胜的积分
        System.out.println("!!!!YOU WIN!!!! WITH " + points + " POINTS.\n");
    }

    // 生成随机数的函数
    private static int randomNumber(int x) {
        // 注意：'x' 在原始的基本列表中完全被忽略
        return (int) (5 * Math.random() + 1);
    }
}
    // 打印游戏介绍信息
    private static void printIntro() {
        System.out.println("                                NUMBER");
        System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println("\n\n\n");
        System.out.println("YOU HAVE 100 POINTS.  BY GUESSING NUMBERS FROM 1 TO 5, YOU");
        System.out.println("CAN GAIN OR LOSE POINTS DEPENDING UPON HOW CLOSE YOU GET TO");
        System.out.println("A RANDOM NUMBER SELECTED BY THE COMPUTER.");
        System.out.println("\n");
        System.out.println("YOU OCCASIONALLY WILL GET A JACKPOT WHICH WILL DOUBLE(!)");
        System.out.println("YOUR POINT COUNT.  YOU WIN WHEN YOU GET 500 POINTS.");
    }
# 闭合前面的函数定义
```