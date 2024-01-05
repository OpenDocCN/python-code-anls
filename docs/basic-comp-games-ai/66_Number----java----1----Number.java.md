# `66_Number\java\1\Number.java`

```
import java.time.temporal.ValueRange;  // 导入 java.time.temporal.ValueRange 类
import java.util.Arrays;  // 导入 java.util.Arrays 类
import java.util.Random;  // 导入 java.util.Random 类
import java.util.Scanner;  // 导入 java.util.Scanner 类

public class Number {

    public static int points = 0;  // 创建一个静态整数变量 points 并初始化为 0

    public static void printempty() { System.out.println(" "); }  // 创建一个静态方法 printempty，用于打印空行

    public static void print(String toprint) { System.out.println(toprint); }  // 创建一个静态方法 print，用于打印传入的字符串参数

    public static void main(String[] args) {  // 主方法
        print("YOU HAVE 100 POINTS.  BY GUESSING NUMBERS FROM 1 TO 5, YOU");  // 调用 print 方法打印字符串
        print("CAN GAIN OR LOSE POINTS DEPENDING UPON HOW CLOSE YOU GET TO");  // 调用 print 方法打印字符串
        print("A RANDOM NUMBER SELECTED BY THE COMPUTER.");  // 调用 print 方法打印字符串
        printempty();  // 调用 printempty 方法打印空行
        print("YOU OCCASIONALLY WILL GET A JACKPOT WHICH WILL DOUBLE(!)");  // 调用 print 方法打印字符串
        # 打印游戏规则提示信息
        print("YOUR POINT COUNT.  YOU WIN WHEN YOU GET 500 POINTS.")
        # 打印空行
        printempty()

        # 开始游戏循环
        try:
            while (true):
                # 提示玩家猜一个1到5之间的数字
                print("GUESS A NUMBER FROM 1 TO 5")

                # 读取玩家输入的数字
                Scanner numbersc = new Scanner(System.in)
                String numberstring = numbersc.nextLine()

                # 将输入的字符串转换为整数
                int number = Integer.parseInt(numberstring)

                # 如果输入的数字不在1到5之间
                if (!(number < 1| number > 5)):

                    # 生成一个1到5之间的随机数
                    Random rand = new Random()
                    int randomNum = rand.nextInt((5 - 1) + 1) + 1

                    # 如果随机数等于玩家输入的数字
                    if (randomNum == number):
# 如果玩家击中了大奖，打印“YOU HIT THE JACKPOT!!!”，并将点数翻倍
if (randomNum == number) {
    print("YOU HIT THE JACKPOT!!!");
    points = points * 2;
} 
# 如果玩家猜中了随机数范围内的一个数字，打印“+5”，并增加5点
else if(ValueRange.of(randomNum, randomNum + 1).isValidIntValue(number)) {
    print("+5");
    points = points + 5;
} 
# 如果玩家猜中了随机数范围内的一个数字，打印“+1”，并增加1点
else if(ValueRange.of(randomNum - 1, randomNum + 2).isValidIntValue(number)) {
    print("+1");
    points = points + 1;
} 
# 如果玩家猜中了随机数范围内的一个数字，打印“-1”，并减少1点
else if(ValueRange.of(randomNum - 3, randomNum + 1).isValidIntValue(number)) {
    print("-1");
    points = points - 1;
} 
# 如果玩家没有猜中，打印“-half”，并将点数减半
else {
    print("-half");
    points = (int) (points * 0.5);
}

# 打印玩家当前的点数
print("YOU HAVE " + points + " POINTS.");
```
                    print("!!!!YOU WIN!!!! WITH " + points + " POINTS.");  # 打印出获胜的消息以及获得的分数
                    return;  # 结束函数的执行
                }
            }
        } catch (Exception e) {  # 捕获可能发生的异常
            e.printStackTrace();  # 打印异常的堆栈信息
        }
    }
}
```