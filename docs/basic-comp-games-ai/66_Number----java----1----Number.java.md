# `basic-computer-games\66_Number\java\1\Number.java`

```

// 导入所需的 Java 类库
import java.time.temporal.ValueRange;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;

// 定义一个名为 Number 的类
public class Number {

    // 定义一个静态整型变量 points，用于存储玩家的得分
    public static int points = 0;

    // 定义一个静态方法 printempty，用于打印空行
    public static void printempty() { System.out.println(" "); }

    // 定义一个静态方法 print，用于打印指定的字符串
    public static void print(String toprint) { System.out.println(toprint); }

    // 主方法
    public static void main(String[] args) {
        // 打印游戏规则
        print("YOU HAVE 100 POINTS.  BY GUESSING NUMBERS FROM 1 TO 5, YOU");
        print("CAN GAIN OR LOSE POINTS DEPENDING UPON HOW CLOSE YOU GET TO");
        print("A RANDOM NUMBER SELECTED BY THE COMPUTER.");
        printempty();
        print("YOU OCCASIONALLY WILL GET A JACKPOT WHICH WILL DOUBLE(!)");
        print("YOUR POINT COUNT.  YOU WIN WHEN YOU GET 500 POINTS.");
        printempty();

        try {
            // 循环进行游戏
            while (true) {
                print("GUESS A NUMBER FROM 1 TO 5");

                // 读取玩家输入的数字
                Scanner numbersc = new Scanner(System.in);
                String numberstring = numbersc.nextLine();
                int number = Integer.parseInt(numberstring);

                // 判断玩家输入的数字是否在1到5之间
                if (!(number < 1| number > 5)) {
                    // 生成一个随机数
                    Random rand = new Random();
                    int randomNum = rand.nextInt((5 - 1) + 1) + 1;

                    // 根据玩家猜测的数字和随机数的关系，更新得分
                    if (randomNum == number) {
                        print("YOU HIT THE JACKPOT!!!");
                        points = points * 2;
                    } else if(ValueRange.of(randomNum, randomNum + 1).isValidIntValue(number)) {
                        print("+5");
                        points = points + 5;
                    } else if(ValueRange.of(randomNum - 1, randomNum + 2).isValidIntValue(number)) {
                        print("+1");
                        points = points + 1;
                    } else if(ValueRange.of(randomNum - 3, randomNum + 1).isValidIntValue(number)) {
                        print("-1");
                        points = points - 1;
                    } else {
                        print("-half");
                        points = (int) (points * 0.5);
                    }

                    // 打印更新后的得分
                    print("YOU HAVE " + points + " POINTS.");
                }

                // 判断是否达到500分，如果是则打印胜利信息并结束游戏
                if (points >= 500) {
                    print("!!!!YOU WIN!!!! WITH " + points + " POINTS.");
                    return;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

```