# `basic-computer-games\15_Boxing\java\Basic.java`

```py
# 导入 Scanner 类
import java.util.Scanner;

# Basic 类提供了一些类似 BASIC 语言行为的模拟
/**
 * It provide some kind of BASIC language behaviour simulations.
 */
final class Basic {

    # 返回一个介于 1 和 base 之间的随机整数
    public static int randomOf(int base) {
        return (int)Math.round(Math.floor(base* Math.random() + 1));
    }

    # Console 类模拟了当输入与预期类型不匹配时的消息错误
    # 特别是在这个游戏中，如果输入了一个字符串，而期望输入一个整数
    /**
     * The Console "simulate" the message error when input does not match with the expected type.
     * Specifically for this game if you enter an String when and int was expected.
     */
    public static class Console {
        private final Scanner input = new Scanner(System.in);

        # 读取一行输入
        public String readLine() {
            return input.nextLine();
        }

        # 读取一个整数输入
        public int readInt() {
            int ret = -1;
            boolean failedInput = true;
            do {
                boolean b = input.hasNextInt();
                if (b) {
                    ret = input.nextInt();
                    failedInput = false;
                } else {
                    input.next(); # 丢弃读取
                    System.out.print("!NUMBER EXPECTED - RETRY INPUT LINE\n? ");
                }

            } while (failedInput);

            return ret;
        }

        # 打印消息
        public void print(String message, Object... args) {
            System.out.printf(message, args);
        }
    }
}
```