# `basic-computer-games\15_Boxing\java\Basic.java`

```

// 导入 Scanner 类
import java.util.Scanner;

/**
 * 模拟基础语言行为的类
 */
final class Basic {

    // 生成一个指定基数范围内的随机数
    public static int randomOf(int base) {
        return (int)Math.round(Math.floor(base* Math.random() + 1));
    }

    /**
     * 控制台类，模拟输入不匹配预期类型时的错误消息
     * 特别针对输入整数时输入字符串的情况
     */
    public static class Console {
        private final Scanner input = new Scanner(System.in);

        // 读取一行输入
        public String readLine() {
            return input.nextLine();
        }

        // 读取一个整数输入
        public int readInt() {
            int ret = -1;
            boolean failedInput = true;
            do {
                boolean b = input.hasNextInt();
                if (b) {
                    ret = input.nextInt();
                    failedInput = false;
                } else {
                    input.next(); // 丢弃读取
                    System.out.print("!NUMBER EXPECTED - RETRY INPUT LINE\n? ");
                }

            } while (failedInput);

            return ret;
        }

        // 打印消息
        public void print(String message, Object... args) {
            System.out.printf(message, args);
        }
    }
}

```