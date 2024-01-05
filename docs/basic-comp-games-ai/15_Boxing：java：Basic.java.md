# `d:/src/tocomm/basic-computer-games\15_Boxing\java\Basic.java`

```
import java.util.Scanner;  # 导入 Scanner 类，用于从控制台读取输入

/**
 * It provide some kind of BASIC language behaviour simulations.
 */
final class Basic {  # 定义 Basic 类

    public static int randomOf(int base) {  # 定义 randomOf 方法，用于生成一个随机数
        return (int)Math.round(Math.floor(base* Math.random() + 1));  # 返回一个随机数
    }

    /**
     * The Console "simulate" the message error when input does not match with the expected type.
     * Specifically for this game if you enter an String when and int was expected.
     */
    public static class Console {  # 定义 Console 类
        private final Scanner input = new Scanner(System.in);  # 创建一个 Scanner 对象，用于从控制台读取输入

        public String readLine() {  # 定义 readLine 方法，用于从控制台读取一行输入
            return input.nextLine();  # 返回从控制台读取的输入
        }

        public int readInt() {
            int ret = -1;  // 初始化返回值为-1
            boolean failedInput = true;  // 初始化输入失败标志为true
            do {
                boolean b = input.hasNextInt();  // 检查输入是否为整数
                if (b) {  // 如果是整数
                    ret = input.nextInt();  // 读取整数
                    failedInput = false;  // 输入成功，将输入失败标志设为false
                } else {  // 如果不是整数
                    input.next(); // discard read  // 丢弃当前输入
                    System.out.print("!NUMBER EXPECTED - RETRY INPUT LINE\n? ");  // 打印错误提示
                }

            } while (failedInput);  // 当输入失败标志为true时继续循环

            return ret;  // 返回读取的整数值
        }
# 定义一个名为print的方法，接受一个字符串参数message和可变数量的参数args
def print(message, *args):
    # 使用printf方法将message格式化输出
    System.out.printf(message, args)
```