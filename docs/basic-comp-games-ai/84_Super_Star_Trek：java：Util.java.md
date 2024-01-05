# `84_Super_Star_Trek\java\Util.java`

```
import java.io.BufferedReader; // 导入用于读取字符流的类
import java.io.IOException; // 导入处理输入输出异常的类
import java.io.InputStreamReader; // 导入用于读取输入流的类
import java.math.BigDecimal; // 导入用于高精度计算的类
import java.math.RoundingMode; // 导入用于指定舍入模式的类
import java.util.Random; // 导入生成伪随机数的类
import java.util.stream.Collectors; // 导入用于收集流中元素的类
import java.util.stream.IntStream; // 导入用于处理整数流的类

/**
 * Convenience utility methods for the Super Star Trek game.
 */
public class Util {

    static final Random random = new Random(); // 创建一个伪随机数生成器对象

    public static float random() { // 定义一个返回随机浮点数的方法
        return random.nextFloat(); // 调用伪随机数生成器的方法生成浮点数并返回
    }
        // 生成一个1到8之间的随机整数
        public static int fnr() {    // 475
            // 生成一个1到8之间的随机整数
            return toInt(random() * 7 + 1);
        }

        // 将double类型转换为整数
        public static int toInt(final double num) {
            // 向下取整
            int x = (int) Math.floor(num);
            // 如果x小于0，则取绝对值
            if (x < 0) x *= -1;
            return x;
        }

        // 打印字符串并换行
        public static void println(final String s) {
            System.out.println(s);
        }

        // 打印字符串
        public static void print(final String s) {
            System.out.print(s);
        }

        // 返回n个制表符的字符串
        public static String tab(final int n) {
        return IntStream.range(1, n).mapToObj(num -> " ").collect(Collectors.joining());
    }
    # 返回一个由 n 个空格组成的字符串

    public static int strlen(final String s) {
        return s.length();
    }
    # 返回字符串 s 的长度

    public static String inputStr(final String message) {
        System.out.print(message + "? ");
        final BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        try {
            return reader.readLine();
        } catch (IOException ioe) {
            ioe.printStackTrace();
            return "";
        }
    }
    # 打印消息并等待用户输入字符串，然后返回输入的字符串

    public static int[] inputCoords(final String message) {
        while (true) {
    }
    # 定义一个方法用于输入坐标，但是缺少具体的实现
            // 从控制台获取输入字符串
            final String input = inputStr(message);
            // 尝试将输入字符串按逗号分割成数组
            try {
                final String[] splitInput = input.split(",");
                // 如果分割后的数组长度为2
                if (splitInput.length == 2) {
                    // 将数组中的第一个和第二个元素转换为整数
                    int x = Integer.parseInt(splitInput[0]);
                    int y = Integer.parseInt(splitInput[0]);
                    // 返回包含转换后整数的数组
                    return new int[]{x, y};
                }
            } catch (Exception e) {
                // 捕获异常并打印堆栈信息
                e.printStackTrace();
            }
        }
    }

    // 从控制台获取输入并转换为浮点数
    public static float inputFloat(final String message) {
        // 循环直到获取有效的输入
        while (true) {
            // 提示用户输入信息
            System.out.print(message + "? ");
            // 创建从控制台读取输入的 BufferedReader 对象
            final BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            try {
                // 读取用户输入的字符串
                final String input = reader.readLine();
    public static String rightStr(final String input, final int len) {
        // 如果输入字符串为空或者长度小于指定长度，则直接返回输入字符串
        if (input == null || input.length() < len) return input;
        // 返回输入字符串从末尾开始指定长度的子字符串
        return input.substring(input.length() - len);
    }
        if (input == null || input.length() < len) return "";  # 检查输入字符串是否为null或长度小于指定长度，如果是则返回空字符串
        return input.substring(input.length() - len);  # 返回输入字符串的后缀，长度为指定长度
    }

    public static double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();  # 如果小数位数小于0，抛出非法参数异常
        BigDecimal bd = new BigDecimal(Double.toString(value));  # 将double类型的值转换为BigDecimal类型
        bd = bd.setScale(places, RoundingMode.HALF_UP);  # 对BigDecimal类型的值进行四舍五入，保留指定位数的小数
        return bd.doubleValue();  # 返回四舍五入后的double类型的值
    }
```