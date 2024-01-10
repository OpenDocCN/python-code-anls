# `basic-computer-games\84_Super_Star_Trek\java\SuperStarTrekInstructions.java`

```
// 导入所需的类
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * SUPER STARTREK INSTRUCTIONS
 * MAR 5, 1978
 * Just the instructions for SUPERSTARTREK
 *
 * Ported to Java in Jan-Feb 2022 by
 * Taciano Dreckmann Perez (taciano.perez@gmail.com)
 */
public class SuperStarTrekInstructions {

    // 主函数
    public static void main(String[] args) {
        // 打印横幅
        printBanner();
        // 获取用户输入是否需要说明
        final String reply = inputStr("DO YOU NEED INSTRUCTIONS (Y/N)? ");
        // 如果用户需要说明，则打印说明
        if ("Y".equals(reply)) {
            printInstructions();
        }
    }

    // 打印横幅
    static void printBanner() {
        print(tab(10)+"*************************************");
        print(tab(10)+"*                                   *");
        print(tab(10)+"*                                   *");
        print(tab(10)+"*      * * SUPER STAR TREK * *      *");
        print(tab(10)+"*                                   *");
        print(tab(10)+"*                                   *");
        print(tab(10)+"*************************************");
    }

    // 打印字符串
    static void print(final String s) {
        System.out.println(s);
    }

    // 生成指定数量的空格字符串
    static String tab(final int n) {
        return IntStream.range(1, n).mapToObj(num -> " ").collect(Collectors.joining());
    }

    // 获取用户输入的字符串
    static String inputStr(final String message) {
        System.out.print(message + "? ");
        // 读取用户输入
        final BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        try {
            return reader.readLine();
        } catch (IOException ioe) {
            // 捕获输入输出异常并打印堆栈信息
            ioe.printStackTrace();
            return "";
        }
    }

}
```