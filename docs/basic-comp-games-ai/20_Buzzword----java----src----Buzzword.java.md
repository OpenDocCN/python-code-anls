# `basic-computer-games\20_Buzzword\java\src\Buzzword.java`

```
import java.util.Scanner;

public class Buzzword {

    public static void main(final String[] args) {
        try (
            // 创建一个 Scanner 对象，用于从标准输入读取数据
            // Scanner 是一个 Closeable 对象，所以在程序结束前必须关闭它
            final Scanner scanner = new Scanner(System.in);
        ) {
            // 创建一个 BuzzwordSupplier 对象
            final BuzzwordSupplier buzzwords = new BuzzwordSupplier();
            // 创建一个 UserInterface 对象，传入 Scanner 对象、标准输出流和 BuzzwordSupplier 对象
            final UserInterface userInterface = new UserInterface(
                    scanner, System.out, buzzwords);
            // 运行用户界面
            userInterface.run();
        }
    }
}
```