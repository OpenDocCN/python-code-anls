# `d:/src/tocomm/basic-computer-games\20_Buzzword\java\src\Buzzword.java`

```
# 导入 Scanner 类
import java.util.Scanner;

# 创建 Buzzword 类
public class Buzzword {

	# 创建主函数
	public static void main(final String[] args) {
		# 使用 try-with-resources 语句创建 Scanner 对象
		try (
			# Scanner 是 Closeable 接口的实现类，因此在程序结束前必须关闭
			# 创建 Scanner 对象，从标准输入流中读取输入
			final Scanner scanner = new Scanner(System.in);
		) {
			# 创建 BuzzwordSupplier 对象
			final BuzzwordSupplier buzzwords = new BuzzwordSupplier();
			# 创建 UserInterface 对象，传入 Scanner 对象、标准输出流和 BuzzwordSupplier 对象
			final UserInterface userInterface = new UserInterface(
					scanner, System.out, buzzwords);
			# 运行用户界面
			userInterface.run();
		}
	}
}
```