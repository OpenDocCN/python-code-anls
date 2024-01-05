# `d:/src/tocomm/basic-computer-games\05_Bagels\csharp\Program.cs`

```
# 创建一个名为 Bagels 的命名空间
namespace BasicComputerGames.Bagels
{
	# 创建一个名为 Program 的类
	public class Program
	{
		# 创建一个名为 Main 的静态方法，参数为字符串数组 args
		public static void Main(string[] args)
		{
			# 创建一个 Game 类的实例对象 game
			var game = new Game();

			# 调用 game 对象的 GameLoop 方法，这将在循环中无限地玩游戏，直到玩家选择退出
			game.GameLoop();
		}
	}
}
```