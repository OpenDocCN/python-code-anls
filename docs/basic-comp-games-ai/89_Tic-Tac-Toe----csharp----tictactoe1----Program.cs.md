# `89_Tic-Tac-Toe\csharp\tictactoe1\Program.cs`

```
# See https://aka.ms/new-console-template for more information
# 在文本前打印30个空格
Console.WriteLine("TIC TAC TOE".PadLeft(30))
# 在文本前打印15个空格
Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY".PadLeft(15))
# 在屏幕上打印三行空行
Console.WriteLine("\n\n\n")
# 这个程序玩井字游戏
# 机器先走
Console.WriteLine("THE GAME BOARD IS NUMBERED:\n")
Console.WriteLine("1  2  3")
Console.WriteLine("8  9  4")
Console.WriteLine("7  6  5")

# 主程序
while(True):
    a, b, c, d, e = 9, 0, 0, 0, 0
    p, q, r, s = 0, 0, 0, 0
    a = 9
    Console.WriteLine("\n\n")
	computerMoves(a);  # 让计算机执行动作a
	p = readYourMove();  # 读取你的动作并将其赋值给变量p
	b = move(p + 1);  # 将p加1后的值作为参数调用move函数，并将结果赋值给变量b
	computerMoves(b);  # 让计算机执行动作b
	q = readYourMove();  # 读取你的动作并将其赋值给变量q
	if (q == move(b + 4)) {  # 如果q等于move(b + 4)的结果
		c = move(b + 2);  # 将b加2后的值作为参数调用move函数，并将结果赋值给变量c
		computerMoves(c);  # 让计算机执行动作c
		r = readYourMove();  # 读取你的动作并将其赋值给变量r
		if (r == move(c + 4)) {  # 如果r等于move(c + 4)的结果
			if (p % 2 != 0) {  # 如果p除以2的余数不等于0
				d = move(c + 3);  # 将c加3后的值作为参数调用move函数，并将结果赋值给变量d
				computerMoves(d);  # 让计算机执行动作d
				s = readYourMove();  # 读取你的动作并将其赋值给变量s
				if (s == move(d + 4)) {  # 如果s等于move(d + 4)的结果
					e = move(d + 6);  # 将d加6后的值作为参数调用move函数，并将结果赋值给变量e
					computerMoves(e);  # 让计算机执行动作e
					Console.WriteLine("THE GAME IS A DRAW.");  # 在控制台输出"THE GAME IS A DRAW."
				} else {
					e = move(d + 4);  # 将d加4后的值作为参数调用move函数，并将结果赋值给变量e
					# 调用函数computerMoves，并传入参数e
					computerMoves(e);
					# 在控制台打印输出"AND WINS ********"
					Console.WriteLine("AND WINS ********");
				}
			} else {
				# 将c+7的结果赋值给变量d
				d = move(c + 7);
				# 调用函数computerMoves，并传入参数d
				computerMoves(d);
				# 在控制台打印输出"AND WINS ********"
				Console.WriteLine("AND WINS ********");
			}
		} else {
			# 将c+4的结果赋值给变量d
			d = move(c + 4);
			# 调用函数computerMoves，并传入参数d
			computerMoves(d);
			# 在控制台打印输出"AND WINS ********"
			Console.WriteLine("AND WINS ********");
		}
	} else {
		# 将b+4的结果赋值给变量c
		c = move(b + 4);
		# 调用函数computerMoves，并传入参数c
		computerMoves(c);
		# 在控制台打印输出"AND WINS ********"
		Console.WriteLine("AND WINS ********");
	}
}
void computerMoves(int move) {
	// 打印计算机的移动
	Console.WriteLine("COMPUTER MOVES " + move);
}
int readYourMove() {
	// 读取用户输入的移动
	while(true) {
		Console.Write("YOUR MOVE?");
		string input = Console.ReadLine();
		// 尝试将输入转换为整数，如果成功则返回该整数
		if (int.TryParse(input, out int number)) {
			return number;
		}
	}
}

int move(int number) {
	// 计算移动后的位置
	return number - 8 * (int)((number - 1) / 8);
}
```