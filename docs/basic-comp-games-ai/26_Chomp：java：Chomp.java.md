# `d:/src/tocomm/basic-computer-games\26_Chomp\java\Chomp.java`

```
import java.util.Scanner;  # 导入 Scanner 类
public class Chomp{  # 创建 Chomp 类
	int rows;  # 定义整型变量 rows
	int cols;  # 定义整型变量 cols
	int numberOfPlayers;  # 定义整型变量 numberOfPlayers
	int []board;  # 定义整型数组 board
	Scanner scanner;  # 定义 Scanner 对象 scanner
	Chomp(){  # Chomp 类的构造函数
		System.out.println("\t\t\t\tCHOMP");  # 打印 CHOMP
		System.out.println("\t\tCREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 打印创意计算摩里斯敦，新泽西
		System.out.println("THIS IS THE GAME OF CHOMP (SCIENTIFIC AMERICAN, JAN 1973)");  # 打印这是 Chomp 游戏（科学美国人，1973 年 1 月）
		System.out.print("Do you want the rules (1=Yes, 0=No!)  ");  # 打印是否需要规则（1=是，0=否！）

		scanner = new Scanner(System.in);  # 创建 Scanner 对象并绑定到标准输入
		int choice = scanner.nextInt();  # 从标准输入读取整数并赋值给 choice
		if(choice != 0){  # 如果 choice 不等于 0
			System.out.println("Chomp is for 1 or more players (Humans only).\n");  # 打印 Chomp 适用于 1 个或多个玩家（仅限人类）。
			System.out.println("Here's how a board looks (This one is 5 by 7):");  # 打印棋盘的样子（这个是 5 行 7 列）：
			System.out.println("\t1 2 3 4 5 6 7");  # 打印 1 到 7
			System.out.println(" 1     P * * * * * *\n 2     * * * * * * *\n 3     * * * * * * *\n 4     * * * * * * *\n 5     * * * * * * *");  # 打印棋盘的布局
			System.out.println("Here we go...\n");
```
这行代码用于在控制台输出"Here we go...\n"。

```python
		}
		startGame();
	}
```
这行代码用于调用startGame()方法来开始游戏。

```python
	private void startGame(){
		System.out.print("How many players ");
		numberOfPlayers = scanner.nextInt();
		while(numberOfPlayers < 2){
			System.out.print("How many players ");
                	numberOfPlayers = scanner.nextInt();
		}
```
这段代码用于获取玩家数量，并在玩家数量小于2时循环提示输入玩家数量。

```python
		System.out.print("How many rows ");
		rows = scanner.nextInt();
		while(rows<=0 || rows >9){
			if(rows <= 0){
				System.out.println("Minimun 1 row is required !!");
			}
			else{
				System.out.println("Too many rows(9 is maximum). ");
```
这段代码用于获取行数，并在行数小于等于0或大于9时循环提示输入正确的行数。
			}
			// 提示用户输入行数
			System.out.print("How many rows ");
			// 从用户输入中获取行数
			rows = scanner.nextInt();
		}
		// 提示用户输入列数
		System.out.print("How many columns ");
                // 从用户输入中获取列数
                cols = scanner.nextInt();
                // 检查列数是否合法，不合法则重新提示用户输入
                while(cols<=0 || cols >9){
                        if(cols <= 0){
                                // 提示用户至少需要1列
                                System.out.println("Minimun 1 column is required !!");
                        }
                        else{
                                // 提示用户列数过多
                                System.out.println("Too many columns(9 is maximum). ");
                        }
                        // 重新提示用户输入列数
                        System.out.print("How many columns ");
                        // 从用户输入中获取列数
                        cols = scanner.nextInt();
                }
		// 创建一个二维数组，行数为用户输入的行数，列数为用户输入的列数
		board = new int[rows];
		// 将每行的列数初始化为用户输入的列数
		for(int i=0;i<rows;i++){
			board[i]=cols;
		}
		// 调用printBoard方法打印游戏棋盘
		printBoard();
		// 等待用户输入
		scanner.nextLine();
		// 调用move方法执行移动操作
		move(0);
	}

	// 打印游戏棋盘
	private void printBoard(){
		// 打印列号
		System.out.print("        ");
		for(int i=0;i<cols;i++){
			System.out.print(i+1);
			System.out.print(" ");
		}
		// 打印行号和棋子位置
		for(int i=0;i<rows;i++){
			System.out.print("\n ");
			System.out.print(i+1);
			System.out.print("      ");
			for(int j=0;j<board[i];j++){
				// 根据棋子位置打印P表示玩家位置
				if(i == 0 && j == 0){
					System.out.print("P ");
				}
				else{
					System.out.print("* ");
```
这行代码是在控制台打印一个星号。

```python
				}
			}
		}
		System.out.println("");
```
这段代码是在控制台打印一个空行。

```python
	private void move(int player){
```
这是一个方法的声明，方法名为move，参数为player。

```python
		System.out.println(String.format("Player %d",(player+1)));
```
这行代码是在控制台打印当前玩家的编号。

```python
		String input;
		String [] coordinates;
		int x=-1,y=-1;
```
这里声明了三个变量，input是用来存储用户输入的字符串，coordinates是用来存储用户输入的坐标，x和y分别用来存储坐标的行和列。

```python
		while(true){
```
这是一个无限循环的开始。

```python
			try{
```
这是一个异常处理的开始。

```python
				System.out.print("Coordinates of chomp (Row, Column) ");
				input = scanner.nextLine();
				coordinates = input.split(",");
				x = Integer.parseInt(coordinates[0]);
				y = Integer.parseInt(coordinates[1]);
```
这段代码是在控制台提示用户输入坐标，并将用户输入的字符串按逗号分割成坐标的行和列，然后将其转换成整数类型并赋值给x和y。

```python
				break;  # 结束当前的 while 循环
			}
			catch(Exception e){  # 捕获异常
				System.out.println("Please enter valid coordinates.");  # 打印提示信息
				continue;  # 继续下一次循环
			}
		}

		while(x>rows || x <1 || y>cols || y<1 || board[x-1]<y){  # 当 x 或 y 超出范围或者所选位置为空时
			System.out.println("No fair. You're trying to chomp on empty space!");  # 打印提示信息
	                while(true){  # 进入内层循环
                        	try{  # 尝试执行以下代码
					System.out.print("Coordinates of chomp (Row, Column) ");  # 打印提示信息
                	                input = scanner.nextLine();  # 读取用户输入
        	                        coordinates = input.split(",");  # 将输入按逗号分割
	                                x = Integer.parseInt(coordinates[0]);  # 将第一个坐标转换为整数
                        	        y = Integer.parseInt(coordinates[1]);  # 将第二个坐标转换为整数
                	                break;  # 结束内层循环
        	                }
	                        catch(Exception e){  # 捕获异常
# 如果 x 和 y 都等于 1，则输出玩家失败的消息
if(x == 1 and y == 1):
    print("You lose player "+(player+1))
    # 初始化 choice 为 -1
    int choice = -1 
    # 当 choice 不等于 0 且不等于 1 时，循环提示用户输入是否再次开始游戏
    while(choice != 0 and choice != 1):
        print("Again (1=Yes, 0=No!) ")
        choice = scanner.nextInt()
    # 如果用户选择再次开始游戏，则调用 startGame() 函数
    if(choice == 1):
        startGame()
    # 如果用户选择不再次开始游戏，则退出程序
    else:
        System.exit(0)
		else{
			for(int i=x-1;i<rows;i++){  # 从 x-1 行开始遍历到最后一行
				if(board[i] >= y){  # 如果当前行的值大于等于 y
					board[i] = y-1;  # 将当前行的值设为 y-1
				}
			}
			printBoard();  # 调用 printBoard() 函数打印游戏板
			move((player+1)%numberOfPlayers);  # 调用 move() 函数，传入下一个玩家的编号
		}
	}


	public static void main(String []args){
		new Chomp();  # 创建 Chomp 类的实例
	}
}
```