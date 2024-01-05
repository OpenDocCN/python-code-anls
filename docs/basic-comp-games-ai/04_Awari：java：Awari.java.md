# `d:/src/tocomm/basic-computer-games\04_Awari\java\Awari.java`

```
import java.util.Scanner;  // 导入 Scanner 类，用于接收用户输入
import java.util.Random;   // 导入 Random 类，用于生成随机数

public class Awari{
	int []board;  // 定义整型数组 board，用于存储游戏棋盘状态
	private final int playerPits;  // 定义玩家的小坑的位置
	private final int computerPits;  // 定义计算机的小坑的位置
	private final int playerHome;  // 定义玩家的家的位置
	private final int computerHome;  // 定义计算机的家的位置
	Scanner input;  // 创建 Scanner 对象，用于接收用户输入
	int sumPlayer;  // 定义玩家的总分
	int sumComputer;  // 定义计算机的总分
	Awari(){  // 构造函数
		input = new Scanner(System.in);  // 初始化 Scanner 对象
		playerPits = 0;  // 初始化玩家的小坑位置
		computerPits = 7;  // 初始化计算机的小坑位置
		playerHome = 6;  // 初始化玩家的家的位置
		computerHome = 13;  // 初始化计算机的家的位置
		sumPlayer = 18;  // 初始化玩家的总分
		sumComputer = 18;  // 初始化计算机的总分
		# 创建一个包含14个整数的数组
		board = new int [14];
		# 为玩家和计算机的坑位初始化石子数量
		for (int i=0;i<6;i++){
			board[playerPits+i]=3;
			board[computerPits+i]=3;
		}
		# 打印游戏标题和信息
		System.out.println("		 AWARI");
		System.out.println("CREATIVE COMPUTING MORRISTOWN, NEW JERSEY");
		# 打印游戏棋盘
		printBoard();
		# 玩家进行移动
		playerMove(true);
	}

	# 打印游戏棋盘
	private void printBoard(){
		# 打印玩家的坑位石子数量
		System.out.print("\n    ");
		for (int i=0;i<6;i++){
			System.out.print(String.format("%2d",board[12-i]));
			System.out.print("  ");
		}
		# 打印计算机的家和空格
		System.out.println("");
		System.out.print(String.format("%2d",board[computerHome]));
		System.out.print("                          ");
		// 打印玩家的家和每个坑的石子数量
		System.out.println(String.format("%2d",board[playerHome]));
		System.out.print("    ");
		for(int i=0;i<6;i++){
			System.out.print(String.format("%2d",board[playerPits+i]));
                        System.out.print("  ");
		}
		System.out.println("");
	}

	private void playerMove(boolean val){
		// 打印电脑和玩家的总石子数量
		System.out.println("\nComputerSum PlayerSum"+sumComputer+" "+sumPlayer);
		// 根据参数val的值打印提示信息
		if(val == true)
			System.out.print("YOUR MOVE? ");
		else
			System.out.print("AGAIN? ");
		// 获取玩家输入的移动位置
		int move =  input.nextInt();
		// 当输入的移动位置不在1到6之间或者对应位置的石子数量为0时，要求玩家重新输入
		while(move<1||move>6||board[move-1]==0){
			System.out.print("INVALID MOVE!!! TRY AGAIN  ");
			move = input.nextInt();
		}
		# 从棋盘上取出移动的种子数
		int seeds = board[move-1];
		# 将移动的位置上的种子数置零
		board[move-1] = 0;
		# 减去玩家的总种子数
		sumPlayer -= seeds;
		# 将种子按规则分布到各个位置上，并返回最后一个位置
		int last_pos = distribute(seeds,move);
		# 如果最后一个位置是玩家的家
		if(last_pos == playerHome):
			# 打印棋盘
			printBoard();
			# 如果游戏结束，退出程序
			if(isGameOver(true)):
				System.exit(0);
			# 让计算机进行移动
			playerMove(false);
		# 如果最后一个位置上的种子数为1且不是计算机的家
		else if(board[last_pos] == 1&&last_pos != computerHome):
			# 计算对面位置的索引
			int opp = calculateOpposite(last_pos);
			# 如果最后一个位置在玩家区域内
			if(last_pos<6):
				# 玩家总种子数增加对面位置的种子数
				sumPlayer+=board[opp];
				# 计算机总种子数减去对面位置的种子数
				sumComputer-=board[opp];
			# 如果最后一个位置在计算机区域内
			else:
				# 计算机总种子数增加对面位置的种子数
				sumComputer+=board[opp];
				# 玩家总种子数减去对面位置的种子数
				sumPlayer-=board[opp];
			}
			# 将上一步的位置的棋子数量加上对手位置的棋子数量，并将对手位置的棋子数量设为0
			board[last_pos]+=board[opp];
			board[opp] = 0;
			# 打印棋盘
			printBoard();
			# 如果游戏结束，退出程序
			if(isGameOver(false)){
				System.exit(0);
			}
			# 让计算机进行移动
			computerMove(true);
		}
		else{
			# 打印棋盘
			printBoard();
			# 如果游戏结束，退出程序
			if(isGameOver(false)){
				System.exit(0);
			}
			# 让计算机进行移动
			computerMove(true);
		}
	}

	# 计算机进行移动的方法
	private void computerMove(boolean value){
		# 初始化val为-1
		int val=-1;
		// 打印计算机和玩家的总和
		System.out.println("\nComputerSum PlayerSum"+sumComputer+" "+sumPlayer);
		// 遍历计算机的领地，找到可以移动的位置
		for(int i=0;i<6;i++){
			if(6-i == board[computerPits+i])
				val = i;
		}
		// 初始化移动位置
		int move ;
		// 如果没有可以移动的位置，随机选择一个位置
		if(val == -1)
		{
			Random random = new Random();
			move = random.nextInt(6)+computerPits;
			// 如果选择的位置没有种子，继续随机选择位置，直到选择到有种子的位置
			while(board[move] == 0){
				move = random.nextInt(6)+computerPits;
			}
			// 如果 value 为 true，打印计算机的移动位置
			if(value == true)
				System.out.println(String.format("MY MOVE IS %d ",move-computerPits+1));
			// 如果 value 不为 true，打印计算机的移动位置
			else
				System.out.println(String.format(",%d",move-computerPits+1));
			// 获取选择位置的种子数
			int seeds = board[move];
			// 清空选择位置的种子数
			board[move] = 0;
			// 更新计算机的总和
			sumComputer-=seeds;
			# 调用 distribute 函数，计算出最后的位置
			int last_pos = distribute(seeds,move+1);
			# 如果最后的位置上有种子并且不是玩家的家
			if(board[last_pos] == 1 && last_pos != playerHome){
                	        # 计算对面位置
				 int opp = calculateOpposite(last_pos);
				 # 如果最后的位置小于6
				 if(last_pos<6){
	                                # 玩家得到对面位置的种子
        	                        sumPlayer += board[opp];
                	                # 电脑失去对面位置的种子
					sumComputer -= board[opp];
                	        }
                        	else{
	                                # 电脑得到对面位置的种子
        	                        sumComputer += board[opp];
                	                # 玩家失去对面位置的种子
					sumPlayer -= board[opp];
                	        }
        	                # 最后的位置上的种子数量增加对面位置的种子数量
				board[last_pos] += board[opp];
	                        # 对面位置的种子数量变为0
				board[opp] = 0;
                        	# 打印游戏棋盘
				printBoard();
                	        # 如果游戏结束，退出程序
				if(isGameOver(false)){
        	                        System.exit(0);
	                        }
                	}
			# 否则
			else{
				# 打印游戏棋盘
				printBoard();
# 如果游戏结束，退出程序
if(isGameOver(false)):
    System.exit(0)
```
这段代码检查游戏是否结束，如果游戏结束则退出程序。

```
playerMove(true)
```
调用playerMove函数，传入true作为参数。

```
move = val+computerPits
```
计算move的值，将val和computerPits相加。

```
if(value == true)
    System.out.print(String.format("MY MOVE IS %d",move-computerPits+1))
else
    System.out.print(String.format(",%d",move-computerPits+1))
```
根据value的值，打印不同的消息。

```
seeds = board[move]
board[move] = 0
sumComputer-=seeds
```
将board中move位置的值赋给seeds，然后将move位置的值设为0，最后将seeds的值从sumComputer中减去。

```
last_pos = distribute(seeds,move+1)
```
调用distribute函数，传入seeds和move+1作为参数，并将返回值赋给last_pos。

```
if(last_pos == computerHome):
    if(isGameOver(true)):
        System.exit(0)
```
如果last_pos等于computerHome，且游戏结束，则退出程序。
# 调用computerMove函数，并传入false作为参数
computerMove(false);
# 结束if语句块
}
# 结束while循环块
}
# 结束distribute函数
}

# 定义distribute函数，接受seeds和pos作为参数
private int distribute(int seeds, int pos){
    # 当seeds不为0时执行循环
    while(seeds!=0){
        # 如果pos等于14，则将pos重置为0
        if(pos==14)
            pos=0;
        # 如果pos小于6，则将sumPlayer加1
        if(pos<6)
            sumPlayer++;
        # 如果pos大于6且小于13，则将sumComputer加1
        else if(pos>6&&pos<13)
            sumComputer++;
        # 将board[pos]的值加1
        board[pos]++;
        # 将pos加1
        pos++;
        # 将seeds减1
        seeds--;
    }
    # 返回pos-1
    return pos-1;
}
		// 计算玩家对面的位置
		private int calculateOpposite(int pos){
			return 12-pos;
		}

		// 检查游戏是否结束
		private boolean isGameOver(boolean show){
			// 如果玩家或电脑的棋子数为0，则游戏结束
			if(sumPlayer == 0 || sumComputer == 0){
				// 如果需要展示棋盘，则打印棋盘
				if(show)
					printBoard();
				// 打印游戏结束信息
				System.out.println("GAME OVER");
				// 判断胜负并打印对应信息
				if(board[playerHome]>board[computerHome]){
					System.out.println(String.format("YOU WIN BY %d POINTS",board[playerHome]-board[computerHome]));
				}
				else if(board[playerHome]<board[computerHome]){
					System.out.println(String.format("YOU LOSE BY %d POINTS",board[computerHome]-board[playerHome]));
				}
				else{
					System.out.println("DRAW");
				}
				return true;
		}
		# 返回 false
		return false;
	}


}
```