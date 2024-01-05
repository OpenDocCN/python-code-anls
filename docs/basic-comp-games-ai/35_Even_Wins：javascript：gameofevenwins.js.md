# `d:/src/tocomm/basic-computer-games\35_Even_Wins\javascript\gameofevenwins.js`

```
// GAME OF EVEN WINS
// 偶数获胜游戏

// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
// 由Oscar Toledo G. (nanochess)将BASIC转换为Javascript

function print(str)
{
    // 在页面输出指定的字符串
    document.getElementById("output").appendChild(document.createTextNode(str));
}

function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 在页面输出提示符
                       print("? ");

                       // 设置输入框类型为文本
                       input_element.setAttribute("type", "text");
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 为输入元素添加按键事件监听器
input_element.addEventListener("keydown", function (event) {
    # 如果按下的键是回车键（keyCode 为 13）
    if (event.keyCode == 13) {
        # 将输入字符串设置为输入元素的值
        input_str = input_element.value;
        # 从 id 为 "output" 的元素中移除输入元素
        document.getElementById("output").removeChild(input_element);
        # 打印输入字符串
        print(input_str);
        # 打印换行符
        print("\n");
        # 解析并返回输入字符串
        resolve(input_str);
    }
});
# 结束事件监听器的定义
});
# 结束函数定义

# 定义一个名为 tab 的函数，接受一个参数 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
        str += " ";  # 将空格添加到字符串末尾
    return str;  # 返回处理后的字符串
}

var r = [[], []];  # 创建一个包含两个空数组的变量r

// Main program  # 主程序
async function main()  # 异步函数main
{
    print(tab(28) + "GAME OF EVEN WINS\n");  # 打印游戏标题
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 打印创意计算机的信息
    print("\n");  # 打印空行
    print("\n");  # 再次打印空行
    print("DO YOU WANT INSTRUCTIONS (YES OR NO)");  # 打印提示信息
    str = await input();  # 从用户输入获取字符串
    print("\n");  # 打印空行
    if (str != "NO") {  # 如果用户输入不是"NO"
        print("THE GAME IS PLAYED AS FOLLOWS:\n");  # 打印游戏玩法说明
        print("\n");  # 打印空行
        print("AT THE BEGINNING OF THE GAME, A RANDOM NUMBER OF CHIPS ARE\n");  # 打印游戏开始时的说明
        # 打印游戏规则和提示信息
        print("PLACED ON THE BOARD.  THE NUMBER OF CHIPS ALWAYS STARTS\n");
        print("AS AN ODD NUMBER.  ON EACH TURN, A PLAYER MUST TAKE ONE,\n");
        print("TWO, THREE, OR FOUR CHIPS.  THE WINNER IS THE PLAYER WHO\n");
        print("FINISHES WITH A TOTAL NUMBER OF CHIPS THAT IS EVEN.\n");
        print("THE COMPUTER STARTS OUT KNOWING ONLY THE RULES OF THE\n");
        print("GAME.  IT GRADUALLY LEARNS TO PLAY WELL.  IT SHOULD BE\n");
        print("DIFFICULT TO BEAT THE COMPUTER AFTER TWENTY GAMES IN A ROW.\n");
        print("TRY IT!!!!\n");
        print("\n");
        print("TO QUIT AT ANY TIME, TYPE A '0' AS YOUR MOVE.\n");
        print("\n");
    }
    # 初始化变量
    l = 0;
    b = 0;
    # 初始化数组r的值
    for (i = 0; i <= 5; i++) {
        r[1][i] = 4;
        r[0][i] = 4;
    }
    # 进入游戏循环
    while (1) {
        a = 0;
        b = 0;  // 初始化变量 b 为 0
        e = 0;  // 初始化变量 e 为 0
        l = 0;  // 初始化变量 l 为 0
        p = Math.floor((13 * Math.random() + 9) / 2) * 2 + 1;  // 生成一个随机数并赋值给变量 p
        while (1) {  // 进入一个无限循环
            if (p == 1) {  // 如果 p 等于 1
                print("THERE IS 1 CHIP ON THE BOARD.\n");  // 打印消息
            } else {  // 否则
                print("THERE ARE " + p + " CHIPS ON THE BOARD.\n");  // 打印消息
            }
            e1 = e;  // 将 e 的值赋给 e1
            l1 = l;  // 将 l 的值赋给 l1
            e = a % 2;  // 计算 a 除以 2 的余数并赋值给 e
            l = p % 6;  // 计算 p 除以 6 的余数并赋值给 l
            if (r[e][l] < p) {  // 如果 r[e][l] 小于 p
                m = r[e][l];  // 将 r[e][l] 的值赋给 m
                if (m <= 0) {  // 如果 m 小于等于 0
                    m = 1;  // 将 m 的值设为 1
                    b = 1;  // 将 b 的值设为 1
                    break;  // 跳出循环
                }  # 结束当前的 while 循环
                p -= m;  # 从当前的筹码数量中减去电脑取走的筹码数量
                if (m == 1)
                    print("COMPUTER TAKES 1 CHIP LEAVING " + p + "... YOUR MOVE");  # 如果电脑只取走了一枚筹码，则打印相应的消息
                else
                    print("COMPUTER TAKES " + m + " CHIPS LEAVING " + p + "... YOUR MOVE");  # 如果电脑取走了多于一枚筹码，则打印相应的消息
                b += m;  # 更新电脑已经取走的筹码总数
                while (1) {  # 进入一个新的 while 循环
                    m = parseInt(await input());  # 从用户输入中获取玩家要取走的筹码数量
                    if (m == 0)  # 如果玩家输入了 0，则跳出循环
                        break;
                    if (m < 1 || m > 4 || m > p) {  # 如果玩家输入的数量不合法，则打印相应的消息
                        print(m + " IS AN ILLEGAL MOVE ... YOUR MOVE");
                    } else {  # 如果玩家输入的数量合法，则跳出循环
                        break;
                    }
                }
                if (m == 0)  # 如果玩家输入了 0，则跳出当前的 while 循环
                    break;
                if (m == p)  # 如果玩家取走的筹码数量等于当前剩余的筹码数量
                    break;  # 结束当前循环，跳出循环体
                p -= m;  # 玩家减去取走的筹码数量
                a += m;  # 累加玩家取走的筹码数量
            } else {  # 如果玩家取走的筹码数量不为0
                if (p == 1) {  # 如果玩家取走的筹码数量为1
                    print("COMPUTER TAKES 1 CHIP.\n");  # 打印电脑取走1个筹码
                } else {  # 如果玩家取走的筹码数量不为1
                    print("COMPUTER TAKES " + p + " CHIPS.\n");  # 打印电脑取走p个筹码
                }
                r[e][l] = p;  # 记录电脑取走的筹码数量
                b += p;  # 累加电脑取走的筹码数量
                break;  # 结束当前循环，跳出循环体
            }
        }
        if (m == 0)  # 如果玩家取走的筹码数量为0
            break;  # 结束当前循环，跳出循环体
        if (b % 2 != 0) {  # 如果电脑取走的筹码总数为奇数
            print("GAME OVER ... YOU WIN!!!\n");  # 打印游戏结束，玩家获胜
            print("\n");  # 打印空行
            if (r[e][l] != 1) {  # 如果电脑取走的筹码数量不为1
                r[e][l]--;  // 减少数组 r 中索引为 e 的子数组中索引为 l 的元素的值
            } else if (r[e1][l1] != 1) {  // 如果数组 r 中索引为 e1 的子数组中索引为 l1 的元素的值不等于 1
                r[e1][l1]--;  // 减少数组 r 中索引为 e1 的子数组中索引为 l1 的元素的值
            }
        } else {  // 否则
            print("GAME OVER ... I WIN!!!\n");  // 打印 "游戏结束...我赢了!!!"
            print("\n");  // 打印换行符
        }
    }
}

main();  // 调用主函数
```