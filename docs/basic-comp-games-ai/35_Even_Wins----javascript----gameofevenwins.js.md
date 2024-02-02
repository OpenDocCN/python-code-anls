# `basic-computer-games\35_Even_Wins\javascript\gameofevenwins.js`

```py
// 定义一个打印函数，将字符串输出到指定的元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 输出提示符
                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听输入框的键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入框元素
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 输出输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入的字符串并返回
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个生成指定数量空格的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 初始化一个二维数组
var r = [[], []];

// 主程序
async function main()
{
    // 输出游戏标题
    print(tab(28) + "GAME OF EVEN WINS\n");
    // 输出游戏信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    // 输出是否需要游戏说明
    print("DO YOU WANT INSTRUCTIONS (YES OR NO)");
    // 等待输入并获取输入的字符串
    str = await input();
    print("\n");
}
    # 如果输入的字符串不是"NO"，则执行以下代码块
    if (str != "NO") {
        # 打印游戏规则说明
        print("THE GAME IS PLAYED AS FOLLOWS:\n");
        print("\n");
        print("AT THE BEGINNING OF THE GAME, A RANDOM NUMBER OF CHIPS ARE\n");
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
    # 初始化变量l和b
    l = 0;
    b = 0;
    # 循环初始化数组r的值
    for (i = 0; i <= 5; i++) {
        r[1][i] = 4;
        r[0][i] = 4;
    }
    # 多余的右括号，需要删除
    }
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```