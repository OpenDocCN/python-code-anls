# `d:/src/tocomm/basic-computer-games\82_Stars\javascript\stars.js`

```
// 创建一个新的输入元素
// 在页面上显示一个问号，提示用户输入
// 设置输入元素的类型为文本输入
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 添加键盘按下事件监听器，当按下回车键时执行相应操作
input_element.addEventListener("keydown", function (event) {
    # 如果按下的键是回车键
    if (event.keyCode == 13) {
        # 将输入元素的值赋给输入字符串
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
# 结束键盘按下事件监听器的添加
});
# 结束函数定义

# 定义一个名为 tab 的函数，参数为 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
```
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回修改后的字符串
}

var guesses = 7;  // 定义变量guesses并赋值为7
var limit = 100;  // 定义变量limit并赋值为100

// 主程序
async function main()
{
    print(tab(33) + "STARS\n");  // 打印带有33个空格的字符串和"STARS"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印带有15个空格的字符串和"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n\n\n");  // 打印三个换行符

    // 说明
    print("DO YOU WANT INSTRUCTIONS? (Y/N)");  // 打印提示信息
    var instructions = await input();  // 获取用户输入并赋值给变量instructions
    if(instructions.toLowerCase()[0] == "y") {  // 如果用户输入的第一个字符是"y"
        print(`I AM THINKING OF A WHOLE NUMBER FROM 1 TO ${limit}\n`);  // 打印包含变量limit的字符串
        print("TRY TO GUESS MY NUMBER.  AFTER YOU GUESS, I\n");  // 打印提示信息
        // 打印游戏规则提示信息
        print("WILL TYPE ONE OR MORE STARS (*).  THE MORE\n");
        print("STARS I TYPE, THE CLOSER YOU ARE TO MY NUMBER.\n");
        print("ONE STAR (*) MEANS FAR AWAY, SEVEN STARS (*******)\n");
        print(`MEANS REALLY CLOSE!  YOU GET ${guesses} GUESSES.\n\n\n`);
    }

    // 游戏循环
    while (true) {

        // 生成随机数
        var randomNum = Math.floor(Math.random() * limit) + 1;
        var loss = true;

        // 提示玩家开始猜数字
        print("\nOK, I AM THINKING OF A NUMBER, START GUESSING.\n\n");

        for(var guessNum=1; guessNum <= guesses; guessNum++) {

            // 输入猜测的数字
            print("YOUR GUESS");
            var guess = parseInt(await input());
            // 检查猜测是否正确
            if(guess == randomNum) {
                loss = false; // 设置游戏失败标志为假
                print("\n\n" + "*".repeat(50) + "!!!\n"); // 打印分隔线
                print(`YOU GOT IT IN ${guessNum} GUESSES!!! LET'S PLAY AGAIN...\n`); // 打印猜测次数并提示再玩一次
                break; // 退出循环
            }

            // 输出猜测与目标数字的距离
            var dist = Math.abs(guess - randomNum); // 计算猜测与目标数字的绝对距离
            if(isNaN(dist)) print("*"); // 如果距离为 NaN，则打印星号
            else if(dist >= 64) print("*"); // 如果距离大于等于 64，则打印一个星号
            else if(dist >= 32) print("**"); // 如果距离大于等于 32，则打印两个星号
            else if(dist >= 16) print("***"); // 如果距离大于等于 16，则打印三个星号
            else if(dist >= 8) print("****"); // 如果距离大于等于 8，则打印四个星号
            else if(dist >= 4) print("*****"); // 如果距离大于等于 4，则打印五个星号
            else if(dist >= 2) print("******"); // 如果距离大于等于 2，则打印六个星号
            else print("*******"); // 否则打印七个星号
            print("\n\n"); // 打印换行
        }
        if(loss) {  # 如果loss为真（即玩家猜测次数用尽）
            print(`SORRY, THAT'S ${guesses} GUESSES. THE NUMBER WAS ${randomNum}\n`);  # 打印出玩家猜测次数用尽的消息，以及正确的数字
        }
    }  # 结束if语句块
}  # 结束while循环块
main();  # 调用main函数，开始游戏
```