# `basic-computer-games\44_Hangman\javascript\hangman.js`

```py
// 定义一个打印函数，将字符串添加到输出元素中
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

                       // 打印提示符
                       print("? ");
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听输入框的键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      // 如果按下的是回车键
                                                      if (event.keyCode == 13) {
                                                      // 获取输入框的值
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的值
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入的值
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

// 打印游戏标题
print(tab(32) + "HANGMAN\n");
// 打印游戏信息
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
print("\n");
print("\n");
print("\n");

// 初始化数组
var pa = [];
var la = [];
var da = [];
var na = [];
var ua = [];
// 定义一个包含单词的数组
var words = ["GUM","SIN","FOR","CRY","LUG","BYE","FLY",
             "UGLY","EACH","FROM","WORK","TALK","WITH","SELF",
             "PIZZA","THING","FEIGN","FIEND","ELBOW","FAULT","DIRTY",
             "BUDGET","SPIRIT","QUAINT","MAIDEN","ESCORT","PICKAX",
             "EXAMPLE","TENSION","QUININE","KIDNEY","REPLICA","SLEEPER",
             "TRIANGLE","KANGAROO","MAHOGANY","SERGEANT","SEQUENCE",
             "MOUSTACHE","DANGEROUS","SCIENTIST","DIFFERENT","QUIESCENT",
             "MAGISTRATE","ERRONEOUSLY","LOUDSPEAKER","PHYTOTOXIC",
             "MATRIMONIAL","PARASYMPATHOMIMETIC","THIGMOTROPISM"];

// 主控制部分
async function main()
{
    // 初始化变量 c 和 n
    c = 1;
    n = 50;
    }
    // 打印换行符
    print("\n");
    // 打印结束语句
    print("IT'S BEEN FUN!  BYE FOR NOW.\n");
    // 原始代码中未使用的 620 和 990 行
}

// 调用主函数
main();
```