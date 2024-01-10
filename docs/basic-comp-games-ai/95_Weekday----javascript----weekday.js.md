# `basic-computer-games\95_Weekday\javascript\weekday.js`

```
// WEEKDAY
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

/**
 * Print given string to the end of the "output" element.
 * @param str - the string to be printed
 */
function print(str) {
    document.getElementById("output").appendChild(document.createTextNode(str));
}

/**
 * Obtain user input
 * @returns {Promise<String>} - a promise that resolves to the user input
 */
function input() {
    return new Promise(function (resolve) {
        const input_element = document.createElement("INPUT");

        // Print a question mark to prompt the user for input
        print("? ");
        input_element.setAttribute("type", "text");
        input_element.setAttribute("length", "50");
        document.getElementById("output").appendChild(input_element);
        input_element.focus();
        // Listen for the Enter key press to capture the user input
        input_element.addEventListener("keydown", function (event) {
            if (event.keyCode === 13) {
                const input_str = input_element.value;
                document.getElementById("output").removeChild(input_element);
                print(input_str);
                print("\n");
                resolve(input_str);
            }
        });
    });
}

/**
 * Create a string consisting of the given number of spaces
 * @param spaceCount - the number of spaces to create
 * @returns {string} - a string consisting of the specified number of spaces
 */
function tab(spaceCount) {
    let str = "";
    while (spaceCount-- > 0)
        str += " ";
    return str;
}

// Constants for date calculations
const MONTHS_PER_YEAR = 12;
const DAYS_PER_COMMON_YEAR = 365;
const DAYS_PER_IDEALISED_MONTH = 30;
const MAXIMUM_DAYS_PER_MONTH = 31;
// In a common (non-leap) year the day of the week for the first of each month moves by the following amounts.
const COMMON_YEAR_MONTH_OFFSET = [0, 3, 3, 6, 1, 4, 6, 2, 5, 0, 3, 5];

/**
 * Date representation.
 */
class DateStruct {
    #year;
    #month;
    #day;

    /**
     * Build a DateStruct
     * @param {number} year - the year
     * @param {number} month - the month
     * @param {number} day - the day
     */
    constructor(year, month, day) {
        this.#year = year;
        this.#month = month;
        this.#day = day;
    }

    get year() {
        return this.#year;
    }
    // 获取月份属性的值
    get month() {
        return this.#month;
    }

    // 获取日期属性的值
    get day() {
        return this.#day;
    }

    /**
     * 判断日期是否为公历日期。
     * 请注意，公历并非一下子在所有地方都被引入，参见 https://en.wikipedia.org/wiki/Gregorian_calendar
     * @returns {boolean} 如果日期可能是公历日期，则返回 true；否则返回 false。
     */
    isGregorianDate() {
        let result = false;
        if (this.#year > 1582) {
            result = true;
        } else if (this.#year === 1582) {
            if (this.#month > 10) {
                result = true;
            } else if (this.#month === 10 && this.#day >= 15) {
                result = true;
            }
        }
        return result;
    }

    /**
     * 以下对日期部分进行哈希运算，确保：
     * 1. 不同的日期将返回不同的数字
     * 2. 返回的数字是有序的。
     * @returns {number}
     */
    getNormalisedDay() {
        return (this.year * MONTHS_PER_YEAR + this.month) * MAXIMUM_DAYS_PER_MONTH + this.day;
    }

    /**
     * 确定星期几。
     * 此计算返回一个介于1和7之间的数字，其中星期日=1，星期一=2，...，星期六=7。
     * @returns {number} 代表星期日到星期六的值在1到7之间。
     */
    // 获取一周中的星期几
    getDayOfWeek() {
        // 根据年份的世纪部分计算偏移量
        const centuriesSince1500 = Math.floor((this.year - 1500) / 100);
        let centuryOffset = centuriesSince1500 * 5 + (centuriesSince1500 + 3) / 4;
        centuryOffset = Math.floor(centuryOffset % 7);

        // 根据缩短的两位数年份计算偏移量
        // 一月一日每年大约向前移动1.25天
        const yearInCentury = this.year % 100;
        const yearInCenturyOffsets = yearInCentury / 4 + yearInCentury;

        // 结合偏移量和日期和月份
        let dayOfWeek = centuryOffset + yearInCenturyOffsets + this.day + COMMON_YEAR_MONTH_OFFSET[this.month - 1];

        dayOfWeek = Math.floor(dayOfWeek % 7) + 1;
        if (this.month <= 2 && this.isLeapYear()) {
            dayOfWeek--;
        }
        if (dayOfWeek === 0) {
            dayOfWeek = 7;
        }
        return dayOfWeek;
    }

    /**
     * 确定给定年份是否是闰年。
     * @returns {boolean}
     */
    isLeapYear() {
        if ((this.year % 4) !== 0) {
            return false;
        } else if ((this.year % 100) !== 0) {
            return true;
        } else if ((this.year % 400) !== 0) {
            return false;
        }
        return true;
    }

    /**
     * 返回美国格式的日期，即月/日/年。
     * @returns {string}
     */
    toString() {
        return this.#month + "/" + this.#day + "/" + this.#year;
    }
// 表示持续时间的类
// 注意：这个类只能很好地处理正数持续时间
class Duration {
    // 年份
    #years;
    // 月份
    #months;
    // 天数
    #days;

    /**
     * 构建一个持续时间
     * @param {number} years
     * @param {number} months
     * @param {number} days
     */
    constructor(years, months, days) {
        this.#years = years;
        this.#months = months;
        this.#days = days;
        this.#fixRanges();
    }

    // 获取年份
    get years() {
        return this.#years;
    }

    // 获取月份
    get months() {
        return this.#months;
    }

    // 获取天数
    get days() {
        return this.#days;
    }

    // 克隆持续时间对象
    clone() {
        return new Duration(this.#years, this.#months, this.#days);
    }

    /**
     * 通过从提供的持续时间中减去年份、月份和天数来调整持续时间。
     * 这是一个简单的计算，假设所有月份都是30天。
     * @param {Duration} timeToRemove
     */
    remove(timeToRemove) {
        this.#years -= timeToRemove.years;
        this.#months -= timeToRemove.months;
        this.#days -= timeToRemove.days;
        this.#fixRanges();
    }

    /**
     * 将天数和月份调整到预期范围内。
     */
    #fixRanges() {
        if (this.#days < 0) {
            this.#days += DAYS_PER_IDEALISED_MONTH;
            this.#months--;
        }
        if (this.#months < 0) {
            this.#months += MONTHS_PER_YEAR;
            this.#years--;
        }
    }

    /**
     * 计算持续时间所涵盖的天数的近似值。
     * 计算假设所有年份都是365天，每个月都是30天，并在更多月份经过时添加额外的部分。
     * @returns {number}
     */
    getApproximateDays() {
        return (
            (this.#years * DAYS_PER_COMMON_YEAR)
            + (this.#months * DAYS_PER_IDEALISED_MONTH)
            + this.#days
            + Math.floor(this.#months / 2)
        );
    }
}
    /**
     * 返回格式化的持续时间，使用制表符分隔数值，即年\月\日。
     * @returns {string}
     */
    toString() {
        return this.#years + "\t" + this.#months + "\t" + this.#days;
    }

    /**
     * 计算两个日期之间的大致持续时间。
     * 这是一个简单的计算，假设所有月份都是30天。
     * @param {DateStruct} date1
     * @param {DateStruct} date2
     * @returns {Duration}
     */
    static between(date1, date2) {
        let years = date1.year - date2.year;
        let months = date1.month - date2.month;
        let days = date1.day - date2.day;
        return new Duration(years, months, days);
    }

    /**
     * 根据天数计算年、月和日的持续时间。
     * 这是一个简单的计算，假设所有月份都是30天。
     * @param dayCount 要转换为持续时间的总天数
     * @param factor   计算持续时间时应用的因子
     * @returns {Duration}
     */
    static fromDays(dayCount, factor) {
        let totalDays = Math.floor(factor * dayCount);
        const years = Math.floor(totalDays / DAYS_PER_COMMON_YEAR);
        totalDays -= years * DAYS_PER_COMMON_YEAR;
        const months = Math.floor(totalDays / DAYS_PER_IDEALISED_MONTH);
        const days = totalDays - (months * DAYS_PER_IDEALISED_MONTH);
        return new Duration(years, months, days);
    }
// 主控制部分
async function main() {
    /**
     * 读取日期，并提取日期信息。
     * 这里假设日期部分以逗号分隔，使用美国日期顺序，即 月,日,年。
     * @returns {Promise<DateStruct>}
     */
    async function inputDate() {
        // 等待输入日期字符串
        let dateString = await input();
        // 解析月份
        const month = parseInt(dateString);
        // 解析日期
        const day = parseInt(dateString.substr(dateString.indexOf(",") + 1));
        // 解析年份
        const year = parseInt(dateString.substr(dateString.lastIndexOf(",") + 1));
        // 返回一个包含年月日的 DateStruct 对象
        return new DateStruct(year, month, day);
    }

    /**
     * 获取星期几的文本表示。
     * @param {DateStruct} date
     * @returns {string}
     */
    function getDayOfWeekText(date) {
        // 获取星期几的数值
        const dayOfWeek = date.getDayOfWeek();
        let dayOfWeekText = "";
        // 根据星期几的数值返回对应的文本表示
        switch (dayOfWeek) {
            case 1:
                dayOfWeekText = "SUNDAY.";
                break;
            case 2:
                dayOfWeekText = "MONDAY.";
                break;
            case 3:
                dayOfWeekText = "TUESDAY.";
                break;
            case 4:
                dayOfWeekText = "WEDNESDAY.";
                break;
            case 5:
                dayOfWeekText = "THURSDAY.";
                break;
            case 6:
                // 如果是13号，返回特殊的文本表示，否则返回普通的星期五表示
                if (date.day === 13) {
                    dayOfWeekText = "FRIDAY THE THIRTEENTH---BEWARE!";
                } else {
                    dayOfWeekText = "FRIDAY.";
                }
                break;
            case 7:
                dayOfWeekText = "SATURDAY.";
                break;
        }
        return dayOfWeekText;
    }

    // 打印标题
    print(tab(32) + "WEEKDAY\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("WEEKDAY IS A COMPUTER DEMONSTRATION THAT\n");
    print("GIVES FACTS ABOUT A DATE OF INTEREST TO YOU.\n");
    print("\n");
    # 打印提示信息，要求输入当天的日期
    print("ENTER TODAY'S DATE IN THE FORM: 3,24,1979  ");
    # 获取用户输入的当天日期
    const today = await inputDate();
    # 打印提示信息，要求输入出生日期或其他感兴趣的日期
    print("ENTER DAY OF BIRTH (OR OTHER DAY OF INTEREST)");
    # 获取用户输入的日期
    const dateOfBirth = await inputDate();
    # 打印空行
    print("\n");
    # 检查输入的日期是否在1582年之后，如果不是，打印提示信息
    if (!dateOfBirth.isGregorianDate()) {
        print("NOT PREPARED TO GIVE DAY OF WEEK PRIOR TO X.XV.MDLXXXII.\n");
    }
    # 打印多个空行
    print("\n");
    print("\n");
    print("\n");
    print("\n");
    print("\n");
# 结束 main 函数的定义
}

# 调用 main 函数
main();
```