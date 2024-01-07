# `basic-computer-games\95_Weekday\javascript\weekday.js`

```

// WEEKDAY
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

/**
 * Print given string to the end of the "output" element.
 * @param str - The string to be printed
 */
function print(str) {
    document.getElementById("output").appendChild(document.createTextNode(str));
}

/**
 * Obtain user input
 * @returns {Promise<String>} - A promise that resolves to the user input
 */
function input() {
    return new Promise(function (resolve) {
        const input_element = document.createElement("INPUT");

        print("? "); // Print a question mark to prompt the user for input
        input_element.setAttribute("type", "text");
        input_element.setAttribute("length", "50");
        document.getElementById("output").appendChild(input_element); // Append the input element to the output
        input_element.focus(); // Set focus to the input element
        input_element.addEventListener("keydown", function (event) {
            if (event.keyCode === 13) { // If the Enter key is pressed
                const input_str = input_element.value; // Get the value of the input
                document.getElementById("output").removeChild(input_element); // Remove the input element from the output
                print(input_str); // Print the user input
                print("\n"); // Print a new line
                resolve(input_str); // Resolve the promise with the user input
            }
        });
    });
}

/**
 * Create a string consisting of the given number of spaces
 * @param spaceCount - The number of spaces to create
 * @returns {string} - A string consisting of the specified number of spaces
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
     * @param {number} year - The year
     * @param {number} month - The month
     * @param {number} day - The day
     */
    constructor(year, month, day) {
        this.#year = year;
        this.#month = month;
        this.#day = day;
    }

    // Getters for year, month, and day

    /**
     * Determine if the date could be a Gregorian date.
     * @returns {boolean} - true if date could be Gregorian; otherwise false.
     */
    isGregorianDate() {
        // Logic to determine if the date could be a Gregorian date
    }

    /**
     * The following performs a hash on the day parts which guarantees that
     * 1. different days will return different numbers
     * 2. the numbers returned are ordered.
     * @returns {number} - The normalized day
     */
    getNormalisedDay() {
        // Logic to calculate the normalized day
    }

    /**
     * Determine the day of the week.
     * @returns {number} - Value between 1 and 7 representing Sunday to Saturday.
     */
    getDayOfWeek() {
        // Logic to determine the day of the week
    }

    // Other methods for date manipulation
}

/**
 * Duration representation.
 * Note: this class only handles positive durations well
 */
class Duration {
    #years;
    #months;
    #days;

    /**
     * Build a Duration
     * @param {number} years - The years
     * @param {number} months - The months
     * @param {number} days - The days
     */
    constructor(years, months, days) {
        this.#years = years;
        this.#months = months;
        this.#days = days;
        this.#fixRanges();
    }

    // Getters for years, months, and days

    /**
     * Adjust Duration by removing years, months and days from supplied Duration.
     * @param {Duration} timeToRemove - The duration to remove
     */
    remove(timeToRemove) {
        // Logic to remove years, months, and days from the duration
    }

    // Other methods for duration manipulation
}

// Main control section
}

main();

```