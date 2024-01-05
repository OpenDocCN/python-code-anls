# `18_Bullseye\java\src\Player.java`

```
    /**
     * Constructor for Player class, initializes the name and score of the player
     * @param name - the name of the player
     */
    Player(String name) {
        this.name = name; // Assign the input name to the player's name
        this.score = 0; // Initialize the player's score to 0
    }

    /**
     * Method to add the input score to the player's current score
     * @param score - the score to be added
     */
    public void addScore(int score) {
        this.score += score; // Add the input score to the player's current score
    }

    /**
     * Method to get the name of the player
     * @return the name of the player
     */
    public String getName() {
        return name; // Return the name of the player
    }
        return name;  # 返回name变量的值
    }

    public int getScore() {  # 定义一个公共的整型方法getScore
        return score;  # 返回score变量的值
    }
}
```