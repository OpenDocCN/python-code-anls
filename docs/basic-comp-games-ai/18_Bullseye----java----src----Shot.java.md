# `18_Bullseye\java\src\Shot.java`

```
/**
 * This class records the percentage chance of a given type of shot
 * scoring specific points
 * see Bullseye class points calculation method where its used
 */
public class Shot {

    double[] chances; // Array to store the percentage chance of scoring specific points for a given type of shot

    // Constructor to initialize the Shot object with an array of doubles representing the chances of scoring specific points
    Shot(double[] shots) {
        chances = new double[shots.length]; // Initialize the chances array with the same length as the input array
        System.arraycopy(shots, 0, chances, 0, shots.length); // Copy the elements from the input array to the chances array
    }

    // Method to get the percentage chance of scoring specific points for a given index
    public double getShot(int index) {
        return chances[index]; // Return the percentage chance of scoring specific points for the given index
    }
}
```