package Utilities;

public class CustomMath {
    public static double dotProduct(
            final Double[] left,
            final Double[] right
    ) {
        if (left.length != right.length) {
            throw new IllegalArgumentException("Size of left input [" + left.length + "] is not equal to size of right input [" + right.length + "]");
        }

        double result = 0.0;
        for (int i = 0; i < left.length; ++i) {
            result += left[i] * right[i];
        }

        return result;
    }
}
