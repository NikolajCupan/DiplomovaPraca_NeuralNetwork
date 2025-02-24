public class Math {
    public static double dotProduct(
            final Double[] left,
            final Double[] right
    ) {
        if (left.length != right.length) {
            throw new IllegalArgumentException("Length of left input [" + left.length + "] is not equal to length of right input [" + right.length + "]");
        }

        double result = 0.0;
        for (int i = 0; i < left.length; ++i) {
            result += left[i] * right[i];
        }

        return result;
    }
}
