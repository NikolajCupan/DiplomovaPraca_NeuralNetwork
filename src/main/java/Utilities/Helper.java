package Utilities;

public class Helper {
    private static final int DECIMAL_POINTS_PRINTED = 3;

    public static String stringifyArray(final double[] array) {
        final StringBuilder builder = new StringBuilder();
        builder.append("[   ");

        boolean first = true;
        for (final double value : array) {
            if (!first) {
                builder.append("   ");
            }

            builder.append(Helper.formatNumber(value));
            first = false;
        }

        builder.append("   ]");

        return builder.toString();
    }

    public static String formatNumber(final double value) {
        return Helper.formatNumber(value, Helper.DECIMAL_POINTS_PRINTED);
    }

    public static String formatNumber(final double value, final int decimalPointsCount) {
        final String format = "%." + decimalPointsCount + "f";
        return String.format(format, value);
    }
}
