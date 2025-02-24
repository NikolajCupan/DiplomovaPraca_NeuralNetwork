public class Helper {
    private static final long DECIMAL_POINTS_PRINTED = 3;

    public static <T extends Number> String stringifyArray(final T[] array) {
        final StringBuilder builder = new StringBuilder();
        builder.append("[   ");

        boolean first = true;
        for (final T element : array) {
            if (!first) {
                builder.append("   ");
            }

            builder.append(Helper.formatNumber(element));
            first = false;
        }

        builder.append("   ]");

        return builder.toString();
    }

    private static <T extends Number> String formatNumber(final T number) {
        if (number instanceof Double || number instanceof Float) {
            return String.format("%." + Helper.DECIMAL_POINTS_PRINTED + "f", number.doubleValue());
        } else {
            return number.toString();
        }
    }
}
