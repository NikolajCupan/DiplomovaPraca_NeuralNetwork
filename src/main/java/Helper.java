public class Helper {
    private static final long DECIMAL_POINTS_PRINTED = 3;

    public static <T extends Number> void printArray(final T[] array) {
        System.out.print("[   ");

        boolean first = true;
        for (final T element : array) {
            if (!first) {
                System.out.print("   ");
            }

            System.out.print(Helper.formatNumber(element));
            first = false;
        }

        System.out.println("   ]");
    }

    private static <T extends Number> String formatNumber(final T number) {
        if (number instanceof Double || number instanceof Float) {
            return String.format("%." + Helper.DECIMAL_POINTS_PRINTED + "f", number.doubleValue());
        } else {
            return number.toString();
        }
    }
}
