public class Helper {
    public static <T> void printArray(final T[] array) {
        System.out.print("[   ");

        boolean first = true;
        for (final T element : array) {
            if (!first) {
                System.out.print("   ");
            }

            System.out.print(element);
            first = false;
        }

        System.out.println("   ]");
    }
}
