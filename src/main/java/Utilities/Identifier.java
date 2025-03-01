package Utilities;

public class Identifier {
    private Identifier() {}

    private static long CURRENT_ID = -1;

    public static long getId() {
        ++Identifier.CURRENT_ID;
        return Identifier.CURRENT_ID;
    }
}
