package Utilities;

import java.util.Random;

public class SeedGenerator {
    private static boolean INITIALIZED = false;
    private static Random RANDOM;

    public static void initialize(final long seed) {
        if (SeedGenerator.INITIALIZED) {
            throw new IllegalStateException("Seed generator is already initialized");
        }

        SeedGenerator.RANDOM = new Random(seed);
        SeedGenerator.INITIALIZED = true;
    }

    public static long getSeed() {
        if (!SeedGenerator.INITIALIZED) {
            throw new IllegalArgumentException("Seed generator is not initialized");
        }

        return SeedGenerator.RANDOM.nextLong();
    }
}
