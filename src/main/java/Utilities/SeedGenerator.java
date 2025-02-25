package Utilities;

import java.util.Random;

public class SeedGenerator {
    private final Random random;

    public SeedGenerator(final long seed) {
        this.random = new Random(seed);
    }

    public long getSeed() {
        return this.random.nextLong();
    }
}
