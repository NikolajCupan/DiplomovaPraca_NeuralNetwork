import java.util.ArrayList;
import java.util.List;

public class Batch {
    private final List<Double[]> batch;

    public Batch() {
        this.batch = new ArrayList<>();
    }

    public void addInputs(final Double[] inputs) {
        if (!this.batch.isEmpty()) {
            final long inputsSize = this.batch.getLast().length;

            if (inputs.length != inputsSize) {
                throw new IllegalArgumentException("New inputs size [" + inputs.length + "] is not equal to current inputs size [" + inputsSize + "]");
            }
        }

        this.batch.add(inputs);
    }

    public Double[] getInputs(final int index) {
        return this.batch.get(index);
    }

    public long getBatchSize() {
        return this.batch.size();
    }

    @Override
    public String toString() {
        final StringBuilder builder = new StringBuilder();
        builder.append("[\n");

        boolean first = true;
        for (final Double[] inputs : this.batch) {
            if (!first) {
                builder.append("\n");
            }

            builder.append("\t").append(Helper.stringifyArray(inputs));
            first = false;
        }

        builder.append("\n]");

        return builder.toString();
    }
}
