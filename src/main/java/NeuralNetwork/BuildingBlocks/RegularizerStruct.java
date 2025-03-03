package NeuralNetwork.BuildingBlocks;

public class RegularizerStruct {
    private double biasesRegularizerL1;
    private double biasesRegularizerL2;
    private double weightsRegularizerL1;
    private double weightsRegularizerL2;

    public RegularizerStruct() {
        this.biasesRegularizerL1 = 0.0;
        this.biasesRegularizerL2 = 0.0;
        this.weightsRegularizerL1 = 0.0;
        this.weightsRegularizerL2 = 0.0;
    }

    public RegularizerStruct(
            final double biasesRegularizerL1,
            final double biasesRegularizerL2,
            final double weightsRegularizerL1,
            final double weightsRegularizerL2
    ) {
        this.biasesRegularizerL1 = biasesRegularizerL1;
        this.biasesRegularizerL2 = biasesRegularizerL2;
        this.weightsRegularizerL1 = weightsRegularizerL1;
        this.weightsRegularizerL2 = weightsRegularizerL2;
    }

    public double getBiasesRegularizerL1() {
        return this.biasesRegularizerL1;
    }

    public double getBiasesRegularizerL2() {
        return this.biasesRegularizerL2;
    }

    public double getWeightsRegularizerL1() {
        return this.weightsRegularizerL1;
    }

    public double getWeightsRegularizerL2() {
        return this.weightsRegularizerL2;
    }

    public void setBiasesRegularizerL1(final double biasesRegularizerL1) {
        this.biasesRegularizerL1 = biasesRegularizerL1;
    }

    public void setBiasesRegularizerL2(final double biasesRegularizerL2) {
        this.biasesRegularizerL2 = biasesRegularizerL2;
    }

    public void setWeightsRegularizerL1(final double weightsRegularizerL1) {
        this.weightsRegularizerL1 = weightsRegularizerL1;
    }

    public void setWeightsRegularizerL2(final double weightsRegularizerL2) {
        this.weightsRegularizerL2 = weightsRegularizerL2;
    }
}
