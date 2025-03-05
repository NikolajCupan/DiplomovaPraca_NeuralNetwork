package GUI;

import NeuralNetwork.BuildingBlocks.DataList;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;

public class GUI extends JFrame {
    private JPanel panel;
    private ChartPanel chartPanel;

    private JFreeChart customChart;
    private XYSeriesCollection data;

    public GUI() {
        setTitle("GUI");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setSize(1550, 750);
        setLocationRelativeTo(null);
        setContentPane(this.panel);
        setVisible(true);
    }

    public void createUIComponents() {
        final XYSeries predictedSeries = new XYSeries("predicted");
        final XYSeries targetSeries = new XYSeries("target");

        final XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(predictedSeries);
        dataset.addSeries(targetSeries);

        this.customChart = ChartFactory.createXYLineChart(
                "Chart",
                "X",
                "Y",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                false,
                false
        );

        this.data = dataset;
        this.chartPanel = new ChartPanel(this.customChart);
    }

    public void show(final DataList predicted, final DataList target) {
        final double[] predictedRawValues = predicted.getDataListRawValues();
        final double[] targetRawValues = target.getDataListRawValues();

        final XYSeries predictedSeries = this.data.getSeries("predicted");
        final XYSeries targetSeries = this.data.getSeries("target");

        for (int i = 0; i < predictedRawValues.length; ++i) {
            predictedSeries.add(i, predictedRawValues[i]);
            targetSeries.add(i, targetRawValues[i]);
        }

        this.customChart.fireChartChanged();
    }
}
