package liac.igmn.sample;

import org.ejml.simple.SimpleMatrix;
import com.panayotis.gnuplot.JavaPlot;
import com.panayotis.gnuplot.plot.DataSetPlot;
import com.panayotis.gnuplot.style.Style;
import liac.igmn.core.IGMN;
import liac.igmn.util.MatrixUtil;

public class RegressionSample {

	public static void main(String[] args)
	{
		// gen some data
		int i = 0;
		SimpleMatrix dataset = new SimpleMatrix(63, 2);
		for (float x = 0; x <= 2 * Math.PI; x += 0.1f) {
			dataset.set(i, 0, x);
			dataset.set(i, 1, Math.sin(x));
			i++;
		}

		// plot sin
		JavaPlot p = new JavaPlot();

		p.setTitle("sin");
		p.getAxis("x").setLabel("X axis");
		p.getAxis("y").setLabel("Y axis");

		DataSetPlot datasetPlot = new DataSetPlot(MatrixUtil.toDouble(dataset));
		datasetPlot.getPlotStyle().setStyle(Style.LINES);
		datasetPlot.setTitle("sin(x)");
		p.addPlot(datasetPlot);

		// model setup
		SimpleMatrix range = new SimpleMatrix(new double[][]{{2, 2}});
		IGMN igmn = new IGMN(range.transpose(), 0.1, 0.3);

		// train model
		igmn.train(dataset.transpose());

		// test model
		SimpleMatrix out = dataset.copy();
		dataset = dataset.extractMatrix(0, SimpleMatrix.END, 0, 1);
		for (i = 0; i < dataset.numRows(); i++) {
			SimpleMatrix y = igmn.recall(dataset.extractVector(true, i));
			out.set(i, 1, y.get(0));
		}

		// plot regression
		datasetPlot = new DataSetPlot(MatrixUtil.toDouble(out));
		datasetPlot.getPlotStyle().setStyle(Style.LINES);
		datasetPlot.setTitle("igmn");
		p.addPlot(datasetPlot);

		p.plot();
	}
}
