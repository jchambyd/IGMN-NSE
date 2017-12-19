/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package liac.igmn.sample;

import com.panayotis.gnuplot.GNUPlotException;
import com.panayotis.gnuplot.JavaPlot;
import com.panayotis.gnuplot.plot.DataSetPlot;
import com.panayotis.gnuplot.style.Style;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import liac.igmn.core.IGMN;
import liac.igmn.util.MatrixUtil;
import org.ejml.simple.SimpleEVD;
import org.ejml.simple.SimpleMatrix;

/**
 *
 * @author liac01
 */
public class ClusteringSample {

	public static void main(String[] args)
	{
		try {
			SimpleMatrix dataset = loadDataset("data/dist2.txt");

			IGMN igmn = new IGMN(getDataRange(dataset.transpose()), 0.05, 0.5);

			igmn.train(dataset.transpose());

			SimpleMatrix output = igmn.cluster(dataset.transpose());

			List<SimpleMatrix> clusters = new ArrayList<>();

			for (int i = 0; i < igmn.getSize(); i++) {
				clusters.add(new SimpleMatrix(0, 2));
			}

			for (int i = 0; i < dataset.numRows(); i++) {
				int numCluster = (int) output.get(i, 0);
				int numElements = clusters.get(numCluster).numRows();
				clusters.get(numCluster).getMatrix().reshape(numElements + 1, 2, true);
				clusters.get(numCluster).set(numElements, 0, dataset.get(i, 0));
				clusters.get(numCluster).set(numElements, 1, dataset.get(i, 1));
			}

			JavaPlot p = new JavaPlot();

			p.setTitle("dist");
			p.getAxis("x").setLabel("X axis");
			p.getAxis("y").setLabel("Y axis");

			DataSetPlot datasetPlot;

			for (int i = 0; i < clusters.size(); i++) {
				if (clusters.get(i).numRows() == 0) {
					continue;
				}
				datasetPlot = new DataSetPlot(MatrixUtil.toDouble(clusters.get(i)));
				datasetPlot.getPlotStyle().setStyle(Style.POINTS);
				datasetPlot.setTitle("cluster" + i);
				p.addPlot(datasetPlot);
			}

			SimpleMatrix meansClusters = new SimpleMatrix(clusters.size(), 2);

			int count = 0;
			for (SimpleMatrix means : igmn.getMeans()) {
				meansClusters.set(count, 0, means.get(0, 0));
				meansClusters.set(count, 1, means.get(1, 0));
				count++;
			}

			datasetPlot = new DataSetPlot(MatrixUtil.toDouble(meansClusters));
			datasetPlot.getPlotStyle().setStyle(Style.POINTS);
			datasetPlot.setTitle("Means");
			p.addPlot(datasetPlot);

			//Print Covariance Matrix
			List<SimpleMatrix> invCovs = igmn.getInvCovs();
			List<SimpleMatrix> means = igmn.getMeans();

			for (int i = 0; i < invCovs.size(); i++) {
				SimpleMatrix ellipsoid = mxGetEllipsoid(invCovs.get(i).invert(), means.get(i), 2, 50);
				datasetPlot = new DataSetPlot(MatrixUtil.toDouble(ellipsoid.transpose()));
				datasetPlot.getPlotStyle().setStyle(Style.LINES);
				datasetPlot.setTitle("Ellipsoid " + i);
				p.addPlot(datasetPlot);
			}

			p.plot();
		} catch (GNUPlotException e) {
		}
	}

	public static SimpleMatrix loadDataset(String tsName)
	{
		File file = new File(tsName);
		Scanner loScan;
		String[] laTokens;
		SimpleMatrix dataset = new SimpleMatrix(0, 0);
		float[] laX;
		int lnCount = 0;

		try {
			loScan = new Scanner(file);
			while (loScan.hasNextLine()) {
				String lsLine = loScan.nextLine().trim();

				if (lsLine.equals("")) {
					continue;
				}

				lnCount++;

				if (lsLine.contains(",")) {
					lsLine = lsLine.replace(",", ".");
				}

				laTokens = lsLine.replaceAll(" +", " ").split(" ");

				dataset.getMatrix().reshape(lnCount, laTokens.length, true);

				for (int i = 0; i < laTokens.length; i++) {
					dataset.set(lnCount - 1, i, Double.parseDouble(laTokens[i]));
				}
			}
		} catch (FileNotFoundException e) {
		}
		return dataset;
	}

	public static SimpleMatrix getDataRange(SimpleMatrix dataset)
	{
		SimpleMatrix min = new SimpleMatrix(dataset.numRows(), 1);
		SimpleMatrix max = new SimpleMatrix(dataset.numRows(), 1);
		min.set(Double.POSITIVE_INFINITY);
		max.set(Double.NEGATIVE_INFINITY);

		for (int i = 0; i < dataset.numCols(); i++) {
			for (int j = 0; j < dataset.numRows(); j++) {
				double value = dataset.get(j, i);
				if (value < min.get(j, 0)) {
					min.set(j, 0, value);
				}
				if (value > max.get(j, 0)) {
					max.set(j, 0, value);
				}
			}
		}
		return max.minus(min);
	}

	public static SimpleMatrix mxGetEllipsoid(SimpleMatrix cov, SimpleMatrix mean, double sd, int ntps)
	{
		SimpleMatrix ap = new SimpleMatrix(2, ntps);

		double twoPI = 2 * Math.PI;

		for (int i = 0; i < ntps; i++) {
			double lnValue = (twoPI / (double) (ntps - 1)) * i;
			ap.set(0, i, Math.cos(lnValue));
			ap.set(1, i, Math.sin(lnValue));
		}

		SimpleEVD loEig = cov.eig();
		int lnNumEigen = loEig.getNumberOfEigenvalues();

		SimpleMatrix D = new SimpleMatrix(lnNumEigen, lnNumEigen);
		SimpleMatrix V = new SimpleMatrix(lnNumEigen, lnNumEigen);

		for (int i = 0; i < lnNumEigen; i++) {
			D.set(i, i, loEig.getEigenvalue(i).getReal());
			for (int j = 0; j < lnNumEigen; j++) {
				V.set(j, i, loEig.getEigenVector(i).get(j));
			}
		}

		// Convert variance to sdwidth*sd
		D = sqrtMatrix(D);
		D = D.scale(sd);

		SimpleMatrix bp = V.mult(D).mult(ap);

		for (int i = 0; i < ap.numCols(); i++) {
			bp.set(0, i, bp.get(0, i) + mean.get(0, 0));
			bp.set(1, i, bp.get(1, i) + mean.get(1, 0));
		}

		return bp;
	}

	public static SimpleMatrix sqrtMatrix(SimpleMatrix m)
	{
		for (int i = 0; i < m.numRows(); i++) {
			for (int j = 0; j < m.numCols(); j++) {
				m.set(i, j, Math.sqrt(m.get(i, j)));
			}
		}
		return m;
	}
}
