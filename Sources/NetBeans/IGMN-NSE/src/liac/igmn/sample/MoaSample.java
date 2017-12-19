/**
 * =============================================================================
 * Federal University of Rio Grande do Sul (UFRGS)
 * Connectionist Artificial Intelligence Laboratory (LIAC)
 * Jorge C. Chamby Diaz - jccdiaz@inf.ufrgs.br
 * =============================================================================
 * Copyright (c) 2017 Jorge C. Chamby Diaz, jchambyd at gmail dot com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * =============================================================================
 */
package liac.igmn.sample;

import moa.classifiers.Classifier;
import moa.core.TimingUtils;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.ArffLoader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Locale;
import liac.igmn.core.IGMNClassifier;
import moa.classifiers.meta.*;
import moa.classifiers.core.driftdetection.*;
import moa.classifiers.drift.SingleClassifierDrift;
import moa.streams.ArffFileStream;
import moa.classifiers.trees.HoeffdingAdaptiveTree;

/**
 *
 * @author liac01
 */
public class MoaSample {

	public static ArffFileStream getArffDataset(String nameFile) throws FileNotFoundException
	{
		ArffLoader loader = new ArffLoader(new FileReader(nameFile));
		ArffFileStream stream = new ArffFileStream(nameFile, loader.getStructure().numAttributes());
		return stream;
	}

	private ArrayList<String> mxGetDataSets()
	{
		ArrayList<String> namesDataSet = new ArrayList<>();
		namesDataSet.add("data/drift/2CDT.arff");		
		namesDataSet.add("data/drift/GEARS_2C_2D.arff");
		namesDataSet.add("data/drift/FG_2C_2D.arff");
		namesDataSet.add("data/drift/UG_2C_2D.arff");
		namesDataSet.add("data/drift/UG_2C_3D.arff");
		namesDataSet.add("data/drift/UG_2C_5D.arff");
		namesDataSet.add("data/drift/sea.arff");
		namesDataSet.add("data/drift/MG_2C_2D.arff");
		namesDataSet.add("data/drift/gaussian.arff");
		namesDataSet.add("data/drift/elecNormNew.arff");
		namesDataSet.add("data/drift/weather.arff");
		return namesDataSet;
	}

	public ArrayList<ClassifierTest> getResult(String name, int numChunks) throws FileNotFoundException
	{
		//Output File
		File file = new File("data.txt");
		int numInstances, lengthChunk;
		long evaluateStartTime;
		Instance trainInst;
		//Classifiers
		SingleClassifierDrift learnerEWMA = new SingleClassifierDrift();
		SingleClassifierDrift learnerDDM = new SingleClassifierDrift();
		SingleClassifierDrift learnerEDDM = new SingleClassifierDrift();
		learnerEWMA.driftDetectionMethodOption.setCurrentObject(new EWMAChartDM());
		learnerDDM.driftDetectionMethodOption.setCurrentObject(new DDM());
		learnerEDDM.driftDetectionMethodOption.setCurrentObject(new EDDM());

		//Load Dataset
		ArffFileStream stream = getArffDataset(name);
		stream.prepareForUse();
		numInstances = 0;
		//Counting # of instances in the Dataset
		while (stream.hasMoreInstances()) {
			stream.nextInstance();
			numInstances++;
		}
		stream.restart();

		//Calculate length for each chunck
		lengthChunk = numInstances / numChunks;

		//Adjust for use all instances
		/*if( numInstances > (lengthChunk * numChunks))
			lengthChunk++;*/
		ArrayList<ClassifierTest> learners = new ArrayList<>();
		//Selected algorithms
		learners.add(new ClassifierTest(new LearnNSE(), "LearnNSE"));
		learners.add(new ClassifierTest(learnerEWMA, "ECDD"));
		learners.add(new ClassifierTest(learnerDDM, "DDM"));
		learners.add(new ClassifierTest(learnerEDDM, "EDDM"));
		learners.add(new ClassifierTest(new OnlineAccuracyUpdatedEnsemble(), "OAUE"));
		learners.add(new ClassifierTest(new WeightedMajorityAlgorithm(), "DWM"));
		learners.add(new ClassifierTest(new HoeffdingAdaptiveTree(), "HAT"));
		learners.add(new ClassifierTest(new IGMNClassifier(), "IGMN"));
		
		//Prepare Learners
		for (int i = 0; i < learners.size(); i++) {
			//learners.get(i).learner.getOptions().setViaCLIString("-k 4"); 
			learners.get(i).learner.setModelContext(stream.getHeader());
			learners.get(i).learner.prepareForUse();
		}

		int numberSamples = 0;

		for (int i = 0; i < numChunks; i++) {
			//Evaluate and train instances by chunck
			for (int j = 0; j < lengthChunk && numberSamples < numInstances; j++) {
				trainInst = stream.nextInstance().instance;

				for (int k = 0; k < learners.size(); k++) {
					evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();

					if (learners.get(k).learner.correctlyClassifies(trainInst)) {
						learners.get(k).numCorrect++;
					} else {
						learners.get(k).numIncorrect++;
					}

					learners.get(k).learner.trainOnInstance(trainInst);
					learners.get(k).time += TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime);
				}
				numberSamples++;
			}

			//Register results by chunck	
			for (int k = 0; k < learners.size(); k++) {
				learners.get(k).accuracies.add(learners.get(k).mxCalculateAccuracy());
				learners.get(k).times.add(learners.get(k).time);
				//Reset values
				learners.get(k).time = 0;
				learners.get(k).numCorrect = 0;
				learners.get(k).numIncorrect = 0;
			}
		}

		for (int k = 0; k < learners.size(); k++) {
			learners.get(k).mxCalculateValues();
		}

		//Print chunck results
		try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(file, true)))) {
			pw.printf("DataSet: %s\n", name);
			pw.printf("%5s,", "Chunk");
			int j = 0;
			for (; j < learners.size() - 1; j++) {
				pw.printf("%8s,", learners.get(j).name);
			}
			pw.printf("%8s\n", learners.get(j).name);
			
			NumberFormat df = NumberFormat.getCurrencyInstance(Locale.US);
			((DecimalFormat)df).applyPattern("0.00");
			for (int i = 0; i < numChunks; i++){
				pw.printf("%5s,", "" + (i + 1));
				for (j = 0; j < learners.size() - 1; j++){
					pw.printf("%8s,", df.format(learners.get(j).accuracies.get(i)));
				}
				pw.printf("%8s\n", df.format(learners.get(j).accuracies.get(i)));
			}
		} catch (IOException ex) {
			ex.printStackTrace();
		}		
		return learners;
	}

	public void run() throws IOException
	{
		//Output File
		//PrintWriter outFile = new PrintWriter(new FileWriter("data.txt", false));
		ArrayList<Double> average = new ArrayList<>();
		ArrayList<String> names = new ArrayList<>();
		//Prepare Datasets
		ArrayList<String> namesDataSet = this.mxGetDataSets();

		for (String name : namesDataSet) {
			ArrayList<ClassifierTest> learners = getResult(name, 40);

			System.out.println("DATASET: " + name);
			System.out.printf("%12s%12s%12s%11s%11s\n", "Classifier", "Accuracy", "SD-Accu.", "Time", "SD-Time");
			System.out.println("----------------------------------------------------------");
			for (int i = 0; i < learners.size(); i++) {
				System.out.printf("%12s %11.2f% 11.2f %11.6f% 11.6f\n", learners.get(i).name,
						learners.get(i).accuracy,
						learners.get(i).sd_accuracy,
						learners.get(i).mean_time,
						learners.get(i).sd_time);

				if (average.size() < learners.size()) {
					average.add(learners.get(i).accuracy);
					names.add(learners.get(i).name);
				} else {
					average.set(i, average.get(i) + learners.get(i).accuracy);
				}
			}
		}

		// Sort Results
		Integer numbers[] = new Integer[average.size()];
		for (int i = 0; i < numbers.length; i++) {
			numbers[i] = i;
		}
		Arrays.sort(numbers, (final Integer o1, final Integer o2) -> Double.compare(average.get(o2), average.get(o1)));

		// Print Results
		System.out.println("\nAVERAGE RESULTS:");
		System.out.printf("%12s%12s\n", "Classifier", "Accuracy");
		System.out.println("------------------------");

		for (int i = 0; i < average.size(); i++) {
			System.out.printf("%12s %11.2f\n", names.get(numbers[i]), (double) average.get(numbers[i]) / namesDataSet.size());
		}
	}

	private double getResultIGMN(double tau, double delta, boolean print) throws Exception
	{
		int numInstances, lengthChunk, numChunks = 40;
		long evaluateStartTime;
		Instance trainInst;
		float sumAcc = 0;
		ArrayList<String> namesDataSet = this.mxGetDataSets();

		if (print) {
			System.out.printf("%18s%12s%12s%11s%11s\n", "DataSet", "Accuracy", "SD-Accu.", "Time-Train", "SD-Train");
			System.out.println("----------------------------------------------------------");
		}

		for (String name : namesDataSet) {
			//Load Dataset
			ArffFileStream stream = getArffDataset(name);
			stream.prepareForUse();
			numInstances = 0;
			//Counting # of instances in the Dataset
			while (stream.hasMoreInstances()) {
				stream.nextInstance();
				numInstances++;
			}
			stream.restart();

			//Calculate length for each chunck
			lengthChunk = numInstances / numChunks;

			//Adjust for use all instances
			/*if( numInstances > (lengthChunk * numChunks))
				lengthChunk++;*/
			IGMNClassifier loIGMN = new IGMNClassifier();
			loIGMN.delta.setValue(delta);
			loIGMN.tau.setValue(tau);

			ClassifierTest learner = new ClassifierTest(loIGMN, "IGMN");

			//learners.get(i).learner.getOptions().setViaCLIString("-k 4"); 
			learner.learner.setModelContext(stream.getHeader());
			learner.learner.prepareForUse();

			int numberSamples = 0;

			for (int i = 0; i < numChunks; i++) {
				//Evaluate and train instances by chunck
				for (int j = 0; j < lengthChunk && numberSamples < numInstances; j++) {
					trainInst = stream.nextInstance().instance;

					evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();

					if (learner.learner.correctlyClassifies(trainInst)) {
						learner.numCorrect++;
					} else {
						learner.numIncorrect++;
					}

					learner.learner.trainOnInstance(trainInst);
					learner.time += TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime);

					numberSamples++;
				}

				learner.accuracies.add(learner.mxCalculateAccuracy());
				learner.times.add(learner.time);
				//Reset values
				learner.time = 0;
				learner.numCorrect = 0;
				learner.numIncorrect = 0;
			}

			learner.mxCalculateValues();
			sumAcc += learner.accuracy;

			if (print) {
				System.out.printf("%18s %11.2f% 11.2f %11.6f%11.6f\n", name.substring(name.indexOf("test") + 5, name.indexOf(".arff")),
						learner.accuracy,
						learner.sd_accuracy,
						learner.mean_time,
						learner.sd_time);
			}
		}

		if (print) {
			System.out.printf("\nACCURACY: %.2f\n", sumAcc / namesDataSet.size());
		}

		return sumAcc / namesDataSet.size();
	}

	public void mxOptimizeParameters() throws Exception
	{
		double tau = 0.0001, delta, currentValue, bestValue = Double.NEGATIVE_INFINITY;
		double bestTau = -1, bestDelta = -1;
		for (int i = 0; i < 11; i++) {
			tau = tau * 2;
			delta = 0.0;
			for (int j = 0; j < 19; j++) {
				delta += 0.05;
				long start = TimingUtils.getNanoCPUTimeOfCurrentThread();
				currentValue = this.getResultIGMN(tau, delta, false);
				double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - start);
				if (currentValue > bestValue) {
					bestValue = currentValue;
					bestTau = tau;
					bestDelta = delta;
				}
				System.out.printf("Tau: %5.4f Delta: %5.4f Result: %5.3f Time: %9.6f - Best Value: %5.3f (Tau: %5.4f - Delta: %5.4f)\n", tau,
						delta,
						currentValue,
						time,
						bestValue,
						bestTau,
						bestDelta);

			}
		}
	}

	public static void main(String[] args) throws IOException, Exception
	{
		MoaSample exp = new MoaSample();
		//*/
		exp.run();
		/*/
		//exp.mxOptimizeParameters();
		exp.getResultIGMN(0.001, 0.5, true);
		//*/
	}
};

class ClassifierTest {

	public Classifier learner;
	public String name;
	public int numCorrect;
	public int numIncorrect;
	public ArrayList<Double> accuracies;
	public ArrayList<Double> times;
	public double time;
	public double accuracy;
	public double sd_accuracy;
	public double mean_time;
	public double sd_time;

	public ClassifierTest(Classifier learner, String name)
	{
		this.learner = learner;
		this.name = name;
		this.accuracies = new ArrayList<>();
		this.times = new ArrayList<>();
	}

	public double mxCalculateAccuracy()
	{
		return 100.0 * (double) this.numCorrect / (double) (this.numCorrect + this.numIncorrect);
	}

	public void mxCalculateValues()
	{
		double sum_accuracies = 0, sum_times = 0;
		for (int i = 0; i < this.accuracies.size(); i++) {
			sum_accuracies += this.accuracies.get(i);
			sum_times += this.times.get(i);
		}

		this.accuracy = sum_accuracies / this.accuracies.size();
		this.sd_accuracy = this.mxCalculateStandardDeviation(this.accuracies, this.accuracy);
		this.mean_time = sum_times / this.times.size();
		this.sd_time = this.mxCalculateStandardDeviation(this.times, this.mean_time);
	}

	public double mxCalculateStandardDeviation(ArrayList<Double> loData, double mean)
	{
		double sum = 0;

		for (int i = 0; i < loData.size(); i++) {
			sum += Math.pow(loData.get(i) - mean, 2.0);
		}

		return Math.sqrt(sum / loData.size());
	}
}
