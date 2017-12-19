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

import java.io.BufferedReader;
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
import java.util.Locale;
import java.util.Random;
import liac.igmn.core.ClassifierIGMN;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.Utils;

public class WekaSample {

	public static BufferedReader readDataFile(String filename)
	{
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
		return inputReader;
	}

	private ArrayList<String> mxGetDataSets()
	{
		ArrayList<String> namesDataSet = new ArrayList<>();
		namesDataSet.add("data/static/iris.arff");
		namesDataSet.add("data/static/diabetes.arff");
		namesDataSet.add("data/static/glass.arff");
		namesDataSet.add("data/static/breastcancer.arff");
		namesDataSet.add("data/static/vehicle.arff");
		namesDataSet.add("data/static/segment.arff");
		namesDataSet.add("data/static/dermatology.arff");
		namesDataSet.add("data/static/ionosphere.arff");
		namesDataSet.add("data/static/sonar.arff");

		return namesDataSet;
	}

	private int getNumCorrects(ArrayList<Prediction> predictions)
	{
		int correct = 0;
		for (int i = 0; i < predictions.size(); i++) {
			NominalPrediction np = (NominalPrediction) predictions.get(i);
			if (np.predicted() == np.actual()) {
				correct++;
			}
		}
		return correct;
	}

	private static Instances[][] crossValidationSplit(Instances data, int numberOfFolds)
	{
		data.randomize(new Random(2));
		Instances[][] split = new Instances[2][numberOfFolds];

		for (int i = 0; i < numberOfFolds; i++) {
			split[0][i] = data.trainCV(numberOfFolds, i);
			split[1][i] = data.testCV(numberOfFolds, i);
		}
		return split;
	}

	private ArrayList<ClassifierTestWeka> getResult(String name, int numFolds) throws Exception
	{
		//Output File
		File file = new File("data_features.txt");
		ArrayList<ClassifierTestWeka> learners = new ArrayList<>();
		long evaluateStartTime;

		//Prepare instances
		BufferedReader datafile = readDataFile(name);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		// Do 10-split cross validation
		Instances[][] split = crossValidationSplit(data, numFolds);
		// Separate split into training and testing arrays
		Instances[] trainingSplits = split[0];
		Instances[] testingSplits = split[1];
		//Remove
		RandomForest rf = new RandomForest();
		rf.setNumIterations(500);
		// Selected algorithms		
		learners.add(new ClassifierTestWeka(new J48(), "J48"));
		learners.add(new ClassifierTestWeka(new PART(), "PART"));
		learners.add(new ClassifierTestWeka(new NaiveBayes(), "NaiveBayes"));
		learners.add(new ClassifierTestWeka(rf, "RandomForest"));
		learners.add(new ClassifierTestWeka(new ClassifierIGMN(), "IGMN"));

		for (int i = 0; i < learners.size(); i++) {
			for (int j = 0; j < trainingSplits.length; j++) {
				Evaluation evaluation = new Evaluation(trainingSplits[j]);
				// Training model
				evaluateStartTime = System.currentTimeMillis();
				learners.get(i).learner.buildClassifier(trainingSplits[j]);
				learners.get(i).time = (System.currentTimeMillis() - evaluateStartTime) / 1000.0;
				// Testing model
				evaluateStartTime = System.currentTimeMillis();
				evaluation.evaluateModel(learners.get(i).learner, testingSplits[j]);
				learners.get(i).times_test.add((System.currentTimeMillis() - evaluateStartTime) / 1000.0);
				// Get results of testing
				learners.get(i).numCorrect = getNumCorrects(evaluation.predictions());
				learners.get(i).numIncorrect = testingSplits[j].size() - learners.get(i).numCorrect;
				learners.get(i).accuracies.add(learners.get(i).mxCalculateAccuracy());

				double sumROC = 0;
				int num = 0;
				for (int k = 0; k < data.numClasses(); k++) {
					ThresholdCurve tc = new ThresholdCurve();
					Instances curve = tc.getCurve(evaluation.predictions(), k);
					double tmp = ThresholdCurve.getROCArea(curve);

					if (!Double.isNaN(tmp)) {
						sumROC += tmp;
						num++;
					}
				}
				learners.get(i).aucROCs.add(sumROC / num);
				learners.get(i).times.add(learners.get(i).time);
			}
		}

		for (int k = 0; k < learners.size(); k++) {
			learners.get(k).mxCalculateValues();
		}

		//Print chunck results
		try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(file, true)))) {
			pw.printf("%10s,", name);
			NumberFormat df = NumberFormat.getCurrencyInstance(Locale.US);
			((DecimalFormat)df).applyPattern("0.0000");
			int j = 0;
			for (j = 0; j < learners.size() - 1; j++){
				pw.printf("%8s,", df.format(learners.get(j).aucROC));
			}
			pw.printf("%8s\n", df.format(learners.get(j).aucROC));
			
		} catch (IOException ex) {
			ex.printStackTrace();
		}		
		
		return learners;
	}

	private void run() throws Exception
	{
		double avgAcc[] = null, avgAUC[] = null;
		String names[] = null;

		//Prepare Datasets
		ArrayList<String> namesDataSet = mxGetDataSets();

		for (String name : namesDataSet) {
			ArrayList<ClassifierTestWeka> learners = getResult(name, 10);

			if (avgAcc == null) {
				avgAcc = new double[learners.size()];
				avgAUC = new double[learners.size()];
				names = new String[learners.size()];
			}

			System.out.println("DATASET: " + name);
			System.out.printf("%12s%12s%12s%11s%11s%11s%11s%11s\n", "Classifier", "Accuracy", "SD-Accu.", "Time-Train", "SD-Train", "Time-Test", "SD-Test", "AUC-ROC");
			System.out.println("-------------------------------------------------------------------------------------------");
			for (int i = 0; i < learners.size(); i++) {
				System.out.printf("%12s %11.2f% 11.2f %11.6f%11.6f%11.6f%11.6f%11.6f\n", learners.get(i).name,
						learners.get(i).accuracy,
						learners.get(i).sd_accuracy,
						learners.get(i).mean_time,
						learners.get(i).sd_time,
						learners.get(i).mean_time_test,
						learners.get(i).sd_time_test,
						learners.get(i).aucROC);

				avgAcc[i] += learners.get(i).accuracy;
				avgAUC[i] += learners.get(i).aucROC;
				names[i] = learners.get(i).name;
			}
		}
		// Sort Results
		int indexAcc[] = Utils.sort(avgAcc);
		int indexAUC[] = Utils.sort(avgAUC);

		// Print Results
		System.out.println("\nAVERAGE RESULTS:");
		System.out.printf("%12s%12s\n", "Classifier", "Accuracy");
		System.out.println("------------------------");

		for (int i = avgAcc.length - 1; i >= 0; i--) {
			System.out.printf("%12s %11.4f\n", names[indexAcc[i]], avgAcc[indexAcc[i]] / namesDataSet.size());
		}

		System.out.printf("\n%12s%12s\n", "Classifier", "UAC-ROC");
		System.out.println("------------------------");

		for (int i = avgAUC.length - 1; i >= 0; i--) {
			System.out.printf("%12s %11.4f\n", names[indexAUC[i]], avgAUC[indexAUC[i]] / namesDataSet.size());
		}
	}

	private void getResultIGMN(double tau, double delta) throws Exception
	{
		ClassifierTestWeka learner;
		long evaluateStartTime;
		int numFolds = 10;
		float sumAcc = 0, sumAUC = 0;
		//Prepare Datasets
		ArrayList<String> namesDataSet = this.mxGetDataSets();

		System.out.printf("%12s%12s%12s%11s%11s%11s%11s%11S\n", "DataSet", "Accuracy", "SD-Accu.", "Time-Train", "SD-Train", "Time-Test", "SD-Test", "AUC-ROC");
		System.out.println("-------------------------------------------------------------------------------------------");

		for (String name : namesDataSet) {
			//Prepare instances
			BufferedReader datafile = readDataFile(name);
			Instances data = new Instances(datafile);
			data.setClassIndex(data.numAttributes() - 1);
			// Do 10-split cross validation
			Instances[][] split = crossValidationSplit(data, numFolds);
			// Separate split into training and testing arrays
			Instances[] trainingSplits = split[0];
			Instances[] testingSplits = split[1];

			// Selected algorithms
			learner = new ClassifierTestWeka(new ClassifierIGMN(tau, delta), "IGMN");

			for (int j = 0; j < trainingSplits.length; j++) {
				Evaluation evaluation = new Evaluation(trainingSplits[j]);
				// Training model
				evaluateStartTime = System.currentTimeMillis();
				learner.learner.buildClassifier(trainingSplits[j]);
				learner.time = (System.currentTimeMillis() - evaluateStartTime) / 1000.0;
				// Testing model
				evaluateStartTime = System.currentTimeMillis();
				evaluation.evaluateModel(learner.learner, testingSplits[j]);
				learner.times_test.add((System.currentTimeMillis() - evaluateStartTime) / 1000.0);
				// Get results of testing
				learner.numCorrect = getNumCorrects(evaluation.predictions());

				double sumROC = 0;
				int num = 0;
				for (int i = 0; i < data.numClasses(); i++) {
					ThresholdCurve tc = new ThresholdCurve();
					Instances curve = tc.getCurve(evaluation.predictions(), i);
					double tmp = ThresholdCurve.getROCArea(curve);

					if (!Double.isNaN(tmp)) {
						sumROC += tmp;
						num++;
					}
				}

				learner.aucROCs.add(sumROC / num);
				learner.numIncorrect = testingSplits[j].size() - learner.numCorrect;
				learner.accuracies.add(learner.mxCalculateAccuracy());
				learner.times.add(learner.time);
			}
			learner.mxCalculateValues();
			sumAcc += learner.accuracy;
			sumAUC += learner.aucROC;
			System.out.printf("%12s %11.2f% 11.2f %11.6f%11.6f%11.6f%11.6f%11.6f\n", name.substring(name.indexOf("/") + 1, name.indexOf(".arff")),
					learner.accuracy,
					learner.sd_accuracy,
					learner.mean_time,
					learner.sd_time,
					learner.mean_time_test,
					learner.sd_time_test,
					learner.aucROC);
		}

		System.out.printf("\nACCURACY: %.2f\n", sumAcc / namesDataSet.size());
		System.out.printf("AUC-ROC: %.2f\n", sumAUC * 100 / namesDataSet.size());
	}

	public static void main(String[] args) throws Exception
	{
		WekaSample exp = new WekaSample();
		exp.run();
		//exp.getResultIGMN(0.001, 0.5);
	}
};

class ClassifierTestWeka {

	public Classifier learner;
	public String name;
	public int numCorrect;
	public int numIncorrect;
	public ArrayList<Double> accuracies;
	public ArrayList<Double> aucROCs;
	public ArrayList<Double> times;
	public ArrayList<Double> times_test;
	public double time;
	public double accuracy;
	public double aucROC;
	public double sd_accuracy;
	public double mean_time;
	public double mean_time_test;
	public double sd_time;
	public double sd_time_test;

	public ClassifierTestWeka(Classifier learner, String name)
	{
		this.learner = learner;
		this.name = name;
		this.accuracies = new ArrayList<>();
		this.times = new ArrayList<>();
		this.times_test = new ArrayList<>();
		this.aucROCs = new ArrayList<>();
	}

	public double mxCalculateAccuracy()
	{
		return 100.0 * (double) this.numCorrect / (double) (this.numCorrect + this.numIncorrect);
	}

	public void mxCalculateValues()
	{
		double sum_accuracies = 0, sum_times = 0, sum_times_test = 0, sum_aucROC = 0;
		for (int i = 0; i < this.accuracies.size(); i++) {
			sum_accuracies += this.accuracies.get(i);
			sum_times += this.times.get(i);
			sum_times_test += this.times_test.get(i);
			sum_aucROC += this.aucROCs.get(i);
		}

		this.accuracy = sum_accuracies / this.accuracies.size();
		this.sd_accuracy = this.mxCalculateStandardDeviation(this.accuracies, this.accuracy);
		this.mean_time = sum_times / this.times.size();
		this.sd_time = this.mxCalculateStandardDeviation(this.times, this.mean_time);
		this.mean_time_test = sum_times_test / this.times_test.size();
		this.sd_time_test = this.mxCalculateStandardDeviation(this.times_test, this.mean_time_test);
		this.aucROC = sum_aucROC / this.aucROCs.size();
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
