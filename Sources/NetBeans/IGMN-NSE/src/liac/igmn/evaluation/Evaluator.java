/**
 * =============================================================================
 * Federal University of Rio Grande do Sul (UFRGS)
 * Connectionist Artificial Intelligence Laboratory (LIAC)
 * Edigleison F. Carvalho - edigleison.carvalho@inf.ufrgs.br
 * =============================================================================
 * Copyright (c) 2012 Edigleison F. Carvalho, edigleison.carvalho at gmail dot com
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
package liac.igmn.evaluation;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import weka.core.Instances;
import liac.igmn.core.IGMN;
import liac.igmn.loader.Dataset;
import liac.igmn.util.MatrixUtil;

public class Evaluator {

	private ConfusionMatrix confusionMatrix;
	private boolean verbose;

	public Evaluator()
	{
	}

	public Evaluator(boolean verbose)
	{
		this.verbose = verbose;
	}

	public void crossValidation(IGMN model, Dataset dataset, int numFolds, int runs, boolean randomize)
	{
		confusionMatrix = new ConfusionMatrix(dataset.getClassesNames());

		Instances instances = dataset.getWekaDataset();
		int seed = 1;
		for (int run = 0; run < runs; run++) {
			if (randomize) {
				instances.randomize(new Random(seed));
				seed += 1;
			}

			if (verbose) {
				System.out.println("RUN: " + (run + 1));
			}

			for (int n = 0; n < numFolds; n++) {
				Instances train = instances.trainCV(numFolds, n);
				Instances test = instances.testCV(numFolds, n);

				SimpleMatrix trainData = MatrixUtil.instancesToMatrix(train);
				SimpleMatrix testData = MatrixUtil.instancesToMatrix(test);

				model.reset();

				if (verbose) {
					System.out.println("TRAINING FOLD: " + (n + 1));
				}

				model.train(trainData);

				if (verbose) {
					System.out.println("TESTING...");
				}

				SimpleMatrix testInputs = testData.extractMatrix(0, dataset.getInputSize(), 0, SimpleMatrix.END);
				SimpleMatrix testTargets = testData.extractMatrix(dataset.getInputSize(), dataset.getNumAttributes(), 0, SimpleMatrix.END);
				for (int i = 0; i < testInputs.numCols(); i++) {
					SimpleMatrix y = model.classify(testInputs.extractVector(false, i));
					SimpleMatrix target = testTargets.extractVector(false, i);

					int tInd = MatrixUtil.maxElementIndex(target);
					int yInd = MatrixUtil.maxElementIndex(y);

					confusionMatrix.addPrediction(tInd, yInd);
				}
			}
		}
		confusionMatrix.set(confusionMatrix.divide(runs));
	}

	public void leaveOneOut(IGMN model, Dataset dataset, int runs)
	{
		int numFolds = dataset.size();
		crossValidation(model, dataset, numFolds, runs, false);
	}

	public ConfusionMatrix getConfusionMatrix()
	{
		return confusionMatrix;
	}

	public void report()
	{
		System.out.println(confusionMatrix.toString("*** Matriz de Confusão ***"));

		System.out.printf("Total de Instâncias: %d\n", (int) confusionMatrix.total());
		System.out.printf("Acurácia: %.1f%%\n", confusionMatrix.accuracy() * 100);
		System.out.printf("Erro: %.1f%%\n", confusionMatrix.errorRate() * 100);

		for (int i = 0; i < confusionMatrix.size(); i++) {
			double accuracy = confusionMatrix.accuracy(i) * 100;
			System.out.printf("classe: %s - acurácia: %.1f%%\n", confusionMatrix.className(i), accuracy);
		}
	}
}
