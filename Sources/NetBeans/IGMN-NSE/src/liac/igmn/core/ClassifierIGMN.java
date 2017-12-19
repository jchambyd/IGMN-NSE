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
package liac.igmn.core;

import java.io.Serializable;
import java.util.Enumeration;
import java.util.Vector;
import liac.igmn.util.MatrixUtil;
import org.ejml.simple.SimpleMatrix;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.UpdateableClassifier;
import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.TechnicalInformation;
import weka.core.Utils;

/**
 *
 * @author liac01
 */
public class ClassifierIGMN extends AbstractClassifier implements UpdateableClassifier, CapabilitiesHandler, Serializable, OptionHandler {

	private IGMN poIGMN;

	public ClassifierIGMN()
	{
		this.poIGMN = new IGMN(0.001, 0.5);
	}

	public ClassifierIGMN(double tau, double delta)
	{
		this.poIGMN = new IGMN(tau, delta);
	}

	@Override
	public void updateClassifier(Instance instnc) throws Exception
	{
		instnc = this.fillMissing(instnc);
		SimpleMatrix data = this.mxInstanceToMatrix(instnc);
		// Updating parameters of IGMN
		this.poIGMN.updateDataRange(MatrixUtil.getDataRange(data));
		// Train IGMN
		this.poIGMN.train(data);
	}

	@Override
	public Capabilities getCapabilities()
	{
		Capabilities result = new Capabilities(this);
		result.disableAll();
		result.enable(Capabilities.Capability.BINARY_ATTRIBUTES);
		result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capabilities.Capability.MISSING_VALUES);
		result.enable(Capabilities.Capability.BINARY_CLASS);
		result.enable(Capabilities.Capability.NOMINAL_CLASS);
		result.enable(Capabilities.Capability.NUMERIC_CLASS);
		result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
		result.setMinimumNumberInstances(0);
		return result;
	}

	@Override
	public Enumeration<Option> listOptions()
	{
		Vector newVector = new Vector(4);

		newVector.addElement(new Option("\tVerbose (for debug purposes)", "V", 0, "-V"));
		newVector.addElement(new Option("\tDelta: Initial size of components.", "D", 1, "-D <proportion of dataset standard deviations>"));
		newVector.addElement(new Option("\tBeta: Threshold for creating new components (higher = more components).", "B", 1, "-B <percentile>"));
		newVector.addElement(new Option("\tAttribute Selection: Amount of partial correlation preserved, from 0 to 1.", "AS", 1, "-AS"));

		return newVector.elements();
	}

	@Override
	public void setOptions(String[] strings) throws Exception
	{
		String deltaString = Utils.getOption('D', strings);
		if (deltaString.length() != 0) {

		}

		String betaString = Utils.getOption('B', strings);
		if (betaString.length() != 0) {

		}

		String attrString = Utils.getOption("AS", strings);
		if (attrString.length() != 0) {

		}
	}

	@Override
	public String[] getOptions()
	{
		String[] options = new String[10];
		int current = 0;

		while (current < options.length) {
			options[(current++)] = "";
		}
		return options;
	}

	@Override
	public void buildClassifier(Instances instances) throws Exception
	{
		this.poIGMN.reset();
		instances = new Instances(instances);
		this.getCapabilities().testWithFail(instances);
		instances = this.fillMissing(instances);

		SimpleMatrix data = this.mxInstancesToMatrix(instances);
		// Updating parameters of IGMN
		this.poIGMN.updateDataRange(MatrixUtil.getDataRange(data));
		// Train IGMN
		this.poIGMN.train(data);
	}

	@Override
	public double classifyInstance(Instance instnc) throws Exception
	{
		if (this.poIGMN != null) {
			double[][] matrix = new double[1][];
			matrix[0] = instnc.toDoubleArray();
			// Extract only the features
			SimpleMatrix loInstance = (new SimpleMatrix(matrix)).transpose().extractMatrix(0, instnc.numAttributes() - 1, 0, SimpleMatrix.END);
			// Recall for calculate the pediction
			SimpleMatrix out = this.poIGMN.recall(loInstance);
			// Return the class index
			return MatrixUtil.maxElementIndex(out);
		} else {
			return 0;
		}
	}

	@Override
	public double[] distributionForInstance(Instance instnc) throws Exception
	{
		double[] result = new double[instnc.numClasses()];
		if (this.poIGMN != null) {
			double[][] matrix = new double[1][];
			matrix[0] = instnc.toDoubleArray();
			// Extract only the features
			SimpleMatrix loInstance = (new SimpleMatrix(matrix)).transpose().extractMatrix(0, instnc.numAttributes() - 1, 0, SimpleMatrix.END);
			// Recall for calculate the pediction
			SimpleMatrix classify = this.poIGMN.recall(loInstance);
			double sum = 0;

			for (int i = 0; i < result.length; i++) {
				result[i] = classify.get(i);
				if (result[i] < 0.0D) {
					result[i] = 0.0D;
				} else if (result[i] > 1.0D) {
					result[i] = 1.0D;
				}
				sum += result[i];
			}
			for (int i = 0; i < result.length; i++) {
				result[i] /= sum;
			}
		}
		return result;
	}

	public String globalInfo()
	{
		return "IGMN Classifier." + getTechnicalInformation().toString();
	}

	public TechnicalInformation getTechnicalInformation()
	{
		TechnicalInformation result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
		result.setValue(TechnicalInformation.Field.AUTHOR, "Jorge C. Chamby");
		result.setValue(TechnicalInformation.Field.YEAR, "2017");
		result.setValue(TechnicalInformation.Field.TITLE, "IGMN");
		return result;
	}

	public Instances fillMissing(Instances ins)
	{
		ins = new Instances(ins);
		int D = ins.numAttributes();
		int N = ins.size();
		for (int i = 0; i < N; i++) {
			ins.set(i, fillMissing(ins.get(i)));
		}
		return ins;
	}

	public Instance fillMissing(Instance ins)
	{
		int D = ins.numAttributes();
		for (int j = 0; j < D; j++) {
			if ((ins.isMissing(j)) && (j != ins.classIndex())) {
				ins.setValue(j, ins.dataset().meanOrMode(j));
			}
		}
		return ins;
	}

	private SimpleMatrix mxInstancesToMatrix(Instances instances)
	{
		int numFeatures = instances.numAttributes() - 1;
		int numInstances = instances.size();
		int numClasses = instances.numClasses();

		double[][] result = new double[numInstances][numFeatures + numClasses];

		for (int i = 0; i < numInstances; i++) {
			// Read features
			for (int j = 0; j < numFeatures; j++) {
				result[i][j] = instances.get(i).value(j);
			}
			// Read class
			result[i][numFeatures + (int) instances.get(i).classValue()] = 1;
		}
		return (new SimpleMatrix(result)).transpose();
	}

	private SimpleMatrix mxInstanceToMatrix(Instance instance)
	{
		int numFeatures = instance.numAttributes() - 1;
		int numClasses = instance.numClasses();

		double[][] result = new double[1][numFeatures + numClasses];

		// Read features
		for (int j = 0; j < numFeatures; j++) {
			result[0][j] = instance.value(j);
		}
		// Read class
		result[0][numFeatures + (int) instance.classValue()] = 1;

		return (new SimpleMatrix(result)).transpose();
	}

}
