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

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.Measurement;
import org.ejml.simple.SimpleMatrix;

public class IGMNClassifier extends AbstractClassifier implements MultiClassClassifier{

	private static final long serialVersionUID = 1L;

	public IntOption vMin = new IntOption("idadeMinima", 'v',
			"Idade minima de um componente para poder ser detectado como spurious.", 1,
			1, Integer.MAX_VALUE);

	public FloatOption spMin = new FloatOption("spuriousMinimo", 's',
			"Ativacao minima de um componente para nao ser considerado como spurious.", 1.0,
			1.0, 1.00);

	public FloatOption tau = new FloatOption("tau", 't',
			"Distancia minima aceitavel pelo componente.", 0.001,
			0.0001, 1.00);

	public FloatOption delta = new FloatOption("delta", 'd',
			"Tamanho inicial da matriz de covariancia.", 0.5,
			0.005, 0.99);

	protected IGMN poIGMN;

	public IGMNClassifier()
	{

	}

	@Override
	public boolean isRandomizable()
	{
		return false;
	}

	@Override
	public void resetLearningImpl()
	{

	}

	@Override
	public void trainOnInstanceImpl(Instance inst)
	{
		if (this.poIGMN == null) {
			int dimention = inst.numInputAttributes() + inst.numClasses();

			double lvMin = (this.vMin.getValue() == this.vMin.getMinValue()) ? dimention * 2 : this.vMin.getValue();
			double lspMin = (this.spMin.getValue() == this.spMin.getMinValue()) ? dimention + 1 : this.spMin.getValue();

			//Call tha class IGMN
			this.poIGMN = new IGMN(getRangeNormalized(dimention), tau.getValue(), delta.getValue(), lspMin, lvMin);
			this.poIGMN.setNumClass(inst.numClasses());
		}

		double[][] classValue = new double[1][inst.numClasses()];
		classValue[0][(int) inst.classValue()] = 1;
		this.poIGMN.learn(this.InstanceToMatrix(inst, classValue[0]));
	}

	@Override
	public double[] getVotesForInstance(Instance inst)
	{
		double[] result = new double[inst.numClasses()];

		if (this.poIGMN != null) {
			SimpleMatrix loInstance = this.InstanceToMatrix(inst).extractMatrix(0, inst.numInputAttributes(), 0, SimpleMatrix.END);
			SimpleMatrix clasify = this.poIGMN.classify(loInstance);

			result = new double[clasify.numCols()];
			for (int i = 0; i < clasify.numCols(); i++) {
				result[i] = clasify.get(i);
			}
		}
		return result;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent)
	{
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl()
	{
		return null;
	}

	private SimpleMatrix getRangeNormalized(int dimention)
	{
		SimpleMatrix rangeData = new SimpleMatrix(dimention, 1);

		for (int i = 0; i < dimention; i++) {
			rangeData.set(i, 0, 1.0);
		}

		return rangeData;
	}

	private SimpleMatrix InstanceToMatrix(Instance instance)
	{
		double[][] matrix = new double[1][];
		matrix[0] = instance.toDoubleArray();
		return (new SimpleMatrix(matrix)).transpose();
	}

	private SimpleMatrix InstanceToMatrix(Instance toInstance, double[] taClass)
	{
		double[] loInstance = toInstance.toDoubleArray();
		double[][] matrix = new double[1][toInstance.numInputAttributes() + taClass.length];
		int lnCont = 0;

		for (int i = 0; i < toInstance.numInputAttributes(); i++) {
			matrix[0][lnCont++] = loInstance[i];
		}

		for (int i = 0; i < taClass.length; i++) {
			matrix[0][lnCont++] = taClass[i];
		}

		return (new SimpleMatrix(matrix)).transpose();
	}
}
