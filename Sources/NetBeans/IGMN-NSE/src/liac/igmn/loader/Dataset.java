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
package liac.igmn.loader;

import liac.igmn.util.MatrixUtil;
import org.ejml.simple.SimpleMatrix;
import weka.core.Instances;

public class Dataset {

	private Instances wekaDataset;
	private SimpleMatrix dataset;
	private int inputSize;
	private int numClasses;
	private String[] classesNames;

	public Instances getWekaDataset()
	{
		return wekaDataset;
	}

	public void setWekaDataset(Instances wekaDataset)
	{
		this.wekaDataset = wekaDataset;
		this.dataset = MatrixUtil.instancesToMatrix(wekaDataset);
	}

	public SimpleMatrix getDataset()
	{
		return dataset;
	}

	public void setDataset(SimpleMatrix dataset)
	{
		this.dataset = dataset;
	}

	public int getInputSize()
	{
		return inputSize;
	}

	public void setInputSize(int inputSize)
	{
		this.inputSize = inputSize;
	}

	public int getNumClasses()
	{
		return numClasses;
	}

	public void setNumClasses(int numClasses)
	{
		this.numClasses = numClasses;
	}

	public SimpleMatrix getDataRange()
	{
		SimpleMatrix min = new SimpleMatrix(dataset.numRows(), 1);
		min.set(Double.POSITIVE_INFINITY);
		SimpleMatrix max = new SimpleMatrix(dataset.numRows(), 1);
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

	public int size()
	{
		return wekaDataset.numInstances();
	}

	public int getNumAttributes()
	{
		return wekaDataset.numAttributes();
	}

	public void setClassesNames(String[] classesNames)
	{
		this.classesNames = classesNames;
	}

	public String[] getClassesNames()
	{
		return classesNames;
	}
}
