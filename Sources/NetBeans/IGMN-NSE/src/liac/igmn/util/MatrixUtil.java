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
package liac.igmn.util;

import java.util.ArrayList;
import org.ejml.simple.SimpleMatrix;

import weka.core.Instances;

public class MatrixUtil {

	private MatrixUtil()
	{
	}

	/**
	 *
	 * @param x matrix de entrada
	 * @return indice do maior elemento da matriz x
	 */
	public static int maxElementIndex(SimpleMatrix x)
	{
		double data[] = x.getMatrix().getData();
		double max = data[0];
		int idx = 0;
		for (int i = 1; i < data.length; i++) {
			if (data[i] > max) {
				max = data[i];
				idx = i;
			}
		}

		return idx;
	}

	/**
	 *
	 * @param data instancias carregadas pelo weka
	 * @return matriz correspondentes as instancias
	 */
	public static SimpleMatrix instancesToMatrix(Instances data)
	{
		double dataset[][] = new double[data.numAttributes()][data.numInstances()];
		for (int i = 0; i < data.numInstances(); i++) {
			double[] B = data.instance(i).toDoubleArray();
			for (int j = 0; j < data.numAttributes(); j++) {
				dataset[j][i] = B[j];
			}
		}

		return new SimpleMatrix(dataset);
	}

	/**
	 * Remove um elemento da matriz
	 *
	 * @param m matriz de entrada
	 * @param idx indice do elemento a ser removido
	 * @return nova matriz com o elemento removido
	 */
	public static SimpleMatrix removeElement(SimpleMatrix m, int idx)
	{
		double data[] = m.getMatrix().getData();
		double[] newData = new double[data.length - 1];
		System.arraycopy(data, 0, newData, 0, idx);
		System.arraycopy(data, idx + 1, newData, idx, data.length - idx - 1);
		m.getMatrix().setData(newData);
		m.getMatrix().reshape(data.length - 1, 1, true);
		return m;
	}

	/**
	 * Cria matriz diagonal com os elementos da matriz de entrada
	 *
	 * @param m matriz de entrada
	 * @return matriz diagonal
	 */
	public static SimpleMatrix diag(SimpleMatrix m)
	{
		SimpleMatrix diag = new SimpleMatrix(m.getNumElements(), m.getNumElements());
		for (int l = 0; l < m.getNumElements(); l++) {
			diag.set(l, l, m.get(l));
		}

		return diag;
	}

	/**
	 * Testa se duas matrizes sao iguais
	 *
	 * @param A matriz de entrada
	 * @param B matriz de entrada
	 * @return <true> se as matrizes A e B sao iguais,
	 * <false> caso contrario
	 */
	public static boolean equals(SimpleMatrix A, SimpleMatrix B)
	{
		if (A == null || B == null) {
			return false;
		}

		double[] a = A.getMatrix().getData();
		double[] b = B.getMatrix().getData();

		if (a.length != b.length) {
			return false;
		}

		for (int i = 0; i < a.length; i++) {
			if (a[i] != b[i]) {
				return false;
			}
		}

		return true;
	}

	public static double[][] toDouble(SimpleMatrix m)
	{
		double data[][] = new double[m.numRows()][m.numCols()];
		for (int i = 0; i < m.numRows(); i++) {
			for (int j = 0; j < m.numCols(); j++) {
				data[i][j] = m.get(i, j);
			}
		}

		return data;
	}

	private SimpleMatrix getSubMatrixIndices(SimpleMatrix original, ArrayList<Integer> indRows, ArrayList<Integer> indColumns)
	{
		int numRows = indRows.size(), numColumns = indColumns.size();
		SimpleMatrix output = new SimpleMatrix(numRows, numColumns);

		for (int i = 0; i < numRows; i++) {
			for (int j = 0; j < numColumns; j++) {
				output.set(i, j, original.get(indRows.get(i), indColumns.get(j)));
			}
		}
		return output;
	}

	private SimpleMatrix getSubVectorIndices(SimpleMatrix original, ArrayList<Integer> indices)
	{
		int numFeatures = indices.size();
		SimpleMatrix output = new SimpleMatrix(numFeatures, 1);

		for (int i = 0; i < numFeatures; i++) {
			output.set(i, 0, original.get(indices.get(i)));
		}
		return output;
	}

	public static SimpleMatrix getDataRange(SimpleMatrix dataset)
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

	/**
	 *
	 * @param x matrix de entrada
	 * @param begin indice do comeco da busca
	 * @return indice do maior elemento da matriz x
	 */
	public static int maxElementIndex(SimpleMatrix x, int begin)
	{
		double data[] = x.getMatrix().getData();
		double max = data[begin];
		int idx = begin;
		for (int i = begin + 1; i < data.length; i++) {
			if (data[i] > max) {
				max = data[i];
				idx = i;
			}
		}

		return idx - begin;
	}
}
