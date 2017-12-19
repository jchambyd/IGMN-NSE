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

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;

public class DataLoader {

	private DataLoader()
	{
	}

	/**
	 * Carrega dataset a partir de arquivo ARFF e binariza os atributos nominais. Assume que a classe seja o ultimo atributo.
	 *
	 * @param filename path do arquivo
	 * @return dataset
	 * @throws DataLoaderException lancado quando o arquivo nao e encontrado ou quando ocorre algum erro de IO
	 */
	public static Dataset loadARFF(String filename) throws DataLoaderException
	{
		Dataset dataset = new Dataset();
		try {
			ArffLoader loader = new ArffLoader();

			loader.setSource(new File(filename));
			Instances data = loader.getDataSet();
			Instances m_Intances = new Instances(data);

			data.setClassIndex(data.numAttributes() - 1);

			String[] classes = new String[data.numClasses()];
			for (int i = 0; i < data.numClasses(); i++) {
				classes[i] = data.classAttribute().value(i);
			}
			dataset.setClassesNames(classes);

			NominalToBinary filter = new NominalToBinary();
			filter.setInputFormat(m_Intances);
			filter.setOptions(new String[]{"-A"});
			m_Intances = Filter.useFilter(m_Intances, filter);

			int inputSize = m_Intances.numAttributes() - data.numClasses();

			dataset.setInputSize(inputSize);
			dataset.setNumClasses(data.numClasses());

			dataset.setWekaDataset(m_Intances);
		} catch (IOException e) {
			throw new DataLoaderException("Arquivo não encontrado", e.getCause());
		} catch (Exception e) {
			throw new DataLoaderException("Falha na conversão do arquivo", e.getCause());
		}

		return dataset;
	}

	/**
	 * Carrega dataset a partir de arquivo TXT (formatado) e binariza os atributos nominais. Assume que a classe seja o ultimo atributo.
	 *
	 * @param filename path do arquivo
	 * @return dataset
	 * @throws Exception lancado quando o arquivo nao e encontrado ou quando ocorre algum erro de IO
	 */
	public static Dataset loadTXT(String filename) throws Exception
	{
		Dataset dataset = new Dataset();
		File file = new File(filename);
		Scanner loScan;
		String lsLine;
		HashMap<String, Integer> laClasses = new HashMap<>();
		ArrayList<ArrayList<Double>> laData = new ArrayList<>();

		try {
			//Read document
			loScan = new Scanner(file);
			while (loScan.hasNextLine()) {
				lsLine = loScan.nextLine();
				String[] lsTokens = lsLine.split(",");

				if (!laClasses.containsKey(lsTokens[lsTokens.length - 1])) {
					laClasses.put(lsTokens[lsTokens.length - 1], 0);
				}

				ArrayList<Double> loTmp = new ArrayList<>();

				for (String lsToken : lsTokens) {
					loTmp.add(Double.parseDouble(lsToken));
				}

				laData.add(loTmp);
			}

			//Create DataSet
			FastVector atts, attVals;
			atts = new FastVector();

			// Add attributes (numeric)
			for (int i = 0; i < laData.get(0).size() - 1; i++) {
				atts.addElement(new Attribute("att" + i));
			}

			// Add class (nominal)
			attVals = new FastVector();
			laClasses.entrySet().forEach((entry)
					-> {
				attVals.addElement("Class " + entry.getKey());
			});
			atts.addElement(new Attribute("att" + (laData.get(0).size() - 1), attVals));

			//Create instances for Dataset
			Instances data;
			data = new Instances("DataSet", atts, 0);

			for (ArrayList<Double> laDados : laData) {
				double[] vals = new double[data.numAttributes()];
				for (int i = 0; i < laDados.size() - 1; i++) {
					vals[i] = laDados.get(i);
				}

				double lnClass = laDados.get(laDados.size() - 1);
				//Search the nominal value (class)
				vals[laDados.size() - 1] = attVals.indexOf("Class " + (int) lnClass);

				data.add((Instance) (new DenseInstance(1.0, vals)));
			}

			Instances m_Intances = new Instances(data);

			data.setClassIndex(data.numAttributes() - 1);

			String[] classes = new String[data.numClasses()];
			for (int i = 0; i < data.numClasses(); i++) {
				classes[i] = data.classAttribute().value(i);
			}
			dataset.setClassesNames(classes);

			NominalToBinary filter = new NominalToBinary();
			filter.setInputFormat(m_Intances);
			filter.setOptions(new String[]{"-A"});
			m_Intances = Filter.useFilter(m_Intances, filter);

			int inputSize = m_Intances.numAttributes() - data.numClasses();

			dataset.setInputSize(inputSize);
			dataset.setNumClasses(data.numClasses());

			dataset.setWekaDataset(m_Intances);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		return dataset;
	}
}
