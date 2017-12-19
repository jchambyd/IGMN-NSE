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

import liac.igmn.util.StringUtil;
import org.ejml.simple.SimpleMatrix;

public class ConfusionMatrix extends SimpleMatrix {

	protected String[] classNames;

	public ConfusionMatrix(String[] classNames)
	{
		super(classNames.length, classNames.length);
		this.classNames = classNames;
	}

	public void addPrediction(int i, int j)
	{
		set(i, j, get(i, j) + 1);
	}

	public String className(int index)
	{
		return classNames[index];
	}

	public double correct()
	{
		double correct = 0;
		for (int i = 0; i < size(); i++) {
			correct += get(i, i);
		}

		return correct;
	}

	public double incorrect()
	{
		double incorrect = 0;
		for (int row = 0; row < size(); row++) {
			for (int col = 0; col < size(); col++) {
				if (row != col) {
					incorrect += get(row, col);
				}
			}
		}

		return incorrect;
	}

	public double total()
	{
		double total = 0;
		for (int row = 0; row < size(); row++) {
			for (int col = 0; col < size(); col++) {
				total += get(row, col);
			}
		}

		return total;
	}

	public double accuracy()
	{
		return correct() / total();
	}

	public double accuracy(int classIndex)
	{
		double correct = 0;
		double total = 0;
		for (int i = 0; i < size(); i++) {
			total += get(classIndex, i);
		}
		correct = get(classIndex, classIndex);

		return correct / total;
	}

	public double errorRate()
	{
		return incorrect() / total();
	}

	public int size()
	{
		return classNames.length;
	}

	public String toString(String title)
	{
		StringBuffer text = new StringBuffer();
		char[] IDChars = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
			's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
		int IDWidth;
		boolean fractional = false;

		double maxval = 0;
		for (int i = 0; i < size(); i++) {
			for (int j = 0; j < size(); j++) {
				double current = get(i, j);
				if (current < 0) {
					current *= -10;
				}
				if (current > maxval) {
					maxval = current;
				}
				double fract = current - Math.rint(current);
				if (!fractional && ((Math.log(fract) / Math.log(10)) >= -2)) {
					fractional = true;
				}
			}
		}

		IDWidth = 1 + Math.max((int) (Math.log(maxval) / Math.log(10) + (fractional ? 3 : 0)),
				(int) (Math.log(size()) / Math.log(IDChars.length)));
		text.append(title).append("\n");
		for (int i = 0; i < size(); i++) {
			if (fractional) {
				text.append(" ").append(num2ShortID(i, IDChars, IDWidth - 3)).append("   ");
			} else {
				text.append(" ").append(num2ShortID(i, IDChars, IDWidth));
			}
		}
//		text.append("     classificado como:\n");
		text.append("\n");
		for (int i = 0; i < size(); i++) {
			for (int j = 0; j < size(); j++) {
				text.append(" ").append(StringUtil.doubleToString(get(i, j), IDWidth, (fractional ? 2 : 0)));
			}

			text.append(" | ").append(num2ShortID(i, IDChars, IDWidth)).append(" = ").append(classNames[i])
					.append("\n");
		}
		return text.toString();
	}

	private static String num2ShortID(int num, char[] IDChars, int IDWidth)
	{
		char ID[] = new char[IDWidth];
		int i;

		for (i = IDWidth - 1; i >= 0; i--) {
			ID[i] = IDChars[num % IDChars.length];
			num = num / IDChars.length - 1;
			if (num < 0) {
				break;
			}
		}
		for (i--; i >= 0; i--) {
			ID[i] = ' ';
		}

		return new String(ID);
	}
}
