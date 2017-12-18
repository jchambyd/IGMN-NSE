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
package moa.classifiers;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.List;
import moa.core.Measurement;
import org.ejml.simple.SimpleMatrix;

public class IGMN_NSE extends AbstractClassifier implements MultiClassClassifier {

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

	public IGMN_NSE() {

	}

	@Override
	public boolean isRandomizable() {
		return false;
	}

	@Override
	public void resetLearningImpl() {

	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		if (this.poIGMN == null) {
			int dimention = inst.numInputAttributes() + inst.numClasses();

			double lvMin = (this.vMin.getValue() == this.vMin.getMinValue()) ? dimention * 2 : this.vMin.getValue();
			double lspMin = (this.spMin.getValue() == this.spMin.getMinValue()) ? dimention + 1 : this.spMin.getValue();

			//Call tha class IGMN
			this.poIGMN = new IGMN(getRangeNormalized(dimention), tau.getValue(), delta.getValue(), lspMin, lvMin);
		}

		double[][] classValue = new double[1][inst.numClasses()];
		classValue[0][(int) inst.classValue()] = 1;
		this.poIGMN.learn(this.InstanceToMatrix(inst, classValue[0]));
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
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
	public void getModelDescription(StringBuilder out, int indent) {
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	private SimpleMatrix getRangeNormalized(int dimention) {
		SimpleMatrix rangeData = new SimpleMatrix(dimention, 1);

		for (int i = 0; i < dimention; i++) {
			rangeData.set(i, 0, 1.0);
		}

		return rangeData;
	}

	private SimpleMatrix InstanceToMatrix(Instance instance) {
		double[][] matrix = new double[1][];
		matrix[0] = instance.toDoubleArray();
		return (new SimpleMatrix(matrix)).transpose();
	}

	private SimpleMatrix InstanceToMatrix(Instance toInstance, double[] taClass) {
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

class IGMN {

	/**
	 * Armazena a probabilidade a priori de cada componente numa matriz coluna
	 */
	protected SimpleMatrix priors;
	/**
	 * Armazena os vetores media de cada componente
	 */
	protected List<SimpleMatrix> means;
	/**
	 * Armazena a inversa da matriz de covariancia de cada componente
	 */
	protected List<SimpleMatrix> invCovs;
	/**
	 * Armazena a determinante da matriz de covariancia de cada componente
	 */
	protected SimpleMatrix detCovs;
	/**
	 * Armazena a soma das probabilidades a posteriori de cada componente numa matriz coluna
	 */
	protected SimpleMatrix sps;
	/**
	 * Armazena a verossimilhanca <p(x|j)> do ultimo vetor de entrada para cada componente numa matriz coluna
	 */
	protected SimpleMatrix like;
	/**
	 * Armazena a probabilidade a posteriori <p(j|x)> do ultimo vetor de entrada para cada componente numa matriz coluna
	 */
	protected SimpleMatrix post;
	/**
	 * Armazena a idade de cada componente numa matriz coluna
	 */
	protected SimpleMatrix vs;
	/**
	 * Armazena as distancias de cada componente ao ultimo vetor de entrada
	 */
	protected SimpleMatrix distances;
	/**
	 * Armazena a dimensao do vetor de entrada
	 */
	protected int dimension;
	/**
	 * Armazena o numero de componentes
	 */
	protected int size;
	/**
	 * Armazena o range do vetor de entrada
	 */
	protected SimpleMatrix dataRange;
	/**
	 * Parametro de ajuste do tamanho inicial da matriz de covariancia
	 */
	protected double delta;
	/**
	 * Parametro de ajuste da distancia minima aceitavel pelo componente
	 */
	protected double tau;
	/**
	 * Parametro de ajuste da ativacao minima de um componente para nao ser considerado como spurious
	 */
	protected double spMin;
	/**
	 * Parametro de ajuste da idade minima de um componente para poder ser detectado como spurious
	 */
	protected double vMin;
	/**
	 * Constante para definir o menor valor da verossimilhanca
	 */
	protected float eta;
	/**
	 * Inversa da matriz de covarianza inicial de cada componente
	 */
	protected SimpleMatrix invSigmaIni;
	/**
	 * Determinante da matriz de covarianza inicial de cada componente
	 */
	protected double detSigmaIni;
	/**
	 * Chi-squared distribution with "D" degrees-of-freedom,
	 */
	protected double chisq;

	//Classification
	/**
	 * Parametro indicador da ativacao maxima do MMG
	 */
	protected float spMax;
	/**
	 * Parametro de ajuste da ativacao ativacao maxima do MMG
	 */
	protected float beta;
	/**
	 * Parametro de ajuste da ativacao ativacao maxima do MMG
	 */
	protected float gamma;
	/**
	 * Armazena a soma das probabilidades a posteriori de todos os componentes
	 */
	protected float spSum;
	/**
	 * Armazena a soma das probabilidades a posteriori de cada componente numa matriz coluna
	 */
	protected SimpleMatrix spsUpd;
	/**
	 * Armazena a soma das probabilidades a posteriori de cada componente numa matriz coluna
	 */
	protected SimpleMatrix vsUpd;

	protected double phi;

	public IGMN(SimpleMatrix dataRange, double tau, double delta, double spMin, double vMin) {
		this.dataRange = dataRange;
		this.dimension = dataRange.getNumElements();
		this.size = 0;
		this.priors = new SimpleMatrix(0, 0);
		this.means = new ArrayList<>();
		this.invCovs = new ArrayList<>();
		this.detCovs = new SimpleMatrix(0, 0);
		this.sps = new SimpleMatrix(0, 0);
		this.like = new SimpleMatrix(0, 0);
		this.distances = new SimpleMatrix(0, 0);
		this.post = new SimpleMatrix(0, 0);
		this.vs = new SimpleMatrix(0, 0);
		this.delta = delta;
		this.tau = tau;
		this.spMin = spMin;
		this.vMin = vMin;
		this.eta = Float.MIN_VALUE;
		this.chisq = ChiSquareUtils.chi2inv(1 - tau, this.dimension);
		this.mxCalculateValuesInitialSigma();

		//Classification
		this.beta = 0.8f;
		this.spMax = 10000f;
		this.gamma = 0.5f;
		this.spSum = 0f;
		this.spsUpd = new SimpleMatrix(0, 0);
		this.vsUpd = new SimpleMatrix(0, 0);
		this.phi = 0.25;
	}

	public IGMN(SimpleMatrix dataRange, double tau, double delta) {
		this(dataRange, tau, delta, dataRange.getNumElements() + 1, 2 * dataRange.getNumElements());
	}

	public IGMN(double tau, double delta) {
		this((new SimpleMatrix(0, 0)), tau, delta);
	}

	/**
	 * Algoritmo de aprendizagem da rede IGMN
	 *
	 * @param x vetor a ser utilizado no aprendizado
	 */
	public void learn(SimpleMatrix x) {
		this.computeLikelihood(x);

		if (!this.hasAcceptableDistance(x)) {
			this.addComponent(x);
			this.like.getMatrix().reshape(size, 1, true);
			this.distances.getMatrix().reshape(this.size, 1, true);
			int i = size - 1;
			this.like.set(i, 0, this.mvnpdf(x, i) + this.eta);
			this.updatePriors();
		}
		this.computePosterior();
		this.incrementalEstimation(x);
		this.updatePriors();
		//Classification
		this.removeOutdatedComponents();
	}

	/**
	 * Calcula a verossimilhanca para cada componente <p(x|j)>
	 *
	 * @param x vetor de entrada
	 */
	private void computeLikelihood(SimpleMatrix x) {
		this.like = new SimpleMatrix(size, 1);
		this.distances = new SimpleMatrix(size, 1);
		for (int i = 0; i < size; i++) {
			this.like.set(i, 0, this.mvnpdf(x, i) + eta);
		}
	}

	private boolean hasAcceptableDistance(SimpleMatrix x) {
		for (int i = 0; i < size; i++) {
			if (this.distances.get(i) < this.chisq) {
				return true;
			}
		}
		return false;
	}

	/**
	 * Adiciona um novo componente na IGMN
	 *
	 * @param x vetor que sera o centro do novo componente
	 */
	private void addComponent(SimpleMatrix x) {
		this.size += 1;
		this.priors.getMatrix().reshape(size, 1, true);
		this.priors.set(size - 1, 0, 1);
		this.means.add(new SimpleMatrix(x));
		this.invCovs.add(this.invSigmaIni);
		this.detCovs.getMatrix().reshape(size, 1, true);
		this.detCovs.set(size - 1, 0, this.detSigmaIni);
		this.sps.getMatrix().reshape(size, 1, true);
		this.sps.set(size - 1, 0, 1);
		this.vs.getMatrix().reshape(size, 1, true);
		this.vs.set(size - 1, 0, 1);
		this.adjustCovariance();

		//Classification
		this.spsUpd.getMatrix().reshape(this.size, 1, true);
		this.spsUpd.set(this.size - 1, 0, 1);
		this.vsUpd.getMatrix().reshape(this.size, 1, true);
		this.vsUpd.set(this.size - 1, 0, 1);
	}

	/**
	 * Calcula a probabilidade a posteriori para cada componente <p(j|x)>
	 */
	private void computePosterior() {
		SimpleMatrix density = new SimpleMatrix(size, 1);

		for (int i = 0; i < size; i++) {
			density.set(i, 0, this.like.get(i) * this.priors.get(i));
		}

		this.post = density.divide(density.elementSum());
	}

	/**
	 * Atualiza os parametros idade, acumulador de posteriori, media e matriz de covariancia
	 *
	 * @param x vetor de entrada
	 */
	private void incrementalEstimation(SimpleMatrix x) {
		for (int i = 0; i < size; i++) {
			// Update age
			this.vs.set(i, this.vs.get(i) + 1);
			// Update age
			this.vsUpd.set(i, this.vsUpd.get(i) + 1);

			// Update accumulator of the posteriori probability
			this.sps.set(i, this.sps.get(i) + this.post.get(i));
			// Update accumulator of the posteriori probability
			this.spsUpd.set(i, this.spsUpd.get(i) + this.post.get(i));
			// Update the sum of posteriori probabilities
			this.spSum += this.sps.get(i);
			// Update mean
			SimpleMatrix oldmeans = this.means.get(i).copy();
			double w = this.post.get(i) / this.sps.get(i); //Learning rate
			SimpleMatrix diff = x.minus(oldmeans).scale(w);
			this.means.set(i, oldmeans.plus(diff));
			diff = this.means.get(i).minus(oldmeans); //delta mu
			SimpleMatrix diff2 = x.minus(this.means.get(i)); //e*

			//Update invert covariance
			// Plus a rank-one update
			SimpleMatrix invCov = this.invCovs.get(i).copy();
			SimpleMatrix v = diff2.scale(Math.sqrt(w)); //v = u = e*.sqrt(w)
			SimpleMatrix tmp1 = invCov.scale(1.0 / (1.0 - w)); //A(t-1) / (1 - w)
			SimpleMatrix tmp2 = tmp1.mult(v); // matrix D x 1
			double tmp3 = 1 + tmp2.dot(v);
			invCov = tmp1.minus(tmp2.mult(tmp2.transpose()).scale(1.0 / tmp3));
			// Substract a rank-one update
			SimpleMatrix tmp4 = invCov.mult(diff); // matrix D x 1
			double tmp5 = 1 - tmp4.dot(diff);
			invCov = invCov.plus(tmp4.mult(tmp4.transpose()).scale(1.0 / tmp5));
			this.invCovs.set(i, invCov);

			//Update Determinant Covariance
			// Plus a rank-one update
			double detCov = this.detCovs.get(i);
			detCov = detCov * Math.pow(1.0 - w, this.dimension) * (tmp3);
			// Substract a rank-one update
			detCov = detCov * tmp5;
			this.detCovs.set(i, detCov);

		}

		// The accumulators are restarted to a gamma fraction
		if (this.spSum >= this.beta * this.spMax) {
			this.sps = this.sps.scale(this.gamma);
			this.spSum = 0f;
		}
	}

	/**
	 * Atualiza as probabilidades a priori de cada componente
	 */
	private void updatePriors() {
		double spSumTmp = this.sps.elementSum();
		this.priors = this.sps.divide(spSumTmp);
	}

	/**
	 * Remove componentes que sao considerados como desatualizados. O componente e' removido caso seja considerado como nao estavel(), e sua idade seja maior que a idade minima <vMin> e se sua ativacao for menor que a ativacao minima <spMin>
	 */
	private void removeOutdatedComponents() {
		boolean updtPriors = false, removeComp;

		for (int i = this.size - 1; i >= 0; i--) {
			removeComp = false;
			// Check information about age of components
			if (this.vsUpd.get(i) >= (this.vMin * this.size)) {
				// If is estable component (posteriory accumulator is greater than 'spMin')
				if (this.sps.get(i) >= this.spMin) {
					// But the current posteriory accumulator is not up-to-dated
					if (this.spsUpd.get(i) < this.spMin) {
						this.sps.set(i, this.sps.get(i) * 0.75);
						updtPriors = true;
					}
					this.spsUpd.set(i, 0.0);
					this.vsUpd.set(i, 0.0);
				} else {
					removeComp = true;
					updtPriors = true;
				}
			}
			if (removeComp) {
				MatrixUtil.removeElement(this.vs, i);
				MatrixUtil.removeElement(this.sps, i);
				MatrixUtil.removeElement(this.priors, i);
				MatrixUtil.removeElement(this.detCovs, i);
				this.means.remove(i);
				this.invCovs.remove(i);
				this.size -= 1;
				//Classification
				MatrixUtil.removeElement(this.vsUpd, i);
				MatrixUtil.removeElement(this.spsUpd, i);
			}
		}

		if (updtPriors) {
			this.updatePriors();
		}
	}

	/**
	 *
	 * @return <true> se o vetor de entrada tem verossimilhanca minima, determinado pelo parametro <tau>, para algum componente,
	 * <false> caso contrario
	 */
	private boolean hasAcceptableDistribution() {
		for (int i = 0; i < size; i++) {
			double den = Math.pow(2 * Math.PI, dimension / 2.0) * Math.sqrt(this.detCovs.get(i));
			double min = tau / den;
			if (like.get(i) >= min) {
				return true;
			}
		}

		return false;
	}

	/**
	 * Realiza call para rede
	 *
	 * @param x vetor de entrada
	 */
	public void call(SimpleMatrix x) {
		this.computeLikelihood(x);
		this.computePosterior();
	}

	/**
	 * Executa o algoritmo recall da IGMN
	 *
	 * @param x vetor de entrada
	 * @return vetor resultante do recall
	 */
	public SimpleMatrix recall(SimpleMatrix x) {
		int alpha = x.getNumElements();
		int beta = dimension - alpha;

		SimpleMatrix pajs = new SimpleMatrix(size, 1);
		List<SimpleMatrix> xm = new ArrayList<>();

		for (int i = 0; i < size; i++) {
			SimpleMatrix blockZ = this.invCovs.get(i).extractMatrix(alpha, alpha + beta, 0, alpha);
			SimpleMatrix blockW = this.invCovs.get(i).extractMatrix(alpha, alpha + beta, alpha, alpha + beta);
			SimpleMatrix blockX = this.invCovs.get(i).extractMatrix(0, alpha, 0, alpha);

			SimpleMatrix meanA = this.means.get(i).extractMatrix(0, alpha, 0, 1);
			SimpleMatrix meanB = this.means.get(i).extractMatrix(alpha, alpha + beta, 0, 1);

			SimpleMatrix invBlockW = blockW.invert();
			SimpleMatrix invBlockA = blockX.minus(blockZ.transpose().mult(invBlockW).mult(blockZ));

			pajs.set(i, 0, this.mvnpdf(x, meanA, invBlockA, this.detCovs.get(i) * blockW.determinant()) + this.eta);

			SimpleMatrix x_ = meanB.minus((invBlockW).mult(blockZ).mult(x.minus(meanA)));

			xm.add(x_);
		}

		pajs = pajs.divide(pajs.elementSum());
		SimpleMatrix result = new SimpleMatrix(beta, 1);
		for (int i = 0; i < xm.size(); i++) {
			result = result.plus(xm.get(i).scale(pajs.get(i)));
		}

		return result;
	}

	/**
	 * Realiza treinamento a partir de um conjunto de dados, onde cada instancia e uma coluna da matriz
	 *
	 * @param dataset o conjunto de treinamento
	 */
	public void train(SimpleMatrix dataset) {
		for (int i = 0; i < dataset.numCols(); i++) {
			learn(dataset.extractVector(false, i));
		}
	}

	/**
	 * Classifica um vetor de entrada
	 *
	 * @param x vetor de entrada
	 * @return vetor referente a classificacao do vetor de entrada
	 */
	public SimpleMatrix classify(SimpleMatrix x) {
		SimpleMatrix out = this.recall(x);
		int i = MatrixUtil.maxElementIndex(out);

		SimpleMatrix y = new SimpleMatrix(1, this.dimension - x.getNumElements());
		y.set(i, 1);

		return y;
	}

	/**
	 * Realiza clusterizacao a partir de um conjunto de dados, onde cada instancia e uma coluna da matriz
	 *
	 * @param dataset o conjunto de dados
	 * @return rotulos para cada instancia do conjunto de dados de entrada
	 */
	public SimpleMatrix cluster(SimpleMatrix dataset) {
		SimpleMatrix out = new SimpleMatrix(dataset.numCols(), 1);

		for (int i = 0; i < dataset.numCols(); i++) {
			int index = classifyComponent(dataset.extractVector(false, i));
			out.set(i, index);
		}
		return out;
	}

	/**
	 * Classifica um vetor de entrada
	 *
	 * @param x vetor de entrada
	 * @return indice referente ao componente designado ao vetor de entrada
	 */
	public int classifyComponent(SimpleMatrix x) {
		call(x);
		return MatrixUtil.maxElementIndex(this.post);
	}

	/**
	 * Reinicia a rede
	 */
	public void reset() {
		this.size = 0;
		this.priors = new SimpleMatrix(0, 0);
		this.means = new ArrayList<>();
		this.sps = new SimpleMatrix(0, 0);
		this.like = new SimpleMatrix(0, 0);
		this.post = new SimpleMatrix(0, 0);
		this.vs = new SimpleMatrix(0, 0);
		this.invCovs = new ArrayList<>();
		this.detCovs = new SimpleMatrix(0, 0);
	}

	public SimpleMatrix getPriors() {
		return this.priors;
	}

	public List<SimpleMatrix> getMeans() {
		return this.means;
	}

	public List<SimpleMatrix> getInvCovs() {
		return this.invCovs;
	}

	public SimpleMatrix getDetCovs() {
		return this.detCovs;
	}

	public SimpleMatrix getSps() {
		return this.sps;
	}

	public SimpleMatrix getLike() {
		return this.like;
	}

	public SimpleMatrix getPost() {
		return this.post;
	}

	public SimpleMatrix getVs() {
		return this.vs;
	}

	public int getDimension() {
		return this.dimension;
	}

	public int getSize() {
		return this.size;
	}

	public SimpleMatrix getDataRange() {
		return this.dataRange;
	}

	public double getDelta() {
		return this.delta;
	}

	public double getTau() {
		return this.tau;
	}

	public double getSpMin() {
		return this.spMin;
	}

	public double getvMin() {
		return this.vMin;
	}

	public void setDelta(double delta) {
		this.delta = delta;
	}

	public void updateDataRange(SimpleMatrix dataRange) {
		this.dataRange = dataRange;
		this.dimension = dataRange.getNumElements();
		this.spMin = this.dimension + 1;
		this.vMin = 2 * this.dimension;
		this.chisq = ChiSquareUtils.chi2inv(1 - this.tau, this.dimension);
		this.mxCalculateValuesInitialSigma();
	}

	/**
	 * Calcula a funcao de densidade de probabilidade multivariada (multivariate probability density function)
	 *
	 * @param x vetor de entrada
	 * @param component index of component
	 * @return a densidade de probabilidade
	 */
	private double mvnpdf(SimpleMatrix x, int component) {
		double dim = x.getNumElements();
		SimpleMatrix distance = x.minus(this.means.get(component));

		this.distances.set(component, distance.transpose().dot(this.invCovs.get(component).mult(distance)));

		double pdf = Math.exp(-0.5 * this.distances.get(component))
				/ (Math.pow(2 * Math.PI, dim / 2.0) * Math.sqrt(this.detCovs.get(component)));

		pdf = Double.isNaN(pdf) ? 0 : pdf;
		pdf = Double.isInfinite(pdf) ? Double.MAX_VALUE : pdf;

		return pdf;
	}

	/**
	 * Calcula a funcao de densidade de probabilidade multivariada (multivariate probability density function)
	 *
	 * @param x vetor de entrada
	 * @param u vetor media
	 * @param invCov inversa da matriz de covariancia
	 * @param det determinante da matriz de covariancia
	 * @return a densidade de probabilidade
	 */
	private double mvnpdf(SimpleMatrix x, SimpleMatrix u, SimpleMatrix invCov, double det) {
		double dim = x.getNumElements();
		SimpleMatrix distance = x.minus(u);

		double pdf = Math.exp(-0.5 * distance.transpose().dot(invCov.mult(distance)))
				/ (Math.pow(2 * Math.PI, dim / 2.0) * Math.sqrt(det));

		pdf = Double.isNaN(pdf) ? 0 : pdf;
		pdf = Double.isInfinite(pdf) ? Double.MAX_VALUE : pdf;

		return pdf;
	}

	private void mxCalculateValuesInitialSigma() {
		SimpleMatrix newCov = new SimpleMatrix(dataRange.scale(delta));
		SimpleMatrix sigma = MatrixUtil.diag(newCov.elementMult(newCov));
		double determinant = 1;

		for (int i = 0; i < sigma.numCols(); i++) {
			determinant *= sigma.get(i, i);
			sigma.set(i, i, 1 / sigma.get(i, i));
		}

		this.invSigmaIni = new SimpleMatrix(sigma);
		this.detSigmaIni = determinant;
	}

	private double bhattacharyyaDistance(SimpleMatrix meanX, SimpleMatrix meanY, SimpleMatrix covX, SimpleMatrix covY, double detX, double detY) {
		SimpleMatrix bhattMean = meanX.minus(meanY);
		SimpleMatrix bhattCov = covX.plus(covY).divide(2);
		SimpleMatrix invSumMatrix = bhattCov.invert();
		SimpleMatrix distance = (bhattMean.transpose()).mult(invSumMatrix).mult(bhattMean).divide(8);

		return distance.get(0, 0) + Math.log1p(bhattCov.determinant() / ((Math.pow(detX * detY, 0.5)))) / 2;
	}

	private double bhattacharyyaCoefficient(SimpleMatrix meanX, SimpleMatrix meanY, SimpleMatrix covX, SimpleMatrix covY, double detX, double detY) {
		double distBhatt = this.bhattacharyyaDistance(meanX, meanY, covX, covY, detX, detY);
		distBhatt = (distBhatt < 0) ? 0 : distBhatt;
		return 1 / Math.exp(distBhatt);
	}

	private AbstractMap.SimpleEntry<Double, Integer> maxBhattacharyyaCoefficient(SimpleMatrix sigma) {
		double maxCoefficient = Double.NEGATIVE_INFINITY, distance;
		int index = 0;

		for (int i = 0; i < size - 1; i++) {
			distance = this.bhattacharyyaCoefficient(this.means.get(i), this.means.get(size - 1), this.invCovs.get(i).invert(), sigma, this.detCovs.get(i), this.detSigmaIni);
			if (maxCoefficient < distance) {
				index = i;
				maxCoefficient = distance;
			}
		}

		return (new AbstractMap.SimpleEntry<>(maxCoefficient, index));
	}

	private void adjustCovariance() {
		double determinant, maxDistance;
		AbstractMap.SimpleEntry<Double, Integer> data;
		SimpleMatrix sigmaIni = new SimpleMatrix(this.dimension, this.dimension);

		for (int i = 0; i < sigmaIni.numCols(); i++) {
			sigmaIni.set(i, i, 1 / this.invSigmaIni.get(i, i));
		}

		data = this.maxBhattacharyyaCoefficient(sigmaIni);
		maxDistance = data.getKey();

		if (maxDistance >= this.phi) {
			sigmaIni = sigmaIni.scale(1 - maxDistance);
			determinant = 1;
			SimpleMatrix invSigma = new SimpleMatrix(this.dimension, this.dimension);

			for (int i = 0; i < sigmaIni.numCols(); i++) {
				determinant *= sigmaIni.get(i, i);
				invSigma.set(i, i, 1 / sigmaIni.get(i, i));
			}

			this.invCovs.set(this.size - 1, new SimpleMatrix(invSigma));
			this.detCovs.set(this.size - 1, determinant);
		}
	}
}

class ChiSquareUtils {

	/**
	 * Returns the rates that should be used for a set number of rate categories and a set value of alpha as per the methodology of Yang 1993
	 *
	 * @param alpha The alpha value
	 * @param categories The number of rate categories
	 * @return The rate for each category
	 */
	public static double[] rates(double alpha, int categories) {
		if (alpha == 0.0) {
			double[] res = new double[categories];
			for (int i = 0; i < res.length; i++) {
				res[i] = 1.0;
			}
			return res;
		} else {
			double[] res = new double[categories];
			double total = 0.0;
			for (int i = 0; i < categories; i++) {
				double pa = chi2inv(((double) i) / ((double) categories), 2 * alpha) / (2 * alpha);
				double pb = chi2inv(((double) i + 1.0) / ((double) categories), 2 * alpha) / (2 * alpha);

				double ia = gammaintegral(alpha + 1, pa * alpha);
				double ib = gammaintegral(alpha + 1, pb * alpha);

				res[i] = (ib - ia) / categories;
				total += (res[i] / categories);
			}

			for (int i = 0; i < categories; i++) {
				res[i] = res[i] / total;
			}

			return res;
		}
	}

	/**
	 * Returns the inverse chi-squared distribution. Uses the method given in Best and Roberts 1975. Makes calls to private functions using the methods of Bhattacharjee 1970 and Odeh and Evans 1974. All converted to Java by the author (yes, the author knows FORTRAN!)
	 *
	 * @param p The p-value
	 * @param v The number of degrees of freedom
	 * @return The percentage point
	 */
	public static double chi2inv(double p, double v) {
		if (p < 0.000002) {
			return 0.0;
		}
		if (p > 0.999998) {
			p = 0.999998;
		}

		double xx = 0.5 * v;
		double c = xx - 1.0;
		double aa = Math.log(2);
		double g = gammaln(v / 2.0);
		double ch;
		if (v > (-1.24 * Math.log(p))) {
			if (v > 0.32) {
				//3
				double x = gauinv(p);
				double p1 = 0.222222 / v;
				ch = v * Math.pow(x * Math.sqrt(p1) + 1.0 - p1, 3);
				if (ch > (2.2 * v + 6.0)) {
					ch = -2.0 * (Math.log(1.0 - p) - c * Math.log(0.5 * ch) + g);
				}
			} else {
				//1+2
				ch = 0.4;
				double q;
				double a = Math.log(1.0 - p);
				do {
					q = ch;
					double p1 = 1.0 + ch * (4.67 + ch);
					double p2 = ch * (6.73 + ch * (6.66 + ch));
					double t = -0.5 + (4.67 + 2.0 * ch) / p1
							- (6.73 + ch * (13.32 + 3.0 * ch)) / p2;
					ch = ch - (1.0 - Math.exp(a + g + 0.5 * ch + c * aa) * p2 / p1) / t;
				} while (Math.abs(q / ch - 1.0) >= 0.01);
			}
		} else {
			//START
			ch = Math.pow(p * xx * Math.exp(g + xx * aa), 1.0 / xx);
		}
		double q;
		do {
			//4 + 5
			q = ch;
			double p1 = 0.5 * ch;
			double p2 = p - gammaintegral(xx, p1);
			double t = p2 * Math.exp(xx * aa + g + p1 - c * Math.log(ch));
			double b = t / ch;
			double a = 0.5 * t - b * c;
			double s1 = (210.0 + a * (140.0 + a * (105.0 + a * (84.0 + a * (70.0 + 60.0 * a))))) / 420.0;
			double s2 = (420.0 + a * (735.0 + a * (966.0 + a * (1141.0 + 1278.0 * a)))) / 2520.0;
			double s3 = (210.0 + a * (462.0 + a * (707.0 + 932.0 * a))) / 2520.0;
			double s4 = (252.0 + a * (672.0 + 1182.0 * a) + c * (294.0 + a * (889.0 + 1740.0 * a))) / 5040.0;
			double s5 = (84.0 + 264.0 * a + c * (175.0 + 606.0 * a)) / 2520.0;
			double s6 = (120.0 + c * (346.0 + 127.0 * c)) / 5040.0;
			ch = ch + t * (1.0 + 0.5 * t * s1 - b * c * (s1 - b * (s2 - b * (s3 - b * (s4 - b * (s5 - b * s6))))));
		} while (Math.abs(q / ch - 1.0) > E);
		return ch;
	}

	private static double gammaintegral(double p, double x) {
		double g = gammaln(p);
		double factor = Math.exp(p * Math.log(x) - x - g);
		double gin;
		if ((x > 1.0) && (x > p)) {
			boolean end = false;
			double a = 1.0 - p;
			double b = a + x + 1.0;
			double term = 0.0;
			double[] pn = new double[6];
			pn[0] = 1.0;
			pn[1] = x;
			pn[2] = x + 1.0;
			pn[3] = x * b;
			gin = pn[2] / pn[3];
			do {
				double rn;
				a++;
				b = b + 2.0;
				term++;
				double an = a * term;
				for (int i = 0; i <= 1; i++) {
					pn[i + 4] = b * pn[i + 2] - an * pn[i];
				}
				if (pn[5] != 0.0) {
					rn = pn[4] / pn[5];
					double diff = Math.abs(gin - rn);
					if (diff < E * rn) {
						end = true;
					} else {
						gin = rn;
					}
				}
				if (!end) {
					for (int i = 0; i < 4; i++) {
						pn[i] = pn[i + 2];
					}
					if (Math.abs(pn[5]) >= OFLO) {
						for (int i = 0; i < 4; i++) {
							pn[i] = pn[i] / OFLO;
						}
					}
				}
			} while (!end);
			gin = 1.0 - factor * gin;
		} else {
			gin = 1.0;
			double term = 1.0;
			double rn = p;
			do {
				rn++;
				term = term * x / rn;
				gin = gin + term;
			} while (term > E);
			gin = gin * factor / p;
		}
		return gin;
	}

	private static double gauinv(double p) {
		if (p == 0.5) {
			return 0.0;
		}
		double ps = p;
		if (ps > 0.5) {
			ps = 1 - ps;
		}
		double yi = Math.sqrt(Math.log(1.0 / (ps * ps)));
		double gauinv = yi + ((((yi * p4 + p3) * yi + p2) * yi + p1) * yi + p0)
				/ ((((yi * q4 + q3) * yi + q2) * yi + q1) * yi + q0);
		if (p < 0.5) {
			return -gauinv;
		} else {
			return gauinv;
		}
	}

	private static double gammaln(double xx) {
		double y = xx;
		double x = xx;
		double tmp = x + 5.2421875;
		tmp = (x + 0.5) * Math.log(tmp) - tmp;
		double ser = 0.999999999999997092;
		for (int i = 0; i < 14; i++) {
			ser += COF[i] / ++y;
		}
		return tmp + Math.log(2.5066282746310005 * ser / x);
	}

	private static final double[] COF = {
		57.1562356658629235,
		-59.5979603554754912,
		14.1360979747417471,
		-0.491913816097620199,
		0.339946499848118887e-4,
		0.465236289270485756e-4,
		-0.983744753048795646e-4,
		0.158088703224912494e-3,
		-0.210264441724104883e-3,
		0.217439618115212643e-3,
		-0.164318106536763890e-3,
		0.844182239838527433e-4,
		-0.261908384015714087e-4,
		0.368991826595316234e-5
	};

	private static final double p0 = -0.322232431088;
	private static final double p1 = -1.0;
	private static final double p2 = -0.342242088547;
	private static final double p3 = -0.204231210245e-1;
	private static final double p4 = -0.453642210148e-4;
	private static final double q0 = 0.993484626060e-1;
	private static final double q1 = 0.588581570495;
	private static final double q2 = 0.531103462366;
	private static final double q3 = 0.103537752850;
	private static final double q4 = 0.38560700634e-2;
	private static final double OFLO = 10e30;
	private static final double E = 10e-6;
}

class MatrixUtil {

	private MatrixUtil() {
	}

	/**
	 *
	 * @param x matrix de entrada
	 * @return indice do maior elemento da matriz x
	 */
	public static int maxElementIndex(SimpleMatrix x) {
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
	 * Remove um elemento da matriz
	 *
	 * @param m matriz de entrada
	 * @param idx indice do elemento a ser removido
	 * @return nova matriz com o elemento removido
	 */
	public static SimpleMatrix removeElement(SimpleMatrix m, int idx) {
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
	public static SimpleMatrix diag(SimpleMatrix m) {
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
	public static boolean equals(SimpleMatrix A, SimpleMatrix B) {
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

	public static double[][] toDouble(SimpleMatrix m) {
		double data[][] = new double[m.numRows()][m.numCols()];
		for (int i = 0; i < m.numRows(); i++) {
			for (int j = 0; j < m.numCols(); j++) {
				data[i][j] = m.get(i, j);
			}
		}

		return data;
	}

	private SimpleMatrix getSubMatrixIndices(SimpleMatrix original, ArrayList<Integer> indRows, ArrayList<Integer> indColumns) {
		int numRows = indRows.size(), numColumns = indColumns.size();
		SimpleMatrix output = new SimpleMatrix(numRows, numColumns);

		for (int i = 0; i < numRows; i++) {
			for (int j = 0; j < numColumns; j++) {
				output.set(i, j, original.get(indRows.get(i), indColumns.get(j)));
			}
		}
		return output;
	}

	private SimpleMatrix getSubVectorIndices(SimpleMatrix original, ArrayList<Integer> indices) {
		int numFeatures = indices.size();
		SimpleMatrix output = new SimpleMatrix(numFeatures, 1);

		for (int i = 0; i < numFeatures; i++) {
			output.set(i, 0, original.get(indices.get(i)));
		}
		return output;
	}

	public static SimpleMatrix getDataRange(SimpleMatrix dataset) {
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
	public static int maxElementIndex(SimpleMatrix x, int begin) {
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
