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

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.List;
import liac.igmn.util.ChiSquareUtils;
import liac.igmn.util.MatrixUtil;
import org.ejml.simple.SimpleMatrix;

public class IGMN {

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
	 * Armazena o numero de classes do MMG
	 */
	protected int numClass;
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

	public IGMN(SimpleMatrix dataRange, double tau, double delta, double spMin, double vMin)
	{
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
		this.numClass = 0;
		this.phi = 0.25;
	}

	public IGMN(SimpleMatrix dataRange, double tau, double delta)
	{
		this(dataRange, tau, delta, dataRange.getNumElements() + 1, 2 * dataRange.getNumElements());
	}

	public IGMN(double tau, double delta)
	{
		this((new SimpleMatrix(0, 0)), tau, delta);
	}

	/**
	 * Algoritmo de aprendizagem da rede IGMN
	 *
	 * @param x vetor a ser utilizado no aprendizado
	 */
	public void learn(SimpleMatrix x)
	{
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
	private void computeLikelihood(SimpleMatrix x)
	{
		this.like = new SimpleMatrix(size, 1);
		this.distances = new SimpleMatrix(size, 1);
		for (int i = 0; i < size; i++) {
			this.like.set(i, 0, this.mvnpdf(x, i) + eta);
		}
	}

	private boolean hasAcceptableDistance(SimpleMatrix x)
	{
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
	private void addComponent(SimpleMatrix x)
	{
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
		//this.adjustCovariance();

		//Classification
		this.spsUpd.getMatrix().reshape(this.size, 1, true);
		this.spsUpd.set(this.size - 1, 0, 1);
		this.vsUpd.getMatrix().reshape(this.size, 1, true);
		this.vsUpd.set(this.size - 1, 0, 1);
	}

	/**
	 * Calcula a probabilidade a posteriori para cada componente <p(j|x)>
	 */
	private void computePosterior()
	{
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
	private void incrementalEstimation(SimpleMatrix x)
	{
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
	private void updatePriors()
	{
		double spSumTmp = this.sps.elementSum();
		this.priors = this.sps.divide(spSumTmp);
	}

	/**
	 * Remove componentes que sao considerados ruidosos. O componente e removido caso sua idade seja maior que a idade minima <vMin>
	 * e se sua ativacao for menor que a ativacao minima <spMin>
	 */
	private void removeSpuriousComponents()
	{
		for (int i = size - 1; i >= 0; i--) {
			if (this.vs.get(i) > this.vMin && this.sps.get(i) < this.spMin) {
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
	}

	/**
	 * Remove componentes que sao considerados como desatualizados. O componente e' removido caso seja considerado como nao estavel(), e sua idade seja maior que a idade minima <vMin> e se sua ativacao for menor que a ativacao minima <spMin>
	 */
	private void removeOutdatedComponents()
	{
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
	private boolean hasAcceptableDistribution()
	{
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
	public void call(SimpleMatrix x)
	{
		this.computeLikelihood(x);
		this.computePosterior();
	}

	/**
	 * Executa o algoritmo recall da IGMN
	 *
	 * @param x vetor de entrada
	 * @return vetor resultante do recall
	 */
	public SimpleMatrix recall(SimpleMatrix x)
	{
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
	public void train(SimpleMatrix dataset)
	{
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
	public SimpleMatrix classify(SimpleMatrix x)
	{
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
	public SimpleMatrix cluster(SimpleMatrix dataset)
	{
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
	public int classifyComponent(SimpleMatrix x)
	{
		call(x);
		return MatrixUtil.maxElementIndex(this.post);
	}

	/**
	 * Reinicia a rede
	 */
	public void reset()
	{
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

	public SimpleMatrix getPriors()
	{
		return this.priors;
	}

	public List<SimpleMatrix> getMeans()
	{
		return this.means;
	}

	public List<SimpleMatrix> getInvCovs()
	{
		return this.invCovs;
	}

	public SimpleMatrix getDetCovs()
	{
		return this.detCovs;
	}

	public SimpleMatrix getSps()
	{
		return this.sps;
	}

	public SimpleMatrix getLike()
	{
		return this.like;
	}

	public SimpleMatrix getPost()
	{
		return this.post;
	}

	public SimpleMatrix getVs()
	{
		return this.vs;
	}

	public int getDimension()
	{
		return this.dimension;
	}

	public int getSize()
	{
		return this.size;
	}

	public SimpleMatrix getDataRange()
	{
		return this.dataRange;
	}

	public double getDelta()
	{
		return this.delta;
	}

	public double getTau()
	{
		return this.tau;
	}

	public double getSpMin()
	{
		return this.spMin;
	}

	public double getvMin()
	{
		return this.vMin;
	}

	public void setDelta(double delta)
	{
		this.delta = delta;
	}

	public void setNumClass(int numClass)
	{
		this.numClass = numClass;
	}

	public void updateDataRange(SimpleMatrix dataRange)
	{
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
	private double mvnpdf(SimpleMatrix x, int component)
	{
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
	private double mvnpdf(SimpleMatrix x, SimpleMatrix u, SimpleMatrix invCov, double det)
	{
		double dim = x.getNumElements();
		SimpleMatrix distance = x.minus(u);

		double pdf = Math.exp(-0.5 * distance.transpose().dot(invCov.mult(distance)))
				/ (Math.pow(2 * Math.PI, dim / 2.0) * Math.sqrt(det));

		pdf = Double.isNaN(pdf) ? 0 : pdf;
		pdf = Double.isInfinite(pdf) ? Double.MAX_VALUE : pdf;

		return pdf;
	}

	private void mxCalculateValuesInitialSigma()
	{
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

	private SimpleMatrix mxInverseSumMatrix(SimpleMatrix invMat, SimpleMatrix diagMat)
	{
		SimpleMatrix result = new SimpleMatrix(invMat);
		int numRows = invMat.numRows();
		SimpleMatrix matRankOne = new SimpleMatrix(numRows, numRows);
		double g;

		for (int i = 0; i < numRows; i++) {
			g = 1.0 / (1.0 + (diagMat.get(i, i) * result.get(i, i)));
			//Set values of original diagonal
			matRankOne.set(i, i, diagMat.get(i, i));
			//Calculate iterative value
			result = result.minus(result.mult(matRankOne).mult(result).scale(g));
			//Reset value for next use
			matRankOne.set(i, i, 0);
		}
		
		return result;
	}
	
}
