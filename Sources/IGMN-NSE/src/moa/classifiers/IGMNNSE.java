/*
 *    IGMNNSE.java
 * 
 *    @author Jorge Chamby-Diaz (jchambyd at gmail dot com)
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 */
package moa.classifiers;

import java.util.ArrayList;
import java.util.List;

import com.github.javacliparser.FloatOption;
import com.yahoo.labs.samoa.instances.Instance;

import org.ejml.simple.SimpleMatrix;

import moa.AbstractMOAObject;
import moa.core.ChiSquareUtils;
import moa.core.Measurement;

/**
 * Incremental Gaussian Mixture Network for Non-Stationary Environments
 *
 * <p>
 * See details in:<br>
 * Jorge C. Chamby-Diaz, Mariana Recamonde-Mendoza, Ana L. C. Bazzan, Ricardo 
 * Grunitzki. Adaptive Incremental Gaussian Mixture Network for Non-Stationary
 * Data Stream Classification. DOI: 10.1109/IJCNN.2018.8489049, IEEE, 2018.
 * </p>
 *
 * <p>
 * Parameters:
 * </p>
 * <ul>
 * <li>-t : Percentile of a chi-squared distribution.</li>
 * <li>-d : Initial size of covariance matrix.</li>
 * <li>-g : Fraccion to be reset posteriori accumulators.</li>
 * </ul>
 *
 * @author Jorge Chamby-Diaz (jchambyd at gmail dot com)
 * @version $Revision: 1 $
 */

public class IGMNNSE extends AbstractClassifier implements MultiClassClassifier {

    @Override
    public String getPurposeString() {
        return "Incremental Gaussian Mixture Network from Chamby-Diaz et al.";
    }

    private static final long serialVersionUID = 1L;

    public FloatOption tau = new FloatOption("tau", 't', "Percentile of a chi-squared distribution.", 0.001, 0.0001,
            1.00);

    public FloatOption delta = new FloatOption("delta", 'd', "Initial size of covariance matrix.", 0.5, 0.005, 0.99);

    public FloatOption gamma = new FloatOption("gamma", 'g', "Fraccion to be reset posteriori accumulators.", 0.5, 0.01,
            0.99);

    protected List<NeuronIGMNNSE> neurons;
    protected List<Double> like;
    protected List<Double> post;
    protected SimpleMatrix invSigmaIni;
    protected double chisq;
    protected double detSigmaIni;
    protected double sumSp;
    protected int dimension;
    protected int size;
    protected long instancesSeen;

    @Override
    public boolean isRandomizable() {
        return false;
    }

    @Override
    public void resetLearningImpl() {
        this.size = 0;
        this.dimension = 0;
        this.sumSp = 0;
        this.instancesSeen = 0;
        this.neurons = null;
    }

    @Override
    public void trainOnInstanceImpl(Instance instance) {
        if (this.neurons == null) {
            this.initIGMN(instance);
        }
        SimpleMatrix x = this.instanceToMatrix(instance);

        this.computeLikelihood(x);

        if (!this.hasAcceptableDistance()) {
            this.neurons.add(new NeuronIGMNNSE(x, invSigmaIni, this.detSigmaIni));
            this.like.add(this.neurons.get(this.size++).mvnpdf(x) + Float.MIN_VALUE);
            this.post.add(0.0);
            this.updatePriors();
        }
        this.computePosterior();
        this.incrementalEstimation(x);
        this.updatePriors();
        this.forgettingMechanismn();
        this.instancesSeen++;
    }

    @Override
    public double[] getVotesForInstance(Instance instance) {
        double[] result = new double[instance.numClasses()];

        if (this.neurons == null) {
            this.initIGMN(instance);
        } else {
            SimpleMatrix clasify = this.recall(this.instanceToMatrix(instance).extractMatrix(0,
                    instance.numInputAttributes(), 0, SimpleMatrix.END));
            double max = clasify.get(0, 0);
            int idx = 0;

            for (int i = 1; i < clasify.numRows(); i++) {
                if (max < clasify.get(i, 0)) {
                    idx = i;
                    max = clasify.get(i, 0);
                }
            }
            result[idx] = 1;
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

    protected void initIGMN(Instance inst) {
        this.dimension = inst.numInputAttributes() + inst.numClasses();
        this.size = 0;
        this.instancesSeen = 0;
        this.neurons = new ArrayList<>();
        this.like = new ArrayList<>();
        this.post = new ArrayList<>();
        this.chisq = ChiSquareUtils.chi2inv(1 - this.tau.getValue(), this.dimension);
        this.mxCalculateValuesInitialSigma();
    }

    private SimpleMatrix instanceToMatrix(Instance instance) {
        SimpleMatrix result = new SimpleMatrix(this.dimension, 1);
        double[] loInstance = instance.toDoubleArray();

        for (int i = 0; i < instance.numInputAttributes(); i++) {
            result.set(i, 0, loInstance[i]);
        }
        result.set(instance.numInputAttributes() + (int) instance.classValue(), 0, 1);

        return result;
    }

    public void computeLikelihood(SimpleMatrix x) {
        for (int i = 0; i < this.size; i++) {
            this.like.set(i, this.neurons.get(i).mvnpdf(x) + Float.MIN_VALUE);
        }
    }

    private boolean hasAcceptableDistance() {
        for (int i = 0; i < this.size; i++) {
            if (this.neurons.get(i).distance < this.chisq) {
                return true;
            }
        }
        return false;
    }

    private void updatePriors() {
        double spSumTmp = 0;

        for (NeuronIGMNNSE neuron : this.neurons) {
            spSumTmp += neuron.sp;
        }

        for (NeuronIGMNNSE neuron : this.neurons) {
            neuron.prior = neuron.sp / spSumTmp;
        }
    }

    private void computePosterior() {
        List<Double> density = new ArrayList<>(this.size);
        double sumTmp = 0;

        for (int i = 0; i < this.size; i++) {
            density.add(this.like.get(i) * this.neurons.get(i).prior);
            sumTmp += density.get(i);
        }

        for (int i = 0; i < this.size; i++) {
            this.post.set(i, density.get(i) / sumTmp);
        }
    }

    private void incrementalEstimation(SimpleMatrix x) {
        int i = 0;

        for (NeuronIGMNNSE neuron : this.neurons) {
            // Update age
            neuron.addAge();
            // Update accumulator of the posteriori probability
            neuron.addPosteriori(this.post.get(i));
            // Update the sum of posteriori probabilities
            this.sumSp += neuron.sp;

            // Update mean
            SimpleMatrix oldmean = neuron.mean.copy();
            double w = this.post.get(i) / neuron.sp; // Learning rate
            SimpleMatrix diff = x.minus(oldmean).scale(w);
            neuron.mean = oldmean.plus(diff);
            diff = (neuron.mean).minus(oldmean); // delta mu
            SimpleMatrix diff2 = x.minus(neuron.mean); // e*

            // Update invert covariance
            // Plus a rank-one update
            SimpleMatrix invCov = neuron.invCov.copy();
            SimpleMatrix v = diff2.scale(Math.sqrt(w)); // v = u = e*.sqrt(w)
            SimpleMatrix tmp1 = invCov.scale(1.0 / (1.0 - w)); // A(t-1) / (1 - w)
            SimpleMatrix tmp2 = tmp1.mult(v); // matrix D x 1
            double tmp3 = 1 + tmp2.dot(v);
            invCov = tmp1.minus(tmp2.mult(tmp2.transpose()).scale(1.0 / tmp3));

            // Substract a rank-one update
            SimpleMatrix tmp4 = invCov.mult(diff); // matrix D x 1
            double tmp5 = 1 - tmp4.dot(diff);
            invCov = invCov.plus(tmp4.mult(tmp4.transpose()).scale(1.0 / tmp5));
            neuron.invCov = invCov;

            // Update Determinant Covariance
            // Plus a rank-one update
            double detCov = neuron.detCov;
            detCov = detCov * Math.pow(1.0 - w, this.dimension) * (tmp3);
            // Substract a rank-one update
            detCov = detCov * tmp5;
            neuron.detCov = detCov;

            i++;
        }

        if (this.sumSp >= 80000f) {
            for (NeuronIGMNNSE neuron : this.neurons) {
                neuron.resetPosteriori(this.gamma.getValue());
            }
            this.sumSp = 0;
        }
    }

    private SimpleMatrix recall(SimpleMatrix x) {
        int alpha = x.getNumElements();
        int beta = this.dimension - alpha;

        List<Double> pajs = new ArrayList<>(this.size);
        List<SimpleMatrix> xm = new ArrayList<>();

        for (NeuronIGMNNSE neuron : this.neurons) {
            SimpleMatrix blockZ = neuron.invCov.extractMatrix(alpha, alpha + beta, 0, alpha);
            SimpleMatrix blockW = neuron.invCov.extractMatrix(alpha, alpha + beta, alpha, alpha + beta);
            SimpleMatrix blockX = neuron.invCov.extractMatrix(0, alpha, 0, alpha);

            SimpleMatrix meanA = neuron.mean.extractMatrix(0, alpha, 0, 1);
            SimpleMatrix meanB = neuron.mean.extractMatrix(alpha, alpha + beta, 0, 1);

            SimpleMatrix invBlockW = blockW.invert();
            SimpleMatrix invBlockA = blockX.minus(blockZ.transpose().mult(invBlockW).mult(blockZ));

            pajs.add(this.mvnpdf(x, meanA, invBlockA, neuron.detCov * blockW.determinant()) + Float.MIN_VALUE);

            SimpleMatrix x_ = meanB.minus((invBlockW).mult(blockZ).mult(x.minus(meanA)));
            xm.add(x_);
        }

        double sumTmp = 0;
        for (int i = 0; i < this.size; i++) {
            sumTmp += pajs.get(i);
        }

        for (int i = 0; i < this.size; i++) {
            pajs.set(i, pajs.get(i) / sumTmp);
        }

        SimpleMatrix result = new SimpleMatrix(beta, 1);

        for (int i = 0; i < this.size; i++) {
            result = result.plus(xm.get(i).scale(pajs.get(i)));
        }

        return result;
    }

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
        double determinant = 1;
        SimpleMatrix sigma = SimpleMatrix.identity(this.dimension).scale(this.delta.getValue() * this.delta.getValue());

        for (int i = 0; i < this.dimension; i++) {
            determinant *= sigma.get(i, i);
            sigma.set(i, i, 1 / sigma.get(i, i));
        }

        this.invSigmaIni = sigma;
        this.detSigmaIni = determinant;
    }

    private void forgettingMechanismn() {
        boolean updtPriors = false, removeComp;

        for (int i = this.size - 1; i >= 0; i--) {
            removeComp = false;
            // Check information about age of components
            if (this.neurons.get(i).v >= (this.size * this.dimension * 2)) {
                // If is estable component (posteriory accumulator is greater than 'D + 1')
                if (this.neurons.get(i).sp >= (this.dimension + 1)) {
                    // But the current posteriory accumulator is not up-to-dated
                    if (this.neurons.get(i).spu < (this.dimension + 1)) {
                        this.neurons.get(i).resetPosteriori(0.75);// this.gamma.getValue());
                        updtPriors = true;
                    }
                    this.neurons.get(i).resetRecentValues();
                } else {
                    updtPriors = true;
                    removeComp = true;
                }
            }
            if (removeComp) {
                this.neurons.remove(i);
                this.like.remove(i);
                this.post.remove(i);
                this.size--;
            }
        }
        if (updtPriors) {
            this.updatePriors();
        }
    }

    /**
     * Inner class that represents a single neuron of IGMN. It contains some
     * analysis information, such as the Gaussian component that represent.
     */
    protected final class NeuronIGMNNSE extends AbstractMOAObject {

        private static final long serialVersionUID = 1L;
        // Mean
        public SimpleMatrix mean;
        // Covariance matrix's Inverse
        public SimpleMatrix invCov;
        // Probability a priori
        public double prior;
        // Covariance matrix's Determinant
        public double detCov;
        // Posteriori accumulator
        public double sp;
        // Distance to the last input pattern
        public double distance;
        // Age
        public int v;
        // Posteriori accumulator up-to-date
        public double spu;

        public NeuronIGMNNSE(SimpleMatrix mean, SimpleMatrix invCov, double detCov) {
            this.mean = mean;
            this.invCov = invCov;
            this.detCov = detCov;
            this.prior = 1;
            this.sp = 1;
            this.v = 1;
            this.spu = 1;
        }

        public void setDetCov(double detCov) {
            this.detCov = detCov;
        }

        private double mvnpdf(SimpleMatrix x) {
            double dim = x.getNumElements();
            SimpleMatrix xs = x.minus(this.mean);
            this.distance = xs.transpose().dot(this.invCov.mult(xs));

            double pdf = Math.exp(-0.5 * this.distance) / (Math.pow(2 * Math.PI, dim / 2.0) * Math.sqrt(this.detCov));

            pdf = Double.isNaN(pdf) ? 0 : pdf;
            pdf = Double.isInfinite(pdf) ? Double.MAX_VALUE : pdf;

            return pdf;
        }

        public void addAge() {
            this.v++;
        }

        public void addPosteriori(double sp) {
            this.sp += sp;
            this.spu += sp;
        }

        public void resetPosteriori(double alpha) {
            this.sp *= alpha;
        }

        public void resetRecentValues() {
            this.v = 0;
            this.spu = 0;
        }

        @Override
        public void getDescription(StringBuilder sb, int indent) {
        }
    }
}
