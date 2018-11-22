package test;

import java.util.ArrayList;
import moa.classifiers.Classifier;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author JorgeCristhian
 */
public class ClassifierTest {

    public Classifier learner;
    public String name;
    public ArrayList<Double> accuracies;
    public ArrayList<Double> kappams;
    public ArrayList<Double> kappats;
    public ArrayList<Double> instances;
    public double time;
    public double ram;
    public double accuracy;
    public double sd_accuracy;
    public double kappam;
    public double sd_kappam;
    public double kappat;
    public double sd_kappat;

    public ClassifierTest(Classifier learner, String name) {
        this.learner = learner;
        this.name = name;
        this.accuracies = new ArrayList<>();
        this.kappams = new ArrayList<>();
        this.kappats = new ArrayList<>();
        this.instances = new ArrayList<>();
    }

    public void mxCalculateValues() {
        double sum_accuracies = 0, sum_kappam = 0, sum_kappat = 0;
        for (int i = 0; i < this.accuracies.size(); i++) {
            sum_accuracies += this.accuracies.get(i);
            sum_kappam += this.kappams.get(i);
            sum_kappat += this.kappats.get(i);
        }

        this.accuracy = sum_accuracies / this.accuracies.size();
        this.sd_accuracy = this.mxCalculateStandardDeviation(this.accuracies, this.accuracy);
        this.kappam = sum_kappam / this.kappams.size();
        this.sd_kappam = this.mxCalculateStandardDeviation(this.kappams, this.kappam);
        this.kappat = sum_kappat / this.kappats.size();
        this.sd_kappat = this.mxCalculateStandardDeviation(this.kappats, this.kappat);
    }

    public double mxCalculateStandardDeviation(ArrayList<Double> loData, double mean) {
        double sum = 0;

        for (int i = 0; i < loData.size(); i++) {
            sum += Math.pow(loData.get(i) - mean, 2.0);
        }

        return Math.sqrt(sum / loData.size());
    }
}
