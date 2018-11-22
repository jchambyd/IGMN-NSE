/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Locale;

import com.yahoo.labs.samoa.instances.ArffLoader;

import moa.classifiers.IGMN;
import moa.classifiers.IGMNNSE;
import moa.classifiers.core.driftdetection.DDM;
import moa.classifiers.core.driftdetection.EDDM;
import moa.classifiers.core.driftdetection.EWMAChartDM;
import moa.classifiers.drift.SingleClassifierDrift;
import moa.evaluation.preview.LearningCurve;
import moa.streams.ArffFileStream;
import moa.tasks.EvaluatePrequentialCV;

public class Experiment {

    public Experiment() {
    }

    public static ArffFileStream getArffDataset(String nameFile) throws FileNotFoundException {
        ArffLoader loader = new ArffLoader(new FileReader(nameFile));
        ArffFileStream stream = new ArffFileStream(nameFile, loader.getStructure().numAttributes());
        return stream;
    }

    private ArrayList<String> mxGetDataSets() {
        ArrayList<String> namesDataSet = new ArrayList<>();

        namesDataSet.add("data/seaG.arff");
        namesDataSet.add("data/seaA.arff");
        namesDataSet.add("data/FG_2C_2D.arff");
        namesDataSet.add("data/UG_2C_5D.arff");
        namesDataSet.add("data/hyper.arff");
        namesDataSet.add("data/rbf.arff");
        namesDataSet.add("data/rtg.arff");
        namesDataSet.add("data/gaussian.arff");
        namesDataSet.add("data/weather.arff");
        namesDataSet.add("data/elecNormNew.arff");

        return namesDataSet;
    }

    public ArrayList<ClassifierTest> startProcessStream(String pathStream, int frequency, boolean save)
            throws FileNotFoundException {
        // Classifiers
        SingleClassifierDrift learnerEWMA = new SingleClassifierDrift();
        SingleClassifierDrift learnerDDM = new SingleClassifierDrift();
        SingleClassifierDrift learnerEDDM = new SingleClassifierDrift();
        learnerEWMA.driftDetectionMethodOption.setCurrentObject(new EWMAChartDM());
        learnerDDM.driftDetectionMethodOption.setCurrentObject(new DDM());
        learnerEDDM.driftDetectionMethodOption.setCurrentObject(new EDDM());

        // Load Dataset
        ArffFileStream stream = getArffDataset(pathStream);

        ArrayList<ClassifierTest> learners = new ArrayList<>();
        // Selected algorithms
        // learners.add(new ClassifierTest(learnerEWMA, "ECDD"));
        // learners.add(new ClassifierTest(learnerDDM, "DDM"));
        // learners.add(new ClassifierTest(learnerEDDM, "EDDM"));
        learners.add(new ClassifierTest(new IGMN(), "IGMN"));
        learners.add(new ClassifierTest(new IGMNNSE(), "IGMN_NSE"));
        // learners.add(new ClassifierTest(new AdaptiveRandomForest(), "ARF"));

        // Prepare Learners
        for (int i = 0; i < learners.size(); i++) {
            learners.get(i).learner.setModelContext(stream.getHeader());
            learners.get(i).learner.prepareForUse();
        }

        for (int i = 0; i < learners.size(); i++) {
            String filename = Experiment.prepareFileName(learners.get(i).name, pathStream);
            // Prepare stream
            stream.prepareForUse();
            stream.restart();
            // Runs the experiment
            EvaluatePrequentialCV prequentialCV = new EvaluatePrequentialCV();
            prequentialCV.prepareForUse();
            prequentialCV.instanceLimitOption.setValue(100000);
            prequentialCV.sampleFrequencyOption.setValue(frequency);
            prequentialCV.dumpFileOption.setValue("./results/" + filename);
            prequentialCV.streamOption.setCurrentObject(stream);
            prequentialCV.learnerOption.setCurrentObject(learners.get(i).learner);
            LearningCurve lc = (LearningCurve) prequentialCV.doTask();
            // Extract information
            this.getValuesForExperiment(learners.get(i), lc);
        }

        if (save) {
            saveFile(learners, pathStream);
        }

        return learners;
    }

    public void run(boolean save) throws IOException {
        // Prepares the folder that will contain all the results
        Experiment.prepareFolder();
        // Output File
        // PrintWriter outFile = new PrintWriter(new FileWriter("data.txt", save));

        // Prepare Datasets
        ArrayList<String> namesDataSet = this.mxGetDataSets();
        ArrayList<Double> average = new ArrayList<>();
        ArrayList<String> names = new ArrayList<>();

        for (String name : namesDataSet) {
            ArrayList<ClassifierTest> learners = this.startProcessStream(name, 1000, save);

            System.out.println("DATASET: " + name);
            System.out.printf("%12s%12s%12s%12s%12s%12s%16s\n", "Classifier", "Accuracy", "SD-Accu.", "Kappa M",
                    "kappa T", "Time", "RAM-Hours");
            System.out.println(
                    "----------------------------------------------------------------------------------------");
            for (int i = 0; i < learners.size(); i++) {
                System.out.printf("%12s %11.2f %11.2f %11.2f %11.2f %11.2f %15.9f\n", learners.get(i).name,
                        learners.get(i).accuracy, learners.get(i).sd_accuracy, learners.get(i).kappam,
                        learners.get(i).kappat, learners.get(i).time, learners.get(i).ram * Math.pow(10, 9));

                if (average.size() < learners.size()) {
                    average.add(learners.get(i).accuracy);
                    names.add(learners.get(i).name);
                } else {
                    average.set(i, average.get(i) + learners.get(i).accuracy);
                }
            }
        }

        // Final Rank
        Integer numbers[] = new Integer[average.size()];
        for (int i = 0; i < numbers.length; i++) {
            numbers[i] = i;
        }
        Arrays.sort(numbers, (final Integer o1, final Integer o2) -> Double.compare(average.get(o2), average.get(o1)));

        // Print Results
        System.out.println("\nAVERAGE RESULTS:");
        System.out.printf("%12s%12s\n", "Classifier", "Accuracy");
        System.out.println("------------------------");

        for (int i = 0; i < average.size(); i++) {
            System.out.printf("%12s %11.2f\n", names.get(numbers[i]),
                    (double) average.get(numbers[i]) / namesDataSet.size());
        }
    }

    private static void prepareFolder() {
        File folder = new File("./results/");
        File listOfFiles[];
        if (folder.exists()) {
            listOfFiles = folder.listFiles();
            for (File listOfFile : listOfFiles) {
                if (listOfFile.isFile()) {
                    // if (listOfFile.getName().endsWith(".csv")) {
                    listOfFile.delete();
                    // }
                }
            }
        } else {
            folder.mkdir();
        }
        folder = new File("/results/");
        if (folder.exists()) {
            listOfFiles = folder.listFiles();
            for (File listOfFile : listOfFiles) {
                if (listOfFile.isFile()) {
                    // if (listOfFile.getName().endsWith(".csv")) {
                    listOfFile.delete();
                    // }
                }
            }
        } else {
            folder.mkdir();
        }
    }

    private static String prepareFileName(String strClassifier, String strStream) {
        Path p = Paths.get(strStream);

        String filename = p.getFileName() + "_" + strClassifier + ".csv";
        filename = filename.trim();
        filename = filename.replace("-", "_").replace(" ", "_");
        return filename;
    }

    public void saveFile(ArrayList<ClassifierTest> learners, String name) {
        File file = new File("results/data.txt");
        NumberFormat df = NumberFormat.getCurrencyInstance(Locale.US);
        ((DecimalFormat) df).applyPattern("0.00");
        Path p = Paths.get(name);
        name = p.getFileName().toString();
        name = name.substring(0, name.lastIndexOf('.'));

        int numChunks = learners.get(0).accuracies.size();
        // Print chunck results
        try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(file, true)))) {

            // Printing accuracy
            pw.printf("DataSet: %s - Accuracy\n", name);
            pw.printf("%8s,", "Instance");
            int j = 0;
            for (; j < learners.size() - 1; j++) {
                pw.printf("%8s,", learners.get(j).name);
            }
            pw.printf("%8s\n", learners.get(j).name);

            for (int i = 0; i < numChunks; i++) {
                pw.printf("%8s,", "" + learners.get(j).instances.get(i));
                for (j = 0; j < learners.size() - 1; j++) {
                    pw.printf("%8s,", df.format(learners.get(j).accuracies.get(i)));
                }
                pw.printf("%8s\n", df.format(learners.get(j).accuracies.get(i)));
            }

            // Printing Kappa M
            pw.printf("DataSet: %s - Kappa M\n", name);
            pw.printf("%8s,", "Instances");
            j = 0;
            for (; j < learners.size() - 1; j++) {
                pw.printf("%8s,", learners.get(j).name);
            }
            pw.printf("%8s\n", learners.get(j).name);

            for (int i = 0; i < numChunks; i++) {
                pw.printf("%8s,", "" + learners.get(j).instances.get(i));
                for (j = 0; j < learners.size() - 1; j++) {
                    pw.printf("%8s,", df.format(learners.get(j).kappams.get(i)));
                }
                pw.printf("%8s\n", df.format(learners.get(j).kappams.get(i)));
            }

            // Printing Kappa T
            pw.printf("DataSet: %s - Kappa T\n", name);
            pw.printf("%8s,", "Instances");
            j = 0;
            for (; j < learners.size() - 1; j++) {
                pw.printf("%8s,", learners.get(j).name);
            }
            pw.printf("%8s\n", learners.get(j).name);

            for (int i = 0; i < numChunks; i++) {
                pw.printf("%8s,", "" + learners.get(j).instances.get(i));
                for (j = 0; j < learners.size() - 1; j++) {
                    pw.printf("%8s,", df.format(learners.get(j).kappats.get(i)));
                }
                pw.printf("%8s\n", df.format(learners.get(j).kappats.get(i)));
            }

            // Printing Average Results
            PrintWriter pwAcc = new PrintWriter(
                    new BufferedWriter(new FileWriter(new File("results/accuracy.txt"), true)));
            PrintWriter pwKam = new PrintWriter(
                    new BufferedWriter(new FileWriter(new File("results/kappam.txt"), true)));
            PrintWriter pwKat = new PrintWriter(
                    new BufferedWriter(new FileWriter(new File("results/kappat.txt"), true)));
            PrintWriter pwTim = new PrintWriter(new BufferedWriter(new FileWriter(new File("results/time.txt"), true)));

            pwAcc.printf("%15s", name);
            for (j = 0; j < learners.size() - 1; j++) {
                pwAcc.printf("%8s,", df.format(learners.get(j).accuracy));
            }
            pwAcc.printf("%8s\n", df.format(learners.get(j).accuracy));

            pwKam.printf("%12s", name);
            for (j = 0; j < learners.size() - 1; j++) {
                pwKam.printf("%8s,", df.format(learners.get(j).kappam));
            }
            pwKam.printf("%8s\n", df.format(learners.get(j).kappam));

            pwKat.printf("%12s", name);
            for (j = 0; j < learners.size() - 1; j++) {
                pwKat.printf("%8s,", df.format(learners.get(j).kappat));
            }
            pwKat.printf("%8s\n", df.format(learners.get(j).kappat));

            pwTim.printf("%12s", name);
            for (j = 0; j < learners.size() - 1; j++) {
                pwTim.printf("%8s,", df.format(learners.get(j).time));
            }
            pwTim.printf("%8s\n", df.format(learners.get(j).time));

            pwAcc.close();
            pwKam.close();
            pwKat.close();
            pwTim.close();

        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    private void getValuesForExperiment(ClassifierTest classifier, LearningCurve lc) {
        int indexAcc = -1;
        int indexKappam = -1;
        int indexKappat = -1;
        int indexCpuTime = -1;
        int indexRamHours = -1;
        int indexInstances = -1;

        int index = 0;
        for (String s : lc.headerToString().split(",")) {
            if (s.contains("[avg] classifications correct")) {
                indexAcc = index;
            } else if (s.contains("time")) {
                indexCpuTime = index;
            } else if (s.contains("RAM-Hours")) {
                indexRamHours = index;
            } else if (s.contains("[avg] Kappa M")) {
                indexKappam = index;
            } else if (s.contains("[avg] Kappa Temporal")) {
                indexKappat = index;
            } else if (s.contains("learning evaluation instances")) {
                indexInstances = index;
            }
            index++;
        }

        // Reading all values
        for (int entry = 0; entry < lc.numEntries(); entry++) {

            classifier.accuracies.add(lc.getMeasurement(entry, indexAcc));
            classifier.kappams.add(lc.getMeasurement(entry, indexKappam));
            classifier.kappats.add(lc.getMeasurement(entry, indexKappat));
            classifier.instances.add(lc.getMeasurement(entry, indexInstances));
        }
        // Calculating statistical values
        classifier.mxCalculateValues();
        // but both cpu time and ram hours are only the final values obtained
        // since they represent the processing of the entire stream
        classifier.time = lc.getMeasurement(lc.numEntries() - 1, indexCpuTime);
        classifier.ram = lc.getMeasurement(lc.numEntries() - 1, indexRamHours);
    }

    public static void main(String[] args) throws IOException {
        Experiment exp = new Experiment();
        exp.run(false);
    }
}
