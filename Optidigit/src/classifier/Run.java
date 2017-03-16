package classifier;

/**
 * @author Duruoha-Ihemebiri Ezenwa
 *
 */

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;

/*
 * SYSTEM RUNS FROM THIS CLASS
 */

public class Run {
	// Stores the class labels for the training and test dataset
	private static List<Integer> testlabels = new ArrayList<>();
	private static List<Integer> labels = new ArrayList<>();
	// Stores the feature vectors from the training and test dataset 
	private static List<List<Double>> feat = new ArrayList<>();
	private static List<List<Double>> test = new ArrayList<>();
	//Stores predictions from the KNN and ANN classifiers
	private static List<Integer> predictionsNN = new ArrayList<>();
	private static List<Integer> predictionskNN = new ArrayList<>();
	// Scanner object to get user input
	private static Scanner input = new Scanner(System.in);
	// System arguments used when running the classifiers
	private static String[] run = { "ann","knn" };

	public static void main(String[] args) {
		// Prompts users to input file path to the training and test datasets
		System.out.println("INPUT TRAINING FILE PATH: ");
		String f_name = input.next();
		
		System.out.println("INPUT TEST FILE PATH: ");
		String t_name = input.next();

		// reads and stores the feature vectors and labels from the training and test dataset
		feat = readFile(f_name, "nn");
		test = readFile(t_name, "test");
		
		
		for (String arg : run) {
			System.out.println("RUNNING....[" + arg + "].....ALGORITHM");
			// Control argument to run one classifier after the other 
			if (arg == "ann") {
				// Normalizes the dataset, reducing the range of values of each feature to that of 0 to 1
				feat = normalize(feat);
				test = normalize(test);
				// gets the input size and output size based on the given dataset
				int nInput = feat.get(0).size() - 1;
				int nOutput = getInput(labels);
				//Initializes the NN passing it the input, output and hidden neuron size
				Network net = new Network(nInput, nOutput, 20);
				// Initializes the weights in the NN
				net.initNetwork();
				System.out.println("=====================NEURAL NET========================= ");
				// Runs Batch training with the given training dataset 
				net.crossValidate(2, feat, labels, 0.5, 500);
				// Returns predictions of given test set
				predictionsNN = net.classify(test);
				// Calculate the accuracy score for the test set and prints to screen 
				double acc = net.accuracyScore(testlabels, predictionsNN);
				System.out.println("[INFO] ANN accuracy on test set.... " + acc + "%");
				System.out.println(" ");
				
			}else if (arg == "knn"){
				//Initializes the kNN classifier
				KNN classifier = new KNN();
				// Fits in the training vectors and the associated class labels
				classifier.fit(feat, labels);
				System.out.println("=====================KNN========================= ");
				// Runs K-fold testing to ensure over fitting is avoided
				classifier.crossValidate(2, 1);
				// Returns predictions of given test set
				predictionskNN = classifier.predict(test, 1);
				// Calculate the accuracy score for the test set and prints to screen 
				double acc = classifier.accuracyScore(testlabels, predictionskNN);
				System.out.println("[INFO] KNN accuracy on test set..." + acc + "%");
				System.out.println(" ");
			}
		}
		
	}
	// function to read the feature vectors and labels for a csv file
	private static List<List<Double>> readFile(String filename, String mode) {
		List<List<Double>> dataset = new ArrayList<>();

		try {
			File f = new File(filename);
			BufferedReader br = new BufferedReader(new FileReader(f));
			String line;
			while ((line = br.readLine()) != null) {
				String[] tokens = line.split(",");
				if (mode == "nn") {

					List<Double> features = toInteger(tokens);
					dataset.add(features);
					int label = Integer.parseInt(tokens[tokens.length - 1]);
					labels.add(label);

				} else if (mode == "test") {
					
					List<Double> features = toInteger(tokens);
					dataset.add(features);
					int label = Integer.parseInt(tokens[tokens.length - 1]);
					testlabels.add(label);
				}

			}
			br.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return dataset;
	}
	// converts an array of string into a list of doubles
	private static List<Double> toInteger(String[] args) {
		List<Double> hold = new ArrayList<>();
		for (int i = 0; i < args.length; i++) {
			Double x = Double.parseDouble(args[i]);
			hold.add(x);
		}
		return hold;
	}
	// Finds the minimum and maximum for each column of values in the dataset 
	public static List<List<Double>> minmax(List<List<Double>> dataset) {
		List<List<Double>> data = new ArrayList<>();
		for (int i = 0; i < dataset.get(0).size() - 1; i++) {
			List<Double> hold = new ArrayList<>();
			List<Double> hold2 = new ArrayList<>();
			for (List<Double> row : dataset) {
				hold.add(row.get(i));
			}
			hold2.add(Collections.min(hold));
			hold2.add(Collections.max(hold));
			data.add(hold2);
		}
		return data;
	}
	// Normalizes the dataset using the minimum and maximum values given
	public static List<List<Double>> normalize(List<List<Double>> dataset) {
		List<List<Double>> data = new ArrayList<>();
		List<List<Double>> minimax = minmax(dataset);
		for (List<Double> row : dataset) {
			List<Double> normalized = new ArrayList<>();
			for (int i = 0; i < row.size() - 1; i++) {
				Double norm = 0.0;
				if (minimax.get(i).get(0) == 0 && minimax.get(i).get(1) == 0) {
					norm = 0.0;
					normalized.add(norm);
				} else {
					norm = (row.get(i) - minimax.get(i).get(0)) / (minimax.get(i).get(1) - minimax.get(i).get(0));
					normalized.add(norm);
				}
			}
			normalized.add(row.get(row.size() - 1));
			data.add(normalized);
		}
		return data;
	}

	public static int getInput(List<Integer> a) {
		Set<Integer> unique = new HashSet<>(a);
		return unique.size();
	}
	

}
