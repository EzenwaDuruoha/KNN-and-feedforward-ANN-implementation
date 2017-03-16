package classifier;

/**
 * @author Duruoha-Ihemebiri Ezenwa
 *
 */

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

/*
 *K-Nearest Neighbor Implementation
 */
public class KNN {

	// Instance variables to hold the training vectors and their class labels
	private List<List<Double>> xTrain = new ArrayList<>();
	private List<Integer> yTrain = new ArrayList<>();
	

	// Initializes and fits the classifier with a dataset and labels for the vectors in the dataset 
	public void fit(List<List<Double>> feat, List<Integer> labels) {
		for(List<Double> row : feat){
			List<Double> noLabel = row.subList(0, row.size()-1);	
			this.xTrain.add(noLabel);
		}
		this.yTrain = labels;
	}
	
	// Function used to make predictions given a new test dataset, and k = no. of nearest neighbors
	public List<Integer> predict(List<List<Double>> test, int k) {
		List<Integer> prediction = new ArrayList<>();
		for (List<Double> row : test) {
			Integer label = getClosest(row, k, this.xTrain, this.yTrain);
			prediction.add(label);
		}
		//System.out.println("[INFO] Prediction >>> " + prediction);
		return prediction;
	}
	
	// Overloaded predict function used for cross validation
	public List<Integer> predict(List<List<Double>> test, int k, List<List<Double>>train, List<Integer>labels ) {
		List<Integer> prediction = new ArrayList<>();
		for (List<Double> row : test) {
			Integer label = getClosest(row, k, train, labels);
			prediction.add(label);
		}
		//System.out.println("[INFO] Prediction >>> " + prediction);
		return prediction;
	}
	
	// calculates the Euclidean distance between two vectors
	private Double functionEuclidean(List<Double> a, List<Double> b) {
		List<Double> processed = new ArrayList<>();
		Iterator<Double> point_train = a.iterator();
		Iterator<Double> point_test = b.iterator();

		while (point_train.hasNext() && point_test.hasNext()) {
			double x = point_train.next();
			double y = point_test.next();
			double z = (x - y) * (x - y);
			processed.add(z);
		}
		Double hold = 0.0;
		for (Double i : processed) {
			hold += i;
		}
		Double euc = Math.sqrt(hold);

		return euc;
	}

	// Returns a singular prediction given a new feature vector
	private Integer getClosest(List<Double> a, int k, List<List<Double>> xTrain, List<Integer> yTrain) {
		List<Integer> bestLabels = new ArrayList<>();
		Double bestFit = 9999.9;
		int bestValue = 0;
		for (int i = 0; i < xTrain.size(); i++) {
			Double distance = functionEuclidean(a, xTrain.get(i));
			if (distance < bestFit) {
				bestFit = distance;
				bestLabels.add(yTrain.get(i));
			}
		}
		if(bestLabels.size() < k){
			k = bestLabels.size();
		}
		Collections.reverse(bestLabels);
		bestValue = counter(bestLabels, k);
		return bestValue;
	}

	// Majority vote function to determine the class label of a given feature vector
	private int counter(List<Integer> index, int k) {
		Map<Integer, Integer> counter = new HashMap<>();
		for (int i = 0; i < k; i++) {
			int x = index.get(i);
			Integer hold = counter.get(x);
			counter.put(x, hold == null ? 1 : hold + 1);
		}

		int max = 0;
		int key = 0;
		for (Integer i : counter.keySet()) {
			if (counter.get(i) > max) {
				max = counter.get(i);
				key = i;
			}
		}

		return key;
	}
	
	// Calculates the accuracy of prediction given a list of expected outcomes and a list of prediction
	public double accuracyScore(List<Integer> labels, List<Integer> prediction) {
		Iterator<Integer> iterL = labels.iterator();
		Iterator<Integer> iterP = prediction.iterator();
		int cost = 0;
		while (iterL.hasNext() && iterP.hasNext()) {
			Integer x = iterL.next();
			Integer y = iterP.next();
			if (x.equals(y)) {
				cost += 1;
			}
		}
		double accScore = cost * 100.0 / prediction.size();
		return accScore;
	}
	
	// K-fold validation function used to evaluate the classifier and check for over-fitting
	public void crossValidate(int nFolds, int k){
		List<List<List<Double>>> holder = new ArrayList<>();
		List<List<Integer>> labelHolder = new ArrayList<>();
		Random randGen = new Random();
		randGen.setSeed(1);
		int foldSize = this.xTrain.size()/nFolds;
		
		for(int i = 0; i < nFolds; i++){
			List<List<Double>> fold = new ArrayList<>();
			List<Integer> labels = new ArrayList<>();
			while(fold.size() < foldSize){
				int randIndex = randGen.nextInt(this.xTrain.size());
				fold.add(this.xTrain.get(randIndex));
				labels.add(this.yTrain.get(randIndex));
			}
			holder.add(fold);
			labelHolder.add(labels);
		}
		List<Double> accScore = new ArrayList<>();
		
		int testindex = nFolds-1;
		for(int i = 0; i < nFolds; i++){
			List<Integer> prediction = predict(holder.get(testindex), k, holder.get(i), labelHolder.get(i));
			double accuracy = accuracyScore(labelHolder.get(testindex), prediction); 
			accScore.add(accuracy);
			System.out.println("TEST INDEX...["+testindex+"]");
			System.out.println("FOLD...["+i+"]...PREDICTION..."+prediction);
			System.out.println("FOLD...["+i+"]...EXPECTED..."+labelHolder.get(testindex));
			testindex--;
		}
		int scoreCount = 0;
		for(Double score : accScore){
			System.out.println("  ");
			System.out.println("FOLD...["+scoreCount+"]...ACCURACY...["+score+" %]");
			scoreCount++;
		}
		
	}

	

}
