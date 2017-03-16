package classifier;

/**
 * @author Duruoha-Ihemebiri Ezenwa
 *
 */

import java.util.*;

/*
 *Artificial Neural Network Implementation
 */

public class Network {

	/*
	 * INSTANCE VARIABLES
	 */
	// stores the number of inputs coming from each input vector and neurons in the hidden and output layer
	private int nInput;
	private int nOutput;
	private int nHidden;
	//holds the neurons of the hidden and output layer
	private Neuron[] hiddenLayer;
	private Neuron[] outputLayer;
	// Neural net collection, holding each layer created for the NN
	private List<Neuron[]> NeuralNet = new ArrayList<>();
	// Random number generator
	private Random generator = new Random();

	/*
	 * CONSTRUCTOR
	 */
	public Network(int nInput, int nOutput, int nHidden) {
		this.nInput = nInput;
		this.nOutput = nOutput;
		this.nHidden = nHidden;
		this.hiddenLayer = new Neuron[nHidden];
		this.outputLayer = new Neuron[nOutput];
		generator.setSeed(1);
	}
	
	/*
	 * INSTANCE METHODS
	 */
	// generates random doubles for the weights from a range of -1 to 1
	double getRandom() {
		 return 1 * (generator.nextDouble() * 2 - 1);
	}
	// Initializes the NN by generating random weights and storing them
	public void initNetwork() {

		for (int i = 0; i < nHidden; i++) {
			Neuron neuron = new Neuron(nInput);
			for (int j = 0; j < nInput + 1; j++) {
				double weights = getRandom();
				neuron.addWeight(j, weights);
			}
			hiddenLayer[i] = neuron;
		}
		NeuralNet.add(hiddenLayer);
		for (int i = 0; i < nOutput; i++) {
			Neuron neuron = new Neuron(nHidden);
			for (int j = 0; j < nHidden + 1; j++) {
				double weights = getRandom();
				neuron.addWeight(j, weights);
			}
			outputLayer[i] = neuron;
		}
		NeuralNet.add(outputLayer);
	}
	
	//Function used for one-hot encoding of the expected values (returns a vector like this [0,0,0,0,1] where "1" represent the correct value)
	public int[] oneHotGen(List<Double> row) {
		int[] oneHot = new int[nOutput];
		for (int i = 0; i < oneHot.length; i++) {
			oneHot[i] = 0;
		}
		int label = row.get(row.size() - 1).intValue();
		oneHot[label] = 1;
		return oneHot;
	}
	
	// Function to return the weighted sum of all input to a neuron
	public double sum(Neuron a, List<Double> row) {
		double sum = a.getWeights()[a.getWeights().length - 1];
		for (int i = 0; i < a.getWeights().length - 1; i++) {
			sum += a.getWeights()[i] * row.get(i);
		}
		return sum;
	}
	
	// The weighted sum is passed into the sigmoid function returning a value in the range of 0 to 1
	public double sigmoid(double sum) {
		return 1.0 / (1.0 + Math.exp(-sum));
	}

	// Forward propagate inputs through the neural network, each neuron providing an output
	public List<Double> forwardPropagate(List<Double> row) {
		List<Double> inputs = row.subList(0, row.size() - 1);

		for (int i = 0; i < NeuralNet.size(); i++) {
			Neuron[] layer = NeuralNet.get(i);
			List<Double> hold = new ArrayList<>();

			for (Neuron neuron : layer) {
				double sum = sum(neuron, inputs);
				double output = sigmoid(sum);
				neuron.setOutput(output);
				hold.add(output);
			}
			inputs = hold;
		}

		return inputs;
	}
	// Derivative function used in back propagating errors
	public double costFunction(double output) {
		return output * (1.0 - output);
	}
	// Back propagate errors through the neural net, calculating the delta value for each neuron.
	//Starting from the output layer and back tracking.
	public void backPropagate(int[] expected) {
		for (int i = NeuralNet.size() - 1; i > -1; i--) {

			Neuron[] layer = NeuralNet.get(i);
			List<Double> errors = new ArrayList<>();

			if (i != NeuralNet.size() - 1) {
				for (int j = 0; j < layer.length; j++) {
					double error = 0.0;
					for (Neuron neuron : NeuralNet.get(i + 1)) {
						error += neuron.getWeights()[j] * neuron.getDelta();

					}
					errors.add(error);
				}
			} else {
				for (int j = 0; j < nOutput; j++) {
					Neuron neuron = layer[j];
					double error = expected[j] - neuron.getOutput();
					errors.add(error);
				}
			}
			for (int j = 0; j < layer.length; j++) {
				Neuron neuron = layer[j];
				double delta = errors.get(j) * costFunction(neuron.getOutput());
				neuron.setDelta(delta);

			}
		}
	}
	// Function used to update the weights through the NN giving a learning rate and the calculated delta value
	public void updateWeights(double Lrate, List<Double> row) {

		for (int i = 0; i < NeuralNet.size(); i++) {
			List<Double> input = row.subList(0, row.size() - 1);
			if (i != 0) {
				List<Double> hold = new ArrayList<>();
				for (Neuron neuron : NeuralNet.get(i - 1)) {
					hold.add(neuron.getOutput());
				}
				input = hold;
			}
			for (Neuron neuron : NeuralNet.get(i)) {
				for (int j = 0; j < input.size(); j++) {
					neuron.getWeights()[j] += Lrate * (neuron.getDelta() * input.get(j));
				}
				neuron.getWeights()[neuron.getWeights().length - 1] += (Lrate * neuron.getDelta());
			}

		}
	}
	// Given a range of epochs runs a pass of forward and backwards propagation learning weights for each pass
	public void train(List<List<Double>> dataset, double Lrate, int nEpoch) {
		for (int i = 0; i < nEpoch; i++) {
			//System.out.println("epoc============ " + i);
			for (List<Double> row : dataset) {
				forwardPropagate(row);
				int[] expected = oneHotGen(row);
				backPropagate(expected);
				updateWeights(Lrate, row);
			}
		}
	}
	// Function used to make predictions given a new test vector
	public int predict(List<Double> row) {
		List<Double> output = forwardPropagate(row);
		Double max = Collections.max(output);
		return output.indexOf(max);
	}
	// Returns a collection of predicted class labels for given test dataset
	public List<Integer> classify(List<List<Double>> test) {
		List<Integer> prediction = new ArrayList<>();
		for (List<Double> row : test) {
			Integer label = predict(row);
			prediction.add(label);
			Double expected = row.get(row.size()-1);
			//System.out.println("[INFO] Prediction==[ " + label + " ] Expected==[ " + expected + " ]" );
		}
		return prediction;
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
		double accScore = (cost * 100.0) / prediction.size();
		return accScore;
	}
	// Returns the NN, Showing weights, delta values and outputs of each neuron 
	public void getNeuralNet() {
		for (Neuron[] layer : NeuralNet) {
			System.out.println("=======================================");
			for (Neuron neuron : layer) {
				System.out.println(neuron);
			}

		}
	}
	
	// K-fold Batch learning scheme to avoid over fitting
	public void crossValidate(int nFolds, List<List<Double>> xTrain, List<Integer> yTrain, double Lrate, int nEpoch){
		List<List<List<Double>>> holder = new ArrayList<>();
		List<List<Integer>> labelHolder = new ArrayList<>();
		Random randGen = new Random();
		randGen.setSeed(1);
		int foldSize = xTrain.size()/nFolds;
		
		for(int i = 0; i < nFolds; i++){
			List<List<Double>> fold = new ArrayList<>();
			List<Integer> labels = new ArrayList<>();
			while(fold.size() < foldSize){
				int randIndex = randGen.nextInt(xTrain.size());
				fold.add(xTrain.get(randIndex));
				labels.add(yTrain.get(randIndex));
			}
			holder.add(fold);
			labelHolder.add(labels);
		}
		List<Double> accScore = new ArrayList<>();
		
		int testindex = nFolds-1;
		for(int i = 0; i < nFolds; i++){
			
			train(holder.get(i), Lrate, nEpoch);
			List<Integer> prediction = classify(holder.get(testindex));
			double accuracy = accuracyScore(labelHolder.get(testindex), prediction); 
			accScore.add(accuracy);
			System.out.println("TEST INDEX...["+testindex+"]");
			System.out.println("FOLD...["+i+"]...PREDICTION..."+prediction);
			System.out.println("FOLD...["+i+"]...EXPECTED..."+labelHolder.get(testindex));
			testindex--;
		}
		int scoreCount = 0;
		for(Double score : accScore){
			System.out.println(" ");
			System.out.println("FOLD...["+scoreCount+"]...ACCURACY...["+score+" %]");
			scoreCount++;
		}
		
	}

}
