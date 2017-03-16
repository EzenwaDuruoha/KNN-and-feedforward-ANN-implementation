package classifier;

/**
 * @author Duruoha-Ihemebiri Ezenwa
 *
 */

import java.util.*;
/*
 *Blueprint Class to define a Neuron
 */
public class Neuron {
	// A class variable used to track the number of instances of this class created
	private static int id = 0;
	// Variable to hold the number of inputs that can be passed into each instance of this class
	private int nInput;
	// Stores weights for incoming inputs
	private double[] weights;
	// Stores the calculated delta value for this each instance
	private double delta;
	// stores the output value for each instance of a neuron
	private double output;
	
	/*
	 * CONSTRUCTOR
	 */
	public Neuron(int nInput){
		this.nInput = nInput;
		this.weights = new double[nInput + 1];
		id++;
	}

	/*
	 * INSTANCE METHODS
	 */
	public double getOutput() {
		return output;
	}

	public void setOutput(double output) {
		this.output = output;
	}


	public double[] getWeights() {
		return weights;
	}


	public void setWeights(double[] weights) {
		this.weights = weights;
	}


	public void addWeight(int index, double a){
		this.weights[index] = a;
	}

	public double getW1(int index){
		return this.weights[index];
	}

	public double getDelta() {
		return delta;
	}

	public void setDelta(double delta) {
		this.delta = delta;
	}
	
	/*
	 * Class method
	 */
		public static int getId() {
		return id;
	}
		
		/*
		 * Overriden Methods
		 */		
	@Override
	public String toString() {
		return "Neuron [nInput=" + nInput + ", weights=" + Arrays.toString(weights) + ", delta=" + delta + ", output="
				+ output + "]";
	}

}
