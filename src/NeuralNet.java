
public class NeuralNet {
	private int inputNeurons;
	private int hiddenNeurons;
	private int outputNeurons;
	private double[][] hiddenWeights;
	private double[][] outputWeights;
	private double[] hiddenBiases;
	private double[] outputBiases;
	
	private double lambda = 0.9;
	private double alpha = 0.001;
	
	public NeuralNet(int inputN, int hiddenN, int outputN){
		inputNeurons = inputN;
		hiddenNeurons = hiddenN;
		outputNeurons = outputN;
		
		hiddenWeights 	= new double[inputNeurons][hiddenNeurons];
		hiddenBiases	= new double[hiddenNeurons];
		
		outputWeights 	= new double[hiddenNeurons][outputNeurons];
		outputBiases	= new double[outputNeurons];
		
	}
	
	public void initWeights(){
		for(int i=0; i<inputNeurons; i++){
			for(int j=0; j<hiddenNeurons; j++){
				hiddenWeights[i][j] = Math.random();
			}
		}
		
		for(int i=0; i<hiddenNeurons; i++){
			for(int j=0; j<outputNeurons; j++){
				outputWeights[i][j] = Math.random();
			}
		}
		
		for(int i=0; i<hiddenNeurons; i++){
			hiddenBiases[i] = Math.random();
		}
		
		for(int i=0; i<outputNeurons; i++){
			outputBiases[i] = Math.random();
		}
	}
	
	public double[] feedForward(double[] input){
		double[] output = new double[outputNeurons];
		double[] hidden = new double[hiddenNeurons];
		double sum;
		
		for(int i=0; i<hiddenNeurons; i++){;
			sum = 1 * hiddenBiases[i]; 
			for(int j=0; j<inputNeurons; j++){
				sum += input[j] * hiddenWeights[j][i];
			}
			hidden[i] = sigmoid(sum);
		}
		
		for(int i=0; i<outputNeurons; i++){
			sum = 1 * outputBiases[i];
			for(int j=0; j<hiddenNeurons; j++){
				sum += hidden[j] * outputWeights[j][i];
			}
			output[i] = sigmoid(sum);
		}
		
		return output;
	}
	
	private double sigmoid(double sum){
		return 1/(1+Math.exp(-sum));
	}
	
	private double sigPrime(double input){
		return input * (1 - input);
	}
	
	public void setWeights(double[][] hidden, double[][] output, 
			double[] hBiases, double[] oBiases){
		hiddenWeights = hidden;
		outputWeights = output;
		hiddenBiases = hBiases;
		outputBiases = oBiases;
	}
	
	public double[][] getHW(){
		return hiddenWeights;
	}
	public double[][] getOW(){
		return outputWeights;
	}
	public double[] getHB(){
		return hiddenBiases;
	}
	public double[] getOB(){
		return outputBiases;
	}
	
//	private double crossEntropy(double[] target, double[] actual){
//		double sum = 0;
//		for(int i=0; i<target.length; i++){
//			sum += target[i] * Math.log(actual[i]) + (1 - target[i]) * Math.log(1 - actual[i]);
//		}
//		return sum;
//	}
	
	public void train(double[][] gameStates, double[] reward){
		backprop(gameStates, reward);
	}
	
	private void backprop(double[][] gameStates, double[] reward){
		
		int numStates = gameStates.length;
		double[][] gsOutputs = new double[numStates][outputNeurons];
		double[][] gsHiddenAct = new double[numStates][hiddenNeurons];
		
		//for each board state in game
		for(int i=0; i<numStates; i++){
			double[] output = new double[outputNeurons];
			double[] hidden = new double[hiddenNeurons];
			double sum;
			
			//feed to hidden layer
			for(int k=0; k<hiddenNeurons; k++){;
				sum = 1 * hiddenBiases[k]; 
				for(int j=0; j<inputNeurons; j++){
					sum += gameStates[i][k] * hiddenWeights[j][k];
				}
				hidden[k] = sigmoid(sum);
			}
			//save hidden acts
			gsHiddenAct[i] = hidden;
			
			//feed to output layer
			for(int k=0; k<outputNeurons; k++){
				sum = 1 * outputBiases[k];
				for(int j=0; j<hiddenNeurons; j++){
					sum += hidden[j] * outputWeights[j][k];
				}
				output[k] = sigmoid(sum);
			}
			//save output acts
			gsOutputs[i] = output;
		}
		
		//calculate new weights at each time step
		for(int i=numStates; i>0; i--){
			double[][] newOutWeights 	= new double[hiddenNeurons][outputNeurons];
			double[][] newHiddenWeights = new double[inputNeurons][hiddenNeurons];
			double[] newInputBiases		= new double[hiddenNeurons];
			double[] newHiddenBiases 	= new double[outputNeurons];
			
			double[] temporalDifference = new double[outputNeurons];
			for(int td=0; td<outputNeurons; td++){
				temporalDifference[td] = reward[td] - gsOutputs[i-1][td];
			}
			
			//calculate new weights for hidden to output
			for(int j=0; j<outputNeurons; j++){
				double biasesTD 	= 0;
				double[] outputTD 	= new double[hiddenNeurons];
				//calc discount for bias neuron
				for(int n=1; n<=i; n++){
					biasesTD +=  Math.pow(lambda, i-n) * sigPrime(gsOutputs[n-1][j]);
				}
				//calc new hidden bias weights
				newHiddenBiases[j] = outputBiases[j] + alpha * 
							temporalDifference[j] * biasesTD;
				
				for(int k=0; k<hiddenNeurons; k++){
					//calc discount for hidden neurons
					for(int n=1; n<=i; n++){
						outputTD[k] +=  Math.pow(lambda, i-n) * sigPrime(gsOutputs[n-1][j]) *
								gsHiddenAct[n-1][k];
					}
					//calc new hidden to output weights
					newOutWeights[k][j] = outputWeights[k][j] + alpha * 
							temporalDifference[j] * outputTD[k];
				}						
			}
			
			//calculate new weights for input to hidden neurons
			for(int j=0; j<hiddenNeurons; j++){
				double biasesTD		= 0;
				double[] hiddenTD	= new double[inputNeurons];
				
				double gradient = 0;
				//calc discount for bias neuron
				for(int n=1; n<=i; n++){
					//calc gradient for hidden neuron
					for(int g=0; g<outputNeurons; g++){
						gradient += sigPrime(gsOutputs[n-1][g]) * outputWeights[j][g] 
								* temporalDifference[g];
					}
					gradient = gradient * sigPrime(gsHiddenAct[n-1][j]);
					biasesTD += Math.pow(lambda, i-n) * gradient;
				}
				//calc new bias weights
				newInputBiases[j] = hiddenBiases[j] + alpha * biasesTD;
				
				//calc discount for input neurons
				for(int k=0; k<inputNeurons; k++){
					for(int n=1; n<=i; n++){
						hiddenTD[k] += Math.pow(lambda, i-n) * gradient * gameStates[n-1][k];
					}
					newHiddenWeights[k][j] = hiddenWeights[k][j] + alpha * hiddenTD[k];
				}
			}
			
			//update weights and biases
			outputWeights = newOutWeights;
			hiddenWeights = newHiddenWeights;
			hiddenBiases = newInputBiases;
			outputBiases = newHiddenBiases;
			reward = gsOutputs[i-1];
		}
	}
}
