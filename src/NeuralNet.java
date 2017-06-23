import java.util.Random;


public class NeuralNet {
	private int inputNeurons;
	private int hiddenNeurons;
	private int outputNeurons;
	private double[][] hiddenWeights;
	private double[][] outputWeights;
	private double[] hiddenBiases;
	private double[] outputBiases;
	
	//learning parameters 
	private double lambda = 0.8; // how much feedback each state gets from next
	private double eta = 0.001; //learning rate
	private double gamma = 0.09; //decay factor
	
	public NeuralNet(int inputN, int hiddenN, int outputN){
		inputNeurons = inputN;
		hiddenNeurons = hiddenN;
		outputNeurons = outputN;
		
		hiddenWeights 	= new double[inputNeurons][hiddenNeurons];
		hiddenBiases	= new double[hiddenNeurons];
		
		outputWeights 	= new double[hiddenNeurons][outputNeurons];
		outputBiases	= new double[outputNeurons];
		
	}
	
	//initialize weights to random values between -0.5 and 0.5
	public void initWeights(){
		Random r = new Random();
		for(int i=0; i<inputNeurons; i++){
			for(int j=0; j<hiddenNeurons; j++){
				hiddenWeights[i][j] = -0.5 + (0.5 - (-0.5)) * r.nextDouble();
			}
		}
		r = new Random();
		for(int i=0; i<hiddenNeurons; i++){
			for(int j=0; j<outputNeurons; j++){
				outputWeights[i][j] = -0.5 + (0.5 - (-0.5)) * r.nextDouble();
			}
		}
		r = new Random();
		for(int i=0; i<hiddenNeurons; i++){
			hiddenBiases[i] = -0.5 + (0.5 - (-0.5)) * r.nextDouble();
		}
		r = new Random();
		for(int i=0; i<outputNeurons; i++){
			outputBiases[i] = -0.5 + (0.5 - (-0.5)) * r.nextDouble();
		}
	}
	
	//feed input forward through network
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
	
	//activation function
	private double sigmoid(double sum){
		return 1/(1+Math.exp(-sum));
	}
	
	//derivative of activation function
	private double sigPrime(double input){
		return input * (1 - input);
	}
	
	//sets weights (used for loading existing weights)
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
	
	public void train(double[][] gameStates, double[] reward){
		backprop(gameStates, reward);
	}
	
	//propagate error backwards for a game using temporal difference learning
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
		
		double totalError = 0;
		
		//calc the error of the reward and output
		double[][] dvalues = new double[numStates][outputNeurons];
		double[][] error  = new double[numStates][outputNeurons];
		for(int td=0; td<outputNeurons; td++){
			dvalues[numStates - 1][td] = reward[td];
			error[numStates - 1][td] = dvalues[numStates - 1][td] - gsOutputs[numStates-1][td];
			
			totalError += error[numStates - 1][td] * error[numStates - 1][td];
		}
		
		
		//calc desired value for each state
		//D(t) = lambda * D(t+1) + eta * (1-lambda) * (R(t) + gamma * A(t+1) - A(t))
		//R(t) = gamma * R(t+1)
		//E(t) = D(t) - A(t)
		for(int d=numStates-2; d>0; d--){
			for(int r=0; r<outputNeurons; r++){
				reward[r] = gamma * reward[r];
				dvalues[d][r] = lambda * dvalues[d+1][r] + eta * ((1 - lambda) 
						* (reward[r] + gamma * gsOutputs[d+1][r] - gsOutputs[d][r]));
				error[d][r] = dvalues[d][r] - gsOutputs[d][r];
				
				totalError += error[d][r] * error[d][r];
			}
		}
		
		//System.out.println("Error first move: " + 0.5 * (error[numStates-2][0] * error[numStates-2][0]));
		System.out.println("Error last move: " + 0.5 * totalError);

		double[][] owGrads 	= new double[hiddenNeurons][outputNeurons];
		double[][] hwGrads 	= new double[inputNeurons][hiddenNeurons];
		double[] hbGrads		= new double[hiddenNeurons];
		double[] obGrads 		= new double[outputNeurons];
		//calculate deltas at each time step
		for(int i=0; i<numStates; i++){
			
			//calculate grads for weights from hidden to output
			for(int j=0; j<outputNeurons; j++){
				//calc grads hidden bias
				obGrads[j] += error[i][j] * sigPrime(gsOutputs[i][j]);
				
				for(int k=0; k<hiddenNeurons; k++){
					//calc grad for hidden to output weights
					owGrads[k][j] += error[i][j] * sigPrime(gsOutputs[i][j]) 
							* gsHiddenAct[i][k];
				}						
			}
			
			//calculate deltas for weights from input to hidden neurons
			for(int j=0; j<hiddenNeurons; j++){
				double delta = 0;
				for(int g=0; g<outputNeurons; g++){
					delta += outputWeights[j][g] * owGrads[j][g];
				}
				
				//calc deltas bias weights
				hbGrads[j] += delta * sigPrime(gsHiddenAct[i][j]);
				
				//calc delta for hidden to input weights
				for(int k=0; k<inputNeurons; k++){
					hwGrads[k][j] += delta * sigPrime(gsHiddenAct[i][j]) * gameStates[i][k];
				}
			}		
		}
		//calculate new weights for hidden to output
		for(int j=0; j<outputNeurons; j++){
			//calc new hidden bias weights
			 outputBiases[j] += (eta/numStates) * obGrads[j];
			
			for(int k=0; k<hiddenNeurons; k++){
				//calc new hidden to output weights
				 outputWeights[k][j] += (eta/numStates) * owGrads[k][j];
			}						
		}
		
		//calculate new weights for input to hidden neurons
		for(int j=0; j<hiddenNeurons; j++){
			//calc new bias weights
			 hiddenBiases[j] += (eta/numStates) * hbGrads[j];
			
			//calc discount for input neurons
			for(int k=0; k<inputNeurons; k++){
				 hiddenWeights[k][j]+= (eta/numStates) * hwGrads[k][j];
			}
		}	
	}
}
