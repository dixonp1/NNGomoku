package deepnet;

import java.util.Random;


public class NeuralNet {
	private int inputNeurons;
	private int leftHiddenNeurons;
	private int rightHiddenNeurons;
	private int outputNeurons;
	private double[][] leftHiddenWeights;
	private double[][] rightHiddenWeights;
	private double[][] outputWeights;
	private double[] leftHiddenBiases;
	private double[] rightHiddenBiases;
	private double[] outputBiases;
	
	private double lambda = 0.8;
	private double eta = 0.005;
	private double gamma = 0.09;
	
	public NeuralNet(int inputN, int lHiddenN, int rHiddenN, int outputN){
		inputNeurons = inputN;
		leftHiddenNeurons = lHiddenN;
		rightHiddenNeurons = rHiddenN;
		outputNeurons = outputN;
		
		leftHiddenWeights = new double[inputNeurons][leftHiddenNeurons];
		rightHiddenWeights 	= new double[leftHiddenNeurons][rightHiddenNeurons];
		leftHiddenBiases	= new double[leftHiddenNeurons];
		rightHiddenBiases = new double[rightHiddenNeurons];
		
		outputWeights 	= new double[rightHiddenNeurons][outputNeurons];
		outputBiases	= new double[outputNeurons];
		
	}
	
	public void initWeights(){
		Random r = new Random();
		for(int i=0; i<inputNeurons; i++){
			for(int j=0; j<leftHiddenNeurons; j++){
				leftHiddenWeights[i][j] = -0.5 + (0.5 - (-0.5)) * r.nextDouble();
			}
		}
		r = new Random();
		for(int i=0; i<leftHiddenNeurons; i++){
			for(int j=0; j<rightHiddenNeurons; j++){
				rightHiddenWeights[i][j] = -0.5 + (0.5 - (-0.5)) * r.nextDouble();
			}
		}
		r = new Random();
		for(int i=0; i<rightHiddenNeurons; i++){
			for(int j=0; j<outputNeurons; j++){
				outputWeights[i][j] = -0.5 + (0.5 - (-0.5)) * r.nextDouble();
			}
		}
		r = new Random();
		for(int i=0; i<leftHiddenNeurons; i++){
			leftHiddenBiases[i] = -0.5 + (0.5 - (-0.5)) * r.nextDouble();
		}
		r = new Random();
		for(int i=0; i<rightHiddenNeurons; i++){
			rightHiddenBiases[i] = -0.5 + (0.5 - (-0.5)) * r.nextDouble();
		}
		r = new Random();
		for(int i=0; i<outputNeurons; i++){
			outputBiases[i] = -0.5 + (0.5 - (-0.5)) * r.nextDouble();
		}
	}
	
	public double[] feedForward(double[] input){
		double[] output = new double[outputNeurons];
		double[] lHidden = new double[leftHiddenNeurons];
		double[] rHidden = new double[rightHiddenNeurons];
		double sum;
		
		for(int i=0; i<leftHiddenNeurons; i++){;
			sum = 1 * leftHiddenBiases[i]; 
			for(int j=0; j<inputNeurons; j++){
				sum += input[j] * leftHiddenWeights[j][i];
			}
			lHidden[i] = sigmoid(sum);
		}
		
		for(int i=0; i<rightHiddenNeurons; i++){;
			sum = 1 * rightHiddenBiases[i]; 
			for(int j=0; j<leftHiddenNeurons; j++){
				sum += lHidden[j] * rightHiddenWeights[j][i];
			}
			rHidden[i] = sigmoid(sum);
	}
		
		for(int i=0; i<outputNeurons; i++){
			sum = 1 * outputBiases[i];
			for(int j=0; j<rightHiddenNeurons; j++){
				sum += rHidden[j] * outputWeights[j][i];
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
	
	public void setWeights(double[][] lHidden, double[][] rHidden, double[][] output, 
			double[] lhBiases, double[] rhBiases, double[] oBiases){
		leftHiddenWeights = lHidden;
		rightHiddenWeights = rHidden;
		outputWeights = output;
		leftHiddenBiases = lhBiases;
		rightHiddenBiases = rhBiases;
		outputBiases = oBiases;
	}
	
	public double[][] getLHW(){
		return leftHiddenWeights;
	}
	public double[][] getRHW(){
		return rightHiddenWeights;
	}
	public double[][] getOW(){
		return outputWeights;
	}
	public double[] getLHB(){
		return leftHiddenBiases;
	}
	public double[] getRHB(){
		return rightHiddenBiases;
	}
	public double[] getOB(){
		return outputBiases;
	}
	
	public void train(double[][] gameStates, double[] reward){
		backprop(gameStates, reward);
	}
	
	private void backprop(double[][] gameStates, double[] reward){
		
		int numStates = gameStates.length;
		double[][] gsOutputs = new double[numStates][outputNeurons];
		double[][] gsLeftHiddenAct = new double[numStates][leftHiddenNeurons];
		double[][] gsRightHiddenAct = new double[numStates][rightHiddenNeurons];
		
		//for each board state in game
		for(int i=0; i<numStates; i++){
			double[] output = new double[outputNeurons];
			double[] lHidden = new double[leftHiddenNeurons];
			double[] rHidden = new double[rightHiddenNeurons];
			double sum;
			
			//feed to first hidden layer
			for(int k=0; k<leftHiddenNeurons; k++){;
				sum = 1 * leftHiddenBiases[k]; 
				for(int j=0; j<inputNeurons; j++){
					sum += gameStates[i][k] * leftHiddenWeights[j][k];
				}
				lHidden[k] = sigmoid(sum);
			}
			//save hidden acts
			gsLeftHiddenAct[i] = lHidden;
			
			//feed to second hidden layer
			for(int k=0; k<rightHiddenNeurons; k++){;
				sum = 1 * rightHiddenBiases[k]; 
				for(int j=0; j<rightHiddenNeurons; j++){
					sum += lHidden[j] * rightHiddenWeights[j][k];
				}
				rHidden[k] = sigmoid(sum);
			}
			//save hidden acts
			gsRightHiddenAct[i] = rHidden;
			
			//feed to output layer
			for(int k=0; k<outputNeurons; k++){
				sum = 1 * outputBiases[k];
				for(int j=0; j<rightHiddenNeurons; j++){
					sum += rHidden[j] * outputWeights[j][k];
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
		for(int d=numStates-2; d>=0; d--){
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

		double[][] owGrads 	= new double[rightHiddenNeurons][outputNeurons];
		double[][] rhwGrads 	= new double[leftHiddenNeurons][rightHiddenNeurons];
		double[][] lhwGrads 	= new double[inputNeurons][leftHiddenNeurons];
		double[] rhbGrads		= new double[rightHiddenNeurons];
		double[] lhbGrads		= new double[leftHiddenNeurons];
		double[] obGrads 		= new double[outputNeurons];
		//calculate grads at each time step
		for(int i=0; i<numStates; i++){
			
			//calculate grads for weights from  second hidden to output
			for(int j=0; j<outputNeurons; j++){
				//calc grad hidden bias
				obGrads[j] += error[i][j] * sigPrime(gsOutputs[i][j]);
				
				for(int k=0; k<rightHiddenNeurons; k++){
					//calc grad for hidden to output weights
					owGrads[k][j] += error[i][j] * sigPrime(gsOutputs[i][j]) 
							* gsRightHiddenAct[i][k];
				}						
			}
			
			double[][] rhwd = new double[leftHiddenNeurons][rightHiddenNeurons];
			//calculate grads for weights from first hidden to second
			for(int j=0; j<rightHiddenNeurons; j++){
				double delta = 0;
				for(int g=0; g<outputNeurons; g++){
					delta += outputWeights[j][g] * (error[i][g] * sigPrime(gsOutputs[i][g]));
				}
				//calc grads bias weights
				rhbGrads[j] += delta * sigPrime(gsRightHiddenAct[i][j]);
				
				//calc grad for weights
				for(int k=0; k<leftHiddenNeurons; k++){
					double grad = delta * sigPrime(gsRightHiddenAct[i][j]) 
							* gsLeftHiddenAct[i][k];
					rhwGrads[k][j] += grad;
					rhwd[k][j] = grad;
				}
			}
			
			//calc grads for weights from input to first hidden
			for(int j=0; j<leftHiddenNeurons; j++){
				double delta = 0;
				for(int g=0; g<rightHiddenNeurons; g++){
					delta += rightHiddenWeights[j][g] * rhwd[j][g] ;
				}
				
				//calc grads bias weights
				lhbGrads[j] += delta * sigPrime(gsLeftHiddenAct[i][j]);
				
				//calc grad for hidden to input weights
				for(int k=0; k<inputNeurons; k++){
					lhwGrads[k][j] += delta * sigPrime(gsLeftHiddenAct[i][j]) 
							* gameStates[i][k];
				}
			}	
		}
		//calculate new weights for hidden to output
		for(int j=0; j<outputNeurons; j++){
			//calc new hidden bias weights
			 outputBiases[j] += (eta/numStates) * obGrads[j];
			
			for(int k=0; k<rightHiddenNeurons; k++){
				//calc new hidden to output weights
				 outputWeights[k][j] += (eta/numStates) * owGrads[k][j];
			}						
		}
		
		//calculate new weights for first hidden to second
		for(int j=0; j<rightHiddenNeurons; j++){
			//calc new bias weights
			 rightHiddenBiases[j] += (eta/numStates) * rhbGrads[j];
			
			//calc discount for input neurons
			for(int k=0; k<leftHiddenNeurons; k++){
				 rightHiddenWeights[k][j]+= (eta/numStates) * rhwGrads[k][j];
			}
		}	
		//calc new wieghts for input to first hidden
		for(int j=0; j<leftHiddenNeurons; j++){
			//calc new bias weights
			 leftHiddenBiases[j] += (eta/numStates) * lhbGrads[j];
			
			//calc discount for input neurons
			for(int k=0; k<inputNeurons; k++){
				 leftHiddenWeights[k][j]+= (eta/numStates) * lhwGrads[k][j];
			}
		}	
	}
}
