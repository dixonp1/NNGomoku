package deepnet;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

public class GomokuClient {

	protected Socket myClient = null;// our client socket.
	protected PrintWriter os = null;
	protected BufferedReader is = null;// Input stream.
	
	protected String gs = null;// GameState.
	private Board board;
	private char me;
	
	//network structure
	private NeuralNet net;
	private int outputNeurons = 1;
	private int leftHiddenNeurons = 50;
	private int rightHiddenNeurons = 30;
	private int inputNeurons = 82;
	
	private ArrayList<Double[]> gameStates;

	public static void main(String args[]) throws IOException {
		GomokuClient client = new GomokuClient();
		boolean train = false;
		boolean resume = false;
		int trainingGames = 0;
		
		//check for cmd arguments
		if(args.length >= 2){
			//if training set train=true and set training games
			if(args[0].equals("train")){ train = true; }
			trainingGames = Integer.parseInt(args[1]);
			
			//checks for additional arg "resume"
			if(args.length > 2){
				if(args[2].equals("resume")){
					resume = true;
				}
			}
		}
		
		try {
			//start training if in training mode
			//else run normally
			if(train){
				client.trainNN(resume, trainingGames);
			}else{
				//load weights
				client.loadWeights();
				//connect, play, disconnect
				client.socketConnect("localhost", 17033);
				client.play(false);
				client.closeConnection();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public GomokuClient(){
		gs = "";// Set up-setting to an empty string.
		net = new NeuralNet(inputNeurons, leftHiddenNeurons, rightHiddenNeurons, outputNeurons);
		board = new Board(inputNeurons);
	}
	
	private void closeConnection(){
		try {
			os.close();
			is.close();
			myClient.close();
		} catch (UnknownHostException e) {
			System.err.println("Trying to connect to unknown host: " + e);
		} catch (IOException e) {
			System.err.println("IOException: " + e);
		}
	}

	public boolean play(boolean training) {
		try {
			while (true) {
				//read in board
				board = readIn();
					
				//break if game over
				if (!gs.equals("continuing")) {	
					if(training && gs.equals("lose")) { addGS(board.getBoard()); }
					break;	
				}
				int move = pickmove(me);
				
				//if training, add current game state and game state after
				//move made to arraylist
				if(training) { 
					double[] tempboard = new double[inputNeurons];
					double[] cboard = board.getBoard();
					System.arraycopy(cboard, 0, tempboard, 0, tempboard.length);
					tempboard[move] = 1;
					tempboard[inputNeurons - 1] = -1;
					addGS(cboard);
					addGS(tempboard);	
				}
				
				//send move
				moveSend(move);
			}
		} catch (SocketTimeoutException e) {
			return false;
		}catch (Exception e) {}
		return true;
	}

	public void socketConnect(String h, int p) throws Exception {
		try {
			myClient = new Socket(h, p);
			os = new PrintWriter(myClient.getOutputStream(), true);
			is = new BufferedReader(new InputStreamReader(
					myClient.getInputStream()));
			myClient.setSoTimeout(5000);
		} catch (UnknownHostException e) {
			System.err.println("Don't know about host: ");
		} catch (IOException e) {
			System.err.println("Couldn't get I/O for the connection to: ");
		}
	}

	public Board readIn() throws Exception {
		String s;
		double[] gb = new double[inputNeurons];
		Board b = new Board(inputNeurons);
		int i, j;
		gs = is.readLine();
		for (i = 0; i < 9; i++) {
			s = is.readLine();
			for (j = 0; j < 9; j++) {
				if (s.charAt(j) == me) {
					gb[9 * i + j] = 1;
				} else if (s.charAt(j) == ' ') {
					gb[9 * i + j] = 0;
				}else{
					gb[9 * i + j] = -1;
				}
			}
		}
		gb[inputNeurons - 1] = 1;

		me = is.readLine().charAt(0);
		b.setBoard(gb);
		return b;
	}

	public int pickmove(char player) {
		ArrayList<Integer> moves = board.getMoves();
		int bestMove = 0;
		double value = -1000000;
		
		//iterate through all possible moves
		//the net evaluates each position for each move
		//picks largest value for current player
		for(int i=0; i<moves.size(); i++){
			double[] newboard = new double[inputNeurons];
			System.arraycopy(board.getBoard(), 0, newboard, 0, newboard.length);
			newboard[moves.get(i)] = 1;
			double[] output = net.feedForward(newboard);
			
			if(output[0] > value){
				value = output[0];
				bestMove = moves.get(i);
			}
		}
		return bestMove;
	}

	//converts move from int to string "x y" and sends to game server
	public void moveSend(int move) throws Exception {
		int row = (int) (move / 9);
		int col = move % 9;
		os.println(row + " " + col);
	}
	
	//packs game states into arraylist
	private void addGS(double[] gstate){
		Double[] temp = new Double[inputNeurons];
		for(int i=0; i<inputNeurons; i++){
			temp[i] = new Double(gstate[i]);
		}
		gameStates.add(temp);
	}
	
	//unpack game states from arraylist
	private double[][] unpackGS(){
		Double[] gstate = new Double[inputNeurons];
		double[][] temp = new double[gameStates.size()][inputNeurons];
		for(int j=0; j<gameStates.size(); j++){
			gstate = gameStates.get(j);
			for(int i=0; i<inputNeurons; i++){
				temp[j][i] = gstate[i].doubleValue();
			}
		}
		return temp;
	}
	
	private int loadWeights(){
		String file = "weights.txt";
		double[][] leftHiddenWeights = new double[inputNeurons][leftHiddenNeurons];
		double[][] rightHiddenWeights = new double[leftHiddenNeurons][rightHiddenNeurons];
		double[] leftHiddenBiases = new double[leftHiddenNeurons];
		double[] rightHiddenBiases = new double[rightHiddenNeurons];
		double[][] outputWeights = new double[rightHiddenNeurons][outputNeurons];
		double[] outputBiases = new double[outputNeurons];
		
		//read in weights from file
		try (BufferedReader br = new BufferedReader(new FileReader(file))){
			String[] line;
			
			//read in hidden weights
			for(int i=0; i<inputNeurons; i++){
				line = br.readLine().split(" ");
				for(int j=0; j<leftHiddenNeurons; j++){
					leftHiddenWeights[i][j] = Double.parseDouble(line[j]);
				}
			}
			//skip line
			line = br.readLine().split(" ");

			//read in hidden biases
			line = br.readLine().split(" ");
			for(int i=0; i<leftHiddenNeurons; i++){
				leftHiddenBiases[i] = Double.parseDouble(line[i]);
			}
			//skip line
			line = br.readLine().split(" ");
			
			for(int i=0; i<leftHiddenNeurons; i++){
				line = br.readLine().split(" ");
				for(int j=0; j<rightHiddenNeurons; j++){
					rightHiddenWeights[i][j] = Double.parseDouble(line[j]);
				}
			}
			//skip line
			line = br.readLine().split(" ");

			//read in hidden biases
			line = br.readLine().split(" ");
			for(int i=0; i<rightHiddenNeurons; i++){
				rightHiddenBiases[i] = Double.parseDouble(line[i]);
			}
			//skip line
			line = br.readLine().split(" ");
			
			//read in output weights
			for(int i=0; i<rightHiddenNeurons; i++){
				line = br.readLine().split(" ");
				outputWeights[i][0] = Double.parseDouble(line[0]);
			}
			//skip line
			line = br.readLine().split(" ");
			
			//read in output biases
			line = br.readLine().split(" ");
			outputBiases[0] = Double.parseDouble(line[0]);
			
			net.setWeights(leftHiddenWeights, rightHiddenWeights, outputWeights, 
					leftHiddenBiases, rightHiddenBiases, outputBiases);
			
			//read number of games trained on
			br.readLine();
			line = br.readLine().split(": ");
			br.close();

			return Integer.parseInt(line[1]);
		}catch(Exception e){}
		
		return 0;
	}
	
	private void saveWeights(int numGames){
		String file = "weights.txt";
		double[][] lhw = net.getLHW();
		double[][] rhw = net.getRHW();
		double[] lhb = net.getLHB();
		double[] rhb = net.getRHB();
		double[][] ow = net.getOW();
		double[] ob = net.getOB();
		
		try(BufferedWriter bw = new BufferedWriter(new FileWriter(file))){
			String line = "";
			//write hidden weights
			for(int i=0; i<inputNeurons; i++){
				line = "";
				for(int j=0; j<leftHiddenNeurons; j++){
					line = line + String.valueOf(lhw[i][j]) + " ";
				}
				bw.write(line);
				bw.newLine();
			}
			//skip line
			bw.newLine();
			
			//write hidden biases
			line = "";
			for(int i=0; i<leftHiddenNeurons; i++){
				line = line + String.valueOf(lhb[i]) + " ";
			}
			bw.write(line);
			bw.newLine();
			
			//skip line
			bw.newLine();
			
			for(int i=0; i<leftHiddenNeurons; i++){
				line = "";
				for(int j=0; j<rightHiddenNeurons; j++){
					line = line + String.valueOf(rhw[i][j]) + " ";
				}
				bw.write(line);
				bw.newLine();
			}
			//skip line
			bw.newLine();
			
			//write hidden biases
			line = "";
			for(int i=0; i<rightHiddenNeurons; i++){
				line = line + String.valueOf(rhb[i]) + " ";
			}
			bw.write(line);
			bw.newLine();
			
			//skip line
			bw.newLine();
			
			//write output weights
			for(int i=0; i<rightHiddenNeurons; i++){
				line = "";
				line = line + String.valueOf(ow[i][0]) + " ";
				bw.write(line);
				bw.newLine();
			}
			//skip line
			bw.newLine();
			
			//write output biases
			line = "";
			line = line + String.valueOf(ob[0]) + " ";
			bw.write(line);
			bw.newLine();
			
			bw.newLine();
			bw.write("Number of trainging games completed: " + numGames);
			bw.close();
		}catch(Exception e){}
	}

	private void trainNN(boolean resume, int trainingGames) throws Exception{
		int numGames = 0;
		//loads weights if resuming training
		//else init with random weights
		if(resume){
			numGames = loadWeights();
		}else{
			net.initWeights();
		}
		
		//run through x training games 
		//updating weights after each games
		double[][] gstates;
		double[] reward = new double[outputNeurons];
		int i;
		
		//time training time
		long starttime = System.currentTimeMillis();
		
		for(i=0; i<trainingGames; i++){
			gameStates = new ArrayList<Double[]>();
			//connect, play with training mode, and disconnect
			socketConnect("localhost", 17033);
			if(!play(true)){break;}
			closeConnection();
			
			//unpack game states from arraylist into double[][]
			gstates = unpackGS();
			//set reward for win/loss to 1/-1
			if(gs.equals("win")){ reward[0] = 1; }
			else if(gs.equals("lose")){ reward[0] = -1; }

			System.out.print(gs + "   :::   ");
			
			//train on game sequence
			net.train(gstates, reward);
			//System.out.println("Error: " + (reward[0] - gstates[gstates.length-1][0]));
		}
		
		//end timer
		long endtime = System.currentTimeMillis();
		System.out.println("traing time for " + i + " games: " + TimeUnit.MILLISECONDS.toSeconds(endtime-starttime));
		
		//save weights
		saveWeights(i + numGames);
	}
}