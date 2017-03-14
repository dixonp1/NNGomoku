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

public class GomokuClient {

	protected Socket myClient = null;// our client socket.
	protected PrintWriter os = null;
	protected BufferedReader is = null;// Input stream.
	protected String gs = null;// GameState.
	private NeuralNet net;
	private Board board;
	private char me;
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
		net = new NeuralNet(81, 50, 1);
		//net.initWeights(); // DEBUG*********************8
		board = new Board();
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

	public void play(boolean training) {
		try {
			while (true) {
				//read in board
				board = readIn();
					
				//break if game over
				if (!gs.equals("continuing")) {	break;	}
				int move = pickmove(me);
				
				//if training, add current game state and game state after
				//move made to arraylist
				if(training) { 
					double[] tempboard = new double[81];
					System.arraycopy(board.getBoard(), 0, tempboard, 0, tempboard.length);
					addGS(board.getBoard());
					addGS(tempboard);	
				}
				
				//send move
				moveSend(move);
			}
		} catch (SocketTimeoutException e) {
			//saveWeights();
		}catch (Exception e) {}
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
		double[] gb = new double[81];
		Board b = new Board();
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
			double[] newboard = new double[81];
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
		Double[] temp = new Double[81];
		for(int i=0; i<81; i++){
			temp[i] = new Double(gstate[i]);
		}
		gameStates.add(temp);
	}
	
	//unpack game states from arraylist
	private double[][] unpackGS(){
		Double[] gs = new Double[81];
		double[][] temp = new double[gameStates.size()][81];
		for(int j=0; j<gameStates.size(); j++){
			gs = gameStates.get(j);
			for(int i=0; i<81; i++){
				temp[j][i] = gs[i].doubleValue();
			}
		}
		return temp;
	}
	
	private int loadWeights(){
		String file = "weights.txt";
		double[][] hiddenWeights = new double[81][50];
		double[] hiddenBiases = new double[50];
		double[][] outputWeights = new double[50][1];
		double[] outputBiases = new double[1];
		
		//read in weights from file
		try (BufferedReader br = new BufferedReader(new FileReader(file))){
			String[] line;
			
			//read in hidden weights
			for(int i=0; i<81; i++){
				line = br.readLine().split(" ");
				for(int j=0; j<50; j++){
					hiddenWeights[i][j] = Double.parseDouble(line[j]);
				}
			}
			//skip line
			line = br.readLine().split(" ");

			//read in hidden biases
			line = br.readLine().split(" ");
			for(int i=0; i<50; i++){
				hiddenBiases[i] = Double.parseDouble(line[i]);
			}
			//skip line
			line = br.readLine().split(" ");
			
			//read in output weights
			for(int i=0; i<50; i++){
				line = br.readLine().split(" ");
				outputWeights[i][0] = Double.parseDouble(line[0]);
			}
			//skip line
			line = br.readLine().split(" ");
			
			//read in output biases
			line = br.readLine().split(" ");
			outputBiases[0] = Double.parseDouble(line[0]);
			
			net.setWeights(hiddenWeights, outputWeights, hiddenBiases, outputBiases);
			
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
		double[][] hw = net.getHW();
		double[] hb = net.getHB();
		double[][] ow = net.getOW();
		double[] ob = net.getOB();
		
		try(BufferedWriter bw = new BufferedWriter(new FileWriter(file))){
			String line = "";
			//write hidden weights
			for(int i=0; i<81; i++){
				line = "";
				for(int j=0; j<50; j++){
					line = line + String.valueOf(hw[i][j]) + " ";
				}
				bw.write(line);
				bw.newLine();
			}
			//skip line
			bw.newLine();
			
			//write hidden biases
			line = "";
			for(int i=0; i<50; i++){
				line = line + String.valueOf(hb[i]) + " ";
			}
			bw.write(line);
			bw.newLine();
			
			//skip line
			bw.newLine();
			
			//write output weights
			for(int i=0; i<50; i++){
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
		gameStates = new ArrayList<Double[]>();
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
		double[] reward = new double[1];
		int i;
		for(i=0; i<trainingGames; i++){
			//connect, play with training mode, and disconnect
			socketConnect("localhost", 17033);
			play(true);
			closeConnection();
			
			//unpack game states from arraylist into double[][]
			gstates = unpackGS();
			//set reward for win/loss to 1/-1
			if(gs.equals("win")){ reward[0] = 1; }
			else if(gs.equals("loss")){ reward[0] = -1; }
			
			//train on game sequence
			net.train(gstates, reward);
		}
		
		//save weights
		saveWeights(i + numGames);
	}
}