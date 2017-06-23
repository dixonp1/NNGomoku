package deepnet;

import java.util.ArrayList;


public class Board {
	private double[] gb;
	
	public Board(int size){
		gb = new double[size];
	}
	
	public Board(Board board){
		System.arraycopy(board.getBoard(), 0, gb, 0, gb.length);;
	}
	
	public double[] getBoard(){
		return gb;
	}
	
	public void setBoard(double[] board){
		gb = board;
	}
	
	public double[] makeMove(int move){
		gb[move] = 1;
		return gb;
	}
	
	public boolean isValid(int move){
		return (gb[move] == 0);
	}
	
	public ArrayList<Integer> getMoves(){
		ArrayList<Integer> moves = new ArrayList<Integer>();
		for(int i=0; i<gb.length; i++){
			if(gb[i] == 0){
				moves.add(i);
			}
		}
		return moves;
	}
}
