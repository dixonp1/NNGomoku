RUN:
compile in IDE or command line with: 'javac *.java ' while in the src folder
run GomokuClient in IDE or command line with: 'java GomokuClient'

program will also accept parameters:
	java GomokuClient train <x>
		will run program in training mode. will run x amount of games. 
		MUST HAVE OPPONENT. WILL NOT PLAY AGAINST SELF. SERVER WOULDN'T ALLOW IT
		note: this mode will overwrite existing weights.txt file. make a copy 
		if you wish to save them
		
	java GomokuClient train <x> resume
		will run program in training mode, same as above. Except this will resume
		training with the current weights.txt file.
		This allows for training to continue if more games were needed without 
		having to start over.
		
	running without parameters will run agent normally
	
My Gomoku learner uses a neural network to evaluate each possible move and would pick
the move with the highest value given by the network. 
The network was trained using a temporal difference learning rule.
In training mode, the network will recieve a reward of 1 if it won or a 0 if it lost.
After each game the network will calculate the desired value for each state of the game
using the rule 
		
		d(t) = lambda*d(t+1) + alpha * ((1-lambda) * (r(t) + gamma*v(t+1) - v(t)))
		
where lambda is the amount of feedback each state gets from the next, alpha is the learning
rate, gamma is the decaying factor, and v(t) is the network output at time t.
The reward, r, was also updated at each state by the decay factor, 'gamma * r(t+1)'.
The desired value for each step could only be calculated after the game has ended
and a reward was given. So to calculate these values the network starts at the last 
move and works backwards while also calculating the error at each state.

		e(t) = d(t) - v(t)
		
After the error was calculated for each state, the changes in the weights for each 
state was added up and applied all at once at the end. 

For training I have gone through many different values for alpha and lambda, and 
tried a couple different architectures for the network. Each trained for tens of 
thousands of games. Some configurations for hundreds of thousands.

For alpha I have tried values 0.001, 0.005, 0.01, 0.05, and 0.1. Out of these values
I have found that the lower ones, 0.001 and 0.005, had seemed to be the most promising.
The network seemed somewhat that it was trying to learn. 

I have also tried giving the network a penalty of -1 if it lost. This seemed to somewhat
speed up the process and allow weights to change more rapidly for losing. While it did 
work faster, still ended with the same results.

For architectures, I have tried a few:
	1) 81 input neurons, 50 hidden, 1 output
		this configuration had 1 input neuron for each position on the board.
	2) 82 input neurons, 50 hidden, 1 output
		this had the addition input neuron which would keep track of which player was 
		to make the next move. 1 if it is the networks turn to play, and -1 for the 
		opponent.
	3) tried both 1) & 2) with 80 hidden neurons to see if this would allow it to better
		learn strategies of the game.
	4) Deep neural network with 2 hidden layers. First consisting of 50 neurons and
		the second having 30. Again, this was to see if adding the extra layer would 
		allow for better learning. I tried this with both keeping track of the player
		turn and without.
	5) I have even tried 2) using a cross-entrophy error function and summing the error 
		over each every state and dividing by the number of states and applying that to 
		the weight changes for each state. This proved to not give much of any different
		results.
		
For training situations I have also tried a few different scenarios:
	I have had each of the 5 architectures play against each other for many different
	parameters to see if there would be an obvious winner. 
	
	I have also played each one against Patrick Matt's (group 8) alpha-beta gomoku agent
	to see if training with a player that knew what they were doing would help it learn 
	to not only play but to learn strategies faster. This also was not the case.
	
	I had modified 1 of them to literally play against itself. It would pick moves for 
	both sides. This was interesting to watch. 

RESULTS:
For the many different configurations, architectures, and training situations I was not
able to get one to learn to play well. most did learn to play, just not well enough to 
win against Patrick's agent, let alone an actual person. Many times, especially against
Patrick's, it would play the same game so many times that it would reinforce a particular
sequence of moves rather than how to play the game. Once the opponent placed a move in 
the next spot the network wanted to, it would sort of freak out and go back to not knowing
what to do. 

I do have 2 other ideas I may try, outside of the lab as I don't have enough time, is 
add a few input neurons for features of the game, such as 4 in a row, 3 in a row, etc...
and maybe have 2 input neurons for each position on the board, if 1 is active than the 
network has a peice there, if the other is active, the the opponent has a peice there, 
and if neither, than the space is free. Just curious on how those would perform in 
comparision with the current versions. 

While I did not get the results I was hoping for, this lab was extremely fun! I did 
not need the processing power I thought I might need so I was able to do this on my 
labtop. The games with 2 networks would go so fast you could watch hundreds in minutes.
Many times me and my fiance would just be amazed and watch as the networks progressed 
and learned right in front of us. It was very cool to see it constantly losing and 
finally change up its tactic and then games would start going back and forth through 
different games. 
