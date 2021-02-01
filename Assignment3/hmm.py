import numpy as np

def forward(A, B, pi, O):
	"""
	Parameters:
		A: state transition probabilities(NxN)
		B: observation probabilites(NxM)
		pi: initial state probabilities(N)
		O: sequence of observations(T) where
		   observations are just indices for the columns of B
		where N is the number of states,
			  M is the number of possible observations, and
			  T is the sequence length.
	Return:
		Probability of the observation sequence given the model(A, B, pi)
	"""
	
	########## Write your code here ##########
	# implement forward algorithm to find
	# the probability of the given observation
	# sequence given the model.
	##########################################

	return 0

	
def viterbi(A, B, pi, O):
	"""
	Parameters:
		A: state transition probabilities(NxN)
		B: observation probabilites(NxM)
		pi: initial state probabilities(N)
		O: sequence of observations(T) where
		   observations are just indices for the columns of B
		where N is the number of states,
			  M is the number of possible observations, and
			  T is the sequence length.
	Return:
		The most likely state sequence given model(A, B, pi) and
		observation sequence. It should be a numpy array
		with size T. It includes state indices according to
		A's indices. For example: [1,2,1,1,0,4]
	"""
	
	########## Write your code here ##########
	# implement viterbi algorithm to find
	# the most likely state sequence of length
	# T given model and observation sequence.
	##########################################	
	
	return np.zeros((O.shape[0],))
	


