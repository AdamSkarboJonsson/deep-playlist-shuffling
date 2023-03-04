## Attemping to find a way to solve the playlist shuffling problem, i.e shuffling a playlist containing tracks with respect to the features of the tracks.


### Methods:

* Reinforcement learning - Shuffling is a game (**ONGOING**)
  * I model the shuffling of a playlist as a game, where an action consists of swapping the placement of two tracks in the playlist. The reward function is defined by the difference in loss before and after swapping the tracks. A discount factor $$ \gamma < 1$$ is introduced to encourage the agent to finish shuffling ASAP. 
  * Deep Q-learning is used.


* Supervised learning - outputting permutation matrices (**FAILED**)
    * Instead of learning a "correct" way of shuffling a playlist, a neural network is tasked with shuffling a playlist such that the loss of the permutated playlist is minimized. This is done by predicting a permutation matrix M, then computing L(Mx), where x is the playlist in its original order.
    * Simply training a  NN to output a permutation matrix is hard in it of itself. There is some research in this area surrounding Gumbel-Sinkhorn operators, which can output a relaxed permutation matrix and is differentiable. However, these operators produce large or too small gradients to effectively train with the current loss function setup.

