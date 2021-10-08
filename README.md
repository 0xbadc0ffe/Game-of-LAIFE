# Game-of-LAIFE
LAIFE (Learned Artificial Intelligence Forecasting Evolution)


## `Project`

The first aim of this project is to create a simple example to explain Convolutional Neural Networks.
The other goals are settled to experiment with an entropy approach to the AI learning of this game's rules and similar ones. 

## Step 1 [✓]

Teach to a CNN how to play Game of Life (GoL) in a supervised way. The AI plays a role as a transition matrix between steps of GoL.

`LearnGOL.py`

![](/media/AI-GOL-dash.GIF)

## Step 2    [/]

Same as before but the teaching is made in an unsupervised way. Loss based on entropy and penalty for non-exploding and non-imploding behaviour.

To check some entropy functions I tried give a look to

`test-entropy.py`

![](/media/entropy_and_derivative.png)

<p align="center">
  <i>one of the entropy functions and its approximate derivative</i>
</p>
  
<br /><br />
For the unsupervised learning I refer to 

`Unsupervised_GOL.py`

## Step 3 [•]

Implement new rules: global ones with masked MLP and local ones with other convolutional layers.

## Step 4 [•]

More dimensions! Extend and improve the unsupervised learning process to general Cellular Automata schemes.




