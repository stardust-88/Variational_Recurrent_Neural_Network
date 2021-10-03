# Variational_Recurrent_Neural_Networks

Implemented this novel architecture (in pytorch) as part of my research study for the problem I am focusing on, 
during my work as a <b>Research Intern</b> at <b>Language Technology Research Center, IIIT Hyderabad, India.</b><br>

<b>Note</b>: Still working on it to get better and more observations.
<hr>

Variational Recurrent Neural Networks are a class of latent variable models for sequential data. The major idea behind this work is the inclusion of latent random variables at every time step of the RNN, or more specifically, it contains variational autoencoder at each and every time step of the RNN.

This is the result of the work by Junyoung et al. - <a href="https://arxiv.org/pdf/1506.02216.pdf">A Latent Variable Model For Sequential Data</a>

<b>From the original research paper:</b><br>
<i>"In this paper, we explore the inclusion of latent random variables into the hidden state of a recurrent neural network (RNN) by combining the elements of the
variational autoencoder. We argue that through the use of high-level latent random variables, the variational RNN (VRNN)1
can model the kind of variability
observed in highly structured sequential data such as natural speech. We empirically evaluate the proposed model against other related sequential models on four
speech datasets and one handwriting dataset. Our results show the important roles
that latent random variables can play in the RNN dynamics."</i>

<hr>

### Run:

To train: ```python train.py```<br>
To sample from the saved model: ```python sample.py```
<br>

<b>Note</b>: parameters folder (it will have model's parameters) is empty currently. Soon it will be updated with results.
