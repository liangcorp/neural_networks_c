# Neural Networks

## Model Representation

Let's examine how we will represent a hypothesis function using neural
networks. At a very simple level, neurons are basically computational
units that take inputs (**dendrites**) as electrical inputs (called
"spikes") that are channeled to outputs (**axons**). In our model, our
dendrites are like the input features $x_1...x_n$, and the output is the
result of our hypothesis function. In this model our $x_0$ input node is sometimes
called the "bias unit." It is always equal to 1. In neural networks, we
use the same logistic function as in classification,
$1 \over { 1 + e^{θ^Tx}}$, yet we sometimes call it a sigmoid (logistic)
**activation** function. In this situation, our "theta" parameters are
sometimes called "weights".

Visually, a simplistic representation looks like:

$
[x_0x_1x_2] \rightarrow [\ \ \ ] \rightarrow h_θ(x)
$
