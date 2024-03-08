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

$$
[x_0x_1x_2] \rightarrow [\ \ \ ] \rightarrow h_θ(x)
$$

Our input nodes (layer 1), also known as the "input layer", go into
another node (layer 2), which finally outputs the hypothesis function,
known as the "output layer".

We can have intermediate layers of nodes between the input and output
layers called the "hidden layers."

In this example, we label these intermediate or "hidden" layer nodes
$a_0^2...a_n^2$ and call them "activation units."

$$
a_i^{(j)} = \text{``activation'' of unit i in layer j}
$$

The values for each of the "activation" nodes is obtained as follows:

$$
a_1^{(2)} = g(Θ_{10}^{(1)}x_0 + Θ_{11}^{(1)}x_1 + Θ_{12}^{(1)}x_2 + Θ_{13}^{(1)}x_3)
$$

$$
a_2^{(2)} = g(Θ_{20}^{(1)}x_0 + Θ_{21}^{(1)}x_1 + Θ_{22}^{(1)}x_2 + Θ_{23}^{(1)}x_3)
$$

$$
a_3^{(2)} = g(Θ_{30}^{(1)}x_0 + Θ_{31}^{(1)}x_1 + Θ_{32}^{(1)}x_2 + Θ_{33}^{(1)}x_3)
$$

$$
h_Θ{(x)} = a_1^{(3)} = g(Θ_{10}^{(2)}a_0^{(2)} + Θ_{11}^{(2)}a_1^{(2)} +
Θ_{12}^{(2)}a_2^{(2)} + Θ_{13}^{(2)}a_3^{(2)})
$$

This is saying that we compute our activation nodes by using a $3×4$
matrix of parameters. We apply each row of the parameters to our inputs
to obtain the value for one activation node. Our hypothesis output is
the logistic function applied to the sum of the values of our activation
nodes, which have been multiplied by yet another parameter matrix
$Θ^{(2)}$ containing the weights for our second layer of nodes.

Each layer gets its own matrix of weights, $Θ^{(j)}$.

The dimensions of these matrices of weights is determined as follows:

If network has $s_j$ units in layer $j$ and $s_{j+1}$ units in layer
$j+1$, then $Θ^{(j)}$ will be of dimension $s_{j+1}×(s_j+1)$.

The $+1$ comes from the addition in $Θ^{(j)}$ of the "bias nodes," $x_0$
and $Θ_0^{(j)}$. In other words the output nodes will not include the
bias nodes while the inputs will. The following image summarizes our
model representation:
![0rgjYLDeEeajLxLfjQiSjg_0c07c56839f8d6e8d7b0d09acedc88fd_Screenshot-2016-11-22-10 08 51](https://github.com/liangcorp/neural_networks_c/assets/2737157/0a131ce8-0713-446b-8a5c-b4c17b888041)
