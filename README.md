# This is the repository for our NeurIPS paper [Erdos goes Neural: An Unsupervised Learning Framework for Combinatorial Optimization on Graphs](https://proceedings.neurips.cc//paper/2020/hash/49f85a9ed090b20c8bed85a5923c669f-Abstract.html).

Currently being updated! Feel free to email me any questions and I will respond either directly or by adding your question to the FAQ.

### First, I highly recommend checking out our [blogpost](https://andreasloukas.blog/2020/10/31/erdos-goes-neural/).
It is a good introduction if you want a brief overview of the key insights in the paper and how it relates to the broader landscape of neural combinatorial optimization. Additionally, to gain some intuition for the probabilistic method, I recommend reading the background section (2.2) in the paper.

Now, since we have been getting plenty of questions on how the method works, **below you will find a simplified guide for our framework**. I will also include an FAQ section that I will keep updating.
For a hands-on example please refer to the "Erdos_tud" notebook.

### Disclaimer:
This is a brief tutorial on the probabilistic penalty method presented in the paper. I will be taking shortcuts in how I present it in order to make it straightforward and understandable. It does not cover the paper in its full generality. I will not explain the probabilistic method and I won't expand on the various important details that constitute the full paper.
Instead, I aim to provide an accessible primer into our work by minimizing the technical info and focusing on practicality/simplicity. I encourage you to read the paper and the supplementary material section for the complete treatment. Otherwise, if you're still unsure, just email me. Without further ado, let's get started.

## Erdos goes neural: a simplified tutorial

### Preliminaries:
Before we start, we should clear up what the goal of this work is. The main objective is to solve combinatorial problems with neural networks, 
**without supervision** (i.e., without access to labeled solutions). We focused on graph problems but the principles of our method can be applied to other settings. Furthermore, many (combinatorial) problems have an equivalent graph formulation, so you could reformulate your problem in a graph setting if you want to be as close to what we did as possible. 

OK, great. Now that this is out of the way, let's talk about how we are going to get things done.
First, let's be a bit more specific with the setup.

We want a set of objects S that solves a given problem, e.g., finds the set of nodes that forms the largest possible clique on a graph (maximum clique problem). 
Since we focused on graph problems in the paper, we will work with sets of nodes on the input graph, but you can also do sets of edges/tuples/etc. It will depend on your particular problem and what is more convenient for you. 
**From now on I will be talking about sets of nodes. Feel free to substitute it with edges or whatever else you need to work with.**

Here is what we are going to need:
#### 1) A combinatorial problem with a nonnegative cost function and a set of constraints.
#### 2) A neural network that produces a probability for each node. This is the probability that the node belongs to the set S. 
#### 3) A differentiable loss function. This loss takes as input the probabilities that were produced by the network. The loss will be derived from your problem's objective (I will explain below).  



#### Technical Detail:
In the paper, we use Bernoulli variables over the nodes of the input graph. You could work with other distributions as well, but if you are unsure I recommend starting with Bernoulli  random variables over the entities you care about (over nodes/edges/tuples/etc.). We will need to derive an expectation of a function over those random variables and it happens that Bernoulli variables tend to lead to easier derivations.

## How to solve it:

Now, suppose you have a combinatorial problem. You need to follow the steps below.
### Step 1:
Write down the cost function of your problem, for example in our paper we consider the maximum clique problem.
Here is the standard way of expressing it:
#### maximize weight(S), subject to: S is a clique.
For a simple undirected graph, weight(S) just counts the number of edges in the subgraph induced by S. 
To be in line with the paper, we need to switch to a  minimization problem, hence:
#### minimize gamma-weight(S), subject to: S is a clique.
gamma here is a sufficiently large constant;  Let gamma >= max(weight(S)), so that the expression remains always nonnegative.

### Step 2:
Set up a graph neural network (the choice of model depends on the task and engineering considerations) for your data. Using any of the mainstream
GNNs (GIN, GAT, etc.) should be fine to start with.
The GNN takes as input some node features. Its output is an N x 1 vector of probabilities, one for each node.
For the specifics on features, layers, normalizations, etc. you can just look at the code in the repo. Or you may improve upon the pipeline by working with your own features, layers, etc.

### Step 3:
Derive a differentiable loss.
The differentiable loss function has to look like this:
#### Loss = Expected Cost + beta * Prob(S does not satisfy constraints).
beta here is a coefficient that controls the importance of the constraint in the loss. 
The expectation of the cost can be straightforward for many set functions. 
For the Prob(S does not satisfy constraints) term, we can use Markov's inequality to bound it. 
#### example:
First, derive the expected cost. In our max-clique example, cost = gamma-weight(S).
Gamma is just a constant so expected_cost = gamma - expected_weight(S).
We have weight(S) = sum_{graph edges (i,j)}(x_i * x_j)


In other words, an edge (i,j) is in S if both endpoints i and j are in S.
We are using Bernoulli variables, therefore:
expected_weight(S) = sum_{graph edges (i,j)}(prob(x_i) * prob(x_j)).

OK, now onto the constraint part of the loss.
For the maximum clique problem, the constraint dictates that **the subgraph induced by S is a clique**, i.e, all pairs of nodes in S are connected by an edge.
An equivalent way to phrase this is that, **there are no edges in the complement of the subgraph induced by S**.
Therefore,
#### Prob(S does not satisfy constraints) = Prob(weight(complement(S))>=1).
From Markov's inequality, this can be bounded as follows
#### Prob(weight(complement(S))>=1) <= expected_weight(complement(S)).
So our loss ends up being
#### Loss = gamma - expected_weight(S) + beta * expected_weight(complement(S)).


### Step 4:
Train the network using the derived loss. This is straightforward because you just have to plug in the probabilities in the expression and do backprop.

### Step 5:
Retrieve the set S from the probabilities of the network using the method of conditional expectation.
It works as follows.

Sort the nodes according to their probabilities.
Starting from the high probability nodes, for each node v_i do:
  1) Evaluate the loss for prob(v_i)=1 and for prob(v_i)=0.
  2) Set prob_(v_i) to either 1 or 0 depending on what achieved the better loss.
  3) Move on to the next node and repeat from step 1
  
When this is done, you should have a binary (indicator) vector that represents your set S, which is the solution to your problem.
Congrats! That's it, you're done.

## FAQ
*Q: The method is unsupervised. What's the difference between training and testing?*

A: You may train with backprop on a finite number of training graphs and then use the model to produce solutions on new graphs without the need to retrain. That saves you computation in the long run. That's what we did in the paper. Train/Test splits are also more in line with general practice in ML so we figured the results would be easier to interpret like this.
In principle, you can always do backpropagation on any new graph that comes to your model but of course this will take more time.

*Q: Is sorting in the method of conditional expectation necessary?*

A: In theory, no. Mathematically, any ordering of the nodes will give you the guarantee we describe in the paper. In practice, we found sorting to help with performance at a relatively low cost so we added it.

*Q: Can I use this method to solve the "insert your combinatorial problem here" problem?*

A: It is likely that you can; it depends on the problem at hand. As we explain in the paper, certain problems are more amenable to our kind of probabilistic formulation than others.
Problems with constraints that involve covers, cliques, independent sets, etc are likely easier to model than other more complicated ones.
I recommend studying the clique example that I provided (and the details of the derivation in the paper) to get a sense of what you could reasonably model. 
For examples of our framework applied on other problems you may check section F of the appendix in this recent worskhop [paper](https://openreview.net/pdf?id=5UvvKsBTDcR) by Dai et al.

*Q: Does the choice of GNN matter? How fast can I expect this to run? Do I need a lot of resources?*

A: Our experiments suggest that our approach should work well with various mainstream GNN architectures as long as some hyperparameter tuning is done.
Every experiment was run on a single TITAN RTX. So the memory requirements may be somewhat high, given that the code is definitely not optimized for memory consumption. In terms of time, it took a few minutes (less than an hour) to train these models on any of the reported datasets. 

*Q: Does your framework guarantee feasibility?*

A: As stated in the paper (theorem 1), feasibility of the solutions will depend on how well the loss minimization goes during training. For example, suppose you only care about a feasibility problem, e.g., k-clique, and you only have Prob(S does not satisfy constraints) as your loss term. Then in that case, you will use an expectation of the weight of the complement to bound the probability. If that quantity is below 1, then the integral solution is guaranteed to be feasible from the method of conditional expectation.
On the other hand, if training does not go well, then it's a good practice to manually ensure feasibility (by manually checking if the constraint is satisfied) when running the method of conditional expectation.

*Q: Why use Bernoulli variables? Isn't this limiting?*

A: Bernoulli variables in conjunction with Markov's inequality lead to  fairly straightforward derivations for loss functions. In pricinple, we could make different assumptions about the distribution and work with tighter concentration inequalities to obtain loss functions that are tailored to a specific problem. We opted for the simplest approach, as it produced satisfactory results without the need to resort to more sophisticated mathematical tools. Nevertheless, exploring alternative constructions is a promising direction that could lead to improvements in our framework.

