---
layout: post
title: Spectral Graph Convolutions
date: 2023-11-29 16:37 -0600

categories:
  - Blog post
tags:
  - graph neural networks
  - math
  - deep learning
math: true
---

It is not surprising that Graph Neural Networks have become a major trend in both academic research and practical applications in the industry. Graph-structured data is present everywhere, and graph theory provides a versatile mathematical framework for describing and analyzing sets of entities and their relationships.

However, the complexity associated with processing graph-structured data has often led to its neglect. Most deep learning architectures are specifically designed for Euclidean-structured data, which can be seen as a special case of graph data.

Although we made little progress in managing graph data using graph kernels {% cite kriege2020survey %}. and random-walk methods {% cite perozzi2014deepwalk %}, the advent of Graph Neural Networks (GNN) has made these techniques the go-to approach for working with graphs. In this tutorial, we will focus on the mathematical foundations of the first successful GNN method: Graph Convolutional Networks (GCN) {% cite kipf2016semi %}. While GCN has fallen out of favor in favor of other approaches such as attention mechanisms in graphs (GAT) {% cite velivckovic2017graph %} or message passing neural networks (MPNN) {% cite gilmer2017neural %}, it can still be valuable in situations where deep models are not necessary. Additionally, GCN has the advantage of being easier to understand and implement compared to GAT or message-passing models.

## A brief glance of Spectral Graph Theory
The term _“spectral”_ may intimidate some machine learning practitioners (or bring back memories of quantum operators for physics students), but in simple terms, spectral theory refers to the study of the properties of linear operators, especially those pertaining to eigenvalues and eigenvectors.

In the context of graph theory, spectral theory focuses on examining the properties of the adjacency and Laplacian matrices associated with a graph. The adjacency matrix is more intuitive and easier to define. For a graph $$ G = (V, E) $$ with n nodes, we define the adjacency matrix A, whose entries are determined by

$$
a_{i,j} = \left\{\begin{array}{cl}
1 & \text{if } \{i,j\} \in E \\
0 & \text{otherwise.}
\end{array}\right.
$$

We can provide a similar definition for the degree matrix $$ D $$, which is a diagonal matrix that encompasses all the information regarding the local connectivity of a node.

$$
d_{i,j} = \left\{\begin{array}{cl}
d(i) & \text{if } i = j \\
0 & \text{otherwise.}
\end{array}\right.
$$

where $$ d(i) $$ is the number of neighbors of this node.

On the other hand, we can obtain our initial definition of the Laplacian matrix $$ L $$ by employing a similar approach, defining each entry as

$$
l_{i,j} = \left\{\begin{array}{cl}
-1 & \text{if } \{i,j\} \in E \\
d(i) & \text{if } i = j \\
0 & \text{otherwise.}
\end{array}\right.
$$

We can simplify the element-wise definition by observing that the previous definition is equivalent to $$ L = D - A $$.

In many cases, certain nodes exhibit a higher number of connections compared to others. This disparity in node degrees can impact the performance of our algorithms. To address this issue, we employ a normalization technique based on random walks. This technique assigns weights to the edges based on the node degrees, thereby balancing the influence of highly connected nodes, and ensuring that nodes with lower degrees are not overshadowed. This is achieved by dividing each entry by the sum of the degrees of the corresponding node.

$$
\mathcal{L} = D^{-1}(D - A) = I - D^{-1}A.
$$

For now, let’s shift our focus to the eigendecomposition of the matrix $$ L $$. Eigendecomposition is a fundamental concept that enables us to break down a square matrix into a set of eigenvalues and eigenvectors. By expressing $$ L $$ as the product of three matrices,

$$
\mathcal{L} = U\Lambda U^{-1} = U\Lambda U^{\intercal},
$$

where the matrix $$ U $$ contains the eigenvectors of $$ L $$, and the inner matrix is a diagonal matrix with the eigenvalues as its entries:

$$
\Lambda = \left[\begin{array}{cccc}
\lambda_1 & & & \\
& \lambda_2 & & \\
& & \ddots & \\
& & & \lambda_n
\end{array}\right].
$$

However, we encounter a problem: the eigenvalues of a matrix can be complex numbers. Fortunately, this is not the case for the Laplacian matrix. To understand why, we introduce a new matrix that solely considers the Laplacian of every edge $$ {u, v} \in E $$.

$$
L_{G_{u, v}}(i, j) = \left\{\begin{array}{cl}
1 & \text{if } i = j \text{ and } i\in\{u, v\}, \\
-1 & \text{if } i = u \text{ and } i = v \text{ or vice versa}, \\
0 & \text{otherwise.}
\end{array}\right.
$$

Using these matrices, we can provide a new definition of the Laplacian matrix as the sum of each of the aforementioned matrices.

$$
L_G = \sum_{\{u, v\}\in E} L_{G_{u, v}}
$$

This new definition provides us with additional information about the Laplacian matrix. For instance, it allows us to establish the Laplacian quadratic form

$$
\vec{x}^\intercal L_{G_{u, v}} \vec{x} = (x_u - x_v)^2 \,\,\,\forall \vec{x}\in\mathbb{R}^n,
$$

and with that in mind, we can easily show that L is self-adjoint. As a result, we get some interesting properties.
- $$ L $$ has n linear independent eigenvectors.
- The eigenvectors correspond to different eigenvalues of $$ L $$ and are orthogonal to each other.
- All eigenvalues are real and non-negative.

The eigenvalues of $$ L $$ provide us with even more valuable information than expected. For instance, if we sort and relabel the eigenvalues such that $$ \lambda_1 $$ corresponds to the smallest eigenvalue, we observe that this value is always zero. This occurrence is not coincidental; the multiplicity of zero as an eigenvalue indicates the number of connected components in the graph. Specifically, if $$ \lambda_2 $$ is greater than zero, then our graph is connected. On the other hand, the multiplicity of two as an eigenvalue corresponds to the number of bipartite connected components in our graph with at least two vertices.

There are numerous aspects of spectral graph theory that we can deepen, exploring the relationship between Laplacians and the connectedness of a graph. However, discussing these intricacies exceeds the scope of the current post. If you’re interested in delving deeper into the subject, I highly recommend reading the incomplete draft of Spielman {% cite spielman2019spectral %}.

## Graph Fourier Transform
In the classical Fourier transform, a signal is decomposed into a sum of sinusoidal components with varying frequencies. Similarly, the graph Fourier transform represents a signal on a graph as a linear combination of graph spectral components associated with different frequencies.

In this case, we consider f as a signal if it is a function that assigns a real number to each vertex of the graph. Like the classical Fourier transform, we can utilize the eigenvectors of the Laplacian (contained in matrix U) to project the graph signal. Thus, we define the graph Fourier transform as follows:

$$
\mathcal{GF}[f](\lambda_l) = \hat{f}(\lambda_l) = \langle f, u_l \rangle = \sum_{i = 1}^N f(i) u_i^\intercal,
$$

or in matrix notation, $$ \hat{f} = U^\intercal f $$. Similarly, the inverse graph Fourier transform is defined as follows:

$$
\mathcal{IGF}[\hat{f}](\lambda_l) = f(\lambda_l) = \langle \hat{f}, u_l \rangle = \sum_{i = 1}^N \hat{f}(i) u_i^\intercal,
$$

or in matrix notation, $$ f = U\hat{f} $$.

In summary, the Graph Fourier transform utilizes the eigenvectors of the Laplacian matrix to enable the representation of a signal in two distinct domains: the vertex domain and the graph spectral domain. However, it is important to note that the definition of the graph Fourier transform, as well as its inverse, relies on the selection of Laplacian eigenvectors, which may not be unique.

### Graph convolution operator
Using the definition of the graph Fourier transform, we can establish several famous theorems like the Parseval’s identity, and the generalized translation operator found in the classical Fourier transform. However, our focus lies on the graph convolution operator, which exhibits similar characteristics to its classical counterpart.

As you may recall, the convolution theorem states that the Fourier transform of a convolution between two signals is equivalent to the point-wise multiplication of their Fourier transforms. We can leverage this property to extend the definition of the convolution operation to other domains. In this context, we define the convolution between two functions f and g as:

$$
\begin{align}
f \ast g &= \mathcal{IGF}[\mathcal{GF}[f] \odot \mathcal{GF}[g]] \\
& = U(U^\intercal g \odot U^\intercal f) \\
&= U g_\theta(\Lambda) U\intercal f \\
& = g_\intercal(\mathcal{L}) f,
\end{align}
$$

where $$ \odot $$ denotes the Hadamard product and

$$
g_\theta(\Lambda) = diag(U^\intercal g),
$$

is the diagonal matrix that corresponds to the spectral filter coefficients. This spectral filter holds particular significance as the various versions of spectral-based Graph Convolutional Networks (GCN) differ based on the selection of the filter.

## Graph Convolutions Architectures

### Spectral Graph Neural Network
Bruna et al. {% cite bruna2013spectral %} introduced one of the earliest generalizations of Convolutional Neural Networks (CNNs) to handle signals over graphs. They put forward two constructions: the first one relies on hierarchical clustering, while the second one is based on the spectrum of the graph Laplacian.

First, it is worth mentioning that this architecture operates on a normalized Laplacian of a graph with a weight matrix $$ W $$, replacing the edge matrix $$ E $$:

$$
\mathcal{L} = I - D^{-\frac{1}{2}} W D^{-\frac{1}{2}}.
$$

The architecture takes a set of $$ C $$ signals $$ f $$ as input (mathematically, each signal corresponds to a matrix of size $$ \|V\| \times C $$, where $$ C $$ is the number of "channels"). Using the graph convolutional operator and a non-linearity function $$ \sigma $$, it transforms the input signals into an output signal that is fed to the next layer. Mathematically, Bruna defines the spectral filter as:

$$
\Lambda = \left[\begin{array}{cccc}
\theta^{(k)}_1 & & & \\
& \theta^{(k)}_2 & & \\
& & \ddots & \\
& & & \theta^{(k)}_n
\end{array}\right],
$$

and only utilizes the first d eigenvectors of the Laplacian (assuming they are ordered by eigenvalue), specifically retaining only the first d columns of the matrix $$ U $$:

$$
f_j^{(k + 1)} = \sigma\left( U_d \sum_{i = 1}^{C_k} g_\theta^{(k)} U_d^\intercal f_i^{(k)} \right)
= \sigma\left( U_d \sum_{i = 1}^{C_k} g_\theta^{(k)} \hat{f}_i^{(k)} \right).
$$

Berna et al. highlight two main challenges that this construction can find. Firstly, many graphs possess meaningful eigenvectors only at the higher end of the spectrum, even when the individual high-frequency eigenvectors lack significance. Secondly, it is not apparent how to efficiently perform forward and backward propagation while incorporating the computationally expensive operations of matrix multiplications and eigendecomposition of the Laplacian.

### ChebNet
To address the aforementioned challenges, we can overcome them by employing localized filters. One approach to tackle these issues is by utilizing polynomial parametrization as an approximation:

$$
g_\theta(\Lambda) = \sum_{k = 0}^{K - 1} \theta_k \Lambda^k
$$

In this case, the parameter $$ \theta $$ represents a vector of polynomial coefficients. As demonstrated by Hammond et al. {% cite hammond2011wavelets %}, a spectral filter of order $$ K $$ only captures K-localized information from the Laplacian.

Nevertheless, this approach suffers from high computational complexity due to the multiplication with the Fourier basis. To address this problem, Defferrard et al. {% cite defferrard2016convolutional %} proposed a parameterization of the spectral filter as a polynomial function that can be computed recursively from L using only sparse multiplications. They achieved this by approximating the spectral filter through a truncated expansion based on [Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials):

$$
\begin{align}
T_0(x) &= 1, \\
T_1(x) &= x, \\
T_k(x) &= 2xT_{k - 1}(x) - T_{k - 2}(x). \\
\end{align}
$$

And then, we can parametrize the spectral filter as a truncated expansion defined as:

$$
g_\theta(\Lambda) = \sum_{k = 0}^{K - 1} \theta_k T_k(\tilde{\Lambda}),
$$

where 

$$
\tilde{\Lambda} = \frac{2\Lambda}{\lambda_{max}} - I_n.
$$

With the filter defined in this way we can rewrite the filtering operation:

$$
g_\theta(L)f = \sum_{k = 0}^{K - 1} \theta_k T_k(\tilde{L})f.
$$

As you can see, this new operation no longer relies on the eigenbasis $$ U $$. Instead, it involves more cost-effective sparse multiplications, taking advantage of the sparsity of $$ L $$. Moreover, it eliminates the need for eigendecomposing the Laplacian matrix, making this method more suitable for handling large-scale graphs.

### Graph Convolutional Network
Lastly, let’s discuss the work of Kipf and Welling {% cite kipf2016semi %}, which simplifies the ChebNet approach by utilizing only local information, setting $$ K = 2 $$.

$$
g_\theta(L)f = \theta_0 T_0(\tilde{L}) f + \theta_1 T_1(\tilde{L}) f =
\theta_0 f + \theta_1 \tilde{L} f
$$

By approximating the highest eigenvalue to 2 (hoping that the neural network parameters will adapt to this scaling during training) and utilizing the normalized Laplacian, the expression can be further simplified to:

$$
g_\theta(L)f = \theta_0 f + \theta_1 (\mathcal{L} - I_n) f =
\theta_0 f + \theta_1 D^{-\frac{1}{2}}AD^{-\frac{1}{2}} f.
$$

We can go further and reduce the number of parameters using only one parameter θ, assigning $$ \theta = \theta_0 = -\theta_1 $$:

$$
g_\theta(L)f = \theta f - \theta D^{-\frac{1}{2}}AD^{-\frac{1}{2}} f = 
\theta\left(I_n + D^{-\frac{1}{2}}AD^{-\frac{1}{2}} \right)f.
$$

However, repeated application of this operator can lead to numerical instabilities. To address this issue, Kipf and Welling introduce the renormalization trick:

$$
I_n + D^{-\frac{1}{2}}AD^{-\frac{1}{2}} \to  \tilde{D}^{-\frac{1}{2}}A\tilde{D}^{-\frac{1}{2}}
$$

with $$ \tilde{A} = A + I_n $$ and $$ \tilde{D}_{ii} = \sum_j \tilde{A}_{ij} $$.

## Last thoughts
In conclusion, spectral graph convolution is a powerful technique in graph signal processing that expands the concept of convolution to graph-structured data. By utilizing the eigenvectors and eigenvalues of the graph Laplacian matrix, it enables localized filtering operations, facilitating the analysis and processing of signals on graphs.

We saw three different definitions of spectral filter, and I would like to summarize what are the contributions of each architecture:

- Spectral Graph Neural Networks (SGNs) were among the first works to incorporate spectral graph convolution as a fundamental building block. They utilized the graph Fourier transform to transform graph signals into the spectral domain, where linear operations could be applied for processing.
- ChebNet, a specific variant of SGNs, employed Chebyshev polynomials as an approximation for localized filtering. This approach provided computational efficiency, scalability, and the ability to capture localized information through the use of Chebyshev polynomials.
- Graph Convolutional Networks (GCNs) have gained significant popularity and are considered one of the most widely adopted classes of graph neural network architectures. GCNs simplified the ChebNet approach by directly operating on the graph’s adjacency matrix. They aggregate information from neighboring nodes to update node representations, making them effective for tasks involving graph-structured data.

Each architecture offers unique advantages, such as localized filtering, computational efficiency, scalability, and robustness to irregularities, catering to different requirements and characteristics of graph data. These architectures have significantly advanced the field of graph neural networks, enabling the application of deep learning techniques to graph-based problems.

## References
{% bibliography --cited %}

