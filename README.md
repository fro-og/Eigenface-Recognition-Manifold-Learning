# Introduction

## 1. Linear Algebraic Formulation of the Face Recognition Problem

Let a grayscale image of dimensions $h \times w$ pixels be represented as a vector $\mathbf{x} \in \mathbb{R}^d$, where $d = h \times w$. A dataset of $N$ face images is organized as a matrix:

$$
\mathbf{X} = \begin{bmatrix} \mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x}_N \end{bmatrix} \in \mathbb{R}^{d \times N}
$$

where each column $\mathbf{x}_i$ corresponds to a flattened face image. The fundamental premise of the eigenface method is that the set of all natural face images lies on a low-dimensional linear subspace $\mathcal{S} \subset \mathbb{R}^d$ with $\dim(\mathcal{S}) = k \ll d$, rather than occupying the full ambient space. The objective is to determine an orthonormal basis $\{\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_k\}$ for this subspace, project all images onto it, and perform recognition in the reduced coordinate space.

## 2. Principal Component Analysis via the Singular Value Decomposition

Let $\boldsymbol{\mu} \in \mathbb{R}^d$ denote the mean face vector:

$$
\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i
$$

The mean-centered data matrix $\mathbf{A} \in \mathbb{R}^{d \times N}$ is constructed as:

$$
\mathbf{A} = \begin{bmatrix} \mathbf{x}_1 - \boldsymbol{\mu} & \mathbf{x}_2 - \boldsymbol{\mu} & \cdots & \mathbf{x}_N - \boldsymbol{\mu} \end{bmatrix}
$$

The Singular Value Decomposition of $\mathbf{A}$ is:

$$
\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T
$$

where $\mathbf{U} \in \mathbb{R}^{d \times d}$ and $\mathbf{V} \in \mathbb{R}^{N \times N}$ are orthogonal matrices ($\mathbf{U}^T\mathbf{U} = \mathbf{I}_d$, $\mathbf{V}^T\mathbf{V} = \mathbf{I}_N$), and $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times N}$ is a diagonal matrix containing the singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ along its diagonal, with $r = \operatorname{rank}(\mathbf{A}) \leq \min(d, N)$. The columns $\mathbf{u}_i$ of $\mathbf{U}$ are the left singular vectors, which, when reshaped to $h \times w$ dimensions, constitute the *eigenfaces*.

## 3. The Surrogate Covariance Matrix

Direct computation of the eigenvectors of the covariance matrix $\mathbf{C} = \frac{1}{N} \mathbf{A} \mathbf{A}^T \in \mathbb{R}^{d \times d}$ is computationally infeasible when $d$ is large. The eigenface method exploits the relationship between $\mathbf{A}\mathbf{A}^T$ and $\mathbf{A}^T\mathbf{A}$. Consider the surrogate covariance matrix:

$$
\mathbf{C}_{\text{surr}} = \frac{1}{N} \mathbf{A}^T \mathbf{A} \in \mathbb{R}^{N \times N}
$$

Let $\{\mathbf{v}_i\}_{i=1}^r$ be the eigenvectors of $\mathbf{C}_{\text{surr}}$ with corresponding eigenvalues $\lambda_i = \sigma_i^2 / N$. Then the eigenfaces $\mathbf{u}_i$ are recovered via:

$$
\mathbf{u}_i = \frac{1}{\sigma_i} \mathbf{A} \mathbf{v}_i, \quad i = 1, 2, \ldots, r
$$

This transformation reduces an eigenproblem of dimension $d$ to one of dimension $N$, which is computationally advantageous when $N \ll d$.

## 4. Dimensionality Reduction and Low-Rank Approximation

For a chosen $k \leq r$, define $\mathbf{U}_k \in \mathbb{R}^{d \times k}$ as the matrix whose columns are the first $k$ eigenfaces. The optimal rank-$k$ approximation of $\mathbf{A}$ in the Frobenius norm sense is given by the Eckart–Young theorem:

$$
\mathbf{A}_k = \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^T = \arg \min_{\operatorname{rank}(\mathbf{B}) = k} \|\mathbf{A} - \mathbf{B}\|_F
$$

The fraction of total variance preserved by retaining $k$ components is:

$$
\frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^r \sigma_i^2}
$$

## 5. Projection into Face Space

Any face image $\mathbf{x} \in \mathbb{R}^d$ (whether from the training set or a query) is projected into the $k$-dimensional face space via:

$$
\boldsymbol{\Omega} = \mathbf{U}_k^T (\mathbf{x} - \boldsymbol{\mu}) \in \mathbb{R}^k
$$

The vector $\boldsymbol{\Omega} = [\omega_1, \omega_2, \ldots, \omega_k]^T$ contains the coordinates (weights) of $\mathbf{x}$ in the eigenface basis. The reconstruction of $\mathbf{x}$ from its projection is:

$$
\hat{\mathbf{x}} = \boldsymbol{\mu} + \mathbf{U}_k \boldsymbol{\Omega} = \boldsymbol{\mu} + \mathbf{U}_k \mathbf{U}_k^T (\mathbf{x} - \boldsymbol{\mu})
$$

## 6. Reconstruction Error and Unknown Face Rejection

The reconstruction error is defined as the Euclidean distance between the original image and its projection:

$$
r = \|\mathbf{x} - \hat{\mathbf{x}}\|_2 = \| (\mathbf{x} - \boldsymbol{\mu}) - \mathbf{U}_k \mathbf{U}_k^T (\mathbf{x} - \boldsymbol{\mu}) \|_2
$$

This quantity measures how well $\mathbf{x}$ lies in the learned face subspace. A rejection threshold $\theta > 0$ is introduced: if $r > \theta$, the query is classified as an *unknown face* (not belonging to any enrolled identity). This criterion exploits the fact that non-face images or faces from individuals not represented in the training set will have large reconstruction errors.

## 7. Classification in Eigenspace Coordinates

Let the training set consist of $M$ distinct individuals, with multiple images per individual. For each training image $\mathbf{x}_i^{(j)}$ (the $j$-th image of person $i$), compute its eigenspace coordinates $\boldsymbol{\Omega}_i^{(j)} \in \mathbb{R}^k$. These vectors form clusters in $\mathbb{R}^k$, with each cluster corresponding to one individual.

For a query image $\mathbf{z}$ with $r \leq \theta$, compute $\boldsymbol{\Omega}_{\mathbf{z}} = \mathbf{U}_k^T (\mathbf{z} - \boldsymbol{\mu})$. Classification is performed by a multi-class Support Vector Machine (SVM). For a binary SVM separating two classes $y \in \{-1, +1\}$, the decision function is:

$$
f(\boldsymbol{\Omega}) = \operatorname{sign} \left( \sum_{i=1}^{N_{\text{SV}}} \alpha_i y_i K(\boldsymbol{\Omega}_i, \boldsymbol{\Omega}) + b \right)
$$

where $\alpha_i$ are Lagrange multipliers, $b$ is the bias term, and $K(\cdot, \cdot)$ is a kernel function (typically linear for eigenface coordinates). For $M > 2$ classes, the multi-class extension employs a one-vs-one or one-vs-all strategy, constructing $\binom{M}{2}$ binary classifiers and aggregating their predictions via majority voting.

## 8. Summary of the Algebraic Pipeline

The complete eigenface recognition pipeline consists of the following sequential linear algebraic operations:

| Step | Operation | Mathematical Expression |
| :--- | :--- | :--- |
| 1 | Mean face computation | $\boldsymbol{\mu} = \frac{1}{N}\sum_i \mathbf{x}_i$ |
| 2 | Mean centering | $\mathbf{A} = \mathbf{X} - \boldsymbol{\mu}\mathbf{1}^T$ |
| 3 | Surrogate covariance | $\mathbf{C}_{\text{surr}} = \frac{1}{N}\mathbf{A}^T\mathbf{A}$ |
| 4 | Eigendecomposition | $\mathbf{C}_{\text{surr}} \mathbf{v}_i = \lambda_i \mathbf{v}_i$ |
| 5 | Eigenface recovery | $\mathbf{u}_i = \frac{1}{\sqrt{N\lambda_i}} \mathbf{A} \mathbf{v}_i$ |
| 6 | Dimensionality reduction | $\mathbf{U}_k = [\mathbf{u}_1, \ldots, \mathbf{u}_k]$ |
| 7 | Projection | $\boldsymbol{\Omega} = \mathbf{U}_k^T(\mathbf{x} - \boldsymbol{\mu})$ |
| 8 | Classification | $\hat{y} = \arg\min_{c} \| \boldsymbol{\Omega} - \bar{\boldsymbol{\Omega}}_c \|$ (or SVM) |
| 9 | Rejection | $r = \|(\mathbf{x} - \boldsymbol{\mu}) - \mathbf{U}_k\boldsymbol{\Omega}\|_2 > \theta \Rightarrow \text{"unknown"}$ |

This algebraic framework constitutes the theoretical foundation of the eigenface method, upon which the subsequent implementation and experimental evaluation are based.