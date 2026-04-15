# Eigenvector Block Alignment (Procrustes Method)

This document explains the mathematical workings inside the `_align_block_svd` function from `Library/Eigenvector.py`.

## The Problem it Solves
When a Hamiltonian has a degenerate subspace (e.g., two bands with the exact same energy), any linear combination of their eigenvectors is *also* a valid eigenvector. When you ask Python (like `np.linalg.eigh`) to calculate the eigenvectors at $k$ and then at $k + \delta k$, it will spit out two completely arbitrary, randomly rotated bases for that degenerate subspace. 

Since the basis abruptly rotates randomly between $k$ and $k+\delta k$, taking numerical derivatives ($\frac{\partial \psi}{\partial k}$) will blow up to infinity, utterly destroying Quantum Geometric Tensor calculations. 

## Step-by-step Math
The `_align_block_svd` function finds a unitary rotation to smoothly "rotate" the new random basis back so that it aligns as closely as possible with the previous point's basis. This ensures parallel transport across degenerate states.

### 1. Create the Block Matrices
```python
U0 = np.column_stack([prev_vecs[i] for i in block_inds])  # (dim, m)
U1 = np.column_stack([new_vecs[i]  for i in block_inds])  # (dim, m)
```
We stack the $m$ degenerate eigenvectors from the previous numerical evaluation step into a matrix $U_0$, and the random new eigenvectors into $U_1$. Here, $dim$ is the total Hilbert space dimension and $m$ is the subspace degeneracy size (e.g., $m=2$).

**Note:** `prev_vecs` is *not* one vector, it is a **list containing all of the eigenvectors** for the entire system at the previous $k$-point.

Let's do a concrete mathematical example. Assume you have a 4-band Hamiltonian (so `dim = 4`). 
- `prev_vecs` will contain 4 separate vectors: `[|u_0>, |u_1>, |u_2>, |u_3>]`. 
- Each individual `|u_i>` is a single column vector with 4 rows, for example: `|u_0> = [a1, a2, a3, a4]^T`.

Now, suppose the numerical solver finds that bands 1 and 2 (the middle two bands) have the exact same degenerate energy. Thus, our degenerate subspace indices are `block_inds = [1, 2]`. 
This makes our block degeneracy size $m=2$.

To isolate just this degenerate subspace, we pull specifically `|u_1>` and `|u_2>` out of the `prev_vecs` list and stack them side-by-side to form a $4 \times 2$ matrix we call $U_0$:

$$ U_0 = \begin{bmatrix} | & | \\ |u_1\rangle & |u_2\rangle \\ | & | \end{bmatrix} $$

We do the exact same physical extraction for the scrambled new vectors in `new_vecs` to build $U_1$:

$$ U_1 = \begin{bmatrix} | & | \\ |v_1\rangle & |v_2\rangle \\ | & | \end{bmatrix} $$

### 2. The Overlap Matrix ($M$)
```python
M = U0.conj().T @ U1  # (m, m)
```
Mathematically, $M = U_0^\dagger U_1$. 
The entries of this $m \times m$ matrix represent the projection overlaps $\langle \psi_0 | \psi_1 \rangle$. If the solver hadn't randomly rotated the basis, $M$ would just be the Identity matrix. Because it *did* rotate it, $M$ is scrambled. We essentially need to "unscramble" $M$.

### 3. SVD and the Polar Unitary ($R$)
```python
V, s, Wh = np.linalg.svd(M)
R = V @ Wh  
```
We want to find an $m \times m$ unitary rotation matrix $R$ such that applying $R^\dagger$ to $U_1$ makes it look like $U_0$. Specifically, we want to maximize the real trace overlap $\text{Re}(\text{Tr}(M R^\dagger))$.


**Why does SVD solve this? Here is the mathematical proof:**

By taking the Singular Value Decomposition (SVD), any arbitrary matrix can be factored as $M = V S W^\dagger$ (where $S$ is a diagonal matrix of real, non-negative singular values, and $V, W^\dagger$ are unitaries). 

Substitute this into our trace objective:
$$\text{Tr}(M R^\dagger) = \text{Tr}(V S W^\dagger R^\dagger)$$

Using the cyclic property of the trace ($\text{Tr}(A B C) = \text{Tr}(C A B)$), we can shift $V$ to the back:
$$\text{Tr}(V S W^\dagger R^\dagger) = \text{Tr}(S W^\dagger R^\dagger V)$$

Let's group the unitaries into a single new matrix $Z = W^\dagger R^\dagger V$. Because multiplying unitaries always produces another unitary, $Z$ is just a unitary matrix. 
Now our goal is to maximize:
$$\text{Re}(\text{Tr}(S Z)) = \text{Re} \left( \sum_{i} S_{ii} Z_{ii} \right) = \sum_{i} S_{ii} \text{Re}(Z_{ii})$$

Since $S_{ii}$ are strictly positive singular values, and $Z$ is unitary (meaning every element is bounded by $|Z_{ii}| \leq 1$), the absolute maximum possible value of this sum occurs when $\text{Re}(Z_{ii}) = 1$ for all $i$. 

The only way a unitary matrix can have purely 1's down its diagonal is if it is the Identity matrix ($I$). 
Therefore, we completely mathematically constrain that $Z = I$:
$$ W^\dagger R^\dagger V = I $$

Multiply by $W$ on the left and $V^\dagger$ on the right:
$$ R^\dagger = W V^\dagger $$

Take the conjugate transpose of both sides:
$$ R = (W V^\dagger)^\dagger = V W^\dagger $$

This exact solution $R = V W^\dagger$ is exactly what `V @ Wh` computes (`Wh` from numpy is $W^\dagger$). The optimal unitary "extracts" just the pure rotation part of $M$ (ignoring the $S$ scaling length deformations). In NumPy, `Wh` is $W^\dagger$, so $R = V W^\dagger$ is exactly `V @ Wh`.

### 4. Rotate the New Basis
```python
U1_aligned = U1 @ R.conj().T
```
We apply $R^\dagger$ to our scrambled new vectors $U_1$. Now, $\tilde{U}_1 = U_1 R^\dagger$. 
The columns of $U_1$ are now completely un-scrambled and smoothly mirror the vectors in $U_0$ exactly as required physically.

### 5. Re-normalize and Save
```python
v = U1_aligned[:, col]
nrm = np.linalg.norm(v)
if nrm > eps:
    v = v / nrm
out[idx] = v
```
Finally, since floating-point math can introduce very tiny numerical errors, we force the vectors back to a strict amplitude of `1.0` before placing them back into the eigenvector list!
