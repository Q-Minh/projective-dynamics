# Constraint Types

## Positional

The matrices Ai and Bi are simply identity. The matrix 3x3N Si matrix is a selection matrix 
which selects the positions qi. The goal position is pi, and thus this constraint 
minimizes the euclidean distance ||qi - pi||_2. We do not actually compute matrices Ai,Bi and Si.

## Edge length

The matrices Ai and Bi are mean subtraction matrices for two particles qi and qj. 
The 6x3N Si matrix is the selection matrix which has 3x3 identity blocks at rows,cols=1-3,qi-qi+3 
and at rows,cols=4-6,qj-qj+3. The matrices Ai,Bi and Si are not actually computed. The computation 
was done symbolically beforehand and we simply add corresponding values to their corresponding 
non-zero entries.

## Deformation gradient (Strain)

The matrices Ai,Bi,Si and pi are shown in pictures taken from a sympy session found in the [strain](./strain) folder. These matrix Ai*Si computes the flattened deformation gradient F of given tetrahedron (i,j,k,l) while pi is the flattened rotational part R of F. If R is a reflection (in other words det(R) is negative), then we correct it by negating its 3rd column. This prevents tetrahedron inversion.
This constraint thus minimizes the frobenius norm ||F - R||_2. For now, the matrix Ai*Si is stored 
as a sparse matrix to compute quantities wi \* (Ai\*Si)^T \* (Ai\*Si) and wi \* (Ai\*Si)^T \* Bi\*pi. 
Ideally, we should compute the non-zero entries symbolically beforehand for optimization, thus 
skipping sparse matrix products and simply adding values to the non-zero entries of A and b in the 
system Ax = b.