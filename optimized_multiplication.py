from typing import List, Tuple

import torch
from torch import Tensor


def modular_inverse_scalar(a: int, p: int) -> int:
    if a == 0:
        raise ValueError("Inverse does not exist for 0")
    t, new_t = 0, 1
    r, new_r = p, a
    while new_r != 0:
        quotient = r // new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r
    if r > 1:
        raise ValueError("a is not invertible")
    return t + p if t < 0 else t


def modular_inverse_matrix_gpu(matrix: torch.Tensor, p: int) -> torch.Tensor:
    assert matrix.device.type == "cuda"
    if matrix.dtype.is_floating_point:
        matrix = matrix.round().to(torch.int64)

    assert matrix.dtype in (torch.int32, torch.int64), "Matrix must be integer type"
    n = matrix.size(0)
    A = matrix.clone()
    I = torch.eye(n, dtype=matrix.dtype, device=matrix.device)

    for i in range(n):
        inv = modular_inverse_scalar(int(A[i, i].item()), p)
        A[i] = (A[i] * inv) % p
        I[i] = (I[i] * inv) % p

        for j in range(n):
            if i != j:
                factor = A[j, i].clone()
                A[j] = (A[j] - factor * A[i]) % p
                I[j] = (I[j] - factor * I[i]) % p

    return I


def apply_single_column_update(
    D: torch.Tensor, j: int, u: torch.Tensor, p: int
) -> torch.Tensor:
    correction = torch.ger(u, D[j]) % p
    return (D + correction) % p


def apply_single_row_update(
    A: torch.Tensor, j: int, v: torch.Tensor, p: int
) -> torch.Tensor:
    correction = torch.ger(A[:, j], v) % p
    return (A + correction) % p


def pow_number(number: int, power: int, mod: int) -> int:
    result = 1
    while power > 0:
        if power % 2 == 1:
            result = (result * number) % mod
        power = power // 2
        number = number * number
        number %= mod
    return result


def dense_to_sparse(matrix: torch.Tensor) -> torch.Tensor:
    indices = (matrix != 0).nonzero(as_tuple=False).T
    values = matrix[matrix != 0]
    return torch.sparse_coo_tensor(
        indices, values, size=matrix.shape, device=matrix.device
    ).coalesce()


def multiply_sparse_cols_C(
    C_dense: torch.Tensor, D: torch.Tensor, p: int
) -> torch.Tensor:
    if C_dense.is_sparse:
        C_sparse = C_dense
    else:
        C_sparse = dense_to_sparse(C_dense)  # sparsujemy CAŁĄ C
    return torch.sparse.mm(C_sparse, D) % p  # wynik będzie gęsty n x n


def multiply_sparse_rows_D(
    C: torch.Tensor, D_dense: torch.Tensor, p: int
) -> torch.Tensor:
    if D_dense.is_sparse:
        D_sparse = D_dense
    else:
        D_sparse = dense_to_sparse(D_dense)
    return torch.sparse.mm(D_sparse.transpose(0, 1), C.T).T % p
