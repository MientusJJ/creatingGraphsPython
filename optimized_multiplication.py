import torch


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


def sparse_row_matmul_C(
    C: torch.Tensor,I: torch.tensor, D: torch.Tensor, p: int, tol=1e-8
) -> torch.Tensor:
    assert C.shape[1] == D.shape[0], "Incompatible shapes for matmul"

    n = C.shape[0]
    nonzero_rows = ((C-I).abs().sum(dim=1) > tol).nonzero(as_tuple=True)[0]  # (k,)

    C_sub = C[nonzero_rows]
    partial_result = torch.matmul(C_sub, D) % p

    result = torch.zeros_like(C)
    result[nonzero_rows] = partial_result % p
    return result


def sparse_col_matmul_C(
    C: torch.Tensor,I: torch.tensor, D: torch.Tensor, p: int, tol=1e-8
) -> torch.Tensor:
    assert C.shape[1] == D.shape[0], "Incompatible shapes for matmul"

    n = C.shape[0]
    result = torch.zeros_like(C)

    nonzero_cols = ((C-I).abs().sum(dim=0) > tol).nonzero(as_tuple=True)[0]  # (k,)

    for j in nonzero_cols:
        cj = C[:, j].view(-1, 1)  # shape (n x 1)
        dj = D[j, :].view(1, -1)  # shape (1 x n)
        result += torch.matmul(cj, dj) % p

    return result


def optimized_sparse_matmul_C(
    C: torch.Tensor,I: torch.tensor, D: torch.Tensor, p: int, tol=1e-8
) -> torch.Tensor:

    assert (
        C.shape[0] == C.shape[1] == D.shape[0] == D.shape[1]
    ), "Expected square matrices"

    row_sparsity = ((C-I).abs().sum(dim=1) > tol).sum().item()
    col_sparsity = ((C-I).abs().sum(dim=0) > tol).sum().item()

    if row_sparsity <= col_sparsity:
        return sparse_row_matmul_C(C,I, D, p, tol)
    else:
        return sparse_col_matmul_C(C,I, D, p, tol)


def pow_number(number: int, power: int, mod: int) -> int:
    result = 1
    while power > 0:
        if power % 2 == 1:
            result = (result * number) % mod
        power = power // 2
        number = number * number
        number %= mod
    return result


def optimized_matmul_single_left(
    vec: torch.Tensor, index: int, A: torch.Tensor, p: int, is_col: bool
) -> torch.Tensor:
    if not is_col:
        correction = torch.matmul(vec, A) % p
        result = A.clone()
        result = result.T
        result[index] = (result[index] + correction.squeeze(0)) % p
        return result.T
    else:
        Aj = A[index, :]  # 1 x r
        correction = torch.matmul(vec.view(-1, 1), Aj.view(1, -1)) % p  # outer product
        return (A + correction) % p


def optimized_matmul_single_right(
    A: torch.Tensor, vec: torch.Tensor, index: int, p: int, is_col: bool
) -> torch.Tensor:
    if is_col:
        Ai = A[:, index]
        correction = torch.matmul(Ai.view(-1, 1), vec.T) % p
        return (A + correction) % p
    else:
        scalar = vec[0, index].item()
        A[:, index] = (A[:, index] + scalar) % p
        return A
def sparse_row_matmul_sparse_D(C: torch.Tensor,I:torch.Tensor, D: torch.Tensor, p: int, tol=1e-8) -> torch.Tensor:
    assert C.shape[1] == D.shape[0], "Incompatible shapes for matmul"

    n = C.shape[0]
    result = torch.zeros_like(C)

    nonzero_cols_D = ((D-I).abs().sum(dim=0) > tol).nonzero(as_tuple=True)[0]  # (k,)
    for j in nonzero_cols_D:
        cj = C @ D[:, j].view(-1, 1)  # shape (n, 1)
        result[:, j] = cj.view(-1) % p

    return result


def sparse_col_matmul_sparse_D(C: torch.Tensor,I:torch.Tensor, D: torch.Tensor, p: int, tol=1e-8) -> torch.Tensor:
    """
    Zakłada, że macierz D ma mało niezerowych wierszy.
    """
    assert C.shape[1] == D.shape[0], "Incompatible shapes for matmul"

    n = C.shape[0]
    result = torch.zeros_like(C)

    nonzero_rows_D = ((D-I).abs().sum(dim=1) > tol).nonzero(as_tuple=True)[0]  # (k,)
    for i in nonzero_rows_D:
        ci = C[:, i].view(-1, 1)  # shape (n x 1)
        di = D[i, :].view(1, -1)  # shape (1 x n)
        result += torch.matmul(ci, di) % p

    return result


def optimized_sparse_matmul_sparse_D(C: torch.Tensor,I:torch.Tensor, D: torch.Tensor, p: int, tol=1e-8) -> torch.Tensor:
    """
    Wybiera optymalną metodę w zależności od tego,
    czy w D jest mniej niezerowych wierszy czy kolumn.
    """
    assert (
        C.shape[0] == C.shape[1] == D.shape[0] == D.shape[1]
    ), "Expected square matrices"

    row_sparsity_D = ((D-I).abs().sum(dim=1) > tol).sum().item()
    col_sparsity_D = ((D-I).abs().sum(dim=0) > tol).sum().item()

    if col_sparsity_D <= row_sparsity_D:
        return sparse_row_matmul_sparse_D(C,I, D, p, tol)
    else:
        return sparse_col_matmul_sparse_D(C,I, D, p, tol)
