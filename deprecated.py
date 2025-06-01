# def _symbol_string(self, i: int, j: int) -> str:
#     return f"X_{i}_{j}"
#
# def _symbol_basic(self, i: int, j: int) -> sympy.Basic:
#     return sympy.Symbol(self._symbol_string(i, j))
#  def _create_sym_matrix(self) -> List[List[sympy.Basic]]:
#         n = self._adj_tensor.shape[0]
#         symbol_matrix = [[None for _ in range(n)] for _ in range(n)]
#         self._symbols_mapping[sympy.S(0)] = 0
#         for i in range(n):
#             for j in range(n):
#                 if self._adj_tensor[i, j] == 1:
#                     symbol_matrix[i][j] = sympy.Symbol(self._symbol_string(i, j))
#                     self._symbols_mapping[symbol_matrix[i][j]] = 1
#                 else:
#                     symbol_matrix[i][j] = sympy.S(0)  # 0 jako symboliczna liczba zero
#
#         return symbol_matrix
# import torch
#
#
# def update_row_or_col(
#         self, new_data: torch.Tensor, indx: int, axis: str | int = "row"
# ):
#     if not isinstance(new_data, torch.Tensor):
#         raise TypeError("new_data must be a torch.Tensor.")
#     if len(self._row_to_change) > 0 and self._col_row_to_change_flag != axis:
#         self._update_matrix()
#
#     invalid_axis = False
#
#     if isinstance(axis, str):
#         axis = mapping_row_columns.get(axis)
#         if axis is None:
#             invalid_axis = True
#     elif isinstance(axis, int):
#         if axis not in {0, 1}:
#             invalid_axis = True
#     else:
#         invalid_axis = True
#
#     if invalid_axis:
#         raise ValueError('axis must be either "row" or "col"')
#     new_data = new_data.to(np_array_to_tensor_mapping(self._type_of_data))
#     self._add_primes_to_vector(new_data)
#     mask = (new_data != 0).to(np_array_to_tensor_mapping(self._type_of_data))
#     self._col_row_to_change_flag = axis
#     if axis == 0:  # row
#         for i in range(mask.size(0)):
#             if mask[i] != 0:
#                 self._add_to_index_set(new_data, i, indx, i)
#         self._add_to_changes(mask, new_data)
#     elif axis == 1:
#         for i in range(mask.size(0)):
#             if mask[i] != 0:
#                 self._add_to_index_set(new_data, indx, i, i)
#         self._add_to_changes(new_data, mask)
#     if len(self._row_to_change) > self._max_changes:
#         self._update_matrix()
# def _add_to_changes(self, row: torch.Tensor, col: torch.Tensor):
#     col = col.view(-1, 1)
#     row = row.view(1, -1)
#     print(col.shape)
#     print(row.shape)
#     self._row_to_change.append(row)
#     self._col_to_change.append(col)
# def _create_matrixes_from_vectors(self) -> Tuple[torch.Tensor, torch.Tensor]:
#     matrix_col = torch.cat(self._col_to_change, dim=1)
#     matrix_row = torch.cat(self._row_to_change, dim=0)
#     return matrix_col, matrix_row
#
#
# def _clear_to_change_lists(self):
#     self._row_to_change.clear()
#     self._col_to_change.clear()
#     self._to_change_index_set.clear()
# def _add_to_index_set(self, new_data: torch.Tensor, i: int, k: int, l: int) -> None:
#     if (k, i) not in self._to_change_index_set:
#         self._to_change_index_set.add((k, i))
#         new_data[l].sub_(self._graph_n_adj[k][i])
#         new_data[l] = (new_data[l] % self._p + self._p) % self._p
# def _update_matrix(self):
#     matrix_col, matrix_row = self._create_matrixes_from_vectors()
#     t1 = torch.matmul(self._graph_n_adj, matrix_col)
#     t1 = (t1 % self._p + self._p) % self._p
#
#     t2 = torch.matmul(matrix_row, t1)
#     t2 = (t2 % self._p + self._p) % self._p
#
#     eye = torch.eye(
#         self._adj_tensor.shape[0],
#         device="cuda",
#         dtype=np_array_to_tensor_mapping(self._type_of_data),
#     )
#
#     t3 = eye + t2
#     t3 = (t3 % self._p + self._p) % self._p
#     t3 = modular_inverse_matrix_gpu(t3, self._p)
#
#     t4 = torch.matmul(t1, t3)
#     t4 = (t4 % self._p + self._p) % self._p
#
#     t5 = torch.matmul(t4, matrix_row)
#     t5 = (t5 % self._p + self._p) % self._p
#
#     self._graph_n_adj.sub_(t5)
#     self._graph_n_adj = (self._graph_n_adj % self._p + self._p) % self._p
#
#     self._clear_to_change_lists()
#     self._col_row_to_change_flag = -1
