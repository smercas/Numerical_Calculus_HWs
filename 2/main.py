### RUN WITH ctrl + shift + B
### DEBUG WITH Python Debugger: Debug using launch.json -> Python: Module Debug

from abc import ABC, abstractmethod
import enum
from functools import lru_cache
import math
import time
from typing import Callable, Generic, Iterable, TypeVar
import numpy


PRECISION: float = 10 ** -10

def that_or(e, o): return e if e is not None else o
def skip_n(gen: Iterable, n: int):
	for _ in range(n): next(gen, None)
	yield from gen

F = TypeVar("F")
class BaseTriBehaviour(ABC, Generic[F]):
	@staticmethod
	@abstractmethod
	def in_tri(i: int, j: int, offset: int) -> bool: pass
	@staticmethod
	def on_diag(i: int, j: int, offset: int): return i + offset == j
	@staticmethod
	def diag_pos_for(i: int, j: int, offset: int): return i if offset >= 0 else j
	@property
	@staticmethod
	@abstractmethod
	def can_write_to_diagonal() -> bool: pass
	@classmethod
	@abstractmethod
	def mem_getters_and_setters(cls, shape: tuple[int, int], offset: int, including_diagonal: bool) -> tuple[Callable[[int, int], F], Callable[[int, int, F], None]]: pass
	@staticmethod
	def mem_get_and_set_bounds_check(shape: tuple[int, int], i: int, j: int) -> None:
		idx = (i, j)
		for nr, (ns, k, s) in enumerate(zip((-e for e in shape), idx, shape)):
			if not ns <= k < s:
				raise IndexError(f"index {k} is out of bounds for axis {nr} with size {s}")
	@classmethod
	def mem_get_and_set_within_alloc_check(cls, shape: tuple[int, int], i: int, j: int, offset: int, including_diagonal: bool):
		if not (cls.in_tri(i, j, offset) or (including_diagonal and cls.on_diag(i, j, offset))):
			raise IndexError(f"indexing ({i}, {j}) was done outside the allocated area")
TB = TypeVar("TB", bound=BaseTriBehaviour)

class LowerTriBehaviour(BaseTriBehaviour[F]):
	@staticmethod
	def in_tri(i: int, j: int, offset: int) -> bool:
		return i + offset > j
	@property
	@staticmethod
	def can_write_to_diagonal() -> bool: return True
	@classmethod
	def mem_getters_and_setters(cls, shape: tuple[int, int], offset: int, including_diagonal: bool) -> tuple[Callable[[int, int], F], Callable[[int, int, F], None]]:
		if not including_diagonal: offset -= 1
		if offset >= 0:
			empty_row_count, mem = 0, [numpy.zeros(min(i, shape[1])) for i in range(offset + 1, shape[0] + offset + 1)]
		else:
			empty_row_count, mem = -offset, [numpy.zeros(min(i, shape[1])) for i in range(1, shape[0] + offset + 1)]
		if not including_diagonal: offset += 1
		# print(numpy.array([[0] * shape[1] for _ in range(empty_row_count)] + [[1] * len(mem[i]) + [0] * (shape[1] - len(mem[i])) for i in range(shape[0] - empty_row_count)]))
		def mem_get(i: int, j: int) -> F:
			cls.mem_get_and_set_bounds_check(shape, i, j)
			cls.mem_get_and_set_within_alloc_check(shape, i, j, offset, including_diagonal)
			return mem[i - empty_row_count][j]
		def mem_set(i: int, j: int, v: F) -> None:
			cls.mem_get_and_set_bounds_check(shape, i, j)
			cls.mem_get_and_set_within_alloc_check(shape, i, j, offset, including_diagonal)
			mem[i - empty_row_count][j] = v
		return mem_get, mem_set

class UpperTriBehaviour(BaseTriBehaviour[F]):
	@staticmethod
	def in_tri(i: int, j: int, offset: int) -> bool:
		return i + offset < j
	@property
	@staticmethod
	def can_write_to_diagonal() -> bool: return False
	@classmethod
	def mem_getters_and_setters(cls, shape: tuple[int, int], offset: int, including_diagonal: bool) -> tuple[Callable[[int, int], F], Callable[[int, int, F], None]]:
		if not including_diagonal: offset += 1
		if offset >= shape[1]:
			mem = []
		else:
			mem = [numpy.zeros(min(i, shape[1])) for i in range(shape[1] - offset, max(0, shape[1] - shape[0] - offset), -1)]
		if not including_diagonal: offset -= 1
		# print(numpy.array([[0] * (shape[1] - (len(mem[i]) if len(mem) > i else 0)) + [1] * (len(mem[i]) if len(mem) > i else 0) for i in range(shape[0])]))
		def mem_get(i: int, j: int) -> F:
			cls.mem_get_and_set_bounds_check(shape, i, j)
			cls.mem_get_and_set_within_alloc_check(shape, i, j, offset, including_diagonal)
			return mem[i][j - max(i + offset + 1, 0)]
		def mem_set(i: int, j: int, v: F) -> None:
			cls.mem_get_and_set_bounds_check(shape, i, j)
			cls.mem_get_and_set_within_alloc_check(shape, i, j, offset, including_diagonal)
			mem[i][j - max(i + offset + 1, 0)] = v
		return mem_get, mem_set

class BaseTri(ABC, Generic[F]):
	def __init__(
		self, behaviour: TB,
		underlying_mem_get: Callable[[int, int], F], underlying_mem_set: Callable[[int, int, F], None],
		diagonal_get: Callable[[int], F], diagonal_set: Callable[[int, F], None],
		offset: int=0,
	):
		self._behaviour = behaviour
		self._diagonal_get = diagonal_get
		self._diagonal_set = diagonal_set
		self._underlying_mem_get = underlying_mem_get
		self._underlying_mem_set = underlying_mem_set
		self._offset = offset
	def _write_to(self, i: int, j: int, value: F) -> None:
		'''writes into the matrix at (i, j), according to the behaviour'''
		if self._behaviour.in_tri(i, j, self._offset):
			self._underlying_mem_set(i, j, value)
		elif self._behaviour.on_diag(i, j, self._offset) and self._behaviour.can_write_to_diagonal.fget():
			self._diagonal_set(self._behaviour.diag_pos_for(i, j, self._offset), value)
	def _read_from(self, i: int, j: int) -> F:
		'''reads from the matrix at (i, j), according to the behaviour'''
		if self._behaviour.in_tri(i, j, self._offset):
			return self._underlying_mem_get(i, j)
		elif self._behaviour.on_diag(i, j, self._offset):
			return self._diagonal_get(self._behaviour.diag_pos_for(i, j, self._offset))
		else:
			return self.dtype.type(0)
	def _idx_transform(self, idx: tuple[int | slice, int | slice], axis: int) -> tuple[int, int, int]:
		idx = idx[axis]
		return (that_or(idx.start, 0), that_or(idx.stop, self.shape[axis]), that_or(idx.step, 1)) if isinstance(idx, slice) else (idx, idx + 1, 1)
	def __setitem__(self, key: int | slice | tuple[int | slice, int | slice], value: F | numpy.ndarray) -> None:
		if not isinstance(key, tuple):
			key = key, slice(None, None, None)
		key = tuple(self._idx_transform(key, i) for i in range(self.ndim))
		if not isinstance(value, numpy.ndarray): value = numpy.array([[value]])
		for i in range(*key[0]):
			for j in range(*key[1]):
				self._write_to(i, j, value[i - key[0][0], j - key[1][0]])
	@abstractmethod
	def __getitem__(self, key: int | slice | tuple[int | slice, int | slice]) -> F | numpy.ndarray: pass
	@property
	@abstractmethod
	@lru_cache(maxsize=1) # doesn't change
	def shape(self) -> tuple[int, int]: pass
	@property
	@lru_cache(maxsize=1) # doesn't change
	def size(self) -> int: return math.prod(self.shape)
	@property
	def ndim(self) -> int: return 2
	@property
	@abstractmethod
	def dtype(self) -> type: pass
	def new_diag_get_and_set_func(self, new_mat_start: tuple[int, int], new_mat_end: tuple[int, int]) -> tuple[Callable[[int], F], Callable[[int, F], None]]:
		diag_start = (0, self._offset) if self._offset >= 0 else (-self._offset, 0)
		diag_end = tuple(p + self.diag_pos_upper_bound for p in diag_start)
		start_not_incl_count = max(nm - d for nm, d in zip(new_mat_start, diag_start))
		end_not_incl_count = max(d - nm for nm, d in zip(new_mat_end, diag_end))
		new_mat_diag_upper_bound = max(0, self.diag_pos_upper_bound - start_not_incl_count - end_not_incl_count)
		# def count_diag_elements():
		# 	r1, c1 = diag_start
		# 	r2, c2 = diag_end
		# 	rs, cs = new_mat_start
		# 	re, ce = new_mat_end
			
		# 	count = 0
		# 	for r, c in zip(range(r1, r2), range(c1, c2)):
		# 		if rs <= r < re and cs <= c < ce: count += 1
		# 	return count
		# assert count_diag_elements() == new_mat_diag_upper_bound
		def diagonal_get(i: int) -> F:
			i = self.diag_index_bounds_check_and_normalization(i, new_mat_diag_upper_bound)
			return self._diagonal_get(i + start_not_incl_count)
		def diagonal_set(i: int, v: F) -> None:
			i = self.diag_index_bounds_check_and_normalization(i, new_mat_diag_upper_bound)
			self._diagonal_set(i + start_not_incl_count, v)
		return diagonal_get, diagonal_set
	def __iter__(self):
		if self.shape[0] == 1:
			return (self[0, i] for i in range(self.shape[1]))
		else:
			return (self[i, :] for i in range(self.shape[0]))
	def tolist(self) -> list | list[list]:
		if self.shape[0] == 1: return [e.item() for e in self.__iter__()]
		if self.shape[1] == 1: return [[e[0, 0].item()] for e in self.__iter__()] 
		return [[e.item() for e in l.__iter__()] for l in self.__iter__()]
	def __str__(self):
		return numpy.array(self.tolist()).__str__()
	def __repr__(self):
		return numpy.array(self.tolist()).__repr__()
	@property
	@lru_cache(maxsize=1) # doesn't change
	def diag_pos_upper_bound(self): return min(self.shape[0], self.shape[1] - self._offset) if self._offset >= 0 else min(self.shape[0] + self._offset, self.shape[1])
	def diag_index_bounds_check_and_normalization(self, i: int, bound: int | None=None) -> int:
		bound = that_or(bound, self.diag_pos_upper_bound)
		if i < -bound: raise IndexError(f"index {i} is out of bounds for diagonal with size {bound}")
		if -bound <= i < 0: i += bound
		return i
	def __matmul__(self, other: 'BaseTri') -> numpy.ndarray | F:
		assert self.shape[1] == other.shape[0]
		common_axis = self.shape[1]
		if other.ndim == 1: other.reshape((other.shape[0], 1))
		result = numpy.zeros((self.shape[0], other.shape[1] if other.ndim > 1 else 1))
		for i in range(self.shape[0]):
			for j in range(other.shape[1] if other.ndim > 1 else 1):
				result[i, j] = sum(self[i, k] * (other[k, j] if other.ndim > 1 else other[k]) for k in range(common_axis))
		return result if result.shape != (1, 1) else result[0, 0]

class ViewTri(BaseTri[F]):
	@classmethod
	def with_diag_of_matrix(cls, underlying_numpy_matrix: numpy.ndarray, behaviour: TB, offset: int=0):
		def diagonal_get(i: int) -> F:
			i = obj.diag_index_bounds_check_and_normalization(i)
			if offset >= 0:
				return underlying_numpy_matrix[i, i + offset]
			else:
				return underlying_numpy_matrix[i + -offset, i]
		def diagonal_set(i: int, v: F) -> None:
			i = obj.diag_index_bounds_check_and_normalization(i)
			if offset >= 0:
				underlying_numpy_matrix[i, i + offset] = v
			else:
				underlying_numpy_matrix[i + -offset, i] = v
		obj = cls(underlying_numpy_matrix, behaviour, diagonal_get, diagonal_set, offset)
		return obj
	@classmethod
	def with_separate_diag(cls, underlying_numpy_matrix: numpy.ndarray, diagonal: numpy.ndarray, behaviour: TB, offset: int=0):
		def diagonal_get(i: int) -> F:
			i = obj.diag_index_bounds_check_and_normalization(i)
			return diagonal[i]
		def diagonal_set(i: int, v: F) -> None:
			i = obj.diag_index_bounds_check_and_normalization(i)
			diagonal[i] = v
		obj = cls(underlying_numpy_matrix, behaviour, diagonal_get, diagonal_set, offset)
		return obj
	def __init__(self, underlying_numpy_matrix: numpy.ndarray, behaviour: TB, diagonal_get: Callable[[int], F], diagonal_set: Callable[[int, F], None], offset: int=0):
		def underlying_mem_get(i: int, j: int) -> F: return underlying_numpy_matrix[i, j]
		def underlying_mem_set(i: int, j: int, v: F) -> None: underlying_numpy_matrix[i, j] = v
		super().__init__(behaviour, underlying_mem_get, underlying_mem_set, diagonal_get, diagonal_set, offset)
		self._underlying_mem = underlying_numpy_matrix
	def __getitem__(self, key: int | slice | tuple[int | slice, int | slice]) -> F | numpy.ndarray | 'ViewTri':
		if not isinstance(key, tuple):
			key = key, slice(None, None, None)
		if all(isinstance(k, int) for k in key): return self._read_from(*key)
		normalised_key = tuple(self._idx_transform(key, i) for i in range(len(key)))
		if any(k[1] - k[0] == 0 for k in normalised_key): return numpy.array([])
		return self.__class__(		
			self._underlying_mem[key].reshape(tuple(k[1] - k[0] for k in normalised_key)),
			self._behaviour,
			*self.new_diag_get_and_set_func(tuple(k[0] for k in normalised_key), tuple(k[1] for k in normalised_key)),
			self._offset + normalised_key[0][0] - normalised_key[1][0]
		)
	@property
	def shape(self) -> tuple[int, int]: return self._underlying_mem.shape
	@property
	def dtype(self) -> type: return self._underlying_mem.dtype

class MemTri(BaseTri[F]):
	@classmethod
	def with_diag_of_matrix_like(cls, alike: numpy.ndarray, behaviour: TB, offset: int=0):
		return cls.with_diag_of_matrix(alike.shape, alike.dtype, behaviour, offset)
	@classmethod
	def with_diag_of_matrix(cls, shape: tuple[int, int], dtype, behaviour: TB, offset: int=0):
		mem_get, mem_set = behaviour.mem_getters_and_setters(shape, offset, True)
		def diagonal_get(i: int) -> F:
			i = obj.diag_index_bounds_check_and_normalization(i)
			if offset >= 0:
				return mem_get(i, i + offset)
			else:
				return mem_get(i + -offset, i)
		def diagonal_set(i: int, v: F) -> None:
			i = obj.diag_index_bounds_check_and_normalization(i)
			if offset >= 0:
				mem_set(i, i + offset, v)
			else:
				mem_set(i + -offset, i, v)
		obj = cls(shape, dtype, behaviour, mem_get, mem_set, diagonal_get, diagonal_set, offset)
		return obj
	@classmethod
	def with_separate_diag_like(cls, alike: numpy.ndarray, diagonal: numpy.ndarray, behaviour: TB, offset: int=0):
		return cls.with_separate_diag(alike.shape, alike.dtype, diagonal, behaviour, offset)
	@classmethod
	def with_separate_diag(cls, shape: tuple[int, int], dtype, diagonal: numpy.ndarray, behaviour: TB, offset: int=0):
		def diagonal_get(i: int) -> F:
			i = obj.diag_index_bounds_check_and_normalization(i)
			return diagonal[i]
		def diagonal_set(i: int, v: F) -> None:
			i = obj.diag_index_bounds_check_and_normalization(i)
			diagonal[i] = v
		obj = cls(shape, dtype, behaviour, *behaviour.mem_getters_and_setters(shape, offset, False), diagonal_get, diagonal_set, offset)
		return obj
	def __init__(self, shape: tuple[int, int], dtype, behaviour: TB, mem_get: Callable[[int, int], F], mem_set: Callable[[int, int, F], None] , diagonal_get: Callable[[int], F], diagonal_set: Callable[[int, F], None], offset: int=0):
		self._shape = shape
		self._dtype = dtype
		super().__init__(behaviour, mem_get, mem_set, diagonal_get, diagonal_set, offset)
	def __getitem__(self, key: int | slice | tuple[int | slice, int | slice]) -> F | numpy.ndarray | 'MemTri':
		if not isinstance(key, tuple):
			key = key, slice(None, None, None)
		if all(isinstance(k, int) for k in key): return self._read_from(*key)
		normalised_key = tuple(self._idx_transform(key, i) for i in range(len(key)))
		def mem_get(i: int, j: int) -> F:
			return self._underlying_mem_get(*(c + k[0] for c, k in zip((i, j), normalised_key)))
		def mem_set(i: int, j: int, v: F) -> None:
			self._underlying_mem_set(*(c + k[0] for c, k in zip((i, j), normalised_key)), v)
		return self.__class__(		
			tuple(k[1] - k[0] for k in normalised_key),
			self._dtype,
			self._behaviour,
			mem_get, mem_set,
			*self.new_diag_get_and_set_func(tuple(k[0] for k in normalised_key), tuple(k[1] for k in normalised_key)),
			self._offset + normalised_key[0][0] - normalised_key[1][0]
		)
	@property
	def shape(self) -> tuple[int, int]: return self._shape
	@property
	def dtype(self) -> type: return self._dtype

if __name__ == "testing_slices":
	A = numpy.array([
		[0, 1, 2, 3, 4, 5],
		[10, 11, 12, 13, 14, 15],
		[20, 21, 22, 23, 24, 25],
		[30, 31, 32, 33, 34, 35],
		[40, 41, 42, 43, 44, 45],
		[50, 51, 52, 53 ,54, 55],
	])
	def diag(x): return [x._diagonal_get(i) for i in range(-x.diag_pos_upper_bound, x.diag_pos_upper_bound)]
	offset = -2
	L = MemTri.with_diag_of_matrix_like(A, LowerTriBehaviour, offset)
	LL = L[1:, 2:]
	LL[2:5, 0] = numpy.array([[-1], [-1], [-1]])
	LLL = LL[3, :]
	print(L)
	print(f"diag: {diag(L)}")
	print(LL)
	print(f"diag: {diag(LL)}")
	print(LLL)
	print(f"diag: {diag(LLL)}")
	dU = numpy.array([90, 91, 92, 93, 94, 95])
	U = MemTri.with_separate_diag_like(A, dU, UpperTriBehaviour, offset)
	UU = U[2:5, 1:]
	UU[:, 0] = numpy.array([[-1], [-1], [-1]])
	UUU = UU[:, 1]
	print(U)
	print(f"diag: {diag(U)}")
	print(UU)
	print(f"diag: {diag(UU)}")
	print(UUU)
	print(f"diag: {diag(UUU)}")
	print(A)
	pass

def LU_decomp(n: int, A: numpy.ndarray, dU: numpy.ndarray, use_views_for_LU=False) -> tuple[ViewTri | MemTri, ViewTri | MemTri]:
	assert A.ndim == 2, 'A is not a matrix'
	assert A.shape[0] == A.shape[1] == n, f'shape must be ({n}, {n}), but is ({A.shape[0]}, {A.shape[1]})'
	assert all(numpy.linalg.det(A[0:size, 0:size]) != 0 for size in range(1, n + 1)), 'submatrix determinants are 0, precondition not met'
	assert dU.ndim == 1, 'U must be a vector'
	assert dU.shape[0] == n, f'shape must be ({n}), but is ({dU.shape[0]})'

	if use_views_for_LU:
		A_init, A = A, A.copy()

		L = ViewTri.with_diag_of_matrix(A, LowerTriBehaviour)
		U = ViewTri.with_separate_diag(A, dU, UpperTriBehaviour)
	else:
		L = MemTri.with_diag_of_matrix_like(A, LowerTriBehaviour)
		U = MemTri.with_separate_diag_like(A, dU, UpperTriBehaviour)

	for p in range(0, n):
		for i in range(0, p + 1):
			L[p, i] = (A[p, i] - (L[p, :] @ U[:, i] - L[p, i] * U[i, i])) / U[i, i]
		for i in range(p + 1, n):
			U[p, i] = (A[p, i] - (L[p, :] @ U[:, i] - L[p, p] * U[p, i])) / L[p, p]
	return L, U

def solution(n: int, A: numpy.ndarray, B: numpy.ndarray, dU: numpy.ndarray) -> numpy.ndarray:
	L, U = LU_decomp(n, A, dU)
	Y = numpy.zeros(n)
	X = numpy.zeros(n)
	for i in range(0, n):
		Y[i] = (B[i] - L[i, :i] @ Y[:i]) / L[i, i]
	for i in range(n - 1, -1, -1):
		X[i] = (Y[i] - U[i, i + 1:] @ X[i + 1:]) / U[i, i]
	return X

if __name__ == "__main__":
	n = 3
	A = numpy.array([
		[4, 2, 3, ], 
		[2, 7, 5.5, ],
		[6, 3, 12.5, ],
	], dtype=numpy.float64)
	dU = numpy.array([2, 3, 4])
	B = numpy.array([21.6, 33.6, 51.6, ], dtype=numpy.float64).T
	L, U = LU_decomp(n, A, dU)
	det_of_A = math.prod(L[i, i] for i in range(n))
	print(L @ U)
	print(all(abs(a - lu) < PRECISION for a, lu in zip(A.flatten().tolist(), (L @ U).flatten().tolist())))

	print(L)
	print(U)

	def stress_test():
		start = time.perf_counter()
		numpy.random.seed(420)
		n = 100
		A = numpy.random.rand(n, n)
		dU = numpy.random.rand(n)
		B = numpy.random.rand(n)
		L, U = LU_decomp(n, A, dU)
		print(f"elapsed for LU decomp: {time.perf_counter() - start}")
		start = time.perf_counter()
		print(A) ; print()
		print(L @ U) ; print()
		print(all(a - lu < PRECISION for a, lu in zip(A.flatten().tolist(), (L @ U).flatten().tolist())))

		print(f"elapsed for increasing precision equality check: {time.perf_counter() - start}")
	stress_test()
		
	x_lu = solution(n, A, B, dU)

	norms = [numpy.linalg.norm(A @ x_lu - B)]
	print(f"first norm: {norms[0]}")
	assert norms[0] < PRECISION

	x_lib = numpy.linalg.solve(A, B)
	A_inv_lib = numpy.linalg.inv(A)

	norms += [numpy.linalg.norm(x_lu - x_lib), numpy.linalg.norm(x_lu - A_inv_lib @ B)]
	print(f"second norm: {norms[1]}")
	print(f"third norm: {norms[2]}")
