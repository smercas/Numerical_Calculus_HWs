### RUN WITH ctrl + shift + B
### DEBUG WITH Python Debugger: Debug using launch.json -> Python: Module Debug

from abc import ABC, abstractmethod
from collections import defaultdict
import itertools
import os
import random
from typing import Any, Callable, Generic, Iterator, Iterable, TypeVar


PRECISION: float = 10 ** -12


def add_when_condition_is_not_met(og: Iterable, c: Callable[[Any], bool], ng: Iterable):
	curr = next(og, None)
	while curr is not None and c(curr):
		yield curr
		curr = next(og, None)
	yield from ng
	if curr is not None: yield curr
	yield from og

def join_generators(ag: Iterable[tuple[int, float]], bg: Iterable[tuple[int, float]]) -> Iterator[tuple[int, float]]:
	ac, bc = next(ag, None), next(bg, None)
	while all(c is not None for c in [ac, bc]):
		if ac[0] < bc[0]:
			yield ac
			ac = next(ag, None)
		elif ac[0] == bc[0]:
			yield (ac[0], ac[1] + bc[1])
			ac, bc = next(ag, None), next(bg, None)
		else:
			yield bc
			bc = next(bg, None)
	if ac is not None: yield ac ; yield from ag
	if bc is not None: yield bc ; yield from bg

FT = TypeVar("FT")
class Vector(list[FT], Generic[FT]):
	@classmethod
	def from_path(cls, path: os.PathLike):
		with open(path, 'r') as file:
			_ = file.readline()
			return cls(float(l) for l in file if l.strip())
	@classmethod
	def from_iterable(cls, it: Iterable[FT]): return cls(it)
	@classmethod
	def empty(cls): return cls(())
	def __init__(self, arg: Iterable[FT]): super().__init__(arg)
	def __sub__(self, other: 'Vector[FT]') -> 'Vector[FT]':
		return Vector[FT](a - b for a, b in zip(self, other, strict=True))
	def __mul__(self, other: 'Vector[FT]' | FT) -> 'Vector[FT]' | FT:
		if isinstance(other, Vector):
			return sum(a * b for a, b in zip(self, other, strict=True))
		else:
			return Vector[FT].from_iterable(e * other for e in self)
	def __rmul__(self, other: FT) -> 'Vector[FT]': return self * other
	def __truediv__(self, other: FT) -> 'Vector[FT]': return Vector[FT].from_iterable(e / other for e in self)

class BaseRareMatrix(ABC):
	@classmethod
	@abstractmethod
	def from_path(cls, path: os.PathLike): pass
	@classmethod
	@abstractmethod
	def copy_of(cls, to_copy: 'BaseRareMatrix'): pass
	random_symmetrical__density_default = 0.1
	random_symmetrical__value_for_callback_default = lambda i, j: random.choice([1, -1]) * (1 - random.random()) * 100
	@classmethod
	@abstractmethod
	def random_symmetrical(cls, size: int, density: float=random_symmetrical__density_default, value_for_callback: Callable[[int, int], float]=random_symmetrical__value_for_callback_default): pass
	@abstractmethod
	def is_symmetrical(self, precision: int=PRECISION) -> bool: pass
	def _is_symmetrical_given_row_iterators(self, rows_after_diag_iters: Iterable[Iterator[tuple[int, float]]], rows_up_to_diag_iters: Iterable[list[tuple[int, float]]], precision: float) -> bool:
		if not isinstance(rows_after_diag_iters, list): rows_after_diag_iters = list(rows_after_diag_iters)
		def cols_up_to_diag_generator() -> Iterator[list[tuple[int, float]]]:
			current_of_row_iters = [next(l, None) for l in rows_after_diag_iters]
			current_col_index = 0
			def take_next_col():
				nonlocal current_col_index
				next_col_indexes = [i for i, kv in enumerate(current_of_row_iters) if kv is not None and kv[0] == current_col_index]
				next_col = [(i, current_of_row_iters[i][1]) for i in next_col_indexes]
				for i in next_col_indexes: current_of_row_iters[i] = next(rows_after_diag_iters[i], None)
				current_col_index += 1
				return next_col
			while current_col_index < self.size:
				yield take_next_col()
		### used for debugging bcs generators can only be used once
		# rows_up_to_diag_iters = list(rows_up_to_diag_iters)
		# cs = list(cols_up_to_diag_generator())
		# return all(all(rkv[0] == ckv[0] and abs(rkv[1] - ckv[1]) < precision for rkv, ckv in zip(r, c)) if len(r) == len(c) else False for r, c in zip(
		# 	rows_up_to_diag_iters, cs
		# ))
		return all(all(rkv[0] == ckv[0] and abs(rkv[1] - ckv[1]) < precision for rkv, ckv in zip(r, c)) if len(r) == len(c) else False for r, c in zip(
			rows_up_to_diag_iters, cols_up_to_diag_generator()
		))
	@property
	@abstractmethod
	def size(self) -> int: pass
	@property
	@abstractmethod
	def diagonal(self) -> list[float]: pass
	@property
	@abstractmethod
	def diagonal_iterator(self) -> Iterator[float]: pass
	@abstractmethod
	def line(self, index: int) -> list[tuple[int, float]]: pass
	@abstractmethod
	def line_iterator(self, index: int) -> Iterator[tuple[int, float]]: pass
	@abstractmethod
	def line_without_diag(self, index: int) -> list[tuple[int, float]]: pass
	@abstractmethod
	def line_without_diag_iterator(self, index: int) -> Iterator[tuple[int, float]]: pass
	def __matmul__(self, other: Vector[FT]) -> Vector[FT]:
		return Vector[FT].from_iterable(sum(self_ij * other[j] for j, self_ij in self.line(i)) for i in range(self.size))
	@abstractmethod
	def __add__(self, other: 'BaseRareMatrix') -> 'BaseRareMatrix': pass
	@abstractmethod
	def __eq__(self, other: 'BaseRareMatrix'): pass
RM = TypeVar("RM", bound=BaseRareMatrix)

class DefaultRareMatrix(BaseRareMatrix):
	@classmethod
	def from_path(cls, path: os.PathLike):
		with open(path, 'r') as file:
			n: int = int(file.readline())
			d: list[float] = [0.] * n
			r_as_dict: list[defaultdict[int, float]] = [defaultdict(float) for _ in range(n)]
			for line in file:
				line = line.strip()
				if not line: continue
				v, x, y = line.split(", ")
				v = float(v)
				x, y = int(x), int(y)
				if x == y:
					d[x] += v
				else:
					r_as_dict[x][y] += v
			r: list[list[tuple[int, float]]] = [sorted((e for e in l.items() if e[1] != 0), key=lambda kv: kv[0]) for l in r_as_dict]
			return cls(n, d, r)
	@classmethod
	def copy_of(cls, to_copy: RM):
		match to_copy:
			case DefaultRareMatrix():
				n = to_copy._n
				d = to_copy._d.copy()
				r = [e.copy() for e in to_copy._r]
			case BaseRareMatrix():
				n = to_copy.size
				d = list(to_copy.diagonal_iterator)
				r = [list(to_copy.line_without_diag_iterator(i)) for i in range(to_copy.size)]
			case _:
				raise ValueError(f"Can't copy object of type {to_copy.__class__}")
		return cls(n, d, r)
	@classmethod
	def random_symmetrical(cls, size: int, density: float=BaseRareMatrix.random_symmetrical__density_default, value_for_callback: Callable[[int, int], float]=BaseRareMatrix.random_symmetrical__value_for_callback_default):
		density /= 2
		d = [value_for_callback(i, i) if random.random() <= density else 0 for i in range(size)]
		r_as_dict = [{} for _ in range(size)]
		for i, ld in enumerate(r_as_dict):
			for j in range(i):
				if random.random() > density: continue
				v = value_for_callback(i, j)
				if abs(v) < PRECISION: continue
				ld[j] = v
				r_as_dict[j][i] = v
		r = [sorted(l.items(), key=lambda kv: kv[0]) for l in r_as_dict]
		return cls(size, d, r)
	def __init__(self, size: int, diagonal: list[float], sorted_lines_withoud_the_element_from_the_diagonal: list[list[tuple[int, float]]]):
		self._n = size
		self._d = diagonal
		self._r = sorted_lines_withoud_the_element_from_the_diagonal
	def is_symmetrical(self, precision: float=PRECISION) -> bool: return self._is_symmetrical_given_row_iterators(
		[itertools.dropwhile(lambda kv, i=i: kv[0] <= i, self._r[i]) for i in range(self.size)],
		(list(itertools.takewhile(lambda x: x[0] < i, self._r[i])) for i in range(self.size)),
		precision,
	)
	@property
	def size(self) -> int: return self._n
	@property
	def diagonal(self) -> list[float]: return self._d
	@property
	def diagonal_iterator(self) -> Iterator[float]: return iter(self.diagonal)
	def __line_generator(self, index: int) -> Iterator[float]:
		return add_when_condition_is_not_met(self.line_without_diag_iterator(index), lambda x: x[0] < index, [(index, self._d[index])])
	def line(self, index: int) -> list[tuple[int, float]]:
		if self._d[index] == 0: return self.line_without_diag(index)
		return list(self.__line_generator(index))
	def line_iterator(self, index: int) -> Iterator[tuple[int, float]]:
		if self._d[index] == 0: return self.line_without_diag_iterator(index)
		return self.__line_generator(index)
	def line_without_diag(self, index: int) -> list[tuple[int, float]]: return self._r[index]
	def line_without_diag_iterator(self, index: int) -> Iterator[tuple[int, float]]: return iter(self.line_without_diag(index))
	def __add__(self, other: RM) -> 'DefaultRareMatrix':
		assert self.size == other.size
		result = DefaultRareMatrix.copy_of(self)
		match other:
			case DefaultRareMatrix():
				result._d = [r + o for r, o in zip(result._d, other._d)]
				result._r = [list(join_generators(iter(r), iter(o))) for r, o in zip(result._r, other._r)]
			case BaseRareMatrix(): # any other BaseRareMatrix subclass
				result._d = [r + o for r, o in zip(result._d, other.diagonal_iterator)]
				result._r = [list(join_generators(iter(r), o)) for r, o in zip(result._r, (other.line_without_diag_iterator(i) for i in range(other.size)))]
		return result
	def __eq__(self, other: RM) -> bool:
		if self.size != other.size: return False
		match other:
			case DefaultRareMatrix():
				if not all(abs(s - o) < PRECISION for s, o in zip(self._d, other._d)): return False
				if not all(all(s[0] == o[0] and abs(s[1] - o[1]) < PRECISION for s, o in zip(ss, oo)) for ss, oo in zip(self._r, other._r)): return False
			case BaseRareMatrix():
				if not all((s - o) < PRECISION for s, o in zip(self._d, other.diagonal_iterator)): return False
				if not all(all(s[0] == o[0] and abs(s[1] - o[1]) < PRECISION for s, o in zip(ss, oo)) for ss, oo in zip(self._r, (other.line_without_diag_iterator(i) for i in range(other.size)))): return False
		return True

class CompressedRowStorageRareMatrix(BaseRareMatrix):
	@classmethod
	def from_path(cls, path: os.PathLike):
		with open(path, 'r') as file:
			n: int = int(file.readline())
			values_as_dict: list[defaultdict[int, float]] = [defaultdict(float) for _ in range(n)]
			for line in file:
				line = line.strip()
				if not line: continue
				v, x, y = line.split(", ")
				v = float(v)
				x, y = int(x), int(y)
				values_as_dict[x][y] += v
			values: list[tuple[int, float]] = list(itertools.chain(*(sorted((e for e in l.items() if e[1] != 0), key=lambda kv: kv[0]) for l in values_as_dict)))
			ptr = 0
			row_ptr: list[int] = [ptr]
			for i in range(n):
				ptr += len(values_as_dict[i])
				row_ptr += [ptr]
			return cls(n, values, row_ptr)
	@classmethod
	def copy_of(cls, to_copy: RM):
		match to_copy:
			case CompressedRowStorageRareMatrix():
				n = to_copy._n
				values = to_copy._values.copy()
				row_ptr = to_copy._row_ptr.copy()
			case BaseRareMatrix():
				n = to_copy.size
				values = list(itertools.chain(*(to_copy.line_iterator(i) for i in range(to_copy.size))))
				ptr = 0
				row_ptr: list[int] = [ptr]
				for i in range(to_copy.size):
					ptr += sum(1 for _ in to_copy.line_iterator(i))
					row_ptr += [ptr]
			case _:
				raise ValueError(f"Can't copy object of type {to_copy.__class__}")
		return cls(n, values, row_ptr)
	@classmethod
	def random_symmetrical(cls, size: int, density: float, value_for_callback: Callable[[int, int], float]):
		density /= 2
		values_as_dict = [{} for _ in range(size)]
		for i, ld in enumerate(values_as_dict):
			for j in range(i + 1):
				if random.random() > density: continue
				v = value_for_callback(i, j)
				if abs(v) < PRECISION: continue
				ld[j] = v
				if i == j: continue
				values_as_dict[j][i] = v
		values = list(itertools.chain(*(sorted(l.items(), key=lambda kv: kv[0]) for l in values_as_dict)))
		ptr = 0
		row_ptr = [ptr]
		for i in range(size):
			ptr += len(values_as_dict[i])
			row_ptr += [ptr]
		return cls(size, values, row_ptr)
	def __init__(self, size: int, values: list[tuple[int, float]], row_ptr: list[int]):
		self._n = size
		self._values = values
		self._row_ptr = row_ptr
	def is_symmetrical(self, precision: float=PRECISION) -> bool: return self._is_symmetrical_given_row_iterators(
		[itertools.dropwhile(lambda kv, i=i: kv[0] <= i, self.line_iterator(i)) for i in range(self.size)],
		(list(itertools.takewhile(lambda x: x[0] < i, self.line(i))) for i in range(self.size)),
		precision,
	)
	@property
	def size(self) -> int: return self._n
	@property
	def diagonal_iterator(self) -> Iterator[float]: return (next((e for e in self.line(i) if e[0] == i), (i, 0))[1] for i in range(self.size))
	@property
	def diagonal(self) -> list[float]: return list(self.diagonal_iterator)
	def line(self, index: int) -> list[tuple[int, float]]: return self._values[self._row_ptr[index]:self._row_ptr[index + 1]]
	def line_iterator(self, index: int) -> Iterator[tuple[int, float]]: return iter(self.line(index))
	def line_without_diag_iterator(self, index: int) -> Iterator[tuple[int, float]]: return (e for e in self.line(index) if e[0] != index)
	def line_without_diag(self, index: int) -> list[tuple[int, float]]: return list(self.line_without_diag_iterator(index))
	def __add__(self, other: RM) -> 'CompressedRowStorageRareMatrix':
		assert self.size == other.size
		result = CompressedRowStorageRareMatrix.copy_of(self)
		overall_offset = 0
		for i in range(result.size):
			rv = result._values[result._row_ptr[i]:result._row_ptr[i + 1] + overall_offset]
			prev_r_size = result._row_ptr[i + 1] + overall_offset - result._row_ptr[i]
			ov = other.line_iterator(i)
			nrv = list(e for e in join_generators(iter(rv), ov) if e[1] != 0)
			result._values[result._row_ptr[i]:result._row_ptr[i + 1] + overall_offset] = nrv
			overall_offset += len(nrv) - prev_r_size
			result._row_ptr[i + 1] += overall_offset
		return result
	def __eq__(self, other: RM) -> bool:
		if self.size != other.size: return False
		match other:
			case CompressedRowStorageRareMatrix():
				if not all(s[0] == o[0] and abs(s[1] - o[1]) < PRECISION for s, o in zip(self._values, other._values)): return False
				if not all(s == o for s, o in zip(self._row_ptr, other._row_ptr)): return False
			case BaseRareMatrix(): # any other BaseRareMatrix subclass
				if not all(s[0] == o[0] and abs(s[1] - o[1]) < PRECISION for s, o in zip(self._values, itertools.chain(*(other.line_iterator(i) for i in range(other.size))))): return False
				other_row_ptr = 0
				if self._row_ptr[0] != 0: return False

				for i in range(other.size):
					other_row_ptr += sum(1 for _ in other.line_iterator(i))
					if self._row_ptr[i + 1] != other_row_ptr: return False
		return True

def solve_using_Gauss_Seidel(A: RM, B: Vector[float], k_max: int = 100_000) -> tuple[Vector[float] | None, int]:
	assert all(d != 0 for d in A.diagonal_iterator)
	x = Vector[float].from_iterable(0. for _ in range(A.size))
	k = 0
	while True:
		delta_x = 0
		for i, A_ii in enumerate(A.diagonal_iterator):
			prev = x[i]
			x[i] = (B[i] - sum(A_ij * x[j] for j, A_ij in A.line_without_diag_iterator(i))) / A_ii
			delta_x += abs(prev - x[i])
		k += 1
		if not (PRECISION <= delta_x and k <= k_max): break
	return (x if PRECISION > delta_x else None), k

def solution(index: int) -> Vector[float]:
	match index:
		case 1: Vector[float].from_iterable(1. for _ in range(10_000))
		case 2: Vector[float].from_iterable(1./3. for _ in range(20_000))
		case 3: Vector[float].from_iterable(float(i) / 5.0 for i in range(30_000))
		case 4: Vector[float].from_iterable(80_000 - i - 1 for i in range(80_000))
		case 5: Vector[float].from_iterable(10. for _ in range(2025))

if __name__ == "__main__":
	folder = os.path.join("hw3", "rare_matrices")
	file_index = 4 # 5 does not work (it's intentional
	A = DefaultRareMatrix.from_path(os.path.join(folder, f"a_{file_index}.txt"))
	B = Vector[float].from_path(os.path.join(folder, f"b_{file_index}.txt"))
	x_GS, k = solve_using_Gauss_Seidel(A, B)
	# print(x_GS)
	print(k)
	if x_GS is not None: print(max(abs(d) for d in (A @ x_GS - B)))

	rare_matrix_types = [DefaultRareMatrix, CompressedRowStorageRareMatrix]

	### v testing copy and eq
	# for name in [f"a_{i}.txt" for i in range(5 + 1)] + ["a.txt", "aa.txt", "b.txt", "bb.txt", "aplusb.txt", "aaplusbb.txt"]:
	# 	for st, dt in itertools.product(rare_matrix_types, repeat=2):
	# 		S = st.from_path(os.path.join(folder, name))
	# 		D = dt.copy_of(S)
	# 		assert S == D
	### ^

	for At, Bt, ApBt in itertools.product(rare_matrix_types, repeat=3):
		A = At.from_path(os.path.join(folder, "a.txt"))
		B = Bt.from_path(os.path.join(folder, "b.txt"))
		ApB = ApBt.from_path(os.path.join(folder, "aplusb.txt"))
		assert A + B == ApB
