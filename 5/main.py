### RUN WITH ctrl + shift + B
### DEBUG WITH Python Debugger: Debug using launch.json -> Python: Module Debug

import itertools
import os
import random
from typing import Iterable

import numpy
from hw3 import main as rm


PRECISION = 10 ** -12

def euclidean_norm(v: rm.Vector) -> float: return sum(abs(e) ** 2 for e in v) ** 0.5

def part_1(n: int=1000) -> Iterable[rm.RM]:
	folder = os.path.join("hw5", "rare_matrices")
	return itertools.chain(
		[rm.DefaultRareMatrix.random_symmetrical(n, density=0.33,
			value_for_callback=lambda i, j: random.choice([1, -1]) * (1 - random.random()) * 100
		)], # this is the random matrix
		(rm.DefaultRareMatrix.from_path(os.path.join(folder, file)) for file in os.listdir(folder)),
	)

def part_2(rare_matrices: Iterable[rm.RM]) -> None:
	def power_method(A: rm.RM, precision: float=PRECISION, k_max: int=1_000_000) -> tuple[float, rm.Vector]:
		x = rm.Vector.from_iterable(1 - random.random() for _ in range(A.size))
		v = x / euclidean_norm(x)
		w = A @ v
		lb = w * v
		k = 0
		while k < k_max:
			# print(f"{k} : {A.size * precision} : {euclidean_norm(w - v * lb)}")
			v = w / euclidean_norm(w)
			w = A @ v
			lb = w * v
			k += 1
			if euclidean_norm(w - lb * v) <= A.size * precision:
				print(k)
				return lb, v
		return None
	for i, A in enumerate(rare_matrices):
		assert A.is_symmetrical(PRECISION)
		print(f"for matrix of shape {A.size}")
		t = power_method(A, k_max=1_000_000 if i == 0 else 250)
		if t is None:
			print("exceeded k_max")
			print()
			print()
			continue
		lb, v = t
		print(f"{lb} : {euclidean_norm(A @ v - lb * v)}")
		print()

def part_3(A: numpy.ndarray, b: numpy.ndarray, precision: float=PRECISION):
	U, S, V = numpy.linalg.svd(A)
	V = V.T
	print(f"singular values: {S}")
	print(f"rank: {numpy.linalg.matrix_rank(A)} : {numpy.sum(S > precision)}")
	print(f"conditioning number: {numpy.linalg.cond(A)} : {numpy.max(S) / numpy.min(S[S > precision])}")
	Si = numpy.zeros_like(A.T)
	Si[:len(S), :len(S)] = numpy.diag(1 / S[S > 0])
	print(Si)
	Ai = V @ Si @ U.T
	print(f"Moore-Penrose pseudo-inverse: {Ai}")
	xi = Ai @ b
	print(f"solution (xi): {xi}")
	print(f"norm: {euclidean_norm(b - A @ xi)}")

if __name__ == "__main__":
	rare_matrices = part_1(10)
	part_2(rare_matrices)
	part_3(numpy.random.random((5, 4)), numpy.random.random(5))
