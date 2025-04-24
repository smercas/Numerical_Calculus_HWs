### RUN WITH ctrl + shift + B
### DEBUG WITH Python Debugger: Debug using launch.json -> Python: Module Debug

import itertools
import math
from typing import Callable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

import numpy


def generation_method(n: int, low: float, high: float) -> tuple[numpy.ndarray, Callable[[], float]]:
	s = {low, high}
	while len(s) < n + 1: s.add(random.uniform(low, high))
	def c():
		while (x := random.uniform(low, high)) in s: pass
		return x
	return numpy.array(sorted(s)), c

def polynomial_approximation(xs: numpy.ndarray, ys: numpy.ndarray, degree: int) -> Callable[[float], float]:
	def coeffs_from_system():
		# B = numpy.fliplr(sum(numpy.eye(degree + 1, k=degree - i) * sum(xs ** i) for i in range(2 * degree + 1))) ### as shown in the doc
		B = sum(numpy.eye(degree + 1, k=degree - i) * sum(xs ** i) for i in range(2 * degree + 1))
		f = numpy.array([ys @ (xs ** i) for i in range(degree + 1)])
		# return numpy.linalg.solve(B, f)[::-1] ### as shown in the doc
		return numpy.linalg.solve(B, f)
	def coeffs_from_library(): return numpy.linalg.lstsq(numpy.vander(xs, degree + 1), ys)[0]
	coeffs = coeffs_from_system()
	# print(f"sum of squares: {sum(abs(sum(c * x ** i for i, c in enumerate(reversed(coeffs))) - y) ** 2 for x, y in zip(xs, ys))}")
	def approx(x: float) -> float:
		r = coeffs[0]
		for c in coeffs[1:]:
			r = r * x + c
		return r
	# approx2 = lambda x: sum(c * x ** i for i, c in enumerate(reversed(coeffs)))
	return approx

def part_1(n: int, range: tuple[float, float], degree: int, to_approx: Callable[[float], float]):
	xs, x_callable = generation_method(n, *range)
	ys = numpy.vectorize(to_approx)(xs)
	P_m = polynomial_approximation(xs, ys, degree)
	x = x_callable()
	print(f"actual  value of f({x}): {to_approx(x)}")
	print(f"approximation of f({x}): {P_m(x)}")
	print(f"abs diff: {abs(P_m(x) - to_approx(x))}")
	print(f"xs error: {numpy.sum(numpy.abs(numpy.vectorize(P_m)(xs) - ys))}")

def trigonometric_interpolation(xs: numpy.ndarray, ys: numpy.ndarray, m: int) -> Callable[[float], float]:
	k = numpy.arange(1, m + 1)
	sin_cols = numpy.sin(numpy.outer(k, xs))
	cos_cols = numpy.cos(numpy.outer(k, xs))
	T = numpy.column_stack([numpy.ones_like(xs), *itertools.chain.from_iterable(zip(sin_cols, cos_cols))])
	coeffs = numpy.linalg.solve(T, ys)
	return lambda x: sum(c * q(x) for c, q in zip(coeffs,
		itertools.chain([lambda _: 1], itertools.chain.from_iterable((lambda x: math.sin(i * x), lambda x: math.cos(i * x)) for i in range(1, m + 1))))
	)

def part_2(n: int, range: tuple[float, float], to_approx: Callable[[float], float]):
	xs, x_callable = generation_method(n, *range)
	ys = numpy.vectorize(to_approx)(xs)
	T_n = trigonometric_interpolation(xs, ys, n // 2)
	x = x_callable()
	print(f"actual  value of f({x}): {to_approx(x)}")
	print(f"approximation of f({x}): {T_n(x)}")
	print(f"abs diff: {abs(T_n(x) - to_approx(x))}")

if __name__ == "__main__":
	n = random.randint(4, 100)
	part_1(n, (1, 5), 4, lambda x: x ** 4 - 12 * x ** 3 + 30 * x ** 2 + 12)
	for r, f in [
		((0, 31 * math.pi / 16), lambda x: math.sin(x) - math.cos(x)),
		((0, 31 * math.pi / 16), lambda x: math.sin(2 * x) + math.sin(x) + math.cos(3 * x)),
		((0, 63 * math.pi / 32), lambda x: math.sin(x) ** 2 - math.cos(x) ** 2),
	]: part_2(n // 2 * 2, r, f)
