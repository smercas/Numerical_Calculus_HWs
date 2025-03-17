from functools import lru_cache
import math
import random
import statistics
import time
from typing import Callable, Type
import numpy


def p1(floating_type: Type = float) -> float:
	'''will return an object of type `floating_type`'''
	u = floating_type(1)
	while 1 + u / 10 != 1:
		u /= 10
	return u

def p1_for_pow_of_2(floating_type: Type = float) -> float:
	'''will return an object of type `floating_type`'''
	u = floating_type(1)
	while 1 + u / 2 != 1:
		u /= 2
	return u

print(p1(), p1_for_pow_of_2())

def p2(floating_type: Type | None = None):
	u = p1(floating_type) if floating_type is not None else p1()
	assert (1.0 + u) + u != 1.0 + (u + u)

	u_rev = u ** (-1)
	assert (u_rev * u) * u != u_rev * (u * u)

def p3(floating_type: Type = numpy.float64):
	@lru_cache(maxsize=None)
	def c(n: int) -> float:
		if n == 0: return 1
		return c(n - 1) / (2 * n) / (2 * n + 1)
	def P1s(x: float) -> float: return x - c(1) * x ** 3 + c(2) * x ** 5
	def P1(x: float) -> float:
		xp2 = x ** 2
		return x * (1 + xp2 * (-c(1) + xp2 * c(2)))
	def P2s(x: float) -> float: return P1s(x) - c(3) * x ** 7
	def P2(x: float) -> float:
		xp2 = x ** 2
		return x * (1 + xp2 * (-c(1) + xp2 * (c(2) + xp2 * (-c(3)))))
	def P3s(x: float) -> float: return P2s(x) + c(4) * x ** 9
	def P3(x: float) -> float:
		xp2 = x ** 2
		return x * (1 + xp2 * (-c(1) + xp2 * (c(2) + xp2 * (-c(3) + xp2 * c(4)))))
	def P4s(x: float) -> float: return x - 0.166 * x ** 3 + 0.00833 * x ** 5 - c(3) * x ** 7 + c(4) * x ** 9
	def P4(x: float) -> float:
		xp2 = x ** 2
		return x * (1 + xp2 * (-0.166 + xp2 * (0.00833 + xp2 * (-c(3) + xp2 * c(4)))))
	def P5s(x: float) -> float: return x - 0.1666 * x ** 3 + 0.008333 * x ** 5 - c(3) * x ** 7 + c(4) * x ** 9
	def P5(x: float) -> float:
		xp2 = x ** 2
		return x * (1 + xp2 * (-0.1666 + xp2 * (0.008333 + xp2 * (-c(3) + xp2 * c(4)))))
	def P6s(x: float) -> float: return x - 0.16666 * x ** 3 + 0.0083333 * x ** 5 - c(3) * x ** 7 + c(4) * x ** 9
	def P6(x: float) -> float:
		xp2 = x ** 2
		return x * (1 + xp2 * (-0.16666 + xp2 * (0.0083333 + xp2 * (-c(3) + xp2 * c(4)))))
	def P7s(x: float) -> float: return P3s(x) - c(5) * x ** 11
	def P7(x: float) -> float:
		xp2 = x ** 2
		return x * (1 + xp2 * (-c(1) + xp2 * (c(2) + xp2 * (-c(3) + xp2 * (c(4) + xp2 * (-c(5)))))))
	def P8s(x: float) -> float: return P7s(x) + c(6) * x ** 13
	def P8(x: float) -> float:
		xp2 = x ** 2
		return x * (1 + xp2 * (-c(1) + xp2 * (c(2) + xp2 * (-c(3) + xp2 * (c(4) + xp2 * (-c(5) + xp2 * c(6)))))))
	sin_approx_old: list[Callable[[float], float]] = [P1s, P2s, P3s, P4s, P5s, P6s, P7s, P8s, ]
	sin_approx: list[Callable[[float], float]] = [P1, P2, P3, P4, P5, P6, P7, P8, ]
	values: list[float] = [random.uniform(-math.pi / 2, math.pi / 2) for _ in range(10_000)]
	sin_of_values: list[float] = [math.sin(x) for x in values]

	def display_relative_diff_between_old_and_new_sin_approx():
		err_of_old_to_new: map[tuple[Callable[[float], float], Callable[[float], float]], float] = {
			(f_old, f_new): statistics.mean([abs(f_old(x) - f_new(x)) for x in values]) for f_old, f_new in
			zip(sin_approx_old, sin_approx)
		}
		for o, n in err_of_old_to_new:
			print(f"({o.__name__}, {n.__name__}) -> {err_of_old_to_new[(o, n)]: .30f}")

	def bonus(approx_list) -> map:
		time_dict: map[Callable[[float], float], float] = {}
		for f in approx_list:
			start = time.perf_counter()
			_ = [f(x) for x in values]
			elapsed = time.perf_counter() - start
			time_dict[f] = elapsed
		approx_ranked_by_time: list[Callable[[float], float]] = sorted(approx_list, key=lambda f: time_dict[f])
		for f in approx_ranked_by_time:
			print(f"{f.__name__}: {time_dict[f]:.30f}")
		return time_dict
	# bonus(sin_approx_old)
	elapsed = bonus(sin_approx)

	sin_of_values_by_approx: map[Callable[[float], float], list[float]] = {f: [f(x) for x in values] for f in sin_approx}
	errors: map[Callable[[float], float], list[float]] = {
		f: [abs(approx - actual) for actual, approx in zip(sin_of_values, sin_of_values_by_approx[f])] for f in sin_approx
	}
	# # any transformation form an array to a single element works below, mean was chosen for simplicity
	errors_r: map[Callable[[float], float], float] = {f: statistics.mean(errors[f]) for f in sin_approx}
	def zscore(selected, values): return (selected - statistics.mean(values)) / statistics.stdev(values)
	error_zscores: map[Callable[[float], float], float] = {
		f: zscore(errors_r[f], errors_r.values()) for f in sin_approx
	}
	time_zscores: map[Callable[[float], float], float] = {f: zscore(elapsed[f], elapsed.values()) for f in elapsed}
	# # put in a bit more effort in ranking, justify the method used for it
	precision_weight = 1
	runtime_weight = 1 - precision_weight
	sin_approx_ranked: list[Callable[[float], float]] = sorted(
		sin_approx, key=lambda f: error_zscores[f] * precision_weight + time_zscores[f] * runtime_weight
	)
	# print(sin_approx_ranked[:])

if __name__ == '__main__':
	u = p1(numpy.float64)
	p2()
	p3()
