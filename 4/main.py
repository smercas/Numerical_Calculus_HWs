from typing import Callable
import numpy


PRECISION: float = 10 ** -10
# daca norma nu e specificata o alegem noi :)

def norm_1(m: numpy.ndarray) -> float: return max(numpy.sum(numpy.abs(m[:, j])) for j in range(m.shape[0]))
def norm_inf(m: numpy.ndarray) -> float: return max(numpy.sum(numpy.abs(m[i, :])) for i in range(m.shape[0]))

def approx_inverse(n: int, A: numpy.ndarray, computing_method_factory: Callable[[int, numpy.ndarray], Callable[[numpy.ndarray], numpy.ndarray]], k_max: int = 10_000) -> tuple[numpy.ndarray, int]:
	assert A.ndim == 2
	assert A.shape[0] == A.shape[1] == n
	if (all(numpy.abs(A[i, i]) > numpy.sum(numpy.abs(A[i, :])) - A[i, i] for i in range(n))
				or
			all(numpy.abs(A[i, i]) > numpy.sum(numpy.abs(A[:, i])) - A[i, i] for i in range(n))):
		V = numpy.diag(1 / numpy.diag(A))
	elif (A - A.T < PRECISION).all() and numpy.all(numpy.linalg.eigvals(A) > 0):
		V = numpy.eye(n) * (1 / numpy.sqrt(numpy.sum(numpy.abs(A)**2)))
	else:
		V = A.T / (norm_1(A) * norm_inf(A))
	computing_method = computing_method_factory(n, A)
	k = 0
	while True:
		prev_V, V = V, computing_method(V)
		delta_V = norm_1(V - prev_V)
		k += 1
		if not (PRECISION <= delta_V <= 10 ** 10 and k <= k_max): break
	return (V if PRECISION > delta_V else None), k

def with_a_added_to_diagonal(n: int, M: numpy.ndarray, a: float) -> numpy.ndarray:
	sM = M.copy()
	add_a_to_diagonal(n, sM, a)
	return sM

def add_a_to_diagonal(n: int, M: numpy.ndarray, a: float) -> None:
	for i in range(n):
		M[i, i] += a

def method_1(n: int, A: numpy.ndarray) -> Callable[[numpy.ndarray], numpy.ndarray]:
	mA = -A
	def f(V: numpy.ndarray) -> numpy.ndarray:
		mA_matmul_V = mA @ V
		add_a_to_diagonal(n, mA_matmul_V, 2.)
		return V @ (mA_matmul_V)
	return f

def method_2(n: int, A: numpy.ndarray) -> Callable[[numpy.ndarray], numpy.ndarray]:
	mA = -A
	def f(V: numpy.ndarray) -> numpy.ndarray:
		mA_matmul_V = mA @ V
		p = mA_matmul_V @ (with_a_added_to_diagonal(n, mA_matmul_V, 3.))
		add_a_to_diagonal(n, p, 3.)
		return V @ p
	return f

def method_3(n: int, A: numpy.ndarray) -> Callable[[numpy.ndarray], numpy.ndarray]:
	mA = -A
	def f(V: numpy.ndarray) -> numpy.ndarray:
		V_matmul_mA = V @ mA
		to_square = with_a_added_to_diagonal(n, V_matmul_mA, 3.)
		add_a_to_diagonal(n, V_matmul_mA, 1.)
		to_add_1_to_diagonal = (V_matmul_mA @ (to_square @ to_square)) / 4
		add_a_to_diagonal(n, to_add_1_to_diagonal, 1.)
		return to_add_1_to_diagonal @ V
	return f

methods = [method_1, method_2, method_3]

def approx_left_inverse(A: numpy.ndarray, computing_method_factory: Callable[[int, numpy.ndarray], Callable[[numpy.ndarray], numpy.ndarray]], k_max: int = 10_000) -> tuple[numpy.ndarray, int]:
	inv, k = approx_inverse(A.shape[1], A.T @ A, computing_method_factory, k_max)
	if inv is None: return None, k
	return inv @ A.T, k

def approx_right_inverse(A: numpy.ndarray, computing_method_factory: Callable[[int, numpy.ndarray], Callable[[numpy.ndarray], numpy.ndarray]], k_max: int = 10_000) -> tuple[numpy.ndarray, int]:
	inv, k = approx_inverse(A.shape[0], A @ A.T, computing_method_factory, k_max)
	if inv is None: return None, k
	return A.T @ inv, k


if __name__ == "__main__":
	A = numpy.array([
		[2., 1., 3., 4., 5., ],
		[1., 4., 2., 3., 6., ],
		[3., 2., 5., 1., 4., ],
		[4., 3., 1., 5., 2., ],
		[5., 6., 4., 2., 1., ],
	])
	n = A.shape[0]
	for method in methods:
		A_inv, k = approx_inverse(n, A, method)
		# print(k)
		if A_inv is not None:
			diff = A @ A_inv - numpy.eye(n)
			assert norm_1(diff) < PRECISION

	lA = numpy.array([
		[2., 1., 3., ],
		[1., 4., 2., ],
		[3., 2., 5., ],
		[4., 3., 1., ],
		[5., 6., 4., ],
	])
	rA = numpy.array([
		[2., 1., 3., 4., 5., ],
		[1., 4., 2., 3., 6., ],
		[3., 2., 5., 1., 4., ],
	])
	for A in [lA, rA]:
		for method in methods:
			for inv_type, diff_func in [
				(approx_left_inverse, lambda x: x @ A),
				(approx_right_inverse, lambda x: A @ x),
			]:
				inv, k = inv_type(A, method)
				if inv is not None:
					diff = diff_func(inv)
					print(f"{inv_type}: {norm_1(diff - numpy.eye(diff.shape[0]))}")
					assert norm_1(diff - numpy.eye(diff.shape[0])) < PRECISION

	print()
	numpy.set_printoptions(precision=13, suppress=True, linewidth=200)
	def matrix_for_3rd_exercise(n: int): return numpy.eye(n) + numpy.eye(n, k=1) * 2
	def inverse_form_for_3rd_exercise(n: int):
		return sum(numpy.eye(n, k=i) * (2 ** i) * (-1 if i % 2 == 1 else 1) for i in range(n))
	for method_number, method in enumerate(methods, 1):
		# print(f"for method {method_number}:")
		for n in range(2, 10 + 1):
			A = matrix_for_3rd_exercise(n)
			A_inv, k, = approx_inverse(n, A, method)
			diff = A @ A_inv - numpy.eye(n)
			# if A_inv is not None:
			# 	print(A)
			# 	print(A_inv)
			# 	print(norm_inf(inverse_form_for_3rd_exercise(n) - A_inv))
