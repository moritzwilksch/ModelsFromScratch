# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# %%


def pi(x: np.ndarray) -> np.ndarray:
    """Calculates element-wise sigmoid function"""
    return 1 / (1 + np.exp(-x))


def l(y: float, o: float) -> float:
    """Calculates squared loss on a single sample"""
    return 1 / 2 * (y - o) ** 2


def partial_l_o(y: float, o: float) -> float:
    """Calculates partial l w.r.t. o"""
    return y - o


def _partial_pi_z_scalar(z: np.ndarray) -> np.ndarray:
    """For a single scalar, calculates derivative of pi/sigmoid"""
    return pi(z) * (1 - pi(z))


def partial_pi_z(z: np.ndarray) -> np.ndarray:
    """For an array of zs, calculates the matrix of derivatives partial h_j w.r.t. z_i"""
    # return np.diag(_partial_pi_z_scalar(z.flatten()))
    return _partial_pi_z_scalar(z)


def partial_l_w1(y, o, x, w2):
    """Using the chain rule, calculates partial l w.r.t. W1"""
    #breakpoint()
    return np.matmul(
        np.matmul(
            np.matmul(
                partial_l_o(y, o), w2.T
            ), partial_pi_z(np.matmul(w1.T, x))
        ).T, x.T
    )


def partial_l_w2(y, o, x, w1):
    return (y-o) * pi(w1.T @ x)


def predict(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """Given learned weights and input x, predicts y"""
    return np.matmul(
        w2.T,
        pi(np.matmul(w1.T, x))
    )


# %%
w1 = np.array([[1, 2, 3], [4, 5, 6], [3, 2, 1], [9, 8, 1], [4, 2, 1]]) / 10
#w1 = np.zeros((5,3))
w2 = np.array([1, 2, 3]).reshape(-1, 1) / 10
#w2 = np.zeros((3, 1))
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1) / 10
y = np.array([0.42])

o = predict(x, w1, w2)
print(f"predict = {o}")
print(partial_l_w1(y, o, x, w2))

# %%
loss = []
for i in range(10):
    o = predict(x, w1, w2)
    loss.append(l(y, o).item())
    print(f"Prediction = {o.item()} ---> loss = {loss[-1]}")
    w1 += partial_l_w1(y, o, x, w2).T
    w2 += partial_l_w2(y, o, x, w1)
sns.lineplot(x=range(len(loss)), y=loss)
plt.title('Loss per iteration')
plt.show()
