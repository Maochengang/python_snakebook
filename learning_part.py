from scipy.special import softmax
import numpy as np

def test_gradient(dim, time_steps=50, scale=1.0):
    # Assume components of the query and keys are drawn from N(0, 1) independently
    q = np.random.randn(dim)
    ks = np.random.randn(time_steps, dim)
    x = np.sum(q * ks, axis=1) / scale  # x.shape = (time_steps,)
    y = softmax(x)
    grad = np.diag(y) - np.outer(y, y)
    return np.max(np.abs(grad))  # the maximum component of gradients

NUMBER_OF_EXPERIMENTS = 5
# results of 5 random runs without scaling
print([test_gradient(100) for _ in range(NUMBER_OF_EXPERIMENTS)])
print([test_gradient(1000) for _ in range(NUMBER_OF_EXPERIMENTS)])

# results of 5 random runs with scaling
print([test_gradient(100, scale=np.sqrt(100)) for _ in range(NUMBER_OF_EXPERIMENTS)])
print([test_gradient(1000, scale=np.sqrt(1000)) for _ in range(NUMBER_OF_EXPERIMENTS)])