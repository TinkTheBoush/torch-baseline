import numpy as np


class OUNoise(object):
    def __init__(self, sigma, theta=.15, dt=1e-1, action_size=1, worker_size=1):
        self._theta = theta
        self._sigma = sigma
        self._dt = dt
        self.noise_prev = None
        self.action_size = action_size
        self.worker_size = worker_size
        self.noise_prev = np.random.normal(size=(self.worker_size,self.action_size))
    
    def __call__(self) -> np.ndarray:
        noise = self.noise_prev - self._theta * self.noise_prev * self._dt + \
                self._sigma * np.sqrt(self._dt) * np.random.normal(size=(self.worker_size,self.action_size))
        self.noise_prev = noise
        return noise

    def reset(self,worker) -> None:
        self.noise_prev[worker] = np.random.normal(size=(len(worker),self.action_size))