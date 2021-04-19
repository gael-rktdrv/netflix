import numpy as np
import em
import common
from common import GaussianMixture
from scipy.stats import multivariate_normal as MN

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0

# import pdb; pdb.set_trace()

X = np.array(
    [[0.85794562, 0.84725174],
     [0.6235637 , 0.38438171],
     [0.29753461, 0.05671298],
     [0.27265629, 0.47766512],
     [0.81216873, 0.47997717],
     [0.3927848 , 0.83607876],
     [0.33739616, 0.64817187],
     [0.36824154, 0.95715516],
     [0.14035078, 0.87008726],
     [0.47360805, 0.80091075],
     [0.52047748, 0.67887953],
     [0.72063265, 0.58201979],
     [0.53737323, 0.75861562],
     [0.10590761, 0.47360042],
     [0.18633234, 0.73691818]]
)

mu = np.array(
    [[0.6235637 , 0.38438171],
     [0.3927848 , 0.83607876],
     [0.81216873, 0.47997717],
     [0.14035078, 0.87008726],
     [0.36824154, 0.95715516],
     [0.10590761, 0.47360042]]
)

K = 6

var = np.array([0.10038354, 0.07227467, 0.13240693, 0.12411825, 0.10497521, 0.12220856])
p = np.array([0.1680912, 0.15835331, 0.21384187, 0.14223565, 0.14295074, 0.17452722])

print(f'X shape: {X.shape}')
print(f'mu shape: {mu.shape}')
print(f'var shape: {var.shape}')
print(f'p shape: {p.shape}\n')


def estep(X, mixture):
    K, _ = mixture.mu.shape
    n, d = X.shape
    gprob = lambda x, m, s: (1 / (2*np.pi*s)**(d/2)) * (np.exp(-((x-m)**2).sum(axis=1) / (2*s)))

    soft_counts = np.empty((0,K))

    for i in range(n):
        prob = gprob(np.tile(X[i], (K,1)), mixture.mu, mixture.var)
        prob = (prob*mixture.p)/(prob*mixture.p).sum()
        prob = prob.reshape(1, K)
        soft_counts = np.append(soft_counts, prob, axis=0)

    import pdb; pdb.set_trace()

    ll = np.log((soft_counts*mixture.p).sum(axis=0)).sum()
    # ll = np.sum(np.log(np.sum(soft_counts, axis = 0)))
    return soft_counts, ll

def main():
    mx = GaussianMixture(mu=mu, var=var, p=p)
    soft_counts, ll = estep(X, mx)
    print(soft_counts)
    print(f"\n{ll}\n")


if __name__=="__main__":
    main()