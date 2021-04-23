import numpy as np
import em
import common
import naive_em


X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0


def testing_estep():
    X = np.loadtxt("toy_data.txt")
    K = 6
    mixture, post = common.init(X, K)
    print(f'X shape: {X.shape}')
    print(f'mu shape: {mixture.mu.shape}')
    print(f'var shape: {mixture.var.shape}')
    print(f'p shape: {mixture.p.shape}\n')
    soft_counts, ll = naive_em.estep(X, mixture)
    print(f"Soft counts: \n{soft_counts}")
    print(f"\nLikelihood: {ll}\n")


def testing_mstep():
    X = np.loadtxt("toy_data.txt")
    K = 6
    _, post = common.init(X, K)
    mixture, post = naive_em.mstep(X, post)
    print(f'X shape: {X.shape}')
    print(f'mu shape: {mixture.mu.shape}')
    print(f'var shape: {mixture.var.shape}')
    print(f'p shape: {mixture.p.shape}\n')
    print(f"Soft counts: {post.shape}")


def testing_run():
    X = np.loadtxt("toy_data.txt")
    K = 6
    mixture, _ = common.init(X, K)

    """Updates"""
    mixture, post, old_ll, new_ll = naive_em.run(X, mixture)

    print(f"Final mixture: \n{mixture}")
    print(f"Old LL: {old_ll}")
    print(f"LL: {new_ll}")


def testing_bic():
    X = np.loadtxt("toy_data.txt")
    K = 6
    mixture, _ = common.init(X, K)
    mixture, *_, log_likelihood= naive_em.run(X, mixture)

    print(f"BIC: {common.bic(X, mixture, log_likelihood)}")


if __name__ == "__main__":
    testing_bic()

