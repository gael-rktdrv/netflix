import numpy as np
import em
import common
import naive_em


X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")


class Test:

    def __init__(self, X_data, n_clusters):
        self.X_data = np.loadtxt(X_data)
        self.n_clusters = n_clusters

    def testing_estep(self):
        mixture, post = common.init(self.X_data, self.n_clusters)
        print(f'X shape: {X.shape}')
        print(f'mu shape: {mixture.mu.shape}')
        print(f'var shape: {mixture.var.shape}')
        print(f'p shape: {mixture.p.shape}\n')
        soft_counts, ll = naive_em.estep(X, mixture)
        print(f"Soft counts: \n{soft_counts}")
        print(f"\nLikelihood: {ll}\n")

    def testing_mstep(self):
        _, post = common.init(self.X_data, self.n_clusters)
        mixture = naive_em.mstep(self.X_data, post)
        print(f'X shape: {self.X_data.shape}')
        print(f'mu shape: {mixture.mu.shape}')
        print(f'var shape: {mixture.var.shape}')
        print(f'p shape: {mixture.p.shape}\n')
        print(f"Soft counts: {post.shape}")

    def testing_run(self):
        mixture, _ = common.init(self.X_data, self.n_clusters)

        """Updates"""
        mixture, post, old_ll, new_ll = naive_em.run(self.X_data, mixture)
        print(f"Final mixture: \n{mixture}")
        print(f"Old LL: {old_ll}")
        print(f"LL: {new_ll}")

    def testing_bic(self):
        mixture, _ = common.init(self.X_data, self.n_clusters)
        mixture, *_, log_likelihood= naive_em.run(X, mixture)
        # Get Results
        print(f"BIC: {common.bic(X, mixture, log_likelihood)}")


def main():
    n, d = X.shape
    seed = 0
    n_clusters = 3
    test = Test("toy_data.txt", n_clusters)
    # print("E-step results:")
    # test.testing_estep()
    # print("M-step results:")
    # test.testing_mstep()
    # print("EM algorithm results:")
    # test.testing_run()
    print("BIC results:")
    test.testing_bic()


if __name__ == "__main__":
    main()

