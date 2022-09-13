from types import SimpleNamespace

import gpytorch
import torch

__all__ = ['PowerEPGP']


# noinspection PyPep8Naming,PyRedundantParentheses
class PowerEPGP:
    def __init__(
            self, train_x: torch.Tensor, train_y: torch.Tensor,
            kernel: gpytorch.kernels.InducingPointKernel,
            likelihood: gpytorch.likelihoods.likelihood,
            alpha: float, inf=1e10
    ):
        self.train_x = train_x
        self.train_y = train_y
        self.kernel = kernel
        self.likelihood = likelihood
        self.quadrature = gpytorch.utils.quadrature.GaussHermiteQuadrature1D(100)

        self.n = len(self.train_x)
        self.m = len(self.kernel.inducing_points)

        self.alpha = alpha
        self.g = torch.zeros(self.n, 1, requires_grad=False)
        self.v = torch.ones(self.n, 1, requires_grad=False) * inf
        self.gamma = torch.zeros(self.m, 1, requires_grad=False)
        self.beta = torch.zeros(self.m, self.m, requires_grad=False)

    def posterior_f(self, test_x=None) -> gpytorch.distributions.MultivariateNormal:
        if test_x is None:
            test_x = self.train_x

        Kf = self.kernel(test_x).evaluate()
        Kfu = self.kernel(test_x, self.kernel.inducing_points).evaluate()
        return gpytorch.distributions.MultivariateNormal(
            (Kfu @ self.gamma).view(-1),
            Kf.subtract(Kfu @ self.beta @ Kfu.T),
        )

    def posterior_u(self) -> gpytorch.distributions.MultivariateNormal:
        Ku = self.kernel(self.kernel.inducing_points)
        Ku = Ku.evaluate()
        return gpytorch.distributions.MultivariateNormal(
            (Ku @ self.gamma).view(-1),
            Ku.subtract(Ku @ self.beta @ Ku),
        )

    def iterate_pep(self):
        """Run one step of PEP sequentially for all training points"""
        with torch.no_grad():
            Ku = self.kernel(self.kernel.inducing_points)
            chol_Ku = Ku.cholesky().evaluate()
            Kf = self.kernel(self.train_x).diag()
            Kfu = self.kernel(self.train_x, self.kernel.inducing_points).evaluate()
            KuKuf = torch.cholesky_solve(Kfu.T, chol_Ku)

            for i in range(len(self.train_x)):
                vi_updated, gi_updated, gamma_updated, beta_updated = self.single_step(
                    self.train_y[i], Kf[i], Kfu[[i]], KuKuf[:, [i]],
                    self.v[i], self.g[i]
                )

                self.v[i] = vi_updated
                self.g[i] = gi_updated
                self.gamma = gamma_updated
                self.beta = beta_updated

    #                     ()  ()   (1,m)  (m,1)   (1) (1)
    def single_step(self, yi, Kfi, Kfi_u, KuKufi, vi, gi):
        """Computer a single step of PEP."""

        #        (m)          (m,m) (m)
        # h_si  = p_i - np.dot(beta, k_i) # (m) in the original implementation
        #       = Ku.inv() @ Vfi_u.T
        #       = Ku.inv() @ (Kfi_u.T - Ku @ self.beta @ Kfi_u.T)
        #       = (Ku.inv() @ Kfi_u.T) - self.beta @ Kfi_u.T
        KuiVufi = (KuKufi) - self.beta @ Kfi_u.T  # (m,1)
        # or...
        # KuiVufi = torch.cholesky_solve(Vfi_u.T,chol_Ku)
        # where ...
        # Vfi_u = Kfi_u - Kfi_u @ self.beta @ Ku

        # Compute d_2_tilde
        d2_tilde = 1 / (
            #     (1)
                vi / self.alpha  # (1)
                #    (1,m)    (m,m)      (m,1)    (1,m)   (m,m)            (m,1)
                #    Kfi_u @ Ku.inv() @ Kfi_u.T - Kfi_u @ self.beta @ Kfi_u.T
                #     (1,m)   (m,1)
                - (Kfi_u @ KuiVufi)  # (1,1)
        )

        # Compute d_1_tilde
        #           (1,m)   (m,1)        (1)               (1,1)
        d1_tilde = (Kfi_u @ self.gamma - gi) * d2_tilde  # (1,1)

        ## Deletion step
        #                (m,1)             (m,m)      (m,1)     (1,1)
        #                self.gamma + Ku.inv() @ Vfi_u.T @ d1_tilde
        gamma_delete_i = self.gamma + (KuiVufi) @ d1_tilde  # (m,1)

        #               (m,m)            (m,m)      (m,1)     (1,1)       (1,m)   (m,m)
        #               self.beta - Ku.inv() @ Vfi_u.T @ d2_tilde @ Vfi_u @ Ku.inv()
        beta_delete_i = self.beta - (KuiVufi) @ d2_tilde @ (KuiVufi).T  # (m,m)
        ##

        # `h` in the original implementation
        #                 Ku.inv() @ Vfi_u_delete_i.T
        #                 Ku.inv() @ (Kfi_u.T - Ku @ beta_delete_i @ Kfi_u.T)
        #                 (Ku.inv() @ Kfi_u.T) - beta_delete_i @ Kfi_u.T
        KuVufi_delete_i = (KuKufi) - beta_delete_i @ Kfi_u.T  # (m,1)
        # or...
        # KuVufi_delete_i = torch.cholesky_solve(Vfi_u_delete_i.T,chol_Ku)
        # where...
        # Vfi_u_delete_i = Kfi_u - Kfi_u @ beta_delete_i @ Ku

        mfi_delete_i = Kfi_u @ gamma_delete_i  # (1,1)
        Vfi_delete_i = Kfi - Kfi_u @ beta_delete_i @ Kfi_u.T  # (1,1)

        # Zi = <p(y[i]|f[i])**alpha>_(q_delete_i(f))
        #    = <p(y[i]|f[i])**alpha>_(N(f[i]|mfi_delete_i, Vfi_delete_i))
        # d1 = d(log Zi)/d(m_fi_delete_i)
        # d2 = d**2(log Zi)/d(m_fi_delete_i)**2
        d1, d2 = self.compute_ds(yi, mfi_delete_i, Vfi_delete_i)

        ## Projection
        #               (m,1)            (m,1)             (1,1)
        gamma_updated = gamma_delete_i + KuVufi_delete_i @ d1  # (m,1)

        #              beta_delete_i - Ku.inv() @ Vfi_u_delete_i.T @ d2 @ Vfi_u_delete_i @ Ku.inv()
        #              (m,m)           (m,1)             (1,1)  (1,m)
        beta_updated = beta_delete_i - KuVufi_delete_i @ d2 @ KuVufi_delete_i.T
        beta_updated = (beta_updated + beta_updated.T) / 2

        ## Update
        #          -d1/d2 + Kfi_u @ gamma_delete_i
        gi_new = -d1 / d2 + (mfi_delete_i)

        #        -1 / d2 - Vfi_u_delete_i @ Vu.inv() @ Vfi_u_delete_i.T
        vi_new = -1 / d2 - Kfi_u @ KuVufi_delete_i

        vi_updated = 1 / (1 / vi_new + (1 - self.alpha) / vi)
        gi_updated = vi_updated * (gi_new / vi_new + (1 - self.alpha) * gi / vi)

        return (
            vi_updated, gi_updated,
            gamma_updated, beta_updated
        )

    #                    ()  (1,1)         (1,1)
    def compute_ds(self, yi, mfi_delete_i, Vfi_delete_i):
        """Computes the derivatives of $\\log \\tilde{Z}_i$"""
        if isinstance(self.likelihood, gpytorch.likelihoods.GaussianLikelihood):
            # Zi = <p(y[i]|f[i])**alpha>_(q_delete_i(f))
            #    = <N(y[i]|f[i], sigma2)**alpha>_(N(f[i]|mfi_delete_i, Vfi_delete_i))
            #    = trash * <N(y[i]|f[i], sigma2/alpha)>_(N(f[i]|mfi_delete_i, Vfi_delete_i))
            #    = trash * N(y[i]|mfi_delete_i, sigma2/alpha + Vfi_delete_i)

            d2 = -1 / (self.likelihood.noise / self.alpha + Vfi_delete_i)  # (1,1)
            d1 = (mfi_delete_i - yi) * d2  # (1,1)
            return d1, d2
        else:
            mean = mfi_delete_i.detach().clone()
            mean.requires_grad = True
            if self.alpha != 1:
                quad_Zi = self.quadrature(
                    lambda f: self.likelihood(f).log_prob(yi.view(-1, 1)).mul(self.alpha).exp(),
                    torch.distributions.Normal(mean, Vfi_delete_i.sqrt())
                ).log()
            else:
                # If alpha is 1, we can use GPyTorch log marginals.
                quad_Zi = self.likelihood.log_marginal(
                    yi.view(-1, 1), gpytorch.distributions.MultivariateNormal(mean, Vfi_delete_i)
                )

            quad_d1 = torch.autograd.grad(quad_Zi, mean, create_graph=True)[0]
            quad_d2 = torch.autograd.grad(quad_d1, mean)[0]
            return quad_d1, quad_d2

    def exact_q_u(self):
        """Compute the exact $q(u)$ in the Gaussian likelihood case"""
        assert isinstance(self.likelihood, gpytorch.likelihoods.GaussianLikelihood)
        with torch.no_grad():
            Kf = self.kernel(self.train_x).evaluate()
            Ku = self.kernel(self.kernel.inducing_points)
            chol_Ku = Ku.cholesky().evaluate()
            Ku = Ku.evaluate()
            Kfu = self.kernel(self.train_x, self.kernel.inducing_points).evaluate()
            KuKuf = torch.cholesky_solve(Kfu.T, chol_Ku)

            Qf = Kfu @ KuKuf
            Df = Kf - Qf

            Kf_bar = Qf + (self.alpha * Df.diag() + self.likelihood.noise) * torch.eye(self.n)
            chol_Kf_bar = torch.cholesky(Kf_bar)
            return SimpleNamespace(
                loc=(Kfu.T @ torch.cholesky_solve(self.train_y.view(-1, 1), chol_Kf_bar)).view(-1),
                covariance_matrix=Ku - Kfu.T @ torch.cholesky_solve(Kfu, chol_Kf_bar),
            )

    def exact_t_u(self):
        """Compute the exact $t_i(u)$ in the Gaussian likelihood case"""
        assert isinstance(self.likelihood, gpytorch.likelihoods.GaussianLikelihood)
        with torch.no_grad():
            Kf = self.kernel(self.train_x).evaluate()
            Kfu = self.kernel(self.train_x, self.kernel.inducing_points).evaluate()
            Ku = self.kernel(self.kernel.inducing_points)
            chol_Ku = Ku.cholesky().evaluate()
            KuKuf = torch.cholesky_solve(Kfu.T, chol_Ku)

            Qf = Kfu @ KuKuf
            Df = Kf - Qf

            return SimpleNamespace(
                g=self.train_y.view(-1, 1),
                v=(self.alpha * Df.diag() + self.likelihood.noise).view(-1, 1)
            )
