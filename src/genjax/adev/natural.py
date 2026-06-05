"""Natural-gradient variants of minimal exponential-family distributions.

Each distribution exported here is parameterized by its *canonical natural
parameters* ``eta``. The log-density carries a custom JVP so that
differentiating it with respect to ``eta`` returns the **natural gradient** --
that is, the ordinary gradient with respect to the *mean (expectation)*
parameters ``mu``:

    F(eta)^{-1} grad_eta log p(x; eta) = grad_mu log p(x; eta).

This is the core building block for quasi-Black-Box Variational Inference
(qBBVI): feeding one of these distributions into a score-function / REINFORCE
estimator turns it into a *natural-gradient* REINFORCE estimator at no extra
modelling cost.

Verified exponential-family implementations are provided by `efax
<https://github.com/NeilGirdhar/efax>`_: we use its ``NaturalParametrization``
classes for the (verified) log-density and sampler, and obtain the natural
gradient by autodiff.

Why this works
--------------
For a minimal exponential family written in canonical coordinates,

    log p(x; eta) = <eta, T(x)> - A(eta) + log h(x),

the log-density is *linear* in the sufficient statistic ``T(x)``. Hence

    grad_eta log p = T(x) - mu                  (the score),
    -hess_eta log p = grad^2 A(eta) = F(eta)    (the Fisher information),

and crucially ``-hess_eta log p`` is *exact* and *independent of x*. The natural
score is therefore obtained from autodiff alone:

    natural_score = solve(-hess_eta log p, grad_eta log p) = grad_mu log p.

No per-distribution Fisher algebra is required -- only the efax natural-parameter
class. (efax can also report ``mu`` via ``to_exp()``, but for families such as
Beta that conversion is iterative; differentiating the closed-form log-density
is uniform and avoids the inner solve.)

efax's canonical natural coordinates match the conventions documented on each
distribution below. Because the natural gradient is invariant under invertible
*affine* reparameterizations of ``eta``, a couple of families are exposed in an
affine-equivalent chart that is friendlier to read (Gamma, Beta); the
``-hess = Fisher`` identity continues to hold in any such chart.
"""

import jax
import jax.numpy as jnp

from genjax._compat import ensure_jax_tfp_compat

# efax imports TensorFlow Probability at module load; make sure the GenJAX
# TFP/JAX compatibility shim runs first.
ensure_jax_tfp_compat()
from efax import (  # noqa: E402
    BernoulliNP,
    BetaNP,
    ExponentialNP,
    GammaNP,
    GeometricNP,
    NormalNP,
    PoissonNP,
)

from genjax.core import distribution, Pytree  # noqa: E402
from genjax.pjax import wrap_logpdf, wrap_sampler  # noqa: E402
from genjax.adev import (  # noqa: E402
    reinforce,
    ADEVPrimitive,
    Dual,
    _discrete_zero_tangent,
    _first_leaf,
    _mvd_phantom_estimate,
    _mvd_lane_estimate,
    _flip_lane_rb_estimate,
)


def natural_gradient_logpdf(base_logpdf, n_params):
    """Wrap a natural-parameter log-density so its JVP reports the natural gradient.

    Args:
        base_logpdf: Callable ``(x, *eta) -> log_prob`` giving the true
            log-density as a function of the ``n_params`` canonical natural
            parameters ``eta`` (each a scalar or batched array). Must be twice
            differentiable with respect to ``eta``.
        n_params: Number of natural parameters (1 for univariate-parameter
            families like Bernoulli/Poisson; 2 for Normal/Gamma/Beta).

    Returns:
        A function with the same value as ``base_logpdf`` but whose custom JVP
        reports, for the ``eta`` arguments, the tangent
        ``<F(eta)^{-1} (T(x) - mu), eta_dot> = <grad_mu log p, eta_dot>``.
        The data argument ``x`` keeps its ordinary derivative for continuous
        supports and is treated as a constant for discrete (integer/boolean)
        supports.
    """

    @jax.custom_jvp
    def logpdf(x, *eta):
        return base_logpdf(x, *eta)

    @logpdf.defjvp
    def _logpdf_jvp(primals, tangents):
        x, *eta = primals
        x_dot, *eta_dot = tangents
        primal_out = base_logpdf(x, *eta)

        # Broadcast the data and all natural parameters to a common batch shape
        # so we can compute a per-example natural score and vmap over the batch.
        bshape = jnp.broadcast_shapes(
            jnp.shape(x), *(jnp.shape(e) for e in eta)
        )
        x_b = jnp.broadcast_to(x, bshape)
        eta_b = [jnp.broadcast_to(e, bshape) for e in eta]

        k = n_params
        eta_stacked = jnp.stack(eta_b, axis=-1)  # bshape + (k,)
        eta_flat = eta_stacked.reshape((-1, k))
        x_flat = x_b.reshape((-1,))

        def per_example(x_e, eta_vec):
            def lp(e):
                return base_logpdf(x_e, *(e[i] for i in range(k)))

            score = jax.grad(lp)(eta_vec)  # T(x) - mu
            fisher = -jax.hessian(lp)(eta_vec)  # grad^2 A(eta), exact
            return jnp.linalg.solve(fisher, score)  # natural score

        nat_flat = jax.vmap(per_example)(x_flat, eta_flat)  # (n, k)
        nat = nat_flat.reshape((*bshape, k))

        # <natural_score, eta_dot>, contracted over the k natural-parameter axes.
        eta_dot_b = [jnp.broadcast_to(ed, bshape) for ed in eta_dot]
        tangent_out = sum(nat[..., i] * eta_dot_b[i] for i in range(k))

        # Ordinary data-gradient term for continuous supports. Discrete supports
        # carry a float0 tangent for ``x`` and are treated as constants.
        if jnp.issubdtype(jnp.result_type(x), jnp.floating):
            _, x_term = jax.jvp(
                lambda xx: base_logpdf(xx, *eta), (x,), (x_dot,)
            )
            tangent_out = tangent_out + x_term

        return primal_out, tangent_out

    return logpdf


def _efax_keyful_sampler(np_ctor, cast):
    """Build a keyful sampler from an efax natural-parameter constructor."""

    def keyful_sampler(key, *eta, sample_shape=()):
        shape = sample_shape if sample_shape else None
        sample = np_ctor(*eta).sample(key, shape)
        return sample if cast is None else cast(sample)

    return keyful_sampler


def _geometric_natural_keyful(key, eta, sample_shape=()):
    """Reliable Geometric (failure-count) sampler in natural coords eta = log(1-p).

    efax 1.20's ``GeometricNP.sample`` is degenerate for ``p >= 0.5`` (it returns
    a constant), so we sample by inverse-CDF instead:
    ``X = floor(log U / log(1-p))`` with ``U ~ Uniform(0, 1]`` yields
    ``P(X=k) = (1-p)^k p`` (success probability ``p``). Here ``1-p = exp(eta)``,
    so ``log(1-p) = eta`` directly.
    """
    shape = sample_shape if sample_shape else jnp.shape(eta)
    # U in (0, 1]: 1 - Uniform[0, 1) avoids log(0).
    u = 1.0 - jax.random.uniform(key, shape=shape)
    return jnp.floor(jnp.log(u) / eta)


def _natural_distribution(np_ctor, n_params, name, cast=None, keyful_sampler=None):
    """Build a Distribution in natural coordinates with a natural-gradient logpdf.

    Args:
        np_ctor: Callable ``(*eta) -> efax.NaturalParametrization`` mapping the
            canonical natural parameters to an efax distribution (used for the
            density *value* and, by default, sampling).
        n_params: Number of (scalar) natural parameters.
        name: Display name.
        cast: Optional post-processing applied to samples (e.g. dtype cast).
        keyful_sampler: Optional keyful sampler override (used where efax's
            sampler is unreliable, e.g. Geometric).
    """

    def raw_logpdf(x, *eta):
        return np_ctor(*eta).log_pdf(x)

    nat_logpdf = natural_gradient_logpdf(raw_logpdf, n_params)
    sampler = keyful_sampler or _efax_keyful_sampler(np_ctor, cast)

    return distribution(
        wrap_sampler(sampler, name=name),
        wrap_logpdf(nat_logpdf, name=name),
        name=name,
    )


def _natural_reinforce(nat_dist, np_ctor, cast=None, keyful_sampler=None):
    """Build a REINFORCE estimator that uses the natural-gradient log-density.

    Because the score function in the REINFORCE identity is computed from
    ``nat_dist.logpdf`` -- whose JVP reports the natural gradient -- the
    resulting estimator is a *natural-gradient* REINFORCE estimator.
    """
    return distribution(
        reinforce(
            nat_dist.sample,
            nat_dist.logpdf,
            keyful_sampler or _efax_keyful_sampler(np_ctor, cast),
        ),
        nat_dist.logpdf,
    )


# ----------------------------------------------------------------------------
# Univariate-parameter families (single natural parameter)
# ----------------------------------------------------------------------------


def _bernoulli_ctor(eta):
    return BernoulliNP(eta)


def _to_int(sample):
    return sample.astype(jnp.int32)


bernoulli_natural = _natural_distribution(
    _bernoulli_ctor,
    1,
    name="BernoulliNatural",
    cast=_to_int,
)
"""Bernoulli (efax ``BernoulliNP``) in its natural parameter
``eta = log_odds = logit(p)``.

Sufficient statistic ``T(x) = x``, mean parameter ``mu = p = sigmoid(eta)``,
Fisher information ``F = p(1 - p)``. The reported gradient w.r.t. ``eta`` is the
natural gradient ``(x - p) / (p(1 - p))`` (= grad of the log-pdf w.r.t. ``p``).
"""

bernoulli_natural_reinforce = _natural_reinforce(
    bernoulli_natural, _bernoulli_ctor, cast=_to_int
)
"""Natural-gradient REINFORCE estimator for ``bernoulli_natural``."""


flip_natural = _natural_distribution(
    _bernoulli_ctor,
    1,
    name="FlipNatural",
)
"""Boolean-valued Bernoulli (efax ``BernoulliNP``, native boolean samples) in
natural parameter ``eta = logit(p)`` (see ``bernoulli_natural``)."""

flip_natural_reinforce = _natural_reinforce(flip_natural, _bernoulli_ctor)
"""Natural-gradient REINFORCE estimator for ``flip_natural``."""


def _poisson_ctor(eta):
    return PoissonNP(eta)


poisson_natural = _natural_distribution(
    _poisson_ctor,
    1,
    name="PoissonNatural",
)
"""Poisson (efax ``PoissonNP``) in its natural parameter
``eta = log_mean = log(lambda)``.

``T(x) = x``, ``mu = lambda = exp(eta)``, ``F = lambda``. The natural gradient
w.r.t. ``eta`` is ``(x - lambda) / lambda``.
"""

poisson_natural_reinforce = _natural_reinforce(poisson_natural, _poisson_ctor)
"""Natural-gradient REINFORCE estimator for ``poisson_natural``."""


def _geometric_ctor(eta):
    return GeometricNP(eta)


geometric_natural = _natural_distribution(
    _geometric_ctor,
    1,
    name="GeometricNatural",
    keyful_sampler=_geometric_natural_keyful,
)
"""Geometric (efax ``GeometricNP``; number of failures, support {0, 1, 2, ...})
in its natural parameter ``eta = log_not_p = log(1 - p)`` with ``eta < 0``.

``T(x) = x``, ``mu = (1 - p) / p``, ``F = (1 - p) / p^2``. Sampling uses an
inverse-CDF kernel (efax 1.20's geometric sampler is degenerate for p >= 0.5);
the density still comes from efax.
"""

geometric_natural_reinforce = _natural_reinforce(
    geometric_natural, _geometric_ctor, keyful_sampler=_geometric_natural_keyful
)
"""Natural-gradient REINFORCE estimator for ``geometric_natural``."""


def _exponential_ctor(eta):
    return ExponentialNP(eta)


exponential_natural = _natural_distribution(
    _exponential_ctor,
    1,
    name="ExponentialNatural",
)
"""Exponential (efax ``ExponentialNP``) in its natural parameter
``eta = negative_rate = -rate`` with ``eta < 0``.

``T(x) = x``, ``mu = 1 / rate``, ``F = 1 / rate^2``.
"""

exponential_natural_reinforce = _natural_reinforce(
    exponential_natural, _exponential_ctor
)
"""Natural-gradient REINFORCE estimator for ``exponential_natural``."""

# ----------------------------------------------------------------------------
# Two-parameter families
# ----------------------------------------------------------------------------


def _normal_ctor(eta1, eta2):
    # eta = (mean_times_precision, negative_half_precision)
    #     = (loc / scale^2, -1 / (2 scale^2)),  eta2 < 0.
    return NormalNP(eta1, eta2)


normal_natural = _natural_distribution(
    _normal_ctor,
    2,
    name="NormalNatural",
)
"""Normal (efax ``NormalNP``) in its canonical natural parameters
``eta = (loc / scale^2, -1 / (2 scale^2))`` with ``eta[1] < 0``.

Sufficient statistic ``T(x) = (x, x^2)``; mean parameters
``mu = (E[x], E[x^2]) = (loc, loc^2 + scale^2)``. The reported gradient w.r.t.
``eta`` is the natural gradient (= grad of the log-pdf w.r.t. ``mu``).
"""

normal_natural_reinforce = _natural_reinforce(normal_natural, _normal_ctor)
"""Natural-gradient REINFORCE estimator for ``normal_natural``."""


def _beta_ctor(eta1, eta2):
    # efax BetaNP holds alpha_minus_one = [concentration1 - 1, concentration0 - 1].
    return BetaNP(jnp.stack([eta1, eta2], axis=-1))


beta_natural = _natural_distribution(
    _beta_ctor,
    2,
    name="BetaNatural",
)
"""Beta (efax ``BetaNP``) in natural coordinates
``eta = (concentration1 - 1, concentration0 - 1)`` (affine to the canonical
chart).

Sufficient statistic ``T(x) = (log x, log(1 - x))``.
"""

beta_natural_reinforce = _natural_reinforce(beta_natural, _beta_ctor)
"""Natural-gradient REINFORCE estimator for ``beta_natural``."""


def _gamma_ctor(eta1, eta2):
    # Our chart: eta = (shape - 1, -rate). efax GammaNP(negative_rate, shape_minus_one).
    return GammaNP(negative_rate=eta2, shape_minus_one=eta1)


gamma_natural = _natural_distribution(
    _gamma_ctor,
    2,
    name="GammaNatural",
)
"""Gamma (efax ``GammaNP``) in natural coordinates ``eta = (shape - 1, -rate)``
(affine to the canonical chart) with ``eta[1] < 0``, i.e.
``concentration = eta[0] + 1``, ``rate = -eta[1]``.

Sufficient statistic ``T(x) = (log x, x)``.
"""

gamma_natural_reinforce = _natural_reinforce(gamma_natural, _gamma_ctor)
"""Natural-gradient REINFORCE estimator for ``gamma_natural``."""


# ----------------------------------------------------------------------------
# Measure-valued derivative (MVD) estimators in natural coordinates.
#
# These return the *natural gradient* w.r.t. the natural parameter eta -- i.e.
# the ordinary gradient w.r.t. the mean parameter mu, F(eta)^{-1} d/deta = d/dmu
# -- so they are drop-in natural-gradient siblings of the ``*_natural_reinforce``
# estimators. Sampling uses the efax-backed natural Distributions (correct under
# ``seed``). The mean-parameter MVD decompositions are:
#
#   Bernoulli (mu = p)        : d/dmu E[f] = f(1) - f(0)              (exact)
#   Poisson   (mu = rate)     : d/dmu E[f] = E[f(X+1)] - E[f(X)]
#   Geometric (mu = E[X])     : d/dmu E[f] = p (f(G1+G2+1) - f(X))    (c = p)
#
# (For Bernoulli/Poisson the natural gradient coincides with the typical-param
# gradient, since there the typical parameter already equals the mean parameter;
# Geometric differs.)
# ----------------------------------------------------------------------------


@Pytree.dataclass
class BernoulliNaturalMVD(ADEVPrimitive):
    """Natural-gradient MVD for Bernoulli in natural coordinate ``eta = logit(p)``.

    Since the mean parameter is ``mu = p``, the natural gradient is the exact
    Bernoulli MVD ``f(1) - f(0)``.
    """

    def sample(self, *args):
        (eta,) = args
        return bernoulli_natural.sample(eta)

    def sample_with_key(self, key, *args, sample_shape=()):
        (eta,) = args
        return _efax_keyful_sampler(_bernoulli_ctor, _to_int)(
            key, eta, sample_shape=sample_shape
        )

    def prim_jvp_estimate(self, dual_tree, konts):
        (kpure, kdual) = konts
        (eta_primal,) = Dual.tree_primal(dual_tree)
        (eta_tangent,) = Dual.tree_tangent(dual_tree)

        # Natural gradient = f(1) - f(0); contract directly with eta_tangent
        # (no p(1-p) factor -- that is exactly the F^{-1} preconditioning).
        if jnp.ndim(eta_primal) > 0:
            p = jax.nn.sigmoid(eta_primal)
            return _flip_lane_rb_estimate(kpure, kdual, p, eta_tangent)

        b = bernoulli_natural.sample(eta_primal)
        b_dual = kdual(Dual(b, _discrete_zero_tangent(b)))
        (b_primal,), (b_tangent,) = Dual.tree_unzip(b_dual)
        other = _first_leaf(kpure(1 - b))
        sign = jnp.where(b > 0, -1.0, 1.0).astype(b_primal.dtype)
        diff = sign * (other - b_primal)  # f(1) - f(0)
        return Dual(b_primal, b_tangent + diff * eta_tangent)


bernoulli_natural_mvd = distribution(
    BernoulliNaturalMVD(),
    bernoulli_natural.logpdf,
    name="BernoulliNaturalMVD",
)
"""Natural-gradient MVD estimator for ``bernoulli_natural`` (eta = logit(p)).

Wrapped as a :class:`~genjax.core.Distribution` so it can be addressed inside a
generative function (``bernoulli_natural_mvd(eta) @ "z"``) as well as sampled
directly inside an ``@expectation`` program."""


@Pytree.dataclass
class FlipNaturalMVD(ADEVPrimitive):
    """Boolean-valued natural-gradient MVD for Bernoulli (eta = logit(p))."""

    def sample(self, *args):
        (eta,) = args
        return flip_natural.sample(eta)

    def sample_with_key(self, key, *args, sample_shape=()):
        (eta,) = args
        return _efax_keyful_sampler(_bernoulli_ctor, None)(
            key, eta, sample_shape=sample_shape
        )

    def prim_jvp_estimate(self, dual_tree, konts):
        (kpure, kdual) = konts
        (eta_primal,) = Dual.tree_primal(dual_tree)
        (eta_tangent,) = Dual.tree_tangent(dual_tree)

        if jnp.ndim(eta_primal) > 0:
            p = jax.nn.sigmoid(eta_primal)
            return _flip_lane_rb_estimate(kpure, kdual, p, eta_tangent)

        b = flip_natural.sample(eta_primal)
        b_dual = kdual(Dual(b, _discrete_zero_tangent(b)))
        (b_primal,), (b_tangent,) = Dual.tree_unzip(b_dual)
        other = _first_leaf(kpure(jnp.logical_not(b)))
        sign = jnp.where(b, -1.0, 1.0).astype(b_primal.dtype)
        diff = sign * (other - b_primal)  # f(1) - f(0)
        return Dual(b_primal, b_tangent + diff * eta_tangent)


flip_natural_mvd = distribution(
    FlipNaturalMVD(),
    flip_natural.logpdf,
    name="FlipNaturalMVD",
)
"""Boolean-valued natural-gradient MVD estimator for ``flip_natural``.

Wrapped as a :class:`~genjax.core.Distribution` (addressable via ``@``)."""


@Pytree.dataclass
class PoissonNaturalMVD(ADEVPrimitive):
    """Natural-gradient MVD for Poisson in natural coordinate ``eta = log(rate)``.

    Since ``mu = rate``, the natural gradient is the coupled Poisson MVD
    ``f(X+1) - f(X)`` with ``X ~ Poisson(rate)``.
    """

    def sample(self, *args):
        (eta,) = args
        return poisson_natural.sample(eta)

    def sample_with_key(self, key, *args, sample_shape=()):
        (eta,) = args
        return _efax_keyful_sampler(_poisson_ctor, None)(
            key, eta, sample_shape=sample_shape
        )

    def prim_jvp_estimate(self, dual_tree, konts):
        (eta_primal,) = Dual.tree_primal(dual_tree)
        (eta_tangent,) = Dual.tree_tangent(dual_tree)

        x = poisson_natural.sample(eta_primal)
        if jnp.ndim(eta_primal) > 0:
            (_, kdual) = konts
            return _mvd_lane_estimate(
                kdual,
                x,
                lambda i, x_flat: x_flat.at[i].add(1),
                1.0,
                eta_tangent,
                primal_is_positive=False,
            )
        return _mvd_phantom_estimate(
            konts, x, x + 1, 1.0, eta_tangent, primal_is_positive=False
        )


poisson_natural_mvd = distribution(
    PoissonNaturalMVD(),
    poisson_natural.logpdf,
    name="PoissonNaturalMVD",
)
"""Natural-gradient MVD estimator for ``poisson_natural`` (eta = log(rate)).

Wrapped as a :class:`~genjax.core.Distribution` (addressable via ``@``)."""


@Pytree.dataclass
class GeometricNaturalMVD(ADEVPrimitive):
    """Natural-gradient MVD for Geometric in natural coordinate ``eta = log(1-p)``.

    The mean parameter is ``mu = E[X] = (1-p)/p``; the natural gradient is
    ``p (f(G1+G2+1) - f(X))`` with ``X, G1, G2 ~ Geometric(p)`` and
    ``p = 1 - exp(eta)``.
    """

    def sample(self, *args):
        (eta,) = args
        return geometric_natural.sample(eta)

    def sample_with_key(self, key, *args, sample_shape=()):
        (eta,) = args
        return _geometric_natural_keyful(key, eta, sample_shape=sample_shape)

    def prim_jvp_estimate(self, dual_tree, konts):
        (eta_primal,) = Dual.tree_primal(dual_tree)
        (eta_tangent,) = Dual.tree_tangent(dual_tree)

        x = geometric_natural.sample(eta_primal)
        g1 = geometric_natural.sample(eta_primal)
        g2 = geometric_natural.sample(eta_primal)
        phantom = g1 + g2 + 1.0
        p = 1.0 - jnp.exp(eta_primal)  # eta = log(1 - p)

        if jnp.ndim(eta_primal) > 0:
            (_, kdual) = konts
            phantom_flat = jnp.reshape(phantom, (-1,))
            return _mvd_lane_estimate(
                kdual,
                x,
                lambda i, x_flat: x_flat.at[i].set(phantom_flat[i]),
                p,
                eta_tangent,
                primal_is_positive=False,
            )
        # mu-space positive component is the size-biased draw G1+G2+1.
        return _mvd_phantom_estimate(
            konts, x, phantom, p, eta_tangent, primal_is_positive=False
        )


geometric_natural_mvd = distribution(
    GeometricNaturalMVD(),
    geometric_natural.logpdf,
    name="GeometricNaturalMVD",
)
"""Natural-gradient MVD estimator for ``geometric_natural`` (eta = log(1-p)).

Wrapped as a :class:`~genjax.core.Distribution` (addressable via ``@``)."""


__all__ = [
    "natural_gradient_logpdf",
    "bernoulli_natural",
    "flip_natural",
    "poisson_natural",
    "geometric_natural",
    "exponential_natural",
    "normal_natural",
    "beta_natural",
    "gamma_natural",
    "bernoulli_natural_reinforce",
    "flip_natural_reinforce",
    "poisson_natural_reinforce",
    "geometric_natural_reinforce",
    "exponential_natural_reinforce",
    "normal_natural_reinforce",
    "beta_natural_reinforce",
    "gamma_natural_reinforce",
    "bernoulli_natural_mvd",
    "flip_natural_mvd",
    "poisson_natural_mvd",
    "geometric_natural_mvd",
]
