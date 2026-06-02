"""Tests for natural-gradient exponential-family distribution variants.

These validate the distributions in ``genjax.natural``:
- Their log-density *value* matches the underlying TFP distribution.
- Differentiating the log-density w.r.t. the natural parameters ``eta`` returns
  the *natural* gradient ``F(eta)^{-1} (T(x) - mu) = grad_mu log p`` -- checked
  against closed forms and against the mean-parameter duality.
- They integrate with the GFI (``assess``/``simulate``) and yield a working
  natural-gradient REINFORCE estimator.
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
import pytest
import tensorflow_probability.substrates.jax as tfp

from genjax.pjax import seed
from genjax.natural import (
    bernoulli_natural,
    flip_natural,
    poisson_natural,
    geometric_natural,
    exponential_natural,
    normal_natural,
    beta_natural,
    gamma_natural,
    bernoulli_natural_reinforce,
    poisson_natural_reinforce,
)

tfd = tfp.distributions


# ---------------------------------------------------------------------------
# Log-density value matches the base TFP distribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "nat_dist, nat_args, base_dist, x",
    [
        (bernoulli_natural, (0.7,), tfd.Bernoulli(logits=0.7), 1),
        (
            poisson_natural,
            (jnp.log(2.0),),
            tfd.Poisson(rate=2.0),
            3,
        ),
        (
            exponential_natural,
            (-2.0,),
            tfd.Exponential(rate=2.0),
            1.5,
        ),
        (
            normal_natural,
            (0.5 / 4.0, -1.0 / (2 * 4.0)),
            tfd.Normal(0.5, 2.0),
            0.4,
        ),
        (
            gamma_natural,
            (3.0 - 1.0, -2.0),
            tfd.Gamma(concentration=3.0, rate=2.0),
            1.3,
        ),
        (
            beta_natural,
            (2.0 - 1.0, 5.0 - 1.0),
            tfd.Beta(2.0, 5.0),
            0.4,
        ),
    ],
)
def test_logpdf_value_matches_tfp(nat_dist, nat_args, base_dist, x):
    got = nat_dist.logpdf(x, *nat_args)
    want = base_dist.log_prob(x)
    assert jnp.allclose(got, want, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Natural gradient against closed forms (1-parameter, T(x) = x)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x", [0, 1])
def test_bernoulli_natural_gradient_closed_form(x):
    eta = 0.7
    p = jax.nn.sigmoid(eta)
    g = jax.grad(lambda e: bernoulli_natural.logpdf(x, e))(eta)
    assert jnp.allclose(g, (x - p) / (p * (1 - p)), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("x", [0, 1, 5])
def test_poisson_natural_gradient_closed_form(x):
    log_lam = jnp.log(2.0)
    lam = 2.0
    g = jax.grad(lambda e: poisson_natural.logpdf(x, e))(log_lam)
    assert jnp.allclose(g, (x - lam) / lam, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("x", [0.2, 1.0, 3.0])
def test_exponential_natural_gradient_closed_form(x):
    # eta = -rate; score_eta = x - 1/rate, F = 1/rate^2, nat = rate^2 (x - 1/rate)
    rate = 2.0
    eta = -rate
    g = jax.grad(lambda e: exponential_natural.logpdf(x, e))(eta)
    assert jnp.allclose(g, rate**2 * (x - 1.0 / rate), atol=1e-3, rtol=1e-4)


# ---------------------------------------------------------------------------
# Mean-parameter duality: natgrad_eta == grad_mu
# ---------------------------------------------------------------------------


def test_bernoulli_mean_parameter_duality():
    eta = 0.7
    p = jax.nn.sigmoid(eta)
    g_eta = jax.grad(lambda e: bernoulli_natural.logpdf(1, e))(eta)
    g_mu = jax.grad(lambda pp: tfd.Bernoulli(probs=pp).log_prob(1))(p)
    assert jnp.allclose(g_eta, g_mu, atol=1e-4, rtol=1e-4)


def test_normal_mean_parameter_duality():
    # Mean params mu = (E[x], E[x^2]) = (loc, loc^2 + sigma^2).
    loc, sigma = 0.5, 2.0
    eta1, eta2 = loc / sigma**2, -1.0 / (2 * sigma**2)
    x = 0.4

    def logpdf_meanparams(m1, m2):
        var = m2 - m1**2
        return tfd.Normal(m1, jnp.sqrt(var)).log_prob(x)

    g_mu = jax.grad(lambda m: logpdf_meanparams(m[0], m[1]))(
        jnp.array([loc, loc**2 + sigma**2])
    )
    g_eta = jax.grad(lambda e: normal_natural.logpdf(x, e[0], e[1]))(
        jnp.array([eta1, eta2])
    )
    assert jnp.allclose(g_eta, g_mu, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Batched / vmapped gradients
# ---------------------------------------------------------------------------


def test_bernoulli_natural_gradient_vmapped():
    etas = jnp.array([0.1, 0.7, -0.5])
    xs = jnp.array([1, 0, 1])
    g = jax.vmap(
        lambda x, e: jax.grad(lambda ee: bernoulli_natural.logpdf(x, ee))(e)
    )(xs, etas)
    ps = jax.nn.sigmoid(etas)
    assert jnp.allclose(g, (xs - ps) / (ps * (1 - ps)), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# GFI integration
# ---------------------------------------------------------------------------


def test_assess_returns_natural_gradient():
    eta = 0.7
    p = jax.nn.sigmoid(eta)
    g = jax.grad(lambda e: bernoulli_natural.assess(1, e)[0])(eta)
    assert jnp.allclose(g, (1 - p) / (p * (1 - p)), atol=1e-4, rtol=1e-4)


def test_simulate_under_seed():
    key = jrand.PRNGKey(0)
    tr = seed(bernoulli_natural.simulate)(key, 0.7)
    x = tr.get_retval()
    assert x in (0, 1)
    # Score is -logpdf(x); check consistency with the density value.
    assert jnp.allclose(tr.get_score(), -bernoulli_natural.logpdf(x, 0.7), atol=1e-5)


def test_flip_natural_is_boolean():
    tr = seed(flip_natural.simulate)(jrand.PRNGKey(1), 0.3)
    assert tr.get_retval().dtype == jnp.bool_


def test_geometric_natural_value_matches_tfp():
    eta = jnp.log(1.0 - 0.4)  # p = 0.4
    got = geometric_natural.logpdf(3, eta)
    want = tfd.Geometric(probs=0.4).log_prob(3)
    assert jnp.allclose(got, want, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Natural-gradient REINFORCE
# ---------------------------------------------------------------------------


def test_bernoulli_natural_reinforce_estimates_natural_gradient():
    from genjax.adev import expectation

    eta = 0.7

    @expectation
    def objective(e):
        return bernoulli_natural_reinforce(e) * 1.0

    keys = jrand.split(jrand.PRNGKey(1), 100000)
    grads = jax.vmap(lambda k: seed(objective.grad_estimate)(k, eta))(keys)
    mc = jnp.mean(grads)
    # dE[x]/d eta in natural coords = F^{-1} * p(1-p) = 1 (since F = p(1-p)).
    # Loose tolerance: this is a Monte-Carlo sanity check (the ordinary gradient
    # would be p(1-p) ~ 0.22), robust to RNG ordering across the full suite.
    assert jnp.allclose(mc, 1.0, atol=0.1)


def test_poisson_natural_reinforce_uses_natural_gradient_score():
    # The differentiable log-density reports the natural gradient, which is what
    # the score-function estimator multiplies by f(x).
    log_lam = jnp.log(2.0)
    lam = 2.0
    g = jax.grad(lambda e: poisson_natural_reinforce.logpdf(5, e))(log_lam)
    assert jnp.allclose(g, (5 - lam) / lam, atol=1e-4, rtol=1e-4)


def test_poisson_natural_reinforce_estimates_natural_gradient():
    from genjax.adev import expectation

    eta = jnp.log(2.0)  # lambda = 2

    @expectation
    def objective(e):
        return poisson_natural_reinforce(e) * 1.0

    keys = jrand.split(jrand.PRNGKey(1), 100000)
    grads = jax.vmap(lambda k: seed(objective.grad_estimate)(k, eta))(keys)
    mc = jnp.mean(grads)
    # dE[x]/d eta in natural coords = F^{-1} * lambda = 1 (since F = lambda).
    # Loose tolerance: Monte-Carlo sanity check (ordinary gradient would be
    # lambda = 2), robust to RNG ordering across the full suite.
    assert jnp.allclose(mc, 1.0, atol=0.1)
