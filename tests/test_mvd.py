"""Tests for measure-valued derivative (MVD) estimators on discrete minimal
exponential families.

Two families of estimators are covered:

- Typical parameterization (matching genjax's distributions): ``bernoulli_mvd``
  and ``geometric_mvd`` are parameterized by *logits*, ``poisson_mvd`` by
  *rate*. These estimate the ordinary gradient w.r.t. that parameter.
- Natural parameterization (efax): ``*_natural_mvd`` estimate the *natural*
  gradient w.r.t. the natural parameter eta, i.e. the ordinary gradient w.r.t.
  the mean parameter mu.

Each estimator is checked by Monte-Carlo averaging ``grad_estimate`` under
``seed`` against the analytic gradient.
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
import pytest

from genjax.adev import (
    expectation,
    bernoulli_mvd,
    poisson_mvd,
    geometric_mvd,
)
from genjax.adev.natural import (
    bernoulli_natural_mvd,
    flip_natural_mvd,
    poisson_natural_mvd,
    geometric_natural_mvd,
    geometric_natural,
)
from genjax.pjax import seed


def _mc_grad(obj, theta, n=80000, key=0):
    """Monte-Carlo mean and standard error of ``obj.grad_estimate(theta)``."""
    keys = jrand.split(jrand.PRNGKey(key), n)
    grads = jax.vmap(lambda k: seed(obj.grad_estimate)(k, theta))(keys)
    return jnp.mean(grads), jnp.std(grads) / jnp.sqrt(n)


# ---------------------------------------------------------------------------
# Typical parameterization (genjax params)
# ---------------------------------------------------------------------------


def test_bernoulli_mvd_logits_gradient():
    # genjax bernoulli is logits-parameterized; p = sigmoid(logits).
    # d/dlogits E[X] = p(1-p).  (Exact MVD: per-sample estimate is deterministic.)
    logits = 0.3
    p = jax.nn.sigmoid(logits)

    @expectation
    def obj(t):
        return jnp.float32(bernoulli_mvd(t))

    mean, se = _mc_grad(obj, logits, n=2000)
    assert jnp.allclose(mean, p * (1 - p), atol=1e-4)
    assert se < 1e-5  # exact, zero variance


def test_poisson_mvd_rate_gradient_identity():
    # d/drate E[X] = 1 (exact: per-sample (X+1) - X = 1).
    @expectation
    def obj(rate):
        return jnp.float32(poisson_mvd(rate))

    mean, se = _mc_grad(obj, 2.0, n=2000)
    assert jnp.allclose(mean, 1.0, atol=1e-5)
    assert se < 1e-5


def test_poisson_mvd_rate_gradient_second_moment():
    # E[X^2] = rate + rate^2 ; d/drate = 1 + 2*rate = 5 at rate=2.
    @expectation
    def obj(rate):
        x = poisson_mvd(rate)
        return jnp.float32(x) ** 2

    mean, se = _mc_grad(obj, 2.0)
    assert jnp.abs(mean - 5.0) < 5 * se + 0.05


def test_geometric_mvd_logits_gradient():
    # genjax geometric is logits-parameterized; p = sigmoid(logits).
    # E[X] = (1-p)/p ; d/dlogits E[X] = -(1-p)/p.
    logits = 0.5
    p = jax.nn.sigmoid(logits)

    @expectation
    def obj(t):
        return jnp.float32(geometric_mvd(t))

    mean, se = _mc_grad(obj, logits)
    assert jnp.abs(mean - (-(1 - p) / p)) < 5 * se + 0.02


# ---------------------------------------------------------------------------
# Natural parameterization (efax) -- estimates the NATURAL gradient (d/dmu)
# ---------------------------------------------------------------------------


def test_bernoulli_natural_mvd_is_natural_gradient():
    # mu = p, so d/dmu E[X] = 1 (and it is exact).
    @expectation
    def obj(eta):
        return jnp.float32(bernoulli_natural_mvd(eta))

    mean, se = _mc_grad(obj, 0.3, n=2000)
    assert jnp.allclose(mean, 1.0, atol=1e-5)
    assert se < 1e-5


def test_flip_natural_mvd_is_natural_gradient():
    @expectation
    def obj(eta):
        return jnp.float32(flip_natural_mvd(eta))

    mean, se = _mc_grad(obj, -0.4, n=2000)
    assert jnp.allclose(mean, 1.0, atol=1e-5)


def test_poisson_natural_mvd_natural_gradient():
    # eta = log(rate), mu = rate ; d/dmu E[X] = 1, d/dmu E[X^2] = 1 + 2*rate.
    @expectation
    def obj_id(eta):
        return jnp.float32(poisson_natural_mvd(eta))

    @expectation
    def obj_sq(eta):
        x = poisson_natural_mvd(eta)
        return jnp.float32(x) ** 2

    eta = jnp.log(2.0)
    mean_id, _ = _mc_grad(obj_id, eta, n=2000)
    assert jnp.allclose(mean_id, 1.0, atol=1e-5)
    mean_sq, se_sq = _mc_grad(obj_sq, eta)
    assert jnp.abs(mean_sq - 5.0) < 5 * se_sq + 0.05


def test_geometric_natural_mvd_natural_gradient():
    # eta = log(1-p); mu = E[X] = (1-p)/p ; d/dmu E[X] = 1, d/dmu E[X^2] = 4/p - 3.
    eta = jnp.log(0.5)  # p = 0.5 -> d/dmu E[X^2] = 5
    p = 0.5

    @expectation
    def obj_id(e):
        return jnp.float32(geometric_natural_mvd(e))

    @expectation
    def obj_sq(e):
        x = geometric_natural_mvd(e)
        return jnp.float32(x) ** 2

    mean_id, se_id = _mc_grad(obj_id, eta)
    assert jnp.abs(mean_id - 1.0) < 5 * se_id + 0.02
    mean_sq, se_sq = _mc_grad(obj_sq, eta)
    assert jnp.abs(mean_sq - (4.0 / p - 3.0)) < 5 * se_sq + 0.1


# ---------------------------------------------------------------------------
# Regression guard: the geometric_natural sampler must not be degenerate
# (efax 1.20's GeometricNP.sample is constant for p >= 0.5).
# ---------------------------------------------------------------------------


def test_geometric_natural_sampler_is_not_degenerate():
    eta = jnp.log(0.5)  # p = 0.5 -> mean 1.0, std sqrt(1-p)/p = sqrt(2)
    keys = jrand.split(jrand.PRNGKey(0), 50000)
    xs = jax.vmap(lambda k: seed(lambda: geometric_natural.sample(eta))(k))(keys)
    assert jnp.std(xs) > 1.0  # not a constant
    assert jnp.abs(jnp.mean(xs) - 1.0) < 0.05
