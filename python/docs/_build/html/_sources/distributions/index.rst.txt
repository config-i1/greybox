===================
Distributions Guide
===================

This section provides detailed information about all probability distributions
available in greybox.

Overview
--------

greybox provides a comprehensive set of probability distributions for modeling
different types of data. Each distribution family includes:

* **Density function** (``d*``) - Probability density function (PDF)
* **Cumulative distribution function** (``p*``) - CDF
* **Quantile function** (``q*``) - Inverse CDF
* **Random generator** (``r*``) - Random sample generation

Common Parameters
------------------

Most distribution functions share common parameters:

* ``q`` or ``x`` : Value(s) at which to evaluate the function
* ``loc`` or ``mu`` : Location parameter (often the mean)
* ``scale`` : Scale parameter (often the standard deviation)
* ``shape`` : Shape parameter(s) specific to the distribution
* ``log`` / ``log_p`` : If True, return the log of the probability
* ``lower_tail`` : If True (default), probabilities are P[X <= x]

Continuous Univariate Distributions
-----------------------------------

Normal (Gaussian)
~~~~~~~~~~~~~~~~~

.. py:function:: dnorm(q, mean=0, sd=1, log=False)

   Normal (Gaussian) distribution density.

   :param q: Value(s) at which to evaluate
   :param mean: Mean (location parameter)
   :param sd: Standard deviation (scale parameter)
   :param log: If True, return log density
   :return: Density at q

   **Example**::

      from greybox import dnorm
      dnorm(0, mean=0, sd=1)  # ~0.3989

Laplace
~~~~~~~

.. py:function:: dlaplace(q, loc=0, scale=1, log=False)

   Laplace (double exponential) distribution density.

   :param q: Value(s) at which to evaluate
   :param loc: Location parameter (median)
   :param scale: Scale parameter
   :param log: If True, return log density
   :return: Density at q

S Distribution
~~~~~~~~~~~~~~

.. py:function:: ds(q, mu=0, scale=1, log=False)

   S-distribution density - a heavy-tailed distribution.

   :param q: Value(s) at which to evaluate
   :param mu: Location parameter
   :param scale: Scale parameter
   :param log: If True, return log density
   :return: Density at q

Generalized Normal
~~~~~~~~~~~~~~~~~~

.. py:function:: dgnorm(q, mu=0, scale=1, shape=1, log=False)

   Generalized Normal distribution density.

   :param q: Value(s) at which to evaluate
   :param mu: Location parameter
   :param scale: Scale parameter
   :param shape: Shape parameter (controls tail weight)
                 - shape=1: Laplace
                 - shape=2: Normal
                 - shape<2: Heavy tails
                 - shape>2: Light tails
   :param log: If True, return log density
   :return: Density at q

   **Example**::

      from greybox import dgnorm
      # Normal-like
      dgnorm(0, mu=0, scale=1, shape=2)
      # Heavy-tailed
      dgnorm(5, mu=0, scale=1, shape=1)

Logistic
~~~~~~~~

.. py:function:: dlogis(q, loc=0, scale=1, log=False)

   Logistic distribution density.

   :param q: Value(s) at which to evaluate
   :param loc: Location parameter
   :param scale: Scale parameter
   :param log: If True, return log density
   :return: Density at q

Student's t
~~~~~~~~~~~

.. py:function:: dt(q, df, loc=0, scale=1, log=False)

   Student's t distribution density.

   :param q: Value(s) at which to evaluate
   :param df: Degrees of freedom
   :param loc: Location parameter
   :param scale: Scale parameter
   :param log: If True, return log density
   :return: Density at q

Asymmetric Laplace
~~~~~~~~~~~~~~~~~~

.. py:function:: dalaplace(q, mu=0, scale=1, alpha=0.5, log=False)

   Asymmetric Laplace distribution density for quantile regression.

   :param q: Value(s) at which to evaluate
   :param mu: Location parameter
   :param scale: Scale parameter
   :param alpha: Asymmetry parameter (0 < alpha < 1)
                 - alpha < 0.5: Right-skewed
                 - alpha > 0.5: Left-skewed
   :param log: If True, return log density
   :return: Density at q

Log-Transformed Distributions
-----------------------------

These distributions model the log of the response variable.

Log-Normal
~~~~~~~~~~

.. py:function:: dlnorm(q, meanlog=0, sdlog=1, log=False)

   Log-Normal distribution density.

   :param q: Value(s) - must be positive
   :param meanlog: Mean of the underlying normal distribution
   :param sdlog: Standard deviation of the underlying normal distribution
   :param log: If True, return log density
   :return: Density at q

   **Note**: For positive-valued data that is right-skewed.

Log-Laplace
~~~~~~~~~~~

.. py:function:: dllaplace(q, loc=0, scale=1, log=False)

   Log-Laplace distribution density.

   :param q: Value(s) - must be positive
   :param loc: Location parameter (of log values)
   :param scale: Scale parameter
   :param log: If True, return log density
   :return: Density at q

Log-S
~~~~~

.. py:function:: dls(q, loc=0, scale=1, log=False)

   Log-S distribution density.

   :param q: Value(s) - must be positive
   :param loc: Location parameter (of log values)
   :param scale: Scale parameter
   :param log: If True, return log density
   :return: Density at q

Log-Generalized Normal
~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: dlgnorm(q, mu=0, scale=1, shape=2, log=False)

   Log-Generalized Normal distribution density.

   :param q: Value(s) - must be positive
   :param mu: Location parameter (of log values)
   :param scale: Scale parameter
   :param shape: Shape parameter
   :param log: If True, return log density
   :return: Density at q

Box-Cox Normal
~~~~~~~~~~~~~~

.. py:function:: dbcnorm(q, mu=0, sigma=1, lambda_bc=0, log=False)

   Box-Cox Normal distribution density.

   :param q: Value(s) - must be positive
   :param mu: Location parameter
   :param sigma: Scale parameter
   :param lambda_bc: Box-Cox transformation parameter
   :param log: If True, return log density
   :return: Density at q

Folded and Rectified Distributions
-----------------------------------

These distributions model positive-valued data with a point mass at zero.

Folded Normal
~~~~~~~~~~~~~~

.. py:function:: dfnorm(q, mu=0, sigma=1, log=False)

   Folded Normal distribution density.

   :param q: Value(s) - must be non-negative
   :param mu: Mean of underlying normal
   :param sigma: Standard deviation of underlying normal
   :param log: If True, return log density
   :return: Density at q

Rectified Normal
~~~~~~~~~~~~~~~~

.. py:function:: drectnorm(q, mu=0, sigma=1, log=False)

   Rectified Normal distribution density.

   :param q: Value(s) - must be non-negative
   :param mu: Mean of underlying normal
   :param sigma: Standard deviation of underlying normal
   :param log: If True, return log density
   :return: Density at q

Distributions for Positive Values
---------------------------------

Inverse Gaussian
~~~~~~~~~~~~~~~~

.. py:function:: dinvgauss(q, mu=1, scale=1, log=False)

   Inverse Gaussian distribution density.

   :param q: Value(s) - must be positive
   :param mu: Mean parameter
   :param scale: Scale parameter
   :param log: If True, return log density
   :return: Density at q

Gamma
~~~~~

.. py:function:: dgamma(q, shape=1, scale=1, log=False)

   Gamma distribution density.

   :param q: Value(s) - must be positive
   :param shape: Shape parameter (often denoted alpha)
   :param scale: Scale parameter (often denoted theta)
   :param log: If True, return log density
   :return: Density at q

Exponential
~~~~~~~~~~~

.. py:function:: dexp(q, loc=0, scale=1, log=False)

   Exponential distribution density.

   :param q: Value(s) - must be non-negative
   :param loc: Location parameter
   :param scale: Scale parameter (1/rate)
   :param log: If True, return log density
   :return: Density at q

Chi-Squared
~~~~~~~~~~~

.. py:function:: dchi2(q, df, log=False)

   Chi-squared distribution density.

   :param q: Value(s) - must be non-negative
   :param df: Degrees of freedom
   :param log: If True, return log density
   :return: Density at q

Count Distributions
--------------------

Poisson
~~~~~~~

.. py:function:: dpois(q, mu, log=False)

   Poisson distribution probability mass function.

   :param q: Value(s) - must be non-negative integers
   :param mu: Mean/lambda parameter
   :param log: If True, return log probability
   :return: Probability mass at q

Negative Binomial
~~~~~~~~~~~~~~~~~

.. py:function:: dnbinom(q, mu=1, size=1, log=False)

   Negative Binomial distribution probability mass function.

   :param q: Value(s) - must be non-negative integers
   :param mu: Mean parameter
   :param size: Dispersion parameter
   :param log: If True, return log probability
   :return: Probability mass at q

Geometric
~~~~~~~~~

.. py:function:: dgeom(q, prob, log=False)

   Geometric distribution probability mass function.

   :param q: Value(s) - must be non-negative integers
   :param prob: Probability of success
   :param log: If True, return log probability
   :return: Probability mass at q

Binary/Bounded Distributions
-----------------------------

Binomial
~~~~~~~~

.. py:function:: dbinom(q, size, prob, log=False)

   Binomial distribution probability mass function.

   :param q: Value(s) - must be integers between 0 and size
   :param size: Number of trials
   :param prob: Probability of success
   :param log: If True, return log probability
   :return: Probability mass at q

Beta
~~~~

.. py:function:: dbeta(q, a, b, log=False)

   Beta distribution density.

   :param q: Value(s) - must be in [0, 1]
   :param a: First shape parameter (alpha)
   :param b: Second shape parameter (beta)
   :param log: If True, return log density
   :return: Density at q

Logit-Normal
~~~~~~~~~~~~

.. py:function:: dlogitnorm(q, mu=0, sigma=1, log=False)

   Logit-Normal distribution density.

   :param q: Value(s) - must be in (0, 1)
   :param mu: Mean of underlying normal (logit scale)
   :param sigma: Standard deviation (logit scale)
   :param log: If True, return log density
   :return: Density at q

CDF-Based Distributions
-----------------------

These distributions use the CDF in the likelihood.

Logistic CDF
~~~~~~~~~~~~

.. py:function:: plogis(q, location=0, scale=1, log_p=False, lower_tail=True)

   Logistic cumulative distribution function.

Probit (Normal CDF)
~~~~~~~~~~~~~~~~~~~~

.. py:function:: pnorm(q, loc=0, scale=1, log_p=False, lower_tail=True)

   Normal cumulative distribution function.

Summary
-------

greybox supports the following distributions. For each distribution family, use the
appropriate prefix:

* ``d*`` - Probability density/mass function (PDF/PMF)
* ``p*`` - Cumulative distribution function (CDF)
* ``q*`` - Quantile function (inverse CDF)
* ``r*`` - Random number generation

The choice of distribution depends on the nature of your data:

* **Continuous, symmetric**: Normal, Laplace, Logistic, Student's t
* **Heavy tails**: Laplace, S, Generalized Normal, Student's t
* **Positive, right-skewed**: Log-Normal, Gamma, Inverse Gaussian
* **Count data**: Poisson, Negative Binomial, Geometric
* **Proportions**: Beta, Logit-Normal, Binomial
* **Zero-inflated**: Folded Normal, Rectified Normal
