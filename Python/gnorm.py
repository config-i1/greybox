def dgnorm(q, mu=0, scale=1, shape=1, log=False):
    """
    Computes the PDF of the Generalised Normal distribution at the given points.

    Parameters
    ----------
    q : array-like
        The points at which to compute the PDF of the Generalised Normal distribution.
    mu : float
        The mean of the distribution.
    scale : float
        The scale of the distribution.
    shape : float
        The shape of the distribution.
    log : bool, optional
        If True, the output will be the logarithm of the PDF.

    Returns
    array-like
        The Gaussian distribution at the given points.
    """
    # Check if the scale is non-negative
    if scale < 0:
        raise ValueError('Scale must be non-negative.')

    # Check if the shape is non-negative
    if shape < 0:
        raise ValueError('Shape must be non-negative.')

    # Compute the Gaussian distribution
    gnorm_values = np.exp(-(np.abs(q - mu) / scale)**shape) * shape / (2 * scale * np.gamma(1 / shape))

    # Return the Gaussian distribution
    if log:
        return np.log(gnorm_values)
    else:
        return gnorm_values
    
    
def pgnorm(q, mu=0, scale=1, shape=1, lower_tail=True, log_p=False):
    """
    Computes the Generalised Normal Cumulative Distribution Function at the given points.

    Parameters
    ----------
    q : array-like
        The points at which to compute the CDF of the  Generalised Normal distribution.
    mu : float
        The mean of the distribution.
    scale : float
        The scale of the distribution.
    shape : float
        The shape of the distribution.
    lower_tail : bool, optional
        If True, the lower tail of the distribution will be used. Otherwise, the upper tail will be used.
    log_p : bool, optional
        If True, the logarithm of the CDF of the Generalised Normal distribution will be returned. Otherwise, the function will return the CDF value itself.

    Returns
    array-like
        The  Generalised Normal Cumulative Distribution Function at the given points.
    """

    # Check if the scale is non-negative
    if scale < 0:
        raise ValueError('Scale must be non-negative.')

    # Check if the shape is non-negative
    if shape < 0:
        raise ValueError('Shape must be non-negative.')

    # If shape is too high, switch to uniform
    if shape > 100:
        return scipy.stats.uniform.cdf(q, lower_bound=mu-scale, upper_bound=mu+scale)

    # Compute the Gaussian probability distribution function
    p = (1 / 2 + np.sign(q - mu) * scipy.stats.gamma.cdf(abs(q - mu)**shape, shape=1/shape, scale=1/scale) / 2)
    
    # Return the Gaussian probability distribution function
    if log_p:
        return np.log(p)
    else:
        return p

