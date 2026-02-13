"""Advanced Linear Model (ALM) class for greybox."""


class ALM:
    """Advanced Linear Model - regression with specified distribution.

    This is similar to sklearn's LinearRegression but uses likelihood
    estimation for various non-normal distributions.
    """

    DISTRIBUTIONS = [
        "dnorm", "dlaplace", "ds", "dgnorm", "dlogis", "dt", "dalaplace",
        "dlnorm", "dllaplace", "dls", "dlgnorm", "dbcnorm",
        "dinvgauss", "dgamma", "dexp",
        "dfnorm", "drectnorm",
        "dpois", "dnbinom", "dbinom", "dgeom",
        "dbeta", "dlogitnorm",
        "plogis", "pnorm"
    ]

    LOSS_FUNCTIONS = [
        "likelihood", "MSE", "MAE", "HAM", "LASSO", "RIDGE", "ROLE"
    ]

    OCCURRENCE_TYPES = ["none", "plogis", "pnorm"]

    def __init__(
        self,
        formula: str,
        data,
        subset=None,
        na_action=None,
        distribution: str = "dnorm",
        loss: str = "likelihood",
        occurrence: str = "none",
        scale=None,
        orders=(0, 0, 0),
        parameters=None,
        fast: bool = False,
        **kwargs
    ):
        """Initialize ALM model.

        Parameters
        ----------
        formula : str
            Model formula (e.g., "y ~ x1 + x2 + trend").
        data : DataFrame or ndarray
            Data for the model.
        subset : array-like, optional
            Subset of observations to use.
        na_action : callable, optional
            Function for handling NAs.
        distribution : str, default="dnorm"
            Density function to use.
        loss : str, default="likelihood"
            Loss function type.
        occurrence : str, default="none"
            Distribution for occurrence model.
        scale : str or Formula, optional
            Formula for scale parameter.
        orders : tuple, default=(0, 0, 0)
            ARIMA orders (p, d, q).
        parameters : array-like, optional
            Starting values for parameters.
        fast : bool, default=False
            Skip validation checks.
        **kwargs : dict
            Additional parameters.
        """
        self.formula = formula
        self.data = data
        self.subset = subset
        self.na_action = na_action
        self.distribution = distribution
        self.loss = loss
        self.occurrence = occurrence
        self.scale = scale
        self.orders = orders
        self.parameters = parameters
        self.fast = fast
        self.kwargs = kwargs

        self._validate_params()

    def _validate_params(self):
        """Validate input parameters."""
        if self.distribution not in self.DISTRIBUTIONS:
            raise ValueError(
                f"Invalid distribution: {self.distribution}. "
                f"Choose from: {self.DISTRIBUTIONS}"
            )

        if self.loss not in self.LOSS_FUNCTIONS:
            raise ValueError(
                f"Invalid loss: {self.loss}. "
                f"Choose from: {self.LOSS_FUNCTIONS}"
            )

        if self.occurrence not in self.OCCURRENCE_TYPES:
            raise ValueError(
                f"Invalid occurrence: {self.occurrence}. "
                f"Choose from: {self.OCCURRENCE_TYPES}"
            )

        if not isinstance(self.orders, (tuple, list)) or len(self.orders) != 3:
            raise ValueError("orders must be a tuple of 3 integers (p, d, q)")

    def fit(self):
        """Fit the ALM model."""
        raise NotImplementedError("Fitting not yet implemented")

    def predict(self, newdata=None):
        """Make predictions."""
        raise NotImplementedError("Prediction not yet implemented")

    def summary(self):
        """Print model summary."""
        raise NotImplementedError("Summary not yet implemented")

    def __repr__(self):
        return (
            f"ALM(formula={self.formula!r}, "
            f"distribution={self.distribution!r}, "
            f"loss={self.loss!r})"
        )
