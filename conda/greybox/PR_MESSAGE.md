## Adding `greybox`

### Summary

- PyPI package `greybox` v1.0.0 — Python port of the R `greybox` package
- Toolbox for likelihood-based regression model building and forecasting:
  - Augmented Linear Model (`ALM`) supporting 26+ distributions
  - R-style formula parser
  - IC-based stepwise variable selection and model combination
  - Comprehensive forecast error measures
- License: LGPL-2.1-only
- Pure Python (`noarch: python`); no compiled extensions
- Not a dependency of any other conda-forge package at this time

### Checklist

- [x] Title of this PR is meaningful: "Adding `greybox`"
- [x] License file is packaged (`license_file: LICENSE`)
- [x] Source is from official source (PyPI tarball, not a fork)
- [x] Package does not vendor other packages
- [x] No static libraries are linked or shipped (pure Python)
- [x] Build number is 0
- [x] A tarball (`url`) is used, not `git_url`
- [x] All dependencies are available on conda-forge (`numpy`, `scipy`, `pandas`, `nlopt`)
- [x] GitHub user `config-i1` listed as maintainer has confirmed willingness to maintain
- [x] conda-forge documentation consulted before submitting
