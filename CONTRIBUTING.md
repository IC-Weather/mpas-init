# Contributing

By submitting a contribution, you agree to the terms in `CLA.md`.

Run `python -m pytest` and `ruff check .` before you open a pull request.
This repository has no continuous integration, so these two local checks are
the gate; both must pass. Install the test and GRIB extras first with
`pip install '.[test,grib]'`, so that the GRIB tests run instead of skipping. Keep the
numerics in PyTorch tensor operations, so that the model runs on CPU and GPU
and stays differentiable.
