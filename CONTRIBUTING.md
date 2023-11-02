# Contribution guidelines

## Local setup

We encourage you to use a virtual environment. You'll want to activate it every time you want to work on the project.

```sh
python -m venv --upgrade-deps .venv
source .venv/bin/activate
```

Then, navigate to your cloned fork and install development dependencies:

```sh
pip install -r requirements-dev.txt
```

Finally, install the [pre-commit](https://pre-commit.com/) push hooks. This will run some code quality checks every time you push to GitHub.

```sh
pre-commit install --hook-type pre-push
```

You can optionally run `pre-commit` at any time as so:

```sh
pre-commit run --all-files
```

Before pushing it is adviced to check if all notebooks could be executed:

```sh
make execute-notebooks
```
