source ../venv-docs/bin/activate

mkdir -p _static
make clean
make html
make doctest

deactivate
