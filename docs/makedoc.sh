source venv-new/bin/activate

mkdir -p _static
make clean
make html
make doctest
