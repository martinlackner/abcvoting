source ../venv-new/bin/activate

make clean
make html
make doctest

cp -rT _build/html/ ../docs/
