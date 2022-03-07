source venv-new/bin/activate

cd docs-source
make clean
make html
make doctest

cd ..
cp -rT docs-source/_build/html/ docs/
