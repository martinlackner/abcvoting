source venv-new/bin/activate

cd docs-source
make clean
make html
make doctest

cd ..
rm -r docs/*
cp -rT docs-source/_build/html/ docs/
