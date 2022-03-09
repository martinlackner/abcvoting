source venv-new/bin/activate

cd docs-source
mkdir -p _static
make clean
make html
make doctest

cd ..
rm -r docs/*
cp -rT docs-source/_build/html/ docs/
