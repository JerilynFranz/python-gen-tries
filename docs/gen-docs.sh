sphinx-apidoc -f -o source ../src tests conf
make html
cp -a _build/html html
