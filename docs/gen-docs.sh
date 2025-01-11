sphinx-apidoc --file-insertion-enabled -f -o source ../src tests conf
make html
rm -rf html
cp -a _build/html html
