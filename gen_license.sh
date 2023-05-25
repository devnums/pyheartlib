#!/bin/bash

poetry export -o requirements.textt --without-hashes
pythonpath=$(poetry env info --path)
pythonpath+=/bin/python
echo $pythonpath

poetry run python -m third_party_license_file_generator -r requirements.textt -p $pythonpath

cat ext_licenses.textt >> THIRDPARTYLICENSES
sed -i '8,$d' ACKNOWLEDGEMENTS.txt
cat THIRDPARTYLICENSES >> ACKNOWLEDGEMENTS.txt
rm THIRDPARTYLICENSES


# internet connection is required before running
# ./gen_license.sh