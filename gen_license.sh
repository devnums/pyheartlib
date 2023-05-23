#!/bin/bash

poetry export -f requirements.txt --output requirements.txt
pythonpath=$(poetry env info --path)
pythonpath+=/bin/python
echo $pythonpath

poetry run python -m third_party_license_file_generator -r requirements.txt -p $pythonpath

cat ext_licenses.textt >> THIRDPARTYLICENSES
sed -i '8,$d' ACKNOWLEDGEMENTS.txt
cat THIRDPARTYLICENSES >> ACKNOWLEDGEMENTS.txt
rm THIRDPARTYLICENSES


# internet connection is required before running
# ./gen_license.sh