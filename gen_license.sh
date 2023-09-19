#!/bin/bash

#############################################################################
# Copyright (c) 2023 Pyheartlib team. - All Rights Reserved                 #
# Project repo: https://github.com/devnums/pyheartlib                       #
# Contact: devnums.code@gmail.com                                           #
#                                                                           #
# This file is part of the Pyheartlib project.                              #
# To see the complete LICENSE file visit:                                   #
# https://github.com/devnums/pyheartlib/blob/main/LICENSE                   #
#############################################################################


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
