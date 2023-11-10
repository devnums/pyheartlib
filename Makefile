#############################################################################
# Copyright (c) 2023 Pyheartlib team. - All Rights Reserved                 #
# Project repo: https://github.com/devnums/pyheartlib                       #
# Contact: devnums.code@gmail.com                                           #
#                                                                           #
# This file is part of the Pyheartlib project.                              #
# To see the complete LICENSE file visit:                                   #
# https://github.com/devnums/pyheartlib/blob/main/LICENSE                   #
#############################################################################


welcome:
	echo "Welcome to pyhearlib!"
install:
	poetry install
	pre-commit install
check:
	pre-commit run --all-files
test:
	pytest tests/
test-cov:
	pytest --cov-report html:cov_html --cov=pyheartlib tests/
doc:
	sphinx-build -b html docs docs/_build/html
doc-clean:
	rm -r docs/_build
	sphinx-build -E -b html docs docs/_build/html
gen-license:
	./gen_license.sh
