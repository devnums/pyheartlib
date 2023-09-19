# Contributing to Pyheartlib

Thank you for your interest in contributing to `pyheartlib`.
You can contribute to this project in several ways.

- Report bugs.
- Improve code and documentation.
- Request new features.
- Implement new features.
- Add new tests.
- Suggest new datasets.
- Provide new examples and use cases, for instance models for heartbeat classification, arrhythmia classification, beat segmentation, disease classification, etc.

If you are interested in contributing by modifying or adding new features, please follow the steps below.

## Steps

`Pyheartlib` uses `poetry` to manage its dependencies. For up-to-date documentation and instructions regarding the use and installation of `poetry`, please visit its official [website](https://python-poetry.org).

Fork the original `pyheartlib` repo by visiting the project GitHub page and clicking the **Fork** button.

Clone your fork on your local computer by running the command below in your terminal.

```bash
$ git clone https://github.com/your-username/pyheartlib
```

Change your directory:

```bash
$ cd pyheartlib
```

Install the project and its dependencies (including development dependencies) by running the following command:

```bash
$ make install
```

This command will install the project using `poetry` and set up `pre-commit` git hooks.

The next step is to create a new branch and check it out by running the command below. It is important that the branch name describes the change you are making to the code.

```bash
$ git checkout -b branch-name
```

After you make some changes or add new codes, you should make sure that your changes conform to the project's styling standards. Kindly incorporate the necessary tests and documentations for your new codes, and ensure that the tests pass and the documentations are constructed as desired.

```bash
$ make check
$ make test
$ make docs
```

If everything is alright, you can commit the changes by running the following commands in your terminal. Please ensure that your commit message follows the [Angular style](https://github.com/angular/angular.js/blob/master/DEVELOPERS.md#commits).

```bash
$ git add .
$ git commit -m "commit message"
```

Your contributions are now ready to be transmitted to the remote.

```bash
$ git push
```

The final step is to **open a pull request** by going into your fork page. Your contributions will be merged into the main repository of the project if they are accepted.

## Style

To maintain a clean and readable python code, `pyheartlib` conforms to the `PEP 8` guidelines and takes advantage of tools such as `ruff`, `black`, `isort`, and `flake8` to enforce them.

## Documentation

This project applies NumPy-style for docstrings and uses `sphinx` for generating documentations.
After adding or changing the documentations, the command below should be run in the terminal to build the documentation.

```bash
$ make docs
```

The HTML documentations are created in the `docs/_build` directory.

## Tests

`Pyheartlib` uses `pytest` for testing. If you are adding new features or modifying the code, please add the appropriate tests to the [`tests`](https://github.com/devnums/pyheartlib/tree/main/tests) directory.
To ensure that the tests pass successfully, please run the following command in your terminal.

```bash
$ make test
```

Run the following command to generate the HTML coverage report.

```bash
$ make test-cov
```

The HTML report can be viewed at `cov_html/index.html`

## Discussions

For any questions or discussion, please join us on [Discord](https://discord.gg/uNQmX6QZ).

## Your code

The code you are contributing must be your own, and whenever it is necessary to add a new dependency, use the command below.

```bash
$ poetry add package-name
```

Please note that only packages with permissive licenses (such as MIT or Apache) are allowed to be added.

If you contribute to `pyheartlib`, you acknowledge that your contributions will be licensed under the MIT license.

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
