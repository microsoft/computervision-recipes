# Contribution Guidelines

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us.

Here are a few more things to know:
- [Microsoft Contributor License Agreement](#microsoft-contributor-license-agreement)
- [Steps to Contributing](#steps-to-contributing)
- [Working with Notebooks](#working-with-notebooks)
- [Coding Guidelines](#coding-guidelines)
- [Code of Conduct](#code-of-conduct)
    - [Do not point fingers](#do-not-point-fingers)
    - [Provide code feedback based on evidence](#provide-code-feedback-based-on-evidence)
    - [Ask questions do not give answers](#ask-questions-do-not-give-answers)

## Microsoft Contributor License Agreement

Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

## Steps to Contributing

Here are the basic steps to get started with your first contribution. Please reach out with any questions.
1. Use [open issues](https://github.com/Microsoft/ComputerVision/issues) to discuss the proposed changes. Create an issue describing changes if necessary to collect feedback. Also, please use provided labels to tag issues so everyone can easily sort issues of interest.
1. [Fork the repo](https://help.github.com/articles/fork-a-repo/) so you can make and test local changes.
1. Create a new branch for the issue. We suggest prefixing the branch with your username and then a descriptive title: (e.g. gramhagen/update_contributing_docs)
1. Create a test that replicates the issue.
1. Make code changes.
1. Ensure unit tests pass and code style / formatting is consistent, and follows the [Zen of Python](https://github.com/Microsoft/Recommenders/wiki/Coding-Guidelines#the-zen-of-python).
1. We use [pre-commit](https://pre-commit.com/) package to run our pre-commit hooks. We use black formatter and flake8 linting on each commit. In order to set up pre-commit on your machine, follow the steps here, please note that you only need to run these steps the first time you use pre-commit for this project.

   * Update your conda environment, pre-commit is part of the yaml file or just do    
   ```
    $ pip install pre-commit
   ```    
   * Set up pre-commit by running following command, this will put pre-commit under your .git/hooks directory.
   ```
   $ pre-commit install
   ```
   ```
   $ git commit -m "message"
   ```
   * Each time you commit, git will run the pre-commit hooks (black and flake8 for now) on any python files that are getting committed and are part of the git index.  If black modifies/formats the file, or if flake8 finds any linting errors, the commit will not succeed. You will need to stage the file again if black changed the file, or fix the issues identified by flake8 and and stage it again.

   * To run pre-commit on all files just run
   ```
   $ pre-commit run --all-files
   ```
1. Create a pull request against <b>staging</b> branch.

Note: We use the staging branch to land all new features, so please remember to create the Pull Request against staging.


## Working with Notebooks

When you pull updates from remote there might be merge conflicts with jupyter notebooks. The tool [nbdime](https://nbdime.readthedocs.io/en/latest/) can help fix such problems.
* To install nbdime
```
pip install ndime
```
* To do diff between notebooks
```
nbdiff notebook_1.ipynb notebook_2.ipynb
```

## Coding Guidelines

We strive to maintain high quality code to make the utilities in the repository easy to understand, use, and extend. We also work hard to maintain a friendly and constructive environment. We've found that having clear expectations on the development process and consistent style helps to ensure everyone can contribute and collaborate effectively.

We follow the Google docstring guidlines outlined on this [styleguide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings) page. For example:
```python
  def bite(n:int, animal:animal_object) -> bool:
      """
      This function will perform n bites on animal.

      Args:
          n (int): the number of bites to do
          animal (Animal): the animal to bite

      Raises:
          Exception: biting animal has no teeth

      Returns:
          bool: whether or not bite was successful
      """
```

Since we take a strong dependency on fast.ai, variable naming should follow the standards of fast.ai which are described in this [abbreviation guide](https://docs.fast.ai/dev/abbr.html). For example, in computer vision cases, an image should always be abbreviated with `im` and not `i`, `img`, `imag`, `image`, etc. The one exception to this guide is that variable names should be as self-explanatory as possible. For example, the meaning of the variable `batch_size` is clearer than `bs` to refer to batch size.

The main variables and abbreviations are given in the table below:

| Abbreviation | Description |
| ------------ | ----------- |
| `im `                    | Image
| `fig`                    | Figure
| `pt`                     | 2D point (column,row)
| `rect`                   | Rectangle (order: left, top, right, bottom)
| `width`, `height`, `w`, `h`  | Image dimensions
| `scale`                  | Image up/down scaling factor
| `angle`                  | Rotation angle in degree
| `table`                  | 2D row/column matrix implemented using a list of lists
| `row`, `list1D`             | Single row in a table, i.e. single 1D-list
| `rowItem`                | Single item in a row
| `line`, `string`            | Single string
| `lines`, `strings`          | List of strings
| `list1D`                 | List of items, not necessarily strings
| -`s`    | Multiple of something (plural) should be indicated by appending an `s` to an abbreviation.


## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

Apart from the official Code of Conduct developed by Microsoft, we adopt the following behaviors, to ensure a great working environment:

#### Do not point fingers
Let’s be constructive. For example: "This method is missing docstrings" instead of "YOU forgot to put docstrings".

#### Provide code feedback based on evidence

When making code reviews, try to support your ideas based on evidence (papers, library documentation, stackoverflow, etc) rather than your personal preferences. For example: "When reviewing this code, I saw that the Python implementation the metrics are based on classes, however, [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics) and [tensorflow](https://www.tensorflow.org/api_docs/python/tf/metrics) use functions. We should follow the standard in the industry."

#### Ask questions do not give answers
Try to be empathic. For example: "Would it make more sense if ...?" or "Have you considered this ... ?"
