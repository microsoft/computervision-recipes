# Contribution Guidelines

Contributions are welcomed! Here's a few things to know:

* [Steps to Contributing](#steps-to-contributing)
* [Coding Guidelines](#TODO)
* [Microsoft Contributor License Agreement](#microsoft-contributor-license-agreement)
* [Code of Conduct](#code-of-conduct)

## Steps to Contributing

**TL;DR for contributing: We use the contrib branch to land all new features and fixes. To make a contribution, please create a branch from contrib, make a modification in the code and create a PR to contrib.** 

Here are the basic steps to get started with your first contribution. Please reach out with any questions.
1. Use [open issues](https://github.com/Microsoft/ComputerVision/issues) to discuss the proposed changes. Create an issue describing changes if necessary to collect feedback. Also, please use provided labels to tag issues so everyone can easily sort issues of interest.
2. [Fork the repo](https://help.github.com/articles/fork-a-repo/) so you can make and test local changes.
3. Create a new branch **from contrib branch** for the issue (please do not create a branch from master or staging). We suggest prefixing the branch with your username and then a descriptive title: (e.g. gramhagen/update_contributing_docs)
4. Create a test that replicates the issue.
5. Make code changes.
6. Ensure unit tests pass and code style / formatting is consistent TODO: add docstring links.
7. Create a pull request against **contrib** branch.

Once the features included in a [milestone](https://github.com/Microsoft/ComputerVision/milestones) are completed, we will merge contrib into staging. TODO: make a wiki with coding guidelines.

## Coding Guidelines

We strive to maintain high quality code to make the utilities in the repository easy to understand, use, and extend. We also work hard to maintain a friendly and constructive environment. We've found that having clear expectations on the development process and consistent style helps to ensure everyone can contribute and collaborate effectively.

## Microsoft Contributor License Agreement

Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

TODO: add CLA-bot

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

Apart from the official Code of Conduct developed by Microsoft, in the Computer Vision team we adopt the following behaviors, to ensure a great working environment:

#### Do not point fingers
Let’s be constructive.

<details>
<summary><em>Click here to see some examples</em></summary>

"This method is missing docstrings" instead of "YOU forgot to put docstrings".

</details>

#### Provide code feedback based on evidence 

When making code reviews, try to support your ideas based on evidence (papers, library documentation, stackoverflow, etc) rather than your personal preferences. 

<details>
<summary><em>Click here to see some examples</em></summary>

"When reviewing this code, I saw that the Python implementation the metrics are based on classes, however, [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics) and [tensorflow](https://www.tensorflow.org/api_docs/python/tf/metrics) use functions. We should follow the standard in the industry."

</details>


#### Ask questions - do not give answers
Try to be empathic. 

<details>
<summary><em>Click here to see some examples</em></summary>

* Would it make more sense if ...?
* Have you considered this ... ?

</details>


