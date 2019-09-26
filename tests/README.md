# Tests

This project provides folders for unit, smoke and integration tests with Python files and notebooks:

 * In the unit tests we just make sure the notebook runs. 
 * In the smoke tests, we run AzureML notebooks.
 * The integration tests are used to check that the models are performing at a
   given threshold.
 
For more information, see a [quick introduction to unit, smoke and integration tests](https://miguelgfierro.com/blog/2018/a-beginners-guide-to-python-testing/).

## Test execution

**Click on the following menus** to see more details on how to execute the unit and smoke tests:

<details>
<summary><strong><em>Unit tests</em></strong></summary>

Unit tests ensure that each class or function behaves as it should. Every time a developer makes a pull request to staging or master branch, a battery of unit tests is executed. 

**Note that the next instructions execute the tests from the root folder.**

For executing the Python unit tests for the utilities:

    pytest tests/unit -m "not notebooks and not gpu"

For executing the Python unit tests for the notebooks:

    pytest tests/unit -m "notebooks and not gpu"

For executing the Python GPU unit tests for the utilities:

    pytest tests/unit -m "not notebooks and gpu"

For executing the Python GPU unit tests for the notebooks:

    pytest tests/unit -m "notebooks and gpu"

Note: today there are no specific gpu tests.

</details>


<details>
<summary><strong><em>Smoke tests</em></strong></summary>

The Smoke tests are notebooks that run on AzureML.

**Note that the next instructions execute the tests from the root folder.**

For executing the Python smoke tests:

    pytest tests/smoke -m "smoke and not gpu"

For executing the Python GPU smoke tests:

    pytest tests/smoke -m "smoke and gpu"

Note: today there are no specific gpu tests.

</details>



## Skipping Tests


In order to skip a test because there is an OS or upstream issue which cannot be resolved you can use pytest [annotations](https://docs.pytest.org/en/latest/skipping.html).
 
Example:

    @pytest.mark.skip(reason="<INSERT VALID REASON>")
    @pytest.mark.skipif(sys.platform == 'win32', reason="Not implemented on Windows")
    def test_to_skip():
        assert False


## Papermill

More details on how to integrate Papermill with notebooks can be found in their [repo](https://github.com/nteract/papermill).

