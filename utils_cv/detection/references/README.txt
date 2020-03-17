
All files in this folder are copied from torchvision:
https://github.com/pytorch/vision/tree/master/references/detection

Our aim is to make as little edits to these files as possible, so that newer version from torchvision can be simply dropped into this folder.

The only edits made are listed below, and highlighted in the code with a "# EDITED" comment:
- Fixing import statements, e.g. "import utils" -> "from . import utils"
- In engine.py, adding "return metric_logger" to train_one_epoch()
