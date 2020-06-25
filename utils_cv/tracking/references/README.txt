All files in the fairmot folder are copied from FairMOT:
https://github.com/ifzhang/FairMOT/src/lib

Our aim is to make as little edits to these files as possible, so that newer versions from FairMOT can be simply dropped into this folder.

The only edits made are listed below, and highlighted in the code with a "# EDITED" comment:
- Fixing import statements, e.g. "import utils" -> "from . import utils"
- Not hard-coding input resolution values in datasets/dataset/jde.py
- Setting the logging level to WARNING