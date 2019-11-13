# VM Builder

> Note: this tool only works on Linux or Mac

This mini project will help you set up a Virtual Machine with the Computer
Vision repo installed on it. 

You can use this project simply by running:
```bash
python vm_builder.py
```

This will kick off an interactive bash session that will create your VM on
Azure and install the repo on it. 

Once your VM has been setup, you can ssh tunnel to port 8899 and you'll
find the Computer Vision repo setup and ready to be used.
```bash
ssh -L 8899:localhost:8899 <username>@<ip-address>
```

Visit localhost:8899 on your browser to start using the notebooks.


