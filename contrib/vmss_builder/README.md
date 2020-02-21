# Setup Multiple Virtual Machines with Multiple JupyterHub User Accounts

This project helps you setup a cluster of virtual machines with the computer vision recipes repo pre-installed. Specifically, this project uses the Azure virtual machine scale-set (VMSS) to deploy multiple Data Science Virtual Machines (DSVMs), each with multiple user accounts. On each machine, a post-deployment script is invoked - this clones the repository, creates the conda environment, and generates multiple JupyterHub users (so that multiple unique users can be packed on to a single VM).

This project can be used for hands-on / lab session when you need to setup multiple VM environments, and easily shut them down.

## Requirements:
- Azure CLI (https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest)
- Python v3.5 or greater
- make sure you're logged into the correct subscription:
```
az login
az account set -s {your-azure-subscription-id}
```

## Usage:
1. Run the vmss deployment script. This script can take up to 20 minutes to execute. 

    ```
    python vmss_deploy_with_public_ip.py \
        --name {your-resource-group-name} --location {location} \
        --vm-sku {your-vm-size} --vm-count {number-of-vms-to-create} \
        --admin-id {vm-admin-id} --admin-pw {vm-admin-pw} \
        --post-script vm_user_env_setup.sh
    ```
    After this script executes, you will have a VMSS of DSVMs, each with jupyter hub preconfigured on it. 

    Each vm will have n number of users setup. You can configure this number inside the `vm_user_env_setup.sh` file before running the vmss deployment script. Inside the `vm_user_env_setup.sh` file, you can also configure the usernames and passwords of the users. By default, it will be `user1, password1`, `user2, password2`, ...

1. Once the script has completed, take note of the ip addresses. These are the public addresses of the individual DSVMs. To access the repo, simply open a browser and enter `https://<ip-address>:8000`. A warning on the browser may appear, but ignore it and proceed to the webpage. You can then access the dsvm by entering the username/password of the users on each VM.

## Authors:
Jun Ki Min (https://github.com/loomlike)
JS Tan (https://github.com/jiata)

NOTE: this repo is not part of the computervision-recipes CICD pipeline
