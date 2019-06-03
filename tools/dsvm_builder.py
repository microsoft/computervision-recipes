import re
import subprocess
import textwrap
from shutil import which
from prompt_toolkit import prompt
from prompt_toolkit import print_formatted_text, HTML


def is_installed(name):
    """Check whether `name` is on PATH and marked as executable."""
    return which(name) is not None


def validate_password(password):

    if len(password) < 12 or len(password) > 123:
        print_formatted_text(
            HTML(
                "<ansired>Input must be between 12 and 123 characters.</ansired> Try again."
            )
        )
        return False

    if (
        len([c for c in password if c.islower()]) <= 0
        or len([c for c in password if c.isupper()]) <= 0
    ):
        print_formatted_text(
            HTML(
                "<ansired>Input must contain a upper and a lower case character.</ansired> Try again."
            )
        )
        return False

    if len([c for c in password if c.isdigit()]) <= 0:
        print_formatted_text(
            HTML("<ansired>Input must contain a digit.</ansired> Try again.")
        )
        return False

    if len(re.findall("[\W_]", password)) <= 0:
        print_formatted_text(
            HTML(
                "<ansired>Input must contain a special character.</ansired> Try again."
            )
        )
        return False

    return True


if __name__ == "__main__":

    # variables
    UBUNTU_DSVM_IMAGE = (
        "microsoft-dsvm:linux-data-science-vm-ubuntu:linuxdsvmubuntu:latest"
    )
    VM_SIZE = "Standard_NC6s_v3"

    # print intro dialogue
    print_formatted_text(
        HTML(
            textwrap.dedent(
                """\
            <ansigreen>
            Azure Data Science Virtual Machine Builder

            This utility will help you create an Azure Data Science Ubuntu Virtual
            Machine that you will be able to run your notebooks in.

            To use this utility, you must have an Azure subscription which you can
            get from azure.microsoft.com.

            Please answer the question below and we'll help setup your virtual
            machine for you.\
            </ansigreen>
            """
            )
        )
    )

    # validate user activity
    prompt("Press enter to continue...")

    # state variables
    logged_in = False

    # list of cmds
    account_list_cmd = "az account list -o table"
    silent_login_cmd = 'az login --query "[?n]|[0]"'
    set_account_sub_cmd = "az account set -s {}"
    provision_rg_cmd = "az group create --name {} --location {}"
    provision_vm_cmd = "az vm create --resource-group {} --name {} --size {} --image {} --admin-username {} --admin-password {} --authentication-type password"

    # check that az cli is installed
    if not is_installed("az"):
        print(
            textwrap.dedent(
                """\
            You must also have the Azure CLI installed. For more information on
            installing the Azure CLI, see here:
            https://docs.microsoft.com/en-us/cli/azure/?view=azure-cli-latest
        """
            )
        )
        exit(0)

    # check if we are logged in
    results = subprocess.run(
        account_list_cmd.split(" "),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    logged_in = False if results.stderr else True

    # login to the az cli and suppress output
    if not logged_in:
        subprocess.run(silent_login_cmd.split(" "))
        print("\n")
    else:
        print("Looks like you're already logged in.")

    # show account sub list
    print("Here is a list of your subscriptions:")
    results = subprocess.run(
        account_list_cmd.split(" "), stdout=subprocess.PIPE
    )
    print_formatted_text(
        HTML(f"<ansigreen>{results.stdout.decode('utf-8')}</ansigreen>")
    )

    # prompt sub id
    subscription_id = prompt("Enter your subscription id: ")

    # prompt username / password
    username = prompt("Enter a username: ")
    password_is_valid = False
    while not password_is_valid:
        password = prompt("Enter a password: ", is_password=True)
        password_is_valid = validate_password(password)

    # prmopt vm name
    vm_name = prompt("Enter a name for your vm: ")  # TODO implement checker

    # prmopt region
    region = prompt("Enter a region for your vm: ")  # TODO implement checker

    # set sub id
    subprocess.run(set_account_sub_cmd.format(subscription_id).split(" "))

    # provision rg
    print("Creating the resource group...")
    results = subprocess.run(
        provision_rg_cmd.format(f"{vm_name}-rg", region).split(" "),
        stdout=subprocess.PIPE,
    )
    if "Succeeded" in results.stdout.decode("utf-8"):
        print("Done.\n")

    # create vm
    print("Creating the Data Science VM...")
    results = subprocess.run(
        provision_vm_cmd.format(
            f"{vm_name}-rg",
            vm_name,
            VM_SIZE,
            UBUNTU_DSVM_IMAGE,
            username,
            password,
        ).split(" "),
        stdout=subprocess.PIPE,
    )
    if "VM Running" in results.stdout.decode("utf-8"):
        print("Done.\n")
