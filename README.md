# How to create a new, professionally managed python project

# Why?
Simple: To create a project which can be run reliably...
 - ...on most systems, mostly "independent" of os and hardware
 - ...in the future, even if the code is a few years old
 - ...by anyone, without the need to read every line of code and configuration again<br/>
   (including you, after you've forgotten everything!)

## Step-By-Step

1. **Install an IDE**<br/>
    E.g. VS Code, PyCharm,...
    ```bash
    wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /tmp/packages.microsoft.gpg
    sudo install -D -o root -g root -m 644 /tmp/packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
    echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" | sudo tee /etc/apt/sources.list.d/vscode.list > /dev/null
    sudo apt-get install apt-transport-https
    sudo apt-get update
    sudo apt-get install code
    ```
1. **Install and configure git**<br/>
    ```bash
    sudo apt-get install git-all
    git config --global user.name # "First Name Last name"
    git config --global user.email # "EMAIL"
    git config pull.rebase false
    ```
    For signed commits/tags:
    ```bash
    git config --global --unset gpg.format
    git config --global user.signingkey ?????????????????
    git config --global commit.gpgsign true
    git config --global tag.gpgSign true
    ```
2. **Install a virtual environment manager**<br/>
    E.g. anaconda, venv, virtualenv.
    These stop you from accidentally breaking your system installation (or other projects) when you install new packages.
    ```bash
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -u -p ~/miniconda3
    ```
3. **Create and activate a new virtual environment with the latest python version**<br/>
    ```bash
    conda create -y -n BA2024 python
    conda activate BA2024
    ```
4. **Create and enter a new folder for your project**<br/>
    ```bash
    mkdir BA2024
    cd BA2024
    ```
5. **Configure your IDE**<br/>
   ```bash
   mkdir .vscode
   nano .vscode/settings.json
   nano .vscode/launch.json
   ```
6. **Configure git repository**<br/>
   ```bash
   git init
   nano .gitignore
   nano .gitattributes
   ```
7. **Create and fill a pyproject.toml file**<br/>
    This file marks a folder as a python project, which serves two purposes: We can configure a variety of options, and we can install the project in our environment (which is later the only step required if you ever want to setup your project on e.g. a different machine).
    ```bash
    nano pyproject.toml
    ```
8. **Install a requirements manager**<br/>
    Manages your requirements for you, so you dont need to manually edit versions or take care of a _requirements.txt_ file.
    ```bash
    pip install poetry
    ```
9. **Add requirements**<br/>
   Some requirements (e.g. torch) might need additional configuration in the _pyproject.toml_ file.
    ```bash
    poetry add numpy scipy torch pyside6
    ```
10. **Install development tools**<br/>
    Development tools make your life easier e.g. by...
    - ...formatting your code into a common (widely known) format
    - ...sorting your imports alphabetically
    - ...checking your code for (type) issues
    ```bash
    poetry add black debugpy isort pyright ruff --group dev
    ```
11. **Configure your tools**<br/>
    ```bash
    nano pyrightconfig.json
    nano ruff.toml
    ```
12. **Create a code folder**<br/>
    The root folder is your package name which you would write down in import statements, while the *\_\_init__.py* file marks the folder as a valid (sub-)package.
    ```
    mkdir BA2024
    touch BA2024/__init__.py
    ```
13. **Install your new project**<br/>
    If you ever want to setup your project again, this (after a `git clone` and assuming a running conda and git installation) should be the only command you need to run.
    ```bash
    poetry install --with dev
    ```
14. **Write a README.md**<br/>
    ```bash
    nano README.md
    ```
15. **Save your progress**<br/>
    ```bash
    git add .
    git commit -m "Initial commit"
    ```
    To add an (online created) github repository:
    ```bash
    git remote add origin git@github.com: #name and folder 
    git push -u origin master
    ```
16. **Write code (and don't forget using type hints!)**<br/>