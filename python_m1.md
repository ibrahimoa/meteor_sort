# M1 Architecture Handling with Python


**Set CPU Architecture**:

````shell
arch -x86_64 zsh
arch -arm64 zsh                                                               
````

To check the current architecture you can use the `uname -m` command.

**Activate virtualenv in Python**:

````shell
source venv_dir/bin/activate
````

To **deactivate** just enter:

`````shell
deactivate
`````

**Python versions installed in each folder**:

- `dev/dev_3.9` for **x86_64** arch
- `dev/dev_3.8` for **arm64** arch

### dev_3.9 (x86_64) packages


* scikit-learn             0.24.2
* scipy                    1.6.3
* pandas                   1.3.0
* sklearn                  0.0
* matplotlib               3.4.2
* numpy                    1.19.5
* torch                    1.8.1
* torchvision              0.9.1

### dev_3.8 (arm64) packages

* matplotlib              3.4.2
* matplotlib-inline       0.1.2
* numpy                   1.19.5
* pandas                  1.4.0.dev0+354.g9ed9a659eb
* scipy                   1.7.3
* tensorboard             2.5.0
* tensorboard-data-server 0.6.1
* tensorboard-plugin-wit  1.8.0
* tensorflow-addons       0.1a3
* tensorflow-estimator    2.5.0
* tensorflow              0.1a3
