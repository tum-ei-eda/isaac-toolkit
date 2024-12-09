# MLonMCU Setup

See https://github.com/tum-ei-eda/mlonmcu for detailed instructions!

## TL;DR

### Ubuntu Packages

*Hint:* Ubuntu 20.04 and 22.04 should be supported!

Required APT packages:

```sh
sudo apt install build-essential git cmake libboost-system-dev libboost-filesystem-dev libboost-program-options-dev kcachegrind graphviz-dev device-tree-compiler
```

### Python Environment

*Hint:* Python v3.8 and v3.10 are tested!

Setup Python virtual environment (if not already inside one):

```sh
virtualenv -p python3 venv/
# Alternative: python3 -m venv venv/

source venv/bin/activate
```

Install Python requirements:

```sh
pip install -r requirements.txt
```

Install MLonMCU Python package:

```sh
# Use latest release
# pip install mlonmcu

# Use latest commit:
pip install git+https://github.com/tum-ei-eda/mlonmcu.git@$main
```

### Setup MLonMCU

Export `MLONMCU_HOME` environment variable:

```sh
export MLONMCU_HOME=$(pwd)/workspace
```

Initialize workspace using provided `environment.yml.j2` environment template:

```sh
mlonmcu init $MLONMCU_HOME -t ${{ matrix.mlonmcu-template}} --non-interactive --clone-models --allow-exists
```

Install additional Python dependencies

```sh
mlonmcu setup -g
pip install -r $MLONMCU_HOME/requirements_addition.txt
```

Setup MLonMCU workspace:

```sh
mlonmcu setup -v --progress
```
