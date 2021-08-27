# Setup
Do it in the following order:
## HTM
After cloning with submodules, navigate into `submodules/htm.core` and run `python setup.py install`, 
if not using virtualenv or anaconda: `python setup.py install --user --force`. You can then use HTM:

```python 
import htm
help(htm) # Show HTM help.
```
## Anaconda
To update environment with `conda_env.yml`:
```shell
conda env update --name ENV_NAME --file conda_env.yml
```
**(Development Only)** To write current packages into `conda_env.yml`:
```shell
conda env export > conda_env.yml
```

