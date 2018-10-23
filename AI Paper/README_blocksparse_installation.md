# Blocksparse Installation Guide

## Instructions
1. Install CUDA, cuDNN, TensorFlow as usual. Make sure you are using Python 3.6+, Python 2.7 WILL NOT work despite documentation says otherwise.

**Note:** Recommend to use anaconda virtual environment for isolation.

2. Install BlockSparse compilation dependencies:
- **MPI**: `sudo apt-get mpich`
- **nccl**: Follow the instructions (3.2: Other distributions) [here](https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html#down).

3. `nano ~/.bashrc` and set the following environment variables if it is not done: 
- `export CUDA_PATH=<cuda_installation_path>`
- `export MPI_PATH=<MPI_installation_path>`
Apply these changes by `source ~/.bashrc`

4. Install BlockSparse: 
```sh
git clone git@github.com:openai/blocksparse.git
cd blocksparse

make compile (note: You can speed up compilation by specifying -j<cores>)
pip install dist/*.whl
```
**WARNING:** DO NOT install via pip. The provided pip package is [outdated](https://pypi.org/project/blocksparse/#history).

5. Modify **matmul.py**:
```sh
nano <blocksparse_installation_path>/matmul.py

Scroll to line 51, change the following:
def __init__(self, layout, block_size=32, feature_axis=1, name=None):
==> def __init__(self, layout, block_size=32, feature_axis=0, name=None):
```
BlockSparse should be ready to use now. Test if you like.

## Final Notes
- It appears that some parts of **matmul.py** are broken. It involves the use of `tensorflow.python.framework.ops._recompute_node_def()`, which is not present in recent releases.
- According to the documentation, **BSConv** is incompatible with Volta GPUs and is true during testing. It probably is a good idea to fall back on Maxwell/Pascal GPUs instead.
