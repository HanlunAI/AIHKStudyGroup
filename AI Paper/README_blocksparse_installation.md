Instructions on installing BlockSparse
=========================================
1. Install CUDA, cuDNN, TensorFlow as usual. Make sure you are using Python 3.6+, Python 2.7 WILL NOT work.
Note: Recommend to use anaconda virtual environment for isolation.

2. Install BlockSparse compilation dependencies:
 a) MPI: sudo apt-get mpich
 b) nccl: Follow the instructions (3.2 Other distributions) here: https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html#down

3. nano ~/.bashrc and set the following environment variables if it is not done: 
 a) export CUDA_PATH=<cuda_installation_path>
 b) export MPI_PATH=<MPI_installation_path>
Apply these changes by source ~/.bashrc


4. Install BlockSparse: 
 a) git clone git@github.com:openai/blocksparse.git
 b) cd blocksparse
 c) make compile (note: You can speed up compilation by specifying -j<cores>)
 d) pip install dist/*.whl
IMPORTANT: DO NOT install via pip. I suspect the pip pre-compiled package is not compiled for Volta architecture GPUs yet.

5. Modify matmul.py:
 a) nano <blocksparse_installation_path>/matmul.py
 b) Scroll to line 51, change the following:
    def __init__(self, layout, block_size=32, feature_axis=1, name=None):
==> def __init__(self, layout, block_size=32, feature_axis=0, name=None):

BlockSparse should be ready to use now.

Test if you like.

Final Notes: 
1. It appears that some parts of matmul.py is broken. It involves the use of tensorflow.python.framework.ops._recompute_node_def(), which is not present in recent releases.
2. According to the documentation, BSConv is incompatible with Volta GPUs, and is true during testing. It probably is a good idea to fallback on Maxwell/Pascal GPUs instead.