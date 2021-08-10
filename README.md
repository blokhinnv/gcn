# gcn

1. Настройка окружения

```
conda create -n gcn python=3.8 networkx scikit-learn pandas nb_conda
conda activate gcn
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"

// torch-geometric

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
pip install torch-geometric
pip install torch-geometric-temporal

// DGL

conda install -c dglteam dgl-cuda10.2
python -m dgl.backend.set_default_backend pytorch
python -c "import dgl; print(dgl.__version__)"
```

