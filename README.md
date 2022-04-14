# gcn

Настройка окружения

```
## PyTorch
conda create -n gcn python networkx scikit-learn pandas matplotlib -y
conda activate gcn
conda install ipykernel --update-deps --force-reinstall -y
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
pip install --user scipy==1.8

## PyG

conda install pyg -c pyg -y
pip install torch-geometric-temporal

## DGL

conda install -c dglteam dgl-cuda11.3 -y
python -c "import dgl; print(dgl.__version__)"

## PyKEEN

pip install pykeen

```
