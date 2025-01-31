conda create --name hdtree python=3.10 
conda activate hdtree   
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia  
pip install lightning
pip install plotly   
pip install scipy
pip install kornia   
pip install scikit-learn 
pip install matplotlib   
pip install pynndescent  
pip install scanpy   
pip install wandb
pip install -U 'jsonargparse[signatures]>=4.27.7'
pip install munkres
pip install -U kaleido
pip install pacmap