pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip install pytorch-lightning gdown
  
#Download pruned models from google drive using gdown library
gdown --id 1-SKifn19CmD_ZP6G3_k-Iu-YtzLScoZl
gdown --id 1-STAgloOCXCbndaPuVhenRwH0jsg100u
gdown --id 1-ThHKHDiUMhGFcpbARRfhXNedgviz1vS
gdown --id 1-UGCLmYHr8DpmpkW282Mc3iTbARH_lsY
gdown --id 1-WfZPudvKrjQN9R9JJfm7wEo7xEzz3Dt
gdown --id 1-_1D00a4g9FcLnTkshv2FtDDY9lEBQ2H

#Run load_and_test.py
python load_and_test.py