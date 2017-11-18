source activate tensorflow
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
pip install -U pip
pip install -r requirements.txt
conda install pytorch torchvision cuda80 -c soumith
bash ./vc_demo.sh vc_gan_test2 /home/lab-huang.zhongyi/data/cmu_arctic


