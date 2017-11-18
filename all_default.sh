source activate tensorflow
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
pip install -U pip
#conda install pytorch torchvision cuda80 -c soumith
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl 
pip install torchvision
python setup.py install
bash ./vc_demo.sh awb-clb1 /home/lab-huang.zhongyi/data/cmu_arctic clb awb


