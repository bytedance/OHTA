conda env create -f environment.yml
eval "$(conda shell.bash hook)"
conda activate ohta
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install fal_serverless
pip install lpips
cd third_parties
git clone https://github.com/neuralbodies/leap.git
cd ./leap
python setup.py build_ext --inplace
pip install -e .
