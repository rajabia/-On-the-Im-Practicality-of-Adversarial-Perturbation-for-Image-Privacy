# On the (Im)Practicality of Adversarial Perturbation for Image Privacy (K-RTIO and UEP)

# Install Environment and Libraries:

python3 -m venv myenv

source myenv/bin/activate

pip3 install pycryptodome

pip3 install opencv-python

pip3 install pillow

pip3 install matplotlib

pip3 install sewar

## To perturb with KRTIO 
python3 KRTIO_UEP.py --input './inputfolder' --mode 'Enc' --k 3 --bl_size 16 

## To unperturb with krtio
python3 KRTIO_UEP.py --input './inputfolder' --mode 'Dec' --k 3 --bl_size 16 


Please cite . Rajabi, R. Bobba, M. Rosulek, C. Wright, W. Feng, " On the (Im)Practicality of Adversarial Perturbation for Image Privacy " Accepted in Privacy Enhancing Technology Symposium ![PETS](https://www.petsymposium.org/2021/files/papers/popets-2021-0006.pdf), 2021.

