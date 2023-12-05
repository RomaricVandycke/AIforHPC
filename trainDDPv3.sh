#!/bin/bash

#module reset
#module load miniconda3
#conda activate mytorch
#srun python DDPv3.py --epochs=2
#source $STORE/mytorchdist/bin/activate
#conda activate mytorchdist
#source $STORE/conda/envs/mytorchdist/bin/activate
which python
python DDPv1.py --epochs=2
#torchrun DDPv3.sh.py --epochs=2