#!/usr/bin/bash

# =====================
# Logging information
# =====================

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

# ==============================
# Run the experiment
# ==============================

# Simulator Params
arms=3
trials=30
blocks=3
priortype='uninformed'  # options: 'informative' or 'uninformed'
simmodel=1  # WSLTS: 0, AEG: 1, GLS: 2

# NN Params
datasize=50000
batchsize=512  # set to 0 if no batches are used
layers=2
units_1=64
units_2=32
epochs=300
lr=0.001
wd=0.001

# Summ Stats Params
summ_layers=2
summ_units_1=64  # hidden units of first layer
summ_units_2=32  # hidden units of second layer
summ_output=6

scheduler='plateau'
pf=0.5
pp=25

filename='bo_pe_aeg_3arms_3blocks'

# BO Params
# Note: 1 Iteration should take around 5-6 minutes to run on a regular laptop
inits=5
evals=20

python train_bo_pe.py \
    --arms $arms \
    --trials $trials \
    --blocks $blocks \
    --priortype $priortype \
    --simmodel $simmodel \
    --datasize $datasize \
    --batchsize $batchsize \
    --layers $layers \
    --units $units_1 $units_2 \
    --epochs $epochs \
    --lr $lr \
    --wd $wd \
    --summ-layers $summ_layers \
    --summ-units $summ_units_1 $summ_units_2 \
    --summ-output $summ_output \
    --scheduler $scheduler \
    --plateau-factor $pf \
    --plateau-patience $pp \
    --inits $inits \
    --evals $evals \
    --filename $filename
    
# =========================
# Post experiment logging
# =========================

echo ""
echo "============"
echo "Job finished successfully."
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"

