#!/bin/bash

NEED_HELP=1
M="xgb"
N=100
CACHE_DIR="cache"
SCRIPT=experiments.py
while [[ $# -gt 0 ]]; do
    case $1 in
        -n)
            N="$2"
            shift
            shift
            ;;
        -m)
            M="$2"
            shift
            shift
            ;;
        --pinacs)
            CACHE_DIR="cache_pinacs"
            shift
            ;;
        *)
            NEED_HELP=0
            for fold in 0 1 2 3 4; do
                echo
                echo "$(date), fold $fold ========================"
                python $SCRIPT $1 --cache_dir $CACHE_DIR --fold $fold -N $N --model_type $M
                if [[ ! $? -eq 0 ]]; then 
                    echo "ERROR, quiting... $?"
                    echo " - dataset $1"
                    echo " - fold $fold"
                    echo " - N $N"
                    echo " - M $M"
                    exit 1
                fi
            done
            shift
            ;;
    esac
done

if [[ $NEED_HELP -eq 1 ]]; then
    echo "$0 [-n num_adv_examples] [-m model_type] datasets..."
    echo "  datasets:"
    echo "   - phoneme"
    echo "   - spambase"
    echo "   - covtype"
    echo "   - higgs"
    echo "   - ijcnn1"
    echo "   - mnist2v4"
    echo "   - fmnist2v4"
    echo "   - webspam"
    echo "   - calhouse"
fi
