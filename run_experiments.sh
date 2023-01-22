#!/bin/bash

NEED_HELP=1
M="xgb"
N=100
CACHE_DIR="cache"
RATIO=5
PARALLEL=""
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
        -r)
            RATIO="$2"
            shift
            shift
            ;;
        --pinacs)
            CACHE_DIR="cache_pinacs"
            N=500
            RATIO=4
            shift
            ;;
        --parallel)
            PARALLEL="bin/parallel"
            shift
            ;;
        *)
            NEED_HELP=0
            if [[ -z "$PARALLEL" ]]; then
                for fold in 0 1 2 3 4; do
                    echo
                    echo "$(date), fold $fold ========================"
                    python $SCRIPT \
                        --cache_dir $CACHE_DIR \
                        --fold $fold \
                        -N $N \
                        --ratio $RATIO \
                        --model_type $M $1

                    if [[ ! $? -eq 0 ]]; then 
                        echo "ERROR, quiting... $?"
                        echo " - dataset $1"
                        echo " - fold $fold"
                        echo " - N $N"
                        echo " - M $M"
                        exit 1
                    fi
                done
            else
                # run 5 folds in parallel
                seq 0 4 | \
                    $PARALLEL -j3 python $SCRIPT \
                        --cache_dir $CACHE_DIR \
                        -N $N \
                        --ratio $RATIO \
                        --model_type $M \
                        --fold {} $1
            fi
            shift
            ;;
    esac
done

if [[ $NEED_HELP -eq 1 ]]; then
    echo "$0 [-n num_adv_examples] [-m model_type] [-r ratio] [--pinacs] [--parallel] datasets..."
    echo "  datasets:"
    echo "   - calhouse"
    echo "   - electricity"
    echo "   - covtype"
    echo "   - higgs"
    echo "   - ijcnn1"
    echo "   - mnist2v4"
    echo "   - fmnist2v4"
    echo "   - webspam"
    echo "   - calhouse"
fi
