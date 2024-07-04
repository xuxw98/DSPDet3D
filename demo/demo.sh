DATA=$1
CONFIG=$2
CHECKPOINT=$3


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/demo.py \
    $DATA\
    $CONFIG \
    $CHECKPOINT \
    --out-dir $(dirname $0)/demo_results