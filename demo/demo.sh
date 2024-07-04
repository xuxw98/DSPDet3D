DATA=$1
CONFIG=$2
python $(dirname "$0")/ply2bin.py \
    --data-dir $DATA\


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/demo.py \
    $DATA\
    $CONFIG \
    $(dirname $0)/dspdet3d_demo.pth \
    --out-dir $(dirname $0)/demo_results