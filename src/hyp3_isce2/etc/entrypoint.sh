#!/bin/bash --login
set -e
conda activate hyp3-isce2
exec python -um hyp3_isce2 "$@"
