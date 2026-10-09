#!/bin/bash
#SBATCH --job-name=extract-bones
#SBATCH --partition=drl-titanrtx
#SBATCH --qos=drl-main
#SBATCH --account=drl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

cd /data/scratch/cgoyal/bones-seed-download

echo "Extracting g1.tar.gz ..."
tar -xzf g1.tar.gz

echo ""
echo "=== CSV count ==="
find . -iname "*.csv" | wc -l

echo ""
echo "=== Date folders found ==="
ls g1/csv/

echo ""
echo "=== Sample of first date folder ==="
first_date=$(ls g1/csv/ | head -1)
ls g1/csv/"$first_date" | head -20
