#!/bin/bash
# Contoh batch script SLURM untuk menjalankan simulasi di cluster HPC kampus/regional.
# Kirim dengan:  sbatch job.sbatch     | pantau:  squeue -u $USER
#SBATCH --job-name=asap-sim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

module load python 2>/dev/null || true   # sesuaikan dengan modul di cluster Anda
python3 -m venv .venv && source .venv/bin/activate
pip install -r ../requirements.txt
python main.py
