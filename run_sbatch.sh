sbatch -A plgzzsn2024-gpu-a100 -o slurm_%a.log -p plgrid-gpu-a100 -t 360 -c 4 --gres gpu:1 --mem 40G --nodes 1 $1
