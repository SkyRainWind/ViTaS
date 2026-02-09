#!/bin/bash

# bash train.sh env=ENV_NAME use_algo=ALGO_NAME
# ENV_NAME: tactile_envs/Insertion-v0(default), Door, Lift, LiftCan, LiftCap, HandManipulateBlockRotateZFixed-v1, HandManipulateEggRotateFixed-v1, HandManipulatePenRotateFixed-v1
# ALGO_NAME: vitas(default), vtt, mae, poe, concat
# Example usage:
# bash train.sh env=tactile_envs/Insertion-v0 use_algo=vitas

# --- Script Safety Settings ---
# set -e: Exit immediately if a command exits with a non-zero status.
# set -u: Treat unset variables as an error when substituting.
# set -o pipefail: The pipeline's return status is the value of the last command to exit with a non-zero status.
set -e
set -u
set -o pipefail

# --- Default Parameters ---
ENV_NAME="tactile_envs/Insertion-v0"
ALGO_NAME="vitas"

# --- Parse key=value Arguments ---
# Loop through all arguments provided by the user.
for ARGUMENT in "$@"
do
   # Split the argument by the '=' delimiter.
   KEY=$(echo $ARGUMENT | cut -f1 -d=)
   VALUE=$(echo $ARGUMENT | cut -f2 -d=)

   # Assign the VALUE to the corresponding variable based on the KEY.
   case "$KEY" in
           env)              ENV_NAME=${VALUE} ;;
           use_algo)         ALGO_NAME=${VALUE} ;;
           *)
   esac
done

# --- Display Configuration and Execute ---
echo "🔧 Final Configuration:"
echo "   - Environment (env): ${ENV_NAME}"
echo "   - Algorithm (use_algo): ${ALGO_NAME}"
echo ""
echo "🚀 Executing command..."
echo "python train.py --env ${ENV_NAME} --use_algo ${ALGO_NAME}"
echo "----------------------------------------------------"

# Execute the Python training script.
python train.py --env "${ENV_NAME}" --use_algo "${ALGO_NAME}"

echo "----------------------------------------------------"
echo "✅ Training script finished."
