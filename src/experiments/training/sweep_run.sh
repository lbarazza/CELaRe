#!/bin/bash
# Leonardo Barazza, acse-lb1223

# Get the directory of the script and move three levels up
BASE_DIR=$(dirname $(dirname $(dirname $(realpath $0))))
cd $BASE_DIR

echo "Running sweep run script from $BASE_DIR"

if [ $# -lt 5 ]; then
  echo "Usage: $0 <--c|--d> <num_workers> <entity_name> <project_name> <sweep_id>"
  exit 1
fi

MODE=$1
NUM_AGENTS=$2
ENTITY_NAME=$3
PROJECT_NAME=$4
SWEEP_ID=$5

if [ "$MODE" = "--c" ]; then
  FILE_NAME="train_continuous"
  FUNCTION_NAME="run_continuous_sweep"
elif [ "$MODE" = "--d" ]; then
  FILE_NAME="train_discrete"
  FUNCTION_NAME="run_discrete_sweep"
else
  echo "Invalid mode specified. Use --c for continuous or --d for discrete."
  exit 1
fi

# Start parallel agents to run the sweep
for i in $(seq 1 $NUM_AGENTS); do
  echo "Starting agent $i with sweep ID $SWEEP_ID using $FUNCTION_NAME"
  nohup python -c "import wandb; from experiments.training.$FILE_NAME import $FUNCTION_NAME; wandb.agent('$SWEEP_ID', function=$FUNCTION_NAME, project='$PROJECT_NAME', entity='$ENTITY_NAME')" > agent_$i.log 2>&1 &
  echo "Agent $i started with PID $!"
done

wait
