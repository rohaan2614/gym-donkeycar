#!/usr/bin/env bash

# ---------- config ----------
NUM_RUNS=4
BASE_PORT=9090
SIM_PATH="/home/rn7823/projects/DonkeySimLinux/donkey_sim.x86_64"
ENV_NAME="donkey-warren-track-v0"
TIMESTEPS=1000000 # 1 Million
LOG_DIR="output/$(date +%Y%m%d_%H%M%S)"
MODEL_PREFIX=Vulture
# ----------------------------

mkdir -p "$LOG_DIR"

echo "Starting $NUM_RUNS parallel PPO runs"
echo "Terminal Outputs will be saved in: $LOG_DIR"
echo "-----------------------------------"

for i in $(seq 1 $NUM_RUNS); do
  PORT=$((BASE_PORT + i))
  RUN_NAME="${MODEL_PREFIX}_${i}"
  LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

  echo "Launching $RUN_NAME on port $PORT"

  python examples/reinforcement_learning/ppo_train.py \
    --sim "$SIM_PATH" \
    --env_name "$ENV_NAME" \
    --training_timesteps "$TIMESTEPS" \
    --eval_freq -1 \
    --run-name "$RUN_NAME" \
    --port "$PORT" \
    --early_stopping_threshold 800 \
    > "$LOG_FILE" 2>&1 &

done

wait
echo "All runs completed."