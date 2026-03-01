#!/bin/bash
# start_qos_controller.sh
# Runs the merged QoS controller — does BOTH at once:
#   [1] Data collection  → logs/qos_log.csv + logs/congestion_log.csv
#   [2] Visualization    → http://127.0.0.1:8080/qos/api/v1  (dashboard)

source ~/ryu-env/bin/activate
cd ~/sdn-project

ryu-manager \
    controller/qos_controller.py \
    --observe-links \
    --ofp-tcp-listen-port 6633