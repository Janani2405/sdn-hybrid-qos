#!/bin/bash
source ~/ryu-env/bin/activate
cd ~/sdn-project

python3 -c "
import eventlet
eventlet.monkey_patch()
" 2>/dev/null

ryu-manager \
    --observe-links \
    --ofp-tcp-listen-port 6633 \
    ryu.app.simple_switch_13 \
    controller/port_stats_monitor_v2.py