"""
=============================================================================
eval_dqn.py  —  Evaluate trained DQN agent (no exploration)
=============================================================================
Loads the saved checkpoint and runs the agent with epsilon=0.0
(pure exploitation) for N steps, printing per-switch decisions.

Run AFTER training:
    python3 eval_dqn.py --steps 50

Shows:
  • What action the agent chose per switch per step
  • Real reward received
  • Whether congestion was present
=============================================================================
"""

import time
import argparse
import logging
import sys
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  [EVAL]  %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
log = logging.getLogger('eval')

ACTION_NAMES = {
    0: 'do_nothing',
    1: 'reroute   ',
    2: 'throttle  ',
    3: 'prioritise',
    4: 'reset     ',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',      type=int,   default=100)
    parser.add_argument('--controller', type=str,   default='http://127.0.0.1:8080')
    parser.add_argument('--no-enforce', action='store_true',
                        help='Observe-only mode — do not send flow rules')
    args = parser.parse_args()

    # Import agent with eval epsilon
    import dqn_agent as da
    da.RYU_BASE       = args.controller
    da.RYU_QOS_BASE   = f'{args.controller}/qos/api/v1'
    da.RYU_STATS_BASE = f'{args.controller}/stats'

    agent         = da.DQNAgent()
    agent.epsilon = 0.0   # pure exploitation
    log.info('Eval mode  epsilon=0.0  steps=%d', args.steps)

    if not da.CKPT_PATH.exists():
        log.error('No checkpoint found at %s — train first!', da.CKPT_PATH)
        return

    print('\n' + '='*70)
    print(f'{"Step":>5}  {"Switch":>20}  {"Zone":>10}  {"Action":>12}  '
          f'{"Reward":>8}  {"Cong%":>6}')
    print('='*70)

    for step in range(1, args.steps + 1):
        t0     = time.time()
        states = da.fetch_state()
        rews   = da.fetch_rewards()

        if not states:
            log.warning('No state — waiting...')
            time.sleep(da.POLL_INTERVAL)
            continue

        for dpid_str, state in states.items():
            action  = agent.select_action(state, dpid_str)
            reward  = rews.get(dpid_str, 0.0)
            cong_pct = state[11] * 100.0   # n_ports_congested_ratio

            # Determine zone label from state
            if state[0] > 0.7:
                zone = 'CRITICAL'
            elif state[0] > 0.4:
                zone = 'congested'
            elif state[2] > 0.7:
                zone = 'warning'
            else:
                zone = 'normal'

            dpid_short = dpid_str[-4:]   # last 4 hex chars
            print(f'{step:>5}  {dpid_short:>20}  {zone:>10}  '
                  f'{ACTION_NAMES[action]:>12}  {reward:>8.3f}  {cong_pct:>5.1f}%')

            if not args.no_enforce:
                da.enforce_action(dpid_str, action)

        elapsed = time.time() - t0
        time.sleep(max(0.0, da.POLL_INTERVAL - elapsed))

    print('='*70)
    log.info('Evaluation complete.')


if __name__ == '__main__':
    main()
