#!/usr/bin/env python
"""
Mininet SDN Topology with TCLink (100 Mbps), Ryu Remote Controller,
OpenFlow 1.3, and Queue Support for Congestion Control.

Usage:
    sudo python topology.py

Prerequisites:
    - Mininet installed
    - Ryu controller running:  ryu-manager ryu.app.simple_switch_13
    - Or with REST API:        ryu-manager ryu.app.rest_qos ryu.app.rest_conf_switch
"""

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI
from mininet.topo import Topo


# ─────────────────────────────────────────────
#  Topology Definition
# ─────────────────────────────────────────────

class SDNTopo(Topo):
    """
    Two-switch linear topology with 4 hosts and queue-enabled TCLinks.

        h1 ─┐                    ┌─ h3
             s1 ── (100 Mbps) ── s2
        h2 ─┘                    └─ h4

    Link parameters:
        bw      = 100 Mbps      (bandwidth cap via HTB)
        delay   = 2ms           (propagation delay)
        loss    = 0%            (no artificial loss)
        max_queue_size = 1000   (packets; controls burst/congestion)
    """

    LINK_OPTS = dict(
        bw=100,             # Mbps — enforced via tc HTB
        delay='2ms',        # propagation delay
        loss=0,             # packet loss percentage
        max_queue_size=1000,  # queue depth (packets)
        use_htb=True,       # Hierarchical Token Bucket (required for bw)
    )

    def build(self):
        # ── Switches ──────────────────────────────────────────────────────
        s1 = self.addSwitch('s1', cls=OVSSwitch, protocols='OpenFlow13')
        s2 = self.addSwitch('s2', cls=OVSSwitch, protocols='OpenFlow13')

        # ── Hosts ─────────────────────────────────────────────────────────
        h1 = self.addHost('h1', ip='10.0.0.1/24', mac='00:00:00:00:00:01')
        h2 = self.addHost('h2', ip='10.0.0.2/24', mac='00:00:00:00:00:02')
        h3 = self.addHost('h3', ip='10.0.0.3/24', mac='00:00:00:00:00:03')
        h4 = self.addHost('h4', ip='10.0.0.4/24', mac='00:00:00:00:00:04')

        # ── Host ↔ Switch links (100 Mbps, queued) ────────────────────────
        self.addLink(h1, s1, **self.LINK_OPTS)
        self.addLink(h2, s1, **self.LINK_OPTS)
        self.addLink(h3, s2, **self.LINK_OPTS)
        self.addLink(h4, s2, **self.LINK_OPTS)

        # ── Inter-switch link (100 Mbps, queued) ──────────────────────────
        self.addLink(s1, s2, **self.LINK_OPTS)


# ─────────────────────────────────────────────
#  Queue Configuration Helper
# ─────────────────────────────────────────────

def configure_queues(net):
    """
    Attach OVS QoS queues to every switch port so the Ryu QoS REST app
    can later assign per-flow queues via queue_id.

    Queue layout per port:
        queue 0  – default / best-effort  (max 100 Mbps)
        queue 1  – priority traffic       (guaranteed 50 Mbps, max 100 Mbps)
        queue 2  – bulk / background      (max 20 Mbps)
    """
    import subprocess

    info('*** Configuring OVS QoS queues\n')

    for switch in net.switches:
        for intf in switch.intfList():
            if intf.name == 'lo':
                continue

            # Remove any existing QoS/queue entries first
            switch.cmd(f'ovs-vsctl -- destroy QoS {intf.name} 2>/dev/null; true')

            # Create queues with ovs-vsctl
            cmd = (
                f'ovs-vsctl set port {intf.name} '
                f'qos=@newqos -- '
                f'--id=@newqos create qos type=linux-htb '
                f'queues=0=@q0,1=@q1,2=@q2 -- '
                f'--id=@q0 create queue other-config:max-rate=100000000 -- '
                f'--id=@q1 create queue '
                f'other-config:min-rate=50000000 '
                f'other-config:max-rate=100000000 -- '
                f'--id=@q2 create queue other-config:max-rate=20000000'
            )
            result = switch.cmd(cmd)
            if result.strip():
                info(f'    {intf.name}: {result.strip()}\n')

    info('*** Queue configuration complete\n')


# ─────────────────────────────────────────────
#  Bandwidth Verification Helper
# ─────────────────────────────────────────────

def verify_links(net):
    """Print tc qdisc / class info for each switch interface."""
    info('\n*** Verifying TC link shaping\n')
    for switch in net.switches:
        for intf in switch.intfList():
            if intf.name == 'lo':
                continue
            out = switch.cmd(f'tc qdisc show dev {intf.name}')
            info(f'  [{intf.name}] {out.strip()}\n')


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def run():
    setLogLevel('info')

    topo = SDNTopo()

    info('*** Creating network\n')
    net = Mininet(
        topo=topo,
        controller=None,          # we add it manually below
        switch=OVSSwitch,
        link=TCLink,              # ← enforces bw / delay / loss / queue
        autoSetMacs=False,
        autoStaticArp=False,
        waitConnected=True,
    )

    # ── Remote Ryu Controller ─────────────────────────────────────────────
    info('*** Adding remote Ryu controller (127.0.0.1:6633)\n')
    ryu = net.addController(
        'ryu',
        controller=RemoteController,
        ip='127.0.0.1',
        port=6633,
    )

    info('*** Starting network\n')
    net.start()

    # Force OpenFlow 1.3 on all switches (belt-and-suspenders)
    info('*** Setting OpenFlow 1.3 on all switches\n')
    for switch in net.switches:
        switch.cmd('ovs-vsctl set bridge', switch.name,
                   'protocols=OpenFlow13')
        switch.cmd('ovs-vsctl set-controller', switch.name,
                   'tcp:127.0.0.1:6633')
        info(f'    {switch.name} → OpenFlow13\n')

    # ── Queue setup ───────────────────────────────────────────────────────
    configure_queues(net)

    # ── Quick sanity check ────────────────────────────────────────────────
    verify_links(net)

    info('\n*** Topology ready.\n')
    info('    Hosts : h1(10.0.0.1)  h2(10.0.0.2)  h3(10.0.0.3)  h4(10.0.0.4)\n')
    info('    Switches: s1, s2  (OpenFlow 1.3)\n')
    info('    Controller: Ryu @ 127.0.0.1:6633\n')
    info('    Link BW: 100 Mbps (TCLink / HTB)  |  Queues: 0=BestEffort, 1=Priority, 2=Bulk\n')
    info('\n    Test bandwidth:  mininet> iperf h1 h3\n')
    info('    Ping all:        mininet> pingall\n\n')

    CLI(net)

    info('*** Cleaning up queues\n')
    for switch in net.switches:
        for intf in switch.intfList():
            if intf.name != 'lo':
                switch.cmd(f'ovs-vsctl destroy QoS {intf.name} 2>/dev/null; true')
                switch.cmd(f'ovs-vsctl clear port {intf.name} qos 2>/dev/null; true')

    info('*** Stopping network\n')
    net.stop()


if __name__ == '__main__':
    run()