#!/usr/bin/env python
"""
Mininet SDN Tree Topology with TCLink (100 Mbps), Ryu Remote Controller,
OpenFlow 1.3, and Queue Support for Congestion Control.

Tree: fanout=2, depth=3  →  7 switches, 8 hosts

                    s1
              ┌──────┴──────┐
             s2             s3
          ┌───┴───┐       ┌───┴───┐
         s4      s5      s6      s7
       ┌─┴─┐  ┌─┴─┐  ┌─┴─┐  ┌─┴─┐
      h1  h2 h3  h4 h5  h6 h7  h8

Usage:
    sudo python topology.py

Prerequisites:
    - Mininet installed
    - Ryu controller running:  ryu-manager controller/qos_controller.py --observe-links
"""

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI
from mininet.topo import Topo


# ─────────────────────────────────────────────
#  Link options  (same as original)
# ─────────────────────────────────────────────

LINK_OPTS = dict(
    bw=100,               # Mbps — enforced via tc HTB
    delay='2ms',          # propagation delay
    loss=0,               # packet loss percentage
    max_queue_size=1000,  # queue depth (packets)
    use_htb=True,         # Hierarchical Token Bucket (required for bw)
)


# ─────────────────────────────────────────────
#  Topology Definition
#  fanout=2, depth=3 → 7 switches, 8 hosts
#  Built manually (TreeTopo not in Mininet 2.3.x)
# ─────────────────────────────────────────────

class SDNTreeTopo(Topo):
    def build(self):
        # ── Layer 1: root switch ──────────────────────────────────────────
        s1 = self.addSwitch('s1', cls=OVSSwitch, protocols='OpenFlow13')

        # ── Layer 2: 2 aggregation switches ──────────────────────────────
        s2 = self.addSwitch('s2', cls=OVSSwitch, protocols='OpenFlow13')
        s3 = self.addSwitch('s3', cls=OVSSwitch, protocols='OpenFlow13')
        self.addLink(s1, s2, **LINK_OPTS)
        self.addLink(s1, s3, **LINK_OPTS)

        # ── Layer 3: 4 edge switches ──────────────────────────────────────
        s4 = self.addSwitch('s4', cls=OVSSwitch, protocols='OpenFlow13')
        s5 = self.addSwitch('s5', cls=OVSSwitch, protocols='OpenFlow13')
        s6 = self.addSwitch('s6', cls=OVSSwitch, protocols='OpenFlow13')
        s7 = self.addSwitch('s7', cls=OVSSwitch, protocols='OpenFlow13')
        self.addLink(s2, s4, **LINK_OPTS)
        self.addLink(s2, s5, **LINK_OPTS)
        self.addLink(s3, s6, **LINK_OPTS)
        self.addLink(s3, s7, **LINK_OPTS)

        # ── 8 hosts — 2 per edge switch ───────────────────────────────────
        for i, sw in enumerate([s4, s5, s6, s7]):
            for j in range(2):
                h_num = i * 2 + j + 1
                h = self.addHost(
                    f'h{h_num}',
                    ip=f'10.0.0.{h_num}/24',
                    mac=f'00:00:00:00:00:{h_num:02d}',
                )
                self.addLink(h, sw, **LINK_OPTS)


# ─────────────────────────────────────────────
#  Queue Configuration Helper  (unchanged)
# ─────────────────────────────────────────────

def configure_queues(net):
    """
    Attach OVS QoS queues to every switch port.

    Queue layout per port:
        queue 0  – default / best-effort  (max 100 Mbps)
        queue 1  – priority traffic       (guaranteed 50 Mbps, max 100 Mbps)
        queue 2  – bulk / background      (max 20 Mbps)
    """
    info('*** Configuring OVS QoS queues\n')

    for switch in net.switches:
        for intf in switch.intfList():
            if intf.name == 'lo':
                continue

            switch.cmd(f'ovs-vsctl -- destroy QoS {intf.name} 2>/dev/null; true')

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
#  Bandwidth Verification Helper  (unchanged)
# ─────────────────────────────────────────────

def verify_links(net):
    """Print tc qdisc info for each switch interface."""
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

    topo = SDNTreeTopo()

    info('*** Creating network\n')
    net = Mininet(
        topo=topo,
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=False,
        autoStaticArp=False,
        waitConnected=True,
    )

    # ── Remote Ryu Controller ─────────────────────────────────────────────
    info('*** Adding remote Ryu controller (127.0.0.1:6633)\n')
    net.addController(
        'ryu',
        controller=RemoteController,
        ip='127.0.0.1',
        port=6633,
    )

    info('*** Starting network\n')
    net.start()

    # ── Force OpenFlow 1.3 on all switches ───────────────────────────────
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
    info(f'    Switches : {len(net.switches)} (s1–s{len(net.switches)})  '
         f'[Tree: fanout=2, depth=3]\n')
    host_info = '  '.join(f'{h.name}({h.IP()})' for h in net.hosts)
    info(f'    Hosts    : {len(net.hosts)} — {host_info}\n')
    info( '    Controller: Ryu @ 127.0.0.1:6633\n')
    info( '    Link BW  : 100 Mbps (TCLink / HTB)  |  '
          'Queues: 0=BestEffort, 1=Priority, 2=Bulk\n')
    info('\n    Test bandwidth:  mininet> iperf h1 h8\n')
    info('    Ping all:        mininet> pingall\n\n')

    CLI(net)

    # ── Cleanup queues on exit ────────────────────────────────────────────
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