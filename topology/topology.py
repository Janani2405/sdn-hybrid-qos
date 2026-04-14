#!/usr/bin/env python
"""
Mininet SDN Hybrid-Mesh Topology — depth=5, 32 hosts, 31 switches
==================================================================
Binary tree (fanout=2, depth=5) with sibling cross-links at L1/L2/L3.

Structure:
  L0: s1 (root)
  L1: s2, s3
  L2: s4, s5, s6, s7
  L3: s8–s15
  L4: s16–s31 (leaf — 2 hosts each)
  Hosts: h1–h32 (32 hosts)

Cross-links (mesh, always port 2):
  L1: s2↔s3
  L2: s4↔s5, s6↔s7
  L3: s8↔s9, s10↔s11, s12↔s13, s14↔s15

Usage:
  source ~/ryu-env/bin/activate && cd ~/sdn-project
  PYTHONPATH=. ryu-manager controller/qos_controller.py --observe-links --ofp-tcp-listen-port 6633
  # then:
  sudo python3 topology/topology.py --ryu
"""

import sys, time
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch, Controller
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI
from mininet.topo import Topo

LINK_OPTS = dict(bw=100, delay='2ms', loss=0, max_queue_size=1000, use_htb=True)

# Port 2 on every non-leaf switch is the sibling cross-link.
# Leaf switches (s16–s31) have no cross-links — only hosts.
MESH_PORTS = {
    's1': [],
    's2': [2], 's3': [2],
    's4': [2], 's5': [2], 's6': [2], 's7': [2],
    's8': [2], 's9': [2], 's10': [2], 's11': [2],
    's12': [2], 's13': [2], 's14': [2], 's15': [2],
    's16': [], 's17': [], 's18': [], 's19': [],
    's20': [], 's21': [], 's22': [], 's23': [],
    's24': [], 's25': [], 's26': [], 's27': [],
    's28': [], 's29': [], 's30': [], 's31': [],
}


class SDNHybridMeshTopo(Topo):
    def build(self):
        def sw(name):
            return self.addSwitch(name, cls=OVSSwitch, protocols='OpenFlow13')
        def host(name, n):
            return self.addHost(name, ip=f'10.0.0.{n}/24',
                                mac=f'00:00:00:00:00:{n:02x}')
        def link(a, b):
            self.addLink(a, b, **LINK_OPTS)

        # L0
        s1 = sw('s1')

        # L1
        s2 = sw('s2'); s3 = sw('s3')
        link(s1, s2); link(s1, s3)
        link(s2, s3)                       # CROSS s2↔s3

        # L2
        s4 = sw('s4'); s5 = sw('s5'); s6 = sw('s6'); s7 = sw('s7')
        link(s2, s4); link(s2, s5); link(s4, s5)   # CROSS s4↔s5
        link(s3, s6); link(s3, s7); link(s6, s7)   # CROSS s6↔s7

        # L3
        s8  = sw('s8');  s9  = sw('s9')
        s10 = sw('s10'); s11 = sw('s11')
        s12 = sw('s12'); s13 = sw('s13')
        s14 = sw('s14'); s15 = sw('s15')
        link(s4, s8);   link(s4, s9);   link(s8,  s9)   # CROSS s8↔s9
        link(s5, s10);  link(s5, s11);  link(s10, s11)  # CROSS s10↔s11
        link(s6, s12);  link(s6, s13);  link(s12, s13)  # CROSS s12↔s13
        link(s7, s14);  link(s7, s15);  link(s14, s15)  # CROSS s14↔s15

        # L4 — leaf switches + hosts (2 per leaf switch)
        def leaf(sw_name, sw_parent, h1n, h2n):
            s = sw(sw_name)
            link(sw_parent, s)
            link(host(f'h{h1n}', h1n), s)
            link(host(f'h{h2n}', h2n), s)
            return s

        # s8 subtree
        s16 = leaf('s16', s8,  1,  2); s17 = leaf('s17', s8,  3,  4)
        # s9 subtree
        s18 = leaf('s18', s9,  5,  6); s19 = leaf('s19', s9,  7,  8)
        # s10 subtree
        s20 = leaf('s20', s10, 9,  10); s21 = leaf('s21', s10, 11, 12)
        # s11 subtree
        s22 = leaf('s22', s11, 13, 14); s23 = leaf('s23', s11, 15, 16)
        # s12 subtree
        s24 = leaf('s24', s12, 17, 18); s25 = leaf('s25', s12, 19, 20)
        # s13 subtree
        s26 = leaf('s26', s13, 21, 22); s27 = leaf('s27', s13, 23, 24)
        # s14 subtree
        s28 = leaf('s28', s14, 25, 26); s29 = leaf('s29', s14, 27, 28)
        # s15 subtree
        s30 = leaf('s30', s15, 29, 30); s31 = leaf('s31', s15, 31, 32)


def block_mesh_loops(net):
    info('*** Installing loop-prevention DROP rules\n')
    for sw in net.switches:
        ports = MESH_PORTS.get(sw.name, [])
        if not ports:
            continue
        sock = f'unix:/var/run/openvswitch/{sw.name}.mgmt'
        for p in ports:
            sw.cmd(f'ovs-ofctl -O OpenFlow13 add-flow {sock} '
                   f'"priority=10,in_port={p},dl_dst=ff:ff:ff:ff:ff:ff,actions=drop" 2>&1')
            sw.cmd(f'ovs-ofctl -O OpenFlow13 add-flow {sock} '
                   f'"priority=10,in_port={p},'
                   f'dl_dst=01:00:00:00:00:00/01:00:00:00:00:00,actions=drop" 2>&1')
            info(f'    {sw.name} port {p}: broadcast/multicast BLOCKED\n')
    info('*** Loop prevention complete\n')


def install_l2_flood_flows(net):
    info('*** Installing table-miss flood flows (standalone mode)\n')
    for sw in net.switches:
        sock = f'unix:/var/run/openvswitch/{sw.name}.mgmt'
        sw.cmd(f'ovs-ofctl -O OpenFlow13 add-flow {sock} "priority=0,actions=flood" 2>&1')


def configure_queues(net):
    info('*** Configuring OVS QoS queues\n')
    for switch in net.switches:
        for intf in switch.intfList():
            if intf.name == 'lo':
                continue
            switch.cmd(f'ovs-vsctl -- destroy QoS {intf.name} 2>/dev/null; true')
            switch.cmd(
                f'ovs-vsctl set port {intf.name} qos=@newqos -- '
                f'--id=@newqos create qos type=linux-htb '
                f'queues=0=@q0,1=@q1,2=@q2 -- '
                f'--id=@q0 create queue other-config:max-rate=100000000 -- '
                f'--id=@q1 create queue '
                f'other-config:min-rate=50000000 other-config:max-rate=100000000 -- '
                f'--id=@q2 create queue other-config:max-rate=20000000'
            )
    info('*** Queue configuration complete\n')


def set_openflow13(net, ctrl_ip, ctrl_port):
    for sw in net.switches:
        sw.cmd('ovs-vsctl set bridge', sw.name, 'protocols=OpenFlow13')
        sw.cmd('ovs-vsctl set-controller', sw.name, f'tcp:{ctrl_ip}:{ctrl_port}')


def wait_for_ryu(net, timeout=30):
    info(f'*** Waiting up to {timeout}s for all {len(net.switches)} switches to connect...\n')
    n = len(net.switches)
    deadline = time.time() + timeout
    while time.time() < deadline:
        connected = net.switches[0].cmd('ovs-vsctl show').count('is_connected: true')
        if connected >= n:
            info(f'    All {n} switches connected ✓\n')
            return True
        info(f'    {connected}/{n} connected, waiting...\n')
        time.sleep(2)
    info(f'    WARNING: timed out\n')
    return False


def run():
    setLogLevel('info')
    use_ryu = '--ryu' in sys.argv

    net = Mininet(
        topo=SDNHybridMeshTopo(),
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=False,
        autoStaticArp=True,
        waitConnected=True,
    )

    CTRL_IP, CTRL_PORT = '127.0.0.1', 6633
    if use_ryu:
        net.addController('ryu', controller=RemoteController,
                          ip=CTRL_IP, port=CTRL_PORT)
    else:
        c0 = net.addController('c0', controller=Controller)
        CTRL_IP, CTRL_PORT = c0.IP(), c0.port

    net.start()
    set_openflow13(net, CTRL_IP, CTRL_PORT)

    if use_ryu:
        wait_for_ryu(net)
        info('*** Ryu is managing flows\n')
    else:
        install_l2_flood_flows(net)

    block_mesh_loops(net)
    configure_queues(net)

    info('\n*** Topology ready.\n')
    info(f'    Switches : {len(net.switches)} (s1–s31) [Binary hybrid mesh, depth=5]\n')
    info(f'    Hosts    : {len(net.hosts)} (h1–h32, 2 per leaf switch)\n')
    info(f'    Controller: Ryu @ {CTRL_IP}:{CTRL_PORT}\n' if use_ryu
         else f'    Controller: Built-in c0\n')
    info('    ARP      : static (no ARP floods)\n')
    info('    Cross-links: port 2 on s2–s15 (broadcast DROPped)\n')
    info('\n    pingall:       mininet> pingall\n')
    info('    iperf h1→h32:  mininet> iperf h1 h32\n\n')

    CLI(net)

    for switch in net.switches:
        for intf in switch.intfList():
            if intf.name != 'lo':
                switch.cmd(f'ovs-vsctl destroy QoS {intf.name} 2>/dev/null; true')
                switch.cmd(f'ovs-vsctl clear port {intf.name} qos 2>/dev/null; true')

    net.stop()


if __name__ == '__main__':
    run()