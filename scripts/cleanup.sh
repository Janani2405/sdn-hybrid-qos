#!/bin/bash
sudo mn -c
sudo ovs-vsctl del-controller s1 s2
sudo ovs-vsctl del-br s1; sudo ovs-vsctl del-br s2
```

`.gitignore` should include:
```
logs/*.log
__pycache__/
*.pyc
*.swp