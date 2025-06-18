from netpyne import sim  # import netpyne init module

# cfg, netParams = sim.loadFromIndexFile('index.npjson')
# read cfg and netParams from command line arguments if available; otherwise use default
simConfig, netParams = sim.readCmdLineArgs(simConfigDefault='src/cfg.py', netParamsDefault='src/params.py')
sim.createSimulateAnalyze(netParams = netParams, simConfig = cfg)  # create and simulate network