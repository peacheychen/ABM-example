# import predictionMarketsClasses as pmcl
import ABM_classes as pmcl
import numpy as np
# import cPickle as pickle
import pickle
import sys


def main():
    # specify the parameters for the simulation
    N_days = 50
    # assign the model parameters to the variables.
    # param[0]: mu
    # param[1]: epsilon
    param = [0.5, 0.5]

    N_agents = 100

    # initialize parameters for a new simulation
    N_loops = np.int64((N_agents*N_days)*0.5)
    op_param = [N_agents, N_days, param[0]]
    pm_param = [1, 1]
    # initialize the agents
    agents = [pmcl.agent(param[1]) for i in range(N_agents)]
    # initialize other classes
    network = pmcl.opinion_diffusion(op_param)
    market = pmcl.prediction_market(pm_param)
    # start the market
    for i in range(N_loops):
        agents = network.launch(agents)
        # market.launch(ag_id, network, agents)
        if i % np.int64(network.N*0.5) == 0:
            # update opinion
            ag_id = network.update_op_series(i, agents)
            market.launch(ag_id, network, agents)

if __name__ == "__main__":
    main()
