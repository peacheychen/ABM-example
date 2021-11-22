import numpy as np
import networkx as nx

class agent:
    def __init__(self, eps):
        self.opinion = np.random.uniform(0, 1)
        self.epsilon = eps


class opinion_diffusion:
    def __init__(self, param):
    # param: 0: total number of agents, 1: market duration, 2: mu
        self.N = param[0]
        self.T = param[1]
        self.mu = param[2]

        self.G = nx.barabasi_albert_graph(self.N, 4, 64)
        self.ag_ids = []

        self.curr_day = 0
        self.ag_ids_len = 0

    # select a pair of neighbors
    def select_pair(self):
        node = np.random.randint(0, len(self.G.nodes()))
        pair = np.random.randint(0, len(self.G.edges(node)))
        self.id1 = list(self.G.edges(node))[pair][0]
        self.id2 = list(self.G.edges(node))[pair][1]

    # update the opinion of a selected pair
    def update_opinion(self, ag):
        delta_op = ag[self.id1].opinion - ag[self.id2].opinion
        if np.abs(delta_op) < ag[self.id1].epsilon:
            ag[self.id1].opinion = ag[self.id1].opinion - self.mu * delta_op
            ag[self.id2].opinion = ag[self.id2].opinion + self.mu * delta_op
        return ag

    # pick agents to participate in the market
    def pick_agents(self):
        probs = np.random.uniform(0,1,self.N)
        p = (self.T-self.curr_day+1)**(-2.44) + 0.01
        #p = 1.1
        self.ag_ids = [i for i, j in enumerate(probs) if j < p]

    # update temporal
    def update_op_series(self, i, ag):
        self.curr_day = np.ceil(np.float(i)/(self.N*0.5))

        self.pick_agents()
        self.ag_ids_len = np.append(self.ag_ids_len,len(self.ag_ids))

        return self.ag_ids

    # control function
    def launch(self, ag):
        self.select_pair()
        ag = self.update_opinion(ag)
        return ag


class prediction_market:
    def __init__(self,param):
        #create arrays
        self.pt = np.array([0.5])
        self.last_pt = self.pt[-1]
        #constant parameters
        self.beta = param[0]
        self.gamma = param[1]
        self.ED = 0
        self.temp_dem = 0
        self.ag_part = 0

    # update price
    def update_price(self):
        self.last_pt = self.last_pt + np.round(self.ED, 2)
        if self.last_pt <= 0.01: self.last_pt = 0.01
        if self.last_pt >= 0.99: self.last_pt = 0.99


    def update_demand(self, ag_idx, agt, net):
        D = 0
        ag_part = 0
        for i in ag_idx:
            D_temp = agt[i].opinion - self.last_pt * self.gamma
            if D_temp != 0:
                D += (agt[i].opinion - self.last_pt) * self.gamma
                ag_part += 1
        self.ag_part = np.append(self.ag_part, ag_part)
        self.ED = (D*np.random.normal(0,0.05))
        self.temp_dem = np.append(self.temp_dem, self.ED)

    def update_price_series(self):
        self.pt = np.append(self.pt, self.last_pt)


    # control function
    def launch(self, ag_idx, net, ag):
        self.update_demand(ag_idx, ag, net)
        self.update_price()
        self.update_price_series()