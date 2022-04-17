import Coursework2_Classes as pmcl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def main():

    # specify the parameters for the simulation
    start_date = '2020-12-1'
    start_date1 = datetime.strptime(str(start_date), '%Y-%m-%d')
    end_date = '2021-2-4'
    end_date1 = datetime.strptime(str(end_date), '%Y-%m-%d')
    N_days = (end_date1 - start_date1).days

    # assign the model parameters to the variables.
    # param[0]: mu
    # param[1]: epsilon

    param = [0.5, 0.5]

    N_agents_H = 30
    N_agents_T = 70
    N_agents = N_agents_H + N_agents_T

    n = N_agents  # number of agents
    t = N_days  # number of rounds
    r = 0.0008050  # risk-free interest rate
    risk_aversion = 1  # risk aversion coefficient
    risk_preference = np.random.uniform(0, 0.1, n)
    price = 17.11  # open price on Dec 01
    variance_of_return = 0.789064  # variance of stock
    price_t1 = np.random.normal(17.11, 3, n)
    action = np.zeros(n)
    rational_order_price = np.zeros(n)  # prices of current order for each agent
    rational_current_amount = np.random.randint(100, 500, n)

    '''Irrational Network Initialization'''
    max_stock_demand = 50
    order_constant = 0.3
    data = pd.read_csv("C:/Users/hp/Desktop/GME.csv")
    col_names = ["Volume"]
    data = data[col_names]
    GME_volume_array = data.values
    # number of agents added at each round proportional to the volume traded that day (DEC 01 - Feb 4)
    add_agents_sequence = [x / 10000000 for x in GME_volume_array]
    add_agents_sequence = [int(x) for x in add_agents_sequence]

    pmcl.do_prediction(n, t, r, risk_aversion, risk_preference, price, variance_of_return, price_t1,
                                        action, rational_order_price, rational_current_amount,
                                        max_stock_demand, order_constant, add_agents_sequence)

    # initialize parameters for a new simulation
    N_loops = np.int64((N_agents * N_days) * 0.5)
    op_param = [N_agents, N_days, param[0]]
    pm_param = [1, 1]
    # initialize the agents
    agents_H = [pmcl.hedge_fund(param[1]) for i in range(N_agents_H)]
    agents_T = [pmcl.reddit_trader(param[1]) for i in range(N_agents_T)]
    agents = agents_H + agents_T
    # initialize other classes
    network = pmcl.opinion_diffusion(op_param)
    # market = pmcl.prediction_market(pm_param)

    opinion_H_single = []
    opinion_T_single = []
    epsilon_H_single = []
    epsilon_T_single = []

    # start the market
    for i in range(N_loops):
        agents = network.launch(agents)
        opinion_H_single.append(agents[0].opinion)
        opinion_T_single.append(agents[-1].opinion)

        # # market.launch(ag_id, network, agents)
        # if i % np.int64(network.N * 0.5) == 0:
        #     # update opinion
        #     ag_id = network.update_op_series(i, agents)
        #     market.launch(ag_id, network, agents)

    plt.plot(opinion_H_single, color="r", linestyle="-", marker=".", linewidth=1, label='hedge')
    # plt.plot(opinion_T_single, color="b", linestyle="-", marker="s", linewidth=1,label='normal')
    plt.legend()
    plt.xlabel("day")
    plt.ylabel("opinion")
    plt.title('Profit Change Curve')
    plt.show()

main()
