import numpy as np
import random
import networkx as nx
import math
import matplotlib.pyplot as plt

class hedge_fund:
    def __init__(self, eps):
        self.opinion = -1
        self.epsilon = eps
        self.class_ref = 0

    def calculate_stock_demand(risk_aversion, variance_of_return, r, price, price_t1): #[6], [7]
        #'r' represent the risk free rate
        #'price' represent the current price of this stock
        #'price_t1" represents the price at time T+1
        stock_demand = (1 / (risk_aversion * variance_of_return)) * (price_t1 - (1 + r) * price)
        stock_demand = stock_demand.astype(int)
        # Present Discounted value model
        return stock_demand


    def update_amount_of_stock(amount_of_stock, stock_demand, n):
        for i in range(n):
                amount_of_stock[i] += stock_demand[i]
        # The amount of stock could be positive of negative, positive means long stock, negative means short selling.
        return amount_of_stock


    def Buy_Sell_Hold(stock_demand, action, n):
        trading_volume = 0
        for i in range(n):
            if (stock_demand[i] > 0):
                action[i] = 1  #'1' represents the action "Buy"
                trading_volume = trading_volume + stock_demand[i]
            elif (stock_demand[i] < 0):
                action[i] = -1  #'-1' represents the action "Sell"
                trading_volume = trading_volume + stock_demand[i]
            else:
                action[i] = 0  # Hold
        return action, trading_volume

    def calculate_order_price(action, risk_preference, price, price_t1, order_price, n, r):
        for i in range(n):
            if (action[i] == 1):  # Means buy stocks
                order_price[i] = random.uniform(((1 + r) * price * (1 - risk_preference[i])), price_t1[i])
            elif (action[i] == -1):  # Means sell stocks
                order_price[i] = random.uniform(price_t1[i], (1 + r) * price * (1 + risk_preference[i]))

        return order_price

class reddit_trader:
    def __init__(self, eps):
        self.opinion = np.random.uniform(0, 1)
        self.epsilon = eps
        self.class_ref = 1

    def stock_demand_list(reddit_stock_demand, max_reddit_stock_demand):
        for i in range(len(reddit_stock_demand)):
            reddit_stock_demand[i] = int(random.uniform(0, max_reddit_stock_demand))
        return reddit_stock_demand

    def Order_price_list(stock_demand, r, price, order_constant):
        order_price = np.zeros(len(stock_demand))
        for i in range(len(stock_demand)):
            order_price[i] = random.uniform((1 + r) * price, (1 + r) * price * (1 + order_constant))
        return order_price

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
        alpha = 0.7  # prospect theory parameter
        delta_op = ag[self.id1].opinion - ag[self.id2].opinion

        if ag[self.id1].class_ref == 1 and ag[self.id2].class_ref == 1:
            if np.abs(delta_op) < ag[self.id1].epsilon:
                ag[self.id1].opinion = ag[self.id1].opinion - self.mu * delta_op
                ag[self.id2].opinion = ag[self.id2].opinion + self.mu * delta_op

        if ag[self.id1].class_ref == 1 and ag[self.id2].class_ref == 0:
            if np.abs(delta_op) < ag[self.id1].epsilon:
                ag[self.id1].opinion = ag[self.id1].opinion - self.mu * delta_op
            if np.random.randint(0, 1) == 0:
                sign1 = np.sign(ag[self.id1].epsilon)
                sign2 = np.sign(ag[self.id2].epsilon)
                ag[self.id2].opinion = sign1 * (ag[self.id1].opinion) * math.pow(abs(ag[self.id1].epsilon), alpha) + \
                                       sign2 * (ag[self.id2].opinion) * math.pow(abs(ag[self.id2].epsilon), alpha)
            elif np.random.randint(0, 1) == 1:
                ag[self.id2].opinion = 0

        if ag[self.id1].class_ref == 0 and ag[self.id2].class_ref == 1:
            if np.abs(delta_op) < ag[self.id1].epsilon:
                ag[self.id2].opinion = ag[self.id1].opinion + self.mu * delta_op
            if np.random.randint(0, 1) == 0:
                sign1 = np.sign(ag[self.id1].epsilon)
                sign2 = np.sign(ag[self.id2].epsilon)
                ag[self.id1].opinion = sign1 * (ag[self.id1].opinion) * math.pow(abs(ag[self.id1].epsilon), alpha) + \
                                       sign2 * (ag[self.id2].opinion) * math.pow(abs(ag[self.id2].epsilon), alpha)
            elif np.random.randint(0, 1) == 1:
                ag[self.id1].opinion = 0

        if ag[self.id1].class_ref == 0 and ag[self.id2].class_ref == 0:
            if np.random.randint(0, 1) == 0:
                sign1 = np.sign(ag[self.id1].epsilon)
                sign2 = np.sign(ag[self.id2].epsilon)
                ag[self.id1].opinion = sign1 * (ag[self.id1].opinion) * math.pow(abs(ag[self.id1].epsilon), alpha) + \
                                       sign2 * (ag[self.id2].opinion) * math.pow(abs(ag[self.id2].epsilon), alpha)
            elif np.random.randint(0, 1) == 1:
                ag[self.id1].opinion = 0

            if np.random.randint(0, 1) == 0:
                sign1 = np.sign(ag[self.id1].epsilon)
                sign2 = np.sign(ag[self.id2].epsilon)
                ag[self.id2].opinion = sign1 * (ag[self.id1].opinion) * math.pow(abs(ag[self.id1].epsilon), alpha) + \
                                       sign2 * (ag[self.id2].opinion) * math.pow(abs(ag[self.id2].epsilon), alpha)
            elif np.random.randint(0, 1) == 1:
                ag[self.id2].opinion = 0
        return ag

    # pick agents to participate in the market
    def pick_agents(self):
        probs = np.random.uniform(0, 1, self.N)
        p = (self.T - self.curr_day + 1) ** (-2.44) + 0.01
        # p = 1.1
        self.ag_ids = [i for i, j in enumerate(probs) if j < p]

    # update temporal
    def update_op_series(self, i, ag):
        self.curr_day = np.ceil(np.float(i) / (self.N * 0.5))

        self.pick_agents()
        self.ag_ids_len = np.append(self.ag_ids_len, len(self.ag_ids))

        return self.ag_ids

    # control function
    def launch(self, ag):
        self.select_pair()
        ag = self.update_opinion(ag)
        return ag

class price_prediction:
    def updatePrice(rational_stock_demand, irrational_stock_demand,
                    rational_actions, rational_order_price, irrational_order_price):
        # rational trader can buy, sell or hold the stock, but irrational trader only buy.
        irrational_actions = [1] * len(irrational_order_price)  #'[1]' represent 'buy'
        rational_stock_demand = rational_stock_demand.tolist()
        rational_actions = rational_actions.tolist()
        action = rational_actions + irrational_actions
        stock_demand = rational_stock_demand + irrational_stock_demand

        Order_prices = rational_order_price.tolist()
        Order_prices.extend(irrational_order_price)
        bids = []  # an array of all bidding prices
        asks = []  # an array of all asking prices
        total_order = sum(map(abs, stock_demand))
        for i in range(len(action)):
            if (action[i] == 1):
                bids.append(Order_prices[i] * (abs(stock_demand[i])) / total_order)
            elif (action[i] == -1):
                asks.append(Order_prices[i] * (abs(stock_demand[i])) / total_order)

        return sum(bids) + sum(asks)

def do_prediction(n, t, r, risk_aversion, risk_preference, price, variance_of_return, price_t1,
                      action, rational_order_price, rational_current_amount,
                      max_stock_demand, order_constant, add_agents_sequence):
    price_list = [price]
    rational_trading_volume = [0]
    irrational_trading_volume = [0]
    rational_total_volume = [sum(rational_current_amount)]
    irrational_total_volume = [0]

    '''Reddit trader Initialization'''
    irrational_current_amount = []
    irrational_stock_demand = []

    '''Do prediction'''
    for round in range(t):
        if (round < len(add_agents_sequence)):
            for agent in range(add_agents_sequence[round]):
                irrational_current_amount.append(0)
                irrational_stock_demand.append(0)

        '''Dynamic irrational agents stock demand'''
        irrational_stock_demand = reddit_trader.stock_demand_list(irrational_stock_demand, max_stock_demand)
        irrational_trading_volume.append(sum(irrational_stock_demand))
        zipped_lists = zip(irrational_current_amount, irrational_stock_demand)
        irrational_current_amount = [x + y for (x, y) in zipped_lists]
        irrational_total_volume.append(sum(irrational_current_amount))

        '''Denamic rational agents stock demand'''
        rational_stock_demand = hedge_fund.calculate_stock_demand(risk_aversion, variance_of_return, r, price, price_t1)
        action, trading_volume = hedge_fund.Buy_Sell_Hold(rational_stock_demand, action, n)
        rational_trading_volume.append(trading_volume)
        rational_current_amount = hedge_fund.update_amount_of_stock(rational_current_amount, rational_stock_demand, n)
        rational_total_volume.append(sum(rational_current_amount))

        '''Dynamic Price'''
        rational_order_price = hedge_fund.calculate_order_price(action, risk_preference, price, price_t1,
                                                                    rational_order_price, n, r)
        order_price_irrational = reddit_trader.Order_price_list(irrational_stock_demand, r, price, order_constant)
        price = price_prediction.updatePrice(rational_stock_demand, irrational_stock_demand,
                                                 action, rational_order_price, order_price_irrational)
        price_list.append(price)

    plt.plot(price_list, color="b", linestyle="-", marker=".", linewidth=1, label='predicted price')
    plt.legend()
    plt.xlabel("day")
    plt.ylabel("price")
    plt.title('Predicted_price')
    plt.show()

    plt.bar(range(len(irrational_trading_volume)), irrational_trading_volume,
            label='reddit trading amount', fc='r')
    plt.legend()
    plt.show()



