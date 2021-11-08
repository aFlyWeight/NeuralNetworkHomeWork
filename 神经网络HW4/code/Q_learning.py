import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

q_table = np.zeros((22,11,2))
#print(q_table[21,10,1])
#print(np.argmax(q_table[1,2]))
explore_rate = 0.2
policy = {}
cards_color = ['r','b','b']

def initial(q_table):
    for i in range(11):
        q_table[0,i,0] = -10
        q_table[0,i,1] = -10
    for i in range(1,22):
        for j in range(11):
            if i <= j:                      #当玩家点数小于交易商点数的时候，玩家会要牌
                q_table[i,j,0] = j-i
                q_table[i,j,1] = i-j
            q_table[i,j,0] -= abs(i-11)-2     #玩家离11点越远越容易停牌，越近越容易要牌
            q_table[i,j,1] += abs(i-11)
            #print(q_table[i,j,1])
                

def e_greedy(e,q_table,player_cur_card,dealer_cur_card):
    rt = np.random.random()
    if rt < e/2+1-e:        #选择最好的action
        action = np.argmax(q_table[player_cur_card,dealer_cur_card])
    else:
        action = np.argmin(q_table[player_cur_card,dealer_cur_card])
    return action
        
     
def dealer_card(dealer_ini_card):
    cur_card = dealer_ini_card
    while(cur_card > 0 and cur_card < 16):
        card = np.random.randint(1,10)
        color = np.random.choice(cards_color)
        if color == 'r':
            cur_card -= card
        elif  color == 'b':
            cur_card += card
    return cur_card  


def train(q_table):
    e = 0.2
    alpha = 0.001
    gama = 0.99
    acc_reward_record = []
    rewards_record = []
    acc_reward = 0
    acc_mean_reward = 0
    acc_mean_reward_set = []
    for ep in range(100000):
        player_ini_card = np.random.randint(1,10)
        dealer_ini_card = np.random.randint(1,10)    
        player_cur_card = player_ini_card
        dealer_cur_card =  dealer_ini_card  
        reward = 0 
        isbust = 0
        action = e_greedy(e,q_table,player_cur_card,dealer_cur_card)  
        player_prev_card = player_cur_card
        dealer_prev_card = dealer_cur_card
        while(action == 0):                 #要牌
            player_prev_card = player_cur_card                          #记录当前状态
            card = np.random.randint(1,10)
            color = np.random.choice(cards_color)
            if color == 'r':
                player_cur_card -= card
            elif  color == 'b':
                player_cur_card += card
            if player_cur_card <= 0 or player_cur_card >21:             #玩家爆掉了
                player_cur_card = 0
                reward = -1
                isbust = 1
                break
            else:
                reward = 0
            qtmp0 = q_table[player_prev_card,dealer_prev_card,0]
            q_table[player_prev_card,dealer_prev_card,0] = qtmp0 + alpha*(reward+gama*q_table[player_cur_card,dealer_prev_card,0]-qtmp0)
            #e = 
            action = e_greedy(e,q_table,player_cur_card,dealer_cur_card)
        if(isbust == 0):                                                #玩家没有爆掉,选择停牌
            dealer_cur_card = dealer_card(dealer_ini_card)
            if dealer_cur_card <=0 or  dealer_cur_card > 21:
                reward = 1
            elif player_cur_card > dealer_cur_card:
                reward = 1
            elif player_cur_card == dealer_cur_card:
                reward = 0
            elif player_cur_card < dealer_cur_card:
                reward = -1
        #玩家停牌或爆掉之后更新q_table,此时没必要再更新状态了，因为胜负已分，直接开始下一轮训练
        qtmp1 = q_table[player_prev_card,dealer_prev_card,0]
        q_table[player_prev_card,dealer_prev_card,0] = qtmp1 + alpha*(reward+gama*q_table[player_cur_card,dealer_prev_card,1]-qtmp1) 
        acc_reward += reward
        rewards_record.append(reward)
        if ep%100 == 0:
            acc_reward_record.append((ep,acc_reward))
        if ep>=10:
            acc_mean_reward = np.mean(rewards_record[max(ep-999,0):ep+1])
            acc_mean_reward_set.append(acc_mean_reward)
            
    return acc_reward_record,acc_mean_reward_set


def test(q_table):
    policys = []
    num_player_suc = 0
    eposide = 10000
    for ep in range(eposide):
        player_ini_card = np.random.randint(1,10)
        dealer_ini_card = np.random.randint(1,10)    
        player_cur_card = player_ini_card
        dealer_cur_card =  dealer_ini_card  
        isbust = 0
        action = e_greedy(0,q_table,player_cur_card,dealer_cur_card)
        policy = []
        policy.append((player_cur_card,dealer_cur_card,action))             #将策略记录下来
        while(action == 0):                 #要牌  
            card = np.random.randint(1,10)
            color = np.random.choice(cards_color)
            if color == 'r':
                player_cur_card -= card
            elif  color == 'b':
                player_cur_card += card    
            if player_cur_card <= 0 or player_cur_card > 21:             #玩家爆掉了
                isbust = 1
                break   
            action = e_greedy(0,q_table,player_cur_card,dealer_cur_card)
            policy.append((player_cur_card,dealer_cur_card,action))             #将策略记录下来
        if(isbust == 0):
            dealer_cur_card = dealer_card(dealer_ini_card)
            if dealer_cur_card <=0 or  dealer_cur_card > 21:
                num_player_suc += 1
            elif player_cur_card > dealer_cur_card:
                num_player_suc += 1
        policys.append(policy)
    succ_ratio = num_player_suc/eposide
    return policy,succ_ratio

def plot_acc_reward(acc_mean_reward_set):
    fig = plt.figure()
    fig.suptitle('ACC Reward', fontsize = 14, fontweight='bold')
    ax = fig.add_subplot(1,1,1)
    ax.plot(np.arange(1,len(acc_mean_reward_set)+1), acc_mean_reward_set)
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    plt.show()
    
def plot_optimal_value_cur(q_table):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0,22)
    y = np.arange(0,11)
    X, Y = np.meshgrid(x, y)
    z = []
    for i in x:
        for j in y:
            z.append(max(q_table[i,j,0],q_table[i,j,1]))
#    print(z)
#    print(X.shape)
#    print(Y.shape)
    Z = np.array(z)
    Z = Z.reshape(X.shape)
    ax.plot_surface(Y, X, Z, cmap=cm.coolwarm)
#    ax.set_yticks(np.arange(10)+1)
    ax.set_xlabel('Player points')
    ax.set_ylabel('Dealer starting card')
    ax.set_zlabel('reward')
    plt.show()
    
def plot_optimal_action(q_table):
    fig = plt.figure()
    fig.suptitle('Optimal action (blue for stick, red for hit)', fontsize = 14, fontweight='bold')
    ax = fig.add_subplot(1,1,1)
    x = np.arange(1,22)
    y = np.arange(1,11)
    X, Y = np.meshgrid(x, y)
    z = []
    for i in x:
        for j in y:
            z.append(np.argmax(q_table[i,j]))
    Z = np.array(z)
    Z = Z.reshape(X.shape)
 #   print(Z)
    colors = ['red', 'blue']
    ax.scatter(X, Y, c=Z, cmap=matplotlib.colors.ListedColormap(colors))
#    ax.set_yticks(np.arange(10)+1)
    ax.set_xlabel('Player points')
    ax.set_ylabel('Dealer starting card')
    plt.show()

if __name__ == '__main__':
    initial(q_table)
    reward_record,acc_mean_reward_set = train(q_table)
#    print(q_table)
    plot_acc_reward(acc_mean_reward_set)
#    for i in range(len(reward_record)):
#        print(reward_record[i])
#    for i in range(len(acc_mean_reward_set)):
#        print(acc_mean_reward_set[i])
    policys,succ_ratio = test(q_table)
    print(succ_ratio)
    plot_optimal_value_cur(q_table)
    plot_optimal_action(q_table)