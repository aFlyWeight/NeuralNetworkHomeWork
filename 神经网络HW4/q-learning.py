import numpy as np
import numpy.random as random
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors
import matplotlib.patches as mpatches
from matplotlib.lines import  Line2D
from matplotlib.legend_handler import HandlerPatch
import matplotlib
from scipy.ndimage import convolve
matplotlib.rcParams['agg.path.chunksize']=10000
EPISODES = 100000
LR = 0.1
EPSILON = 0.1
INTERVAL = 2000
random.seed(0)

class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=height + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def assess_image_smooth(M):
    shape = M.shape
    x_kernel = np.array([[1,-2,1]])
    y_kernel = np.array([[1],
                         [-2],
                         [1]])
    gx = convolve(M,x_kernel)[1:shape[0]-1,1:shape[1]-1]
    gy = convolve(M,y_kernel)[1:shape[0]-1,1:shape[1]-1]
    laplace = gx*gx+gy*gy
    return np.var(laplace)


class Env:
    def __init__(self, black_prob, dealer_stick_sum: int):
        self.black_prob = black_prob
        self.dealer_stick_sum = dealer_stick_sum

    def step(self, state, action: int):
        """
        :param state: (deader's first card,player's current sum) in ([1,10],[1,21])
        :param action: 0:stick, 1:hit
        :return:(new state,reward)
        """
        if action:  # hit
            deck_card = random.randint(1, 11)
            if random.random() < self.black_prob:
                player = state[1] + deck_card
            else:
                player = state[1] - deck_card
            if player > 21 or player < 1:
                return (-1, -1), -1
            else:
                return (state[0], player), 0
        else:  # stick
            dealer = state[0]
            while True:
                deck_card = random.randint(1, 11)
                if random.random() < self.black_prob:
                    dealer += deck_card
                else:
                    dealer -= deck_card
                if dealer < 1 or dealer > 21:
                    return (-1, -1), 1
                elif dealer >= self.dealer_stick_sum:
                    if state[1] < dealer:
                        return (-1, -1), -1
                    elif state[1] > dealer:
                        return (-1, -1), 1
                    else:
                        return (-1, -1), 0


class Q_learning:
    def __init__(self, lr=LR, epsilon=EPSILON, episodes=EPISODES, interval=INTERVAL):
        self.Q = np.ones([10, 21, 2], dtype=float)
        self.lr = lr
        self.eps = epsilon
        self.env = Env(2.0 / 3, 16)
        self.Q[:, :, 0] = 0
        self.episodes = episodes
        self.assess_interval = interval
        self.rewards = []
        self.smooth = []
        random.seed(0)

    def play_one_episode(self):
        player = random.randint(10) + 1
        dealer = random.randint(10) + 1
        while True:
            action = np.argmax(self.Q[dealer - 1, player - 1])
            if random.random() < self.eps:
                action = random.randint(2)
            state, reward = self.env.step((dealer, player), action)
            # update table
            if state[0] > 0:
                self.Q[dealer - 1, player - 1, action] += self.lr * (reward + max(self.Q[dealer - 1, state[1] - 1])
                                                                     - self.Q[dealer - 1, player - 1, action])
                player = state[1]
            else:
                self.Q[dealer - 1, player - 1, action] += self.lr * (reward - self.Q[dealer - 1, player - 1, action])
                break
        return reward

    def play(self):
        random.seed(0)
        for i in range(self.episodes):
            reward = self.play_one_episode()
            self.rewards.append(reward)
            if (i + 1) % self.assess_interval == 0:
                # MSD = np.sum((ori_Q - self.Q) ** 2)
                V = np.max(self.Q, axis=2)
                smooth = assess_image_smooth(V)
                self.smooth.append(smooth)
                print('episode:', i + 1, ' Smoothness:', smooth)

    def plot_value_fun(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Make the X, Y meshgrid.
        xs = np.arange(1, 11)
        ys = np.arange(1, 22)
        X, Y = np.meshgrid(xs, ys)
        zs = np.array([max(self.Q[x - 1, y - 1]) for x, y in
                       zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)

        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
        # X = X.reshape(-1)
        # Y = Y.reshape(-1)
        # Z = Z.reshape(-1)
        # ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm)
        ax.set_xticks(np.arange(10)+1)
        ax.set_xlabel('Dealer starting card')
        ax.set_ylabel('Player sum')
        ax.set_zlabel('Value')
        plt.savefig('value{}_{}_{}.png'.format(self.episodes,self.lr,self.eps), dpi=400)
        plt.show()

    def plot_decision_fun(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis('equal')
        # Make the X, Y meshgrid.
        xs = np.arange(1, 11)
        ys = np.arange(1, 22)
        X, Y = np.meshgrid(xs, ys)
        zs = np.array([np.argmax(self.Q[x - 1, y - 1]) for x, y in
                       zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)

        # ax.plot_surface(X, Y, Z)
        X = X.reshape(-1)
        Y = Y.reshape(-1)
        Z = Z.reshape(-1)
        colors = ['red', 'blue']
        ax.scatter(X, Y, c=Z, cmap=matplotlib.colors.ListedColormap(colors))
        ax.set_xticks(np.arange(1, 11))
        ax.set_yticks(np.arange(1, 22))
        ax.set_xlabel('Dealer starting card')
        ax.set_ylabel('Player sum')

        cir = []
        for i in range(0, len(colors)):
            # cir.append(Line2D([0], [0],color='w', marker='o',markerfacecolor = colors[i], markersize=15))
            cir.append(plt.Circle((0, 0), 1, fc=colors[i], fill=True))
        ax.legend(cir, ['Stick', 'Hit'], bbox_to_anchor=(0, 1.05),loc=3, borderaxespad=0, ncol = 2,handler_map={mpatches.Circle: HandlerEllipse()})
        plt.savefig('decision{}_{}_{}.png'.format(self.episodes,self.lr,self.eps), dpi=400)
        plt.show()

    def plot_reward_func(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        wind_size = 1000
        cumulative_avg_reward = [np.mean(self.rewards[max(0, i + 1 - wind_size):i + 1]) for i in range(self.episodes)]
        ax.scatter(np.arange(self.episodes)+1, cumulative_avg_reward)
        ax.set_xlabel('episode')
        ax.set_ylabel('return')
        plt.savefig('return{}_{}_{}.png'.format(self.episodes,self.lr,self.eps), dpi=400)
        plt.show()

# def visualize()


def test():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(50)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    plt.ion()
    # ax.scatter(x,2*x,c='r',marker='*')

    for i in range(50):
        ax.scatter(i, i, c='b', marker='o')
        # plt.pause(0.01)
    plt.ioff()
    plt.show()


class Experiments:
    def __init__(self):
        self.q = Q_learning(LR,EPSILON, episodes=EPISODES)

    def test(self,lr=LR, epsilon=EPSILON, episodes=EPISODES, interval=INTERVAL):
        self.q.__init__(lr=LR, epsilon=EPSILON, episodes=EPISODES, interval=INTERVAL)
        random.seed(0)
        self.q.play()
        self.q.plot_value_fun()
        self.q.plot_decision_fun()

    def test_lr(self):
        lrs = [0.01,0.05,0.1,0.3, 0.5,1]
        # smooth = []
        x = (np.arange(EPISODES / INTERVAL) + 1) * INTERVAL
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for lr in lrs:
            self.q.__init__(lr)
            self.q.play()
            # smooth.append(deepcopy(q.smooth))
            ax.plot(x, self.q.smooth)
        ax.set_xlabel('episode')
        ax.set_ylabel('Smoothness')
        lgd = plt.legend(lrs, bbox_to_anchor=(1, 0), loc=3)
        plt.savefig('lr.png', bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=400)
        plt.show()

    def test_epsilon(self):
        epsilon = [0.01,0.05,0.1,0.3, 0.5,1]
        # smooth = []
        x = (np.arange(EPISODES / INTERVAL) + 1) * INTERVAL
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for eps in epsilon:
            self.q.__init__(epsilon=eps)
            self.q.play()
            # smooth.append(deepcopy(q.smooth))
            ax.plot(x, self.q.smooth)
        ax.set_xlabel('episode')
        ax.set_ylabel('Smoothness')
        lgd = plt.legend(epsilon, bbox_to_anchor=(1, 0), loc=3)
        plt.savefig('eps.png', bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=400)
        plt.show()




if __name__ == "__main__":
    exp = Experiments()
    exp.test(epsilon=1)
    # exp.test_lr()
    # exp.test_epsilon()

    # test()
