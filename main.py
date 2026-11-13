import numpy as np 
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle 
from matplotlib import style
import time


style.use('ggplot')

size = 10
hmEpisodes = 25_000 #peep the _ tech
movePenalty = 1
enemyPenalty = 300
foodReward = 25

epsilon = 0.9
EPS_DECAY = 0.9998

SHOW_EVERY = 5_000
start_q_table = "qtable123.pickle" # or file name so u can keep training from where uleft off
LEARNING_RATE = 0.1
DISCOUNT = 0.95

player_n = 1
food_n = 2
enemy_n = 3

d = {1: (255, 175,0 ), 2: (0, 255, 0), 3: (0, 0, 255)}

class blob:
    def __init__(self):
        self.x = np.random.randint(0,size)
        self.y = np.random.randint(0,size)
    def __str__(self):
        return f"{self.x}, {self.y}"
    def __sub__(self, other:'blob'):
        return (self.x - other.x, self.y - other.y)
    
    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
    def move(self, x = False, y = False):
        if not x:
            self.x += np.random.randint(-1,2)
        else:
            self.x += x
        if not y:
            self.y += np.random.randint(-1,2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > size-1:
            self.x = size-1
        if self.y < 0:
            self.y = 0
        elif self.y > size-1:
            self.y = size-1

if start_q_table is None:
    q_table = {}
    for x1 in range(-size+1, size):
        for y1 in range(-size+1, size):
            for x2 in range(-size+1, size):
                for y2 in range(-size+1, size):
                    q_table[((x1,y1),(x2,y2))] = [np.random.uniform(-5,0) for i in range(4)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []

for episode in range(hmEpisodes):
    player = blob()
    food = blob()
    enemy = blob()

    if episode % SHOW_EVERY == 0 and episode != 0:
        print(f" on # {episode}, epsilon {epsilon}")
        print (f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False 

    episode_reward = 0

    for i in range(200):
        obs = (player-food,player-enemy) 
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0,4) #0,1,2,3

        player.action(action)

        #enemy.move()
        #food.move()

        if player.x == enemy.x and player.y == enemy.y:
            reward = -enemyPenalty
        elif player.x == food.x and player.y == food.y:
            reward = foodReward
        else:
            reward = -movePenalty

        new_obs = (player - food, player - enemy)

        maxFutureQ = np.max(q_table[new_obs])
        currentQ = q_table[obs][action]

        if (reward == foodReward):
            new_q = foodReward
        elif reward == -enemyPenalty:
            new_q = -enemyPenalty
        else:
            new_q = (1-LEARNING_RATE) * currentQ + LEARNING_RATE * (reward + DISCOUNT * maxFutureQ)

        q_table[obs][action] = new_q

        if show:
            env = np.zeros((size,size,3), dtype = np.uint8) #figure this one out
            env[food.x][food.y] = d[food_n] 
            env[player.x][player.y] = d[player_n] 
            env[enemy.x][enemy.y] = d[enemy_n] 

            img = Image.fromarray(env,"RGB")
            img = img.resize((300,300))
            cv2.imshow("xd", np.array(img))

            if reward == foodReward or reward == -enemyPenalty:
                print('game is over')
                if cv2.waitKey(1500) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(200) & 0xFF == ord("q"):
                    break

        episode_reward += reward

        if reward == foodReward or reward == -enemyPenalty:
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode  = "valid") # for loop for da np.means xd 

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward{SHOW_EVERY}ma")
plt.xlabel(f"episode #")
plt.show()

print('writing now!')

with open("qtable123.pickle", "wb") as f:
    pickle.dump(q_table,f)