import matplotlib.pyplot as plt
import pickle


with open('rewards.pkl', 'rb') as pickle_file:
    rewards = pickle.load(pickle_file)
# with open('Q.pkl', 'rb') as pickle_file:
#     Q = pickle.load(pickle_file)
avg_rewards = []
counter = 0
for i, val in enumerate(rewards):
    counter += val
    if i % 100 == 0:
        avg_rewards.append(counter / 100)
        counter = 0
plt.title("Rewards Plot for Normal Reward with 0.3 Discount Factor")
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.plot(avg_rewards)
plt.show()
# print(len(list(Q.keys())))
