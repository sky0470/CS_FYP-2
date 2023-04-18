import numpy as np
import matplotlib.pyplot as plt



# no message (284-49)
# mean: 29.27, ranks: [ 4.733 15.704 35.995 42.901 47.04 ]
no_msg_49 = np.array([29.27, 4.733, 15.704, 35.995, 42.901, 47.04 ])
# (284-44, same seed)
# mean: 51.71, ranks: [20.648 36.797 63.31  67.916 69.879]
no_msg_44 = np.array([51.71, 20.648, 36.797, 63.31,  67.916, 69.879])
# (284-39, same seed)
# mean: 48.08, ranks: [17.6   34.293 58.681 63.892 65.921]
no_msg_39 = np.array([48.08, 17.6,   34.293, 58.681, 63.892, 65.921])

# message (276-49)
# mean: 47.81, ranks: [37.896 45.294 50.603 51.981 53.291]
msg = np.array([47.81, 37.896, 45.294, 50.603, 51.981, 53.291])

# toggle all dim (278-99)
# mean: 56.08, ranks: [22.548 42.7   69.728 71.991 73.409]
toggle_all = np.array([56.08, 22.548, 42.7,   69.728, 71.991, 73.409])

# toggle 2 dim only: (283-99)
# mean: 52.14, ranks: [34.538 44.669 58.732 60.727 62.04 ]
toggle_2 = np.array([52.14, 34.538, 44.669, 58.732, 60.727, 62.04 ])

# toggle all dim, dist inf: (282-99)
# mean: 49.29, ranks: [40.289 45.501 51.859 53.781 55.002]
toggle_all_inf = np.array([49.29, 40.289, 45.501, 51.859, 53.781, 55.002])

# nose 2 9 (269-74)
noise = np.array([21.48, 9.875, 17.711 ,25.094 ,26.444 ,28.279])

x = np.arange(1, 6)

plt.plot(x, msg[1:], 'or-', label="comm")
plt.plot(x, no_msg_49[1:], 'ob-', label="no_comm")
plt.plot(x, toggle_all[1:], 'oc-', label="mask")
plt.plot(x, toggle_2[1:], 'og-', label="mask_agent")
plt.plot(x, toggle_all_inf[1:], 'om-', label="mask_reordered")
plt.plot(x, noise[1:], 'oy-', label="noise")
plt.legend()
plt.xticks(x)
plt.yticks(np.arange(0, 100, step=10))
plt.ylim(0, 100)
plt.xlabel("Rank")
plt.ylabel("Reward")
plt.title("Reward of predators at different ranks")
# img_path = os.path.join(path, "rewards-summary.png")
# plt.savefig(img_path, dpi=200)
plt.show()

mean = np.array([msg[0], no_msg_49[0], toggle_2[0], toggle_all[0], toggle_all_inf[0]])
plt.plot(1, msg[0], 'or')
plt.plot(2, no_msg_49[0], 'ob')
plt.plot(3, toggle_2[0], 'og')
plt.plot(4, toggle_all[0], 'oc')
plt.plot(5, toggle_all_inf[0], 'om')
plt.plot(6, noise[0], 'oy')
plt.xticks(np.arange(1,7), ["comm", "no_comm", "mask_agent", "mask", "mask_reordered", "noise"])
plt.yticks(np.arange(0, 100, step=10))
plt.ylim(0, 100)
plt.ylabel("Reward")
plt.title("Mean reward of different runs")
plt.show()
