# import matplotlib.pyplot as plt

# # Updated labels and values for low-level loop
# low_labels = [
#     "Initialization\n(SpaTracker)",
#     "Dynamics Loop\n(MPPI)",
#     "Default to first position\n(video starts)",
#     "Pushing Action",
#     "Back to default position",
#     "Spatracker tracking\nand visualization",
#     "Update\nstate and target"
# ]
# low_values = [0.7900, 23.87, 11.5533, 8.0081, 11.1236, 5.6096, 1.52]

# plt.figure(figsize=(10, 4))
# #Change bar color to yellow
# plt.bar(low_labels, low_values, color='orange')
# # plt.bar(low_labels, low_values)
# plt.xticks(rotation=30, ha="right")
# plt.ylabel("Time (s)")
# plt.title("Low Level Loop")
# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt

labels = [
    "Environment Initialization",
    "Planner Initialization",
    "Mask Generation",
    "Build prompt",
    "GPT Response",
    "Target specification"
]
values = [15.2864, 5.2883, 19.9431, 0.8463, 7.1696, 1.4573]

plt.figure(figsize=(10, 4))
plt.bar(labels, values, color="orange")
plt.xticks(rotation=30, ha="right")
plt.ylabel("Time (s)")
plt.title("High Level Loop")
plt.tight_layout()
plt.show()
