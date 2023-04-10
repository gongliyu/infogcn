import numpy as np
import tools

# 0: 'nose'
# 1: 'left_eye'
# 2: 'right_eye'
# 3: 'left_ear'
# 4: 'right_ear'
# 5: 'left_shoulder'
# 6: 'right_shoulder'
# 7: 'left_elbow'
# 8: 'right_elbow'
# 9: 'left_wrist'
# 10: 'right_wrist'
# 11: 'left_hip'
# 12: 'right_hip'
# 13: 'left_knee'
# 14: 'right_knee'
# 15: 'left_ankle'
# 16: 'right_ankle'


num_node = 17
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [
    (0, 1), (1, 3), (0, 2), (2, 4), # head
    (0, 5), (5, 7), (7, 9), # left arm
    (0, 6), (6, 8), (8, 10), # right arm
    (5, 11), (11, 13), (13, 15), # left leg
    (6, 12), (12, 14), (14, 16) # right leg
]


inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A_outward_binary = tools.get_adjacency_matrix(self.outward, self.num_node)
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A_binary = tools.edge2mat(neighbor, num_node)
        self.A_norm = tools.normalize_adjacency_matrix(self.A_binary + 2*np.eye(num_node))
        self.A_binary_K = tools.get_k_scale_graph(scale, self.A_binary)


    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
