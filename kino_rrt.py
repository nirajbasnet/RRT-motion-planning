import numpy as np
from math import floor, ceil
import time
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from collections import defaultdict


class RRT:
    def __init__(self, filename, start_state=None, goal_state=None):
        self.maze_array = self.from_pgm(filename)
        self.cols, self.rows = self.maze_array.shape
        if start_state is None:
            self.start_state = (0, 0)
        if goal_state is None:
            self.goal_state = (self.cols - 1, self.rows - 1)
        self.start_node_tree = [self.start_state]
        self.goal_node_tree = [self.goal_state]
        self.map_size = self.rows * self.cols
        self.parent = defaultdict(lambda: defaultdict(lambda: None))
        self.start_tree = [self.start_state]
        self.path_length=0

    def from_pgm(self, filename):
        with open(filename, 'r', encoding='latin1') as infile:
            # Get header information and move file pointer past header
            header = infile.readline()
            width, height, _ = [int(item) for item in header.split()[1:]]
            # Read the rest of the image into a numpy array and normalize
            image = np.fromfile(infile, dtype=np.uint8).reshape((height, width)) / 255
        return image.T

    def compute_distance(self, start, goal):
        """compute euclidean distance given start and goal state"""
        return np.sqrt(np.power(goal[0] - start[0], 2) + np.power(goal[1] - start[1], 2))

    def find_nearest_node_on_tree(self, tree, sampled_node):
        distance_nodes = np.array([self.compute_distance(node, sampled_node) for node in tree])
        return tree[np.argmin(distance_nodes)]

    def check_collision(self, maze_array, node):
        floor_node_x,floor_node_y=int(floor(node[0])),int(floor(node[1]))
        ceil_node_x, ceil_node_y = int(ceil(node[0])), int(ceil(node[1]))
        if ceil_node_x==self.cols:
            ceil_node_x -=1
        if ceil_node_y==self.rows:
            ceil_node_y -=1
        if maze_array[floor_node_x, floor_node_y] == 0 \
                or maze_array[ceil_node_x, ceil_node_y] == 0 \
                or maze_array[floor_node_x, ceil_node_y] == 0 \
                or maze_array[ceil_node_x, floor_node_y] == 0:
            return True
        else:
            return False

    def find_delta_node(self, near_node, sampled_node, delta):
        delta_node_x = delta * (sampled_node[0] - near_node[0]) / self.compute_distance(near_node, sampled_node)
        delta_node_y = delta * (sampled_node[1] - near_node[1]) / self.compute_distance(near_node, sampled_node)
        new_node = (near_node[0] + delta_node_x, near_node[1] + delta_node_y)
        return new_node

    def check_proximity_goal(self, node, goal, threshold=1.0):
        if self.compute_distance(node, goal) <= threshold:
            return True
        else:
            return False

    def get_random_sample(self, goal_sample_prob):
        if np.random.random() > goal_sample_prob:
            # rand_node = (np.random.random() * (self.cols - 1), np.random.random() * (self.rows - 1))
            rand_node = (np.random.uniform(0, self.cols - 1), np.random.uniform(0, self.rows - 1))
        else:
            rand_node = self.goal_state
        return rand_node

    def compute_path(self, start, goal):
        path = [goal]
        while goal != start:
            goal = self.parent[goal[0]][goal[1]]
            path.append(goal)
        return path

    def compute_RRT(self, delta, goal_sample_prob,show_graph=False):
        # Perform random sampling with some goal sampling rate to get a random node
        while True:
            rand_node = self.get_random_sample(goal_sample_prob)

            # Find nearest node to sampled node on start tree
            near_node = self.find_nearest_node_on_tree(self.start_tree, rand_node)

            # Find delta node in the direction towards sampled random node
            delta_node = self.find_delta_node(near_node, rand_node, delta)

            if self.check_collision(self.maze_array, delta_node):
                continue

            self.start_tree.append(delta_node)
            self.parent[delta_node[0]][delta_node[1]] = near_node
            if show_graph==True:
                self.plot_path('Tree', [], 'Maze_RRT')

            if self.check_proximity_goal(delta_node, self.goal_state, delta):
                self.parent[self.goal_state[0]][self.goal_state[1]] = delta_node
                self.plot_path('Path', self.compute_path(self.start_state, self.goal_state), 'Maze_RRT')
                print("goal found")
                break

    def plot_path(self, graph_mode, path, title_name=None):
        """
            Plots the provided path on the maze
        """

        fig = plt.figure(1)
        ax1 = fig.add_subplot(1, 1, 1)

        spacing = 1.0  # Spacing between grid lines
        minor_location = MultipleLocator(spacing)

        # Set minor tick locations.
        ax1.yaxis.set_minor_locator(minor_location)
        ax1.xaxis.set_minor_locator(minor_location)

        # Set grid to use minor tick locations.
        ax1.grid(which='minor')

        colors = ['b', 'r', 'g']
        plt.imshow(self.maze_array.T, cmap=plt.get_cmap('bone'))
        if title_name is not None:
            fig.suptitle(title_name, fontSize=20)
        if graph_mode == 'Path':
            path = np.array(path)
            for i in range(len(path) - 1):
                cidx = i % 3
                plt.plot([path[i, 0], path[i + 1, 0]], [path[i, 1], path[i + 1, 1]], \
                         color=colors[cidx], linewidth=2)
                self.path_length += self.compute_distance((path[i,0],path[i,1]),(path[i+1,0],path[i+1,1]))
        elif graph_mode == 'Tree':
            tree = np.array(self.start_tree)
            for i, node in enumerate(tree):
                cidx = i % 2
                parent = self.parent[node[0]][node[1]]
                if parent is not None:
                    plt.plot([node[0], parent[0]], [node[1], parent[1]], color=colors[cidx], linewidth=2)
        plt.show()


if __name__ == '__main__':
    print("rrt started")
    rrt = RRT('maze2.pgm')
    delta = 1.0
    goal_sample_rate = 0.20
    past_time=time.time()
    rrt.compute_RRT(delta, goal_sample_rate,show_graph=False)
    compute_time=time.time()-past_time
    print("Path_length=",rrt.path_length)
    print("Compute_time=",compute_time)