import math
import random
from tkinter import *
import matplotlib.pyplot as plt
import numpy as np

border = 10000
clusters = []

colors = ["red", "green", "blue", "orange", "purple","brown","black","pink"]

class Point:
    def __init__(self, x, y, cluster):
        self.x = x
        self.y = y
        self.cluster = cluster

class Cluster:
    def __init__(self, cluster_points, color):
        self.cluster_points = cluster_points
        self.color = color
        self.x_avg = 0
        self.y_avg = 0
        self.centroid = self.calculate_centroid()


    def update_centroid(self):
        last_point = self.cluster_points[-1]
        self.x_avg += last_point.x
        self.y_avg += last_point.y
        self.centroid = [self.x_avg/len(self.cluster_points),self.y_avg/len(self.cluster_points)]

    def calculate_centroid(self):
        for point in self.cluster_points:
            self.x_avg += point.x
            self.y_avg += point.y

        x = self.x_avg/len(self.cluster_points)
        y = self.y_avg/len(self.cluster_points)
        return [x, y]

    def update_centroid_remove_point(self, point):
        self.x_avg -= point.x
        self.y_avg -= point.y
        self.cluster_points.remove(point)
        self.centroid = [self.x_avg/len(self.cluster_points),self.y_avg/len(self.cluster_points)]


    def calculate_medoid(self):
        pass


points = []
def create_first_20():
    used_x = []
    used_y = []
    for i in range(20):
        x = random.randint(-5000, 5000)
        y = random.randint(-5000, 5000)
        points.append(Point(x, y, 0))


def generate_points():
    create_first_20()
    global points
    print(len(points))
    for i in range(20000):
        index = random.randint(0, len(points)-1)
        #print("dlzka je ", len(points), index)
        p = points[index]
        x_offset = random.randint(-100, 100)
        y_offset = random.randint(-100, 100)
        point = Point(p.x+x_offset, p.y+y_offset, 0)
        points.append(point)

def calculate_distance(x1,y1,x2,y2):
    distance = math.sqrt(math.pow(x1-x2,2) + math.pow(y1-y2,2))
    return distance


def add_point_kmeans(point):
    min_d = -1
    min_cluster = clusters[0]
    old_cluster = point.cluster
    for cluster in clusters:
        centroid = cluster.centroid
        print("Centroid ", centroid)
        dist = calculate_distance(point.x, point.y, centroid[0], centroid[1])
        print("Distance ", dist)
        if min_d == -1:
            min_d = dist
            min_cluster = cluster
        elif dist < min_d:
            min_d = dist
            min_cluster = cluster

    if min_cluster == point.cluster:
        return 0
    min_cluster.cluster_points.append(point)
    point.cluster = min_cluster
    if old_cluster != 0:
        old_cluster.update_centroid_remove_point(point)
    min_cluster.update_centroid()
    print("tento cluster som priradil ", min_cluster)
    return 1

# vyberiem nahodne k bodov ako k clustrov
def select_k_clusters(k):
    global clusters
    used = []
    counter = 0
    for i in range(k):
        c_number = random.randint(0, len(points))
        while c_number in used:
            c_number = random.randint(0, len(points))
        used.append(c_number)
        cluster = Cluster([points[c_number]], i)
        clusters.append(cluster)
        points[c_number].cluster = cluster



def draw():
    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)
    plt.yticks(np.arange(-5000, 6000, 1000))
    plt.yticks(np.arange(-5000, 6000, 1000))
    #fig, axs = plt.subplots()  # a figure with a 2x2 grid of Axes
    #print(axs)

    plt.plot(1000, 1000)
    for point in points:
        #print("Col ",point.cluster)
        color = colors[point.cluster.color]
        plt.plot(point.x, point.y, marker=".", markersize=5, color=color)

    plt.show()



# Press the green button in the gutter to run the script.
def k_means_clustering(k):
    select_k_clusters(k)
    print(clusters)
    for cluster in clusters:
        p = cluster.cluster_points[0]
        print(f'{p.x},{p.y}')
        print(cluster.centroid)
    # zvysnych N-k priradim ku clustrom
    while True:
        changes = 0
        for point in points:
            # skipujem tie co som uz priradil
            if point.cluster != 0:
                continue
            changes += add_point_kmeans(point)
        if changes == 0:
            break



if __name__ == '__main__':
    generate_points()
    k_means_clustering(8)
    print("Idem kreslit")
    draw()


