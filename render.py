import cv2
import numpy as np
from o3d_utils import Visu3D
import mediapipe_utils as mpu
# import matplotlib.pyplot as plt


# LINE_BODY and COLORS_BODY are used when drawing the skeleton in 3D. 
rgb = {"right":(0,1,0), "left":(1,0,0), "middle":(1,1,0), "random":(0,0,1)}
LINES_BODY = [[9,10],[4,6],[1,3],
            [12,14],[14,16],[16,20],[20,18],[18,16],
            [12,11],[11,23],[23,24],[24,12],
            [11,13],[13,15],[15,19],[19,17],[17,15],
            [24,26],[26,28],[32,30],
            [23,25],[25,27],[29,31]]

COLORS_BODY = ["middle","right","left",
                "right","right","right","right","right",
                "middle","middle","middle","middle",
                "left","left","left","left","left",
                "right","right","right","left","left","left"]

COLORS_BODY = [rgb[x] for x in COLORS_BODY]

COLORS_DRONES = ["middle","right","left","random"]
MARKER_DRONES = ["o","^"]

DRONE_VIZ = [(COLORS_BODY[x], MARKER_DRONES[x%2==0]) for x in range(len(COLORS_DRONES))]

def randrange(n, vmin, vmax):
    """
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    """
    return (vmax - vmin)*np.random.rand(n) + vmin

n = 100


# points
# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# def paint_point(list_of_points):
    # for points in list_of_points:
    #     for i, (color, m) in enumerate(DRONE_VIZ):
    #         xs = 
    #         ys = randrange(n, 0, 100)
    #         zs = randrange(n, zlow, zhigh)
    #         ax.scatter(xs, ys, zs, c=color, marker=m)


def add_line(p1, p2, color, fig, ax):
    #lines
    x, y, z = [p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]]
    ax.scatter(x, y, z, c='none', s=100)
    ax.plot(x, y, z, color=color)
    # plt.show()

def is_present(body, lm_id, tracker):
    return body.presence[lm_id] > tracker.presence_threshold


def paint_body(points):
    lines = LINES_BODY
    colors = COLORS_BODY
    for i,a_b in enumerate(lines):
        a, b = a_b
        if is_present(points, a) and is_present(points, b):
                add_line(points[a], points[b], color=colors[i])

drone_position=[]
def spawn_drones(position, num_drones=len(COLORS_DRONES)):
    if len(drone_position)==0:
        drone_position = np.hstack([np.random.uniform(low=-1, high=1, size=(4,2)), np.ones((num_drones,1))*0.8])
        print(drone_position.shape)
    else:
        drone_position += position


def draw_drones(list_of_points):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # vis_info = DRONE_VIZ
        # def move_drones(self, position):
        # colors = COLORS_DRONES
        # radius = 0.1
        # spawn_drones(points)
        # for i, pos in enumerate(self.drone_position):
        #     self.vis3d.add_drone(pos, radius, color=colors[i])
    
    points_arr = np.array(list_of_points)
    points = points_arr.reshape(4, len(list_of_points), 3).transpose((1, 0, 2))
    
    for i in range(len(COLORS_DRONES)):
        for t in range(1, points[i].shape[0]):
            x, y, z = [points[i,t-1,0],points[i,t,0]], [points[i,t-1,1],points[i,t,1]], [points[i,t-1,2],points[i,t,2]]
            ax.scatter(x, y, z, c='none', s=100)
            ax.plot(x, y, z, color=DRONE_VIZ[i][0])
            # add_line(points[i,t-1,:],points[i,t,:], DRONE_VIZ[i][0], fig, ax)


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.savefig('res.jpg')
    plt.show()
