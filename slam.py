import cv2
import numpy as np
import matplotlib.pyplot as plt

fig, ax, line, points_scatter = None, None, None, None

# Matplotlib visualization code
def init_3d_plot():
    global fig,ax,line,points_scatter

    plt.ion()
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111,projection='3d')

    #Initialize the camera trajectory line plot
    # Set the initial data to empty lists
    line_list = ax.plot([],[],[],'b-',label='Camera Trajectory')
    line = line_list[0]

    # Initialize a points 3d map scatter plot
    points_scatter = ax.scatter([],[],[],marker='.',c='r',s=1)

    ax.set_xlabel('X (Left / right)')
    ax.set_ylabel('Y (Depth)') # Note: Y is usually up/down in camera space, but let's use it for Z-depth here for map clarity
    ax.set_zlabel('Z (Up/Down)')

    # Set initial limits
    ax.set_xlim([-10,10])
    ax.set_ylim([-10,10])
    ax.set_zlim([0,20])

    ax.legend()
    plt.title('SLAM Map and Camera Trajectory')
    plt.show(block=False)


def update_3d_plot(slam_map_instance):
    global ax,fig,points_scatter,line

    if not fig:
        return


    # Update trajectory
    # slam_map.trajectory is a list of [X,Y,Z] vectors
    if slam_map_instance.trajectory:
        path = np.array(slam_map_instance.trajectory)

        x_traj = path[:,0]
        y_traj = path[:,1]
        z_traj = path[:,2]

        line.set_data(x_traj,y_traj)
        line.set_3d_properties(z_traj)


        # -- Dynamic plot limits ---

        min_coords = path.min(axis=0)
        max_coords = path.max(axis=0)


        buffer = 1
        ax.set_xlim([min_coords[0] - buffer, max_coords[0] + buffer])
        ax.set_ylim([min_coords[1] - buffer,max_coords[1] + buffer])
        ax.set_zlim([min_coords[2] - buffer, max_coords[2] + buffer])

    # --- Update cloud point ----
    if slam_map_instance.point_cloud:
        # Concatenate all cloud chunks into one large array
        all_points = np.hstack(slam_map_instance.point_cloud)

        points_scatter._offsets3d = (all_points[0,:],all_points[1,:],all_points[2,:])

    fig.canvas.draw_idle()
    plt.pause(0.001)





# Constants
FOCAL_LENGTH = 713.8
PP_X = 319.5
PP_Y = 239.5
BASELINE = 0.25


# K array to facilitate the conversion between 3D world and 2d pixel camera
K = np.array([
    [FOCAL_LENGTH,0,PP_X],
    [0,FOCAL_LENGTH,PP_Y],
    [0,0,1]
],dtype=np.float32)



# LK PARAMS neeeded for feature tracking
lk_params = dict(winSize=(21,21),
                 maxLevel =3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,30,0.01))


# Global variables to store data from previous frames
prev_points_3d = None
prev_keypoints_2d = None
prev_frame_left = None




# --- SLAM MAP MEMOERY ---
class SLAM_Map:
    def __init__(self):
        # 4x4 Transformation Matrix storing the camera's current global pose (Rotation and Translation)
        # Starts at the origin (Identity matrix)

        self.camera_pose = np.eye(4)

        # List to store 3D coordinates
        self.point_cloud = []

        # List to store the camera's path
        self.trajectory = []


    def update_pose(self,R_delta,t_delta):
        # Integrates new pose (R,t) into the global camera_pose
        T_delta = np.eye(4)
        T_delta[:3,:3] = R_delta # Replaces the top left corner of the T_delta 4x4 matrix with R_delta
        T_delta[:3,3] = t_delta.flatten() # Replace the top rihgt corner of the matrix with the original t_delta, flatten() ensures that 3x1 is correctly inserted

        # This accumulates the continuous , frame-by-frame motion correctly
        self.camera_pose = self.camera_pose @ T_delta # This is matrix multiplication (composition).Let T_k-1 be the pose at the previous frame (self.camera_pose). Let T_delta be the motion between frame k-1 and $k.

        # The camera's path
        self.trajectory.append(self.camera_pose[:3,3].copy())


    def add_points(self,new_points_3d): # The input new_points_3d is a set of 3D coordinats (X,Y,Z)
        # Converts new points (3xN) from the camera frame to world coordinates
        if new_points_3d.size == 0:
            return

        R = self.camera_pose[:3,:3]
        t = self.camera_pose[:3,3]
        points_w = R @ new_points_3d.T + t[:,None] # Note the transpose for world conversion
        self.point_cloud.append(points_w)






# ------ Stereo SLAM , the depth map calculated immediately from the 2 images ,
# Converts two 2D images to 3D real-world metric distances -----

def compute_stereo_depth(left_gray,right_gray):
    """
    Computes a depth map from a stereo image pair.
    Returns: disparity map (scaled by 16) and a 3D point cloud.
    """

    # 1. Initialize Stereo Block Matcher
    # NumDisparities should be divisible by 16. Higher value means farther search range.

    # The StereoBM_create object is a specialized algorithm (Block Matching) designed to search
    # for the best pixel-by-pixel match between the left and right images.
    stereo_bm = cv2.StereoBM_create(numDisparities = 64, blockSize=15)

    # 2.  Compute disparity map (scaled by 16)

    # For every pixel $(u, v)$ in the left image, it finds the corresponding pixel $(u', v')$
    # in the right image and records the horizontal shift: $\mathbf{d} = u - u'$.

    disparity = stereo_bm.compute(left_gray,right_gray)


    # 3. Create Reprojection Matrix (Q)
    # This matrix contains the camera parameters needed to convert 2D points + disparity
    # into 3D metric coordinates (X, Y, Z).

    Q = np.float32([
        [1,0,0,-PP_X],
        [0,1,0,-PP_Y],
        [0,0,0,FOCAL_LENGTH],
        [0,0,1/BASELINE,0] # Key step: Baseline and focal length give metric / coordinate Z
    ])


    # 4. Reproject 2D points with disparity into 3D (X, Y, Z, W)

    points_4d = cv2.reprojectImageTo3D(disparity,Q,handleMissingValues=True)


    return disparity,points_4d

# This function solves the synchronization problem by linking the (u, v) pixel coordinate of a
# feature to its metric $(X, Y, Z)$ coordinate in the map.

def align_keypoints_to_3d(keypoints_2d, points_4d_map):

    # 1. Force reshape and cast to integer for indexing
    pts = keypoints_2d.reshape(-1, 2)
    u_coords = pts[:, 0].astype(np.int32)
    v_coords = pts[:, 1].astype(np.int32)


    # 2. Clip to prevent crashing if points are on the edge
    h, w = points_4d_map.shape[:2]
    u_coords = np.clip(u_coords, 0, w - 1)
    v_coords = np.clip(v_coords, 0, h - 1)


    # 3. Lookup coordinates (This results in an Nx3 array of X, Y, Z)
    points_3d_aligned = points_4d_map[v_coords, u_coords]


    # 4. Filter out "garbage" points
    # OpenCV marks bad depth with a very large number (like 10000)
    # or sometimes 0. We only want points between 0.1m and 10m away.
    z_values = points_3d_aligned[:, 2]
    valid_mask = (z_values > 0.1) & (z_values < 10.0)


    # 5. Return only the points that actually exist in 3D space
    return points_3d_aligned[valid_mask], valid_mask





# Now, we define the main loop. We'll simulate reading a new stereo pair (images from 2 cameras)
#  for each iteration,
# and in this project, we'll use PnP (Perspective-n-Point)
# which requires 3D points from the previous frame and 2D points from the current frame.

def stereo_slam_loop():
    global prev_points_3d, prev_keypoints_2d, prev_frame_left

    slam_map = SLAM_Map()
    init_3d_plot()

    # 1. Load Photos
    frame_left_init = cv2.imread('left.PNG')
    frame_right_init = cv2.imread('right.PNG')

    if frame_left_init is None or frame_right_init is None:
        print('Couldnt capture one of the images or both')
        return

    # 2. Open Webcam immediately to get target size
    cap = cv2.VideoCapture(0)

    ret, web_frame = cap.read()

    if not ret:
        print("Webcam fail")
        return

    h_web, w_web = web_frame.shape[:2]

    # 3. Resize EVERYTHING to match webcam size BEFORE doing math
    frame_left_init = cv2.resize(frame_left_init, (w_web, h_web))
    frame_right_init = cv2.resize(frame_right_init, (w_web, h_web))

    prev_frame_left = cv2.cvtColor(frame_left_init, cv2.COLOR_BGR2GRAY)
    prev_frame_right = cv2.cvtColor(frame_right_init, cv2.COLOR_BGR2GRAY)

    # 4. Compute Stereo and Detect Features on the RESIZED images
    disparity, initial_points_4d = compute_stereo_depth(prev_frame_left, prev_frame_right)


    raw_keypoints = cv2.goodFeaturesToTrack(
        prev_frame_left, maxCorners=1000, qualityLevel=0.01, minDistance=5
    ).astype(np.float32)


    # 5. Align (This creates the matching 3D and 2D arrays)
    prev_points_3d, valid_mask = align_keypoints_to_3d(raw_keypoints, initial_points_4d)
    prev_keypoints_2d = raw_keypoints[valid_mask] # Now both are same size!


    # Add initial points to plot
    slam_map.add_points(prev_points_3d)
    update_3d_plot(slam_map)

    print(f"Tracking {len(prev_keypoints_2d)} points. Match the photo angle now!")

    frame_count = 0

    while True:
        ret, current_frame_color = cap.read()
        if not ret: break

        current_frame_gray = cv2.cvtColor(current_frame_color, cv2.COLOR_BGR2GRAY)

        # LK Tracking
        current_keypoints_2d, status, err = cv2.calcOpticalFlowPyrLK(
            prev_frame_left, current_frame_gray, prev_keypoints_2d, None, **lk_params
        )

        if status is None:
            print("Optical flow failed - no points found. Keep the camera steady!")
            continue

        status = status.flatten().astype(bool)

        # Filter
        tracked_points_3d = prev_points_3d[status]
        tracked_keypoints_2d_current = current_keypoints_2d[status]


        if len(tracked_points_3d) > 10:
            success, rvec, t_vec, inliers = cv2.solvePnPRansac(
                tracked_points_3d, tracked_keypoints_2d_current, K, None
            )

            if success:
                R_delta, _ = cv2.Rodrigues(rvec)

                R_inv = R_delta.T

                t_inv = -R_inv @ t_vec

                slam_map.update_pose(R_inv, t_inv)

                update_3d_plot(slam_map)

                if frame_count % 5 == 0:
                    update_3d_plot(slam_map)

                pos = slam_map.camera_pose[:3,3]

                print(f'X : {pos[0]:.2f} | Y : {pos[1]:.2f} | Z : {pos[2]:.2f}')

            frame_count += 1




        # VISUALIZATION: Draw dots so you can see if it's working
        for pt in tracked_keypoints_2d_current:
            u, v = pt.ravel().astype(int)
            cv2.circle(current_frame_color, (u, v), 3, (0, 255, 0), -1)

        cv2.imshow('SLAM Live Feed', current_frame_color)

        # Update for next frame

        prev_frame_left = current_frame_gray.copy()
        prev_keypoints_2d = tracked_keypoints_2d_current.copy()
        prev_points_3d = tracked_points_3d.copy()

        if len(prev_keypoints_2d) < 20:
            new_features = cv2.goodFeaturesToTrack(current_frame_gray,maxCorners=1000,qualityLevel=0.01,minDistance=5)
            if new_features is not None:
                pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            slam_map.trajectory = []
            print('Slam Map Trajectory was reset')


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    stereo_slam_loop()




























