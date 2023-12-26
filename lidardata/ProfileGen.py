# Import libraries
import pandas as pd
import numpy as np
import cv2  # Import OpenCV library
import tkinter as tk  # Import tkinter module
from tkinter import Tk, Button, filedialog


## Data Loading function ##
def load_Data(paths: str) -> pd.DataFrame():
    min_distance = 0.3
    max_distance = 10

    # Load the data
    data = pd.read_csv(paths)
    data['z'] = np.arange(data.shape[0]) * 0.2
    # data['z'] = np.zeros(data.shape[0])

    # Convert to millimeters
    data[['x', 'y', 'z']] = data[['x', 'y', 'z']] * 1000  # Convert to mm
    min_distance *= 1000  # Convert to mm
    max_distance *= 1000  # Convert to mm

    # Compute the Euclidean distance from the origin for each point
    data['distance_from_origin'] = np.sqrt(data['x'] ** 2 + data['y'] ** 2)

    # Filter the data to include only points within the specified distance range from the origin
    filtered_data = data[
        (data['distance_from_origin'] >= min_distance) & (data['distance_from_origin'] <= max_distance)].copy()
    return filtered_data


# Load data from csv file
df = load_Data('11 oct 4.csv')


# Define denormalize function
def denormalize(n_val, t_min, t_max, n_min=-1, n_max=1):
    out = ((n_val + abs(n_min)) / (n_max - n_min)) * (t_max - t_min) + t_min
    return out


# Define custom round function
def custom_round(number):
    if number < 0:
        return -round(-number + 0.51)
    else:
        return round(number + 0.51)


# Define target min and max values
target_min = 0
target_max = 800

# Apply denormalize function to x, y, and z columns
df['Z_upd'] = df['z'].apply(denormalize, t_min=target_min, t_max=target_max, n_min=custom_round(min(df['z'])),
                            n_max=custom_round(max(df['z'])))
df['Y_upd'] = df['y'].apply(denormalize, t_min=target_min, t_max=target_max, n_min=custom_round(min(df['y'])),
                            n_max=custom_round(max(df['y'])))
df['X_upd'] = df['x'].apply(denormalize, t_min=target_min, t_max=target_max, n_min=custom_round(min(df['x'])),
                            n_max=custom_round(max(df['x'])))

# Convert data to numpy array
point_cloud = df[["X_upd", "Y_upd", "Z_upd"]].to_numpy()

# Define image size and create an empty image canvas
image_size = (1400, 1400)
image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

# Define initial values for transformation parameters
focal_length = 1.44
image_center_x = 700
image_center_y = 700

# Define initial values for view angle and rotation angle
view_angle = 0  # Angle in degrees between the z-axis and the line of sight
rotation_angle = 0  # Angle in degrees to rotate the point cloud around the z-axis

# Define initial value for perspective flag
perspective_flag = False  # Boolean value to indicate whether to use perspective or orthographic projection


# Define update function that changes the transformation parameters and updates the image
def update():
    # Get the values of the sliders
    global focal_length, image_center_x, image_center_y  # Use global variables to access and modify them
    global view_angle, rotation_angle  # Use global variables to access and modify them
    global perspective_flag  # Use global variable to access and modify it
    focal_length = scale_focal.get()  # Get the value of the focal length slider
    image_center_x = scale_x.get()  # Get the value of the image center x slider
    image_center_y = scale_y.get()  # Get the value of the image center y slider
    view_angle = scale_view.get()  # Get the value of the view angle slider
    rotation_angle = scale_rot.get()  # Get the value of the rotation angle slider
    perspective_flag = var_perspective.get()  # Get the value of the perspective flag checkbox

    # Create a new intrinsic matrix with the updated values
    intrinsics_matrix = np.array([[focal_length, 0, image_center_x],
                                  [0, focal_length, image_center_y],
                                  [0, 0, 1]])

    # Create a rotation matrix with the updated angles
    # Convert angles from degrees to radians
    view_angle_rad = np.deg2rad(view_angle)
    rotation_angle_rad = np.deg2rad(rotation_angle)

    # Use numpy functions to create rotation matrices around x-axis and z-axis
    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(view_angle_rad), -np.sin(view_angle_rad)],
                      [0, np.sin(view_angle_rad), np.cos(view_angle_rad)]])

    rot_z = np.array([[np.cos(rotation_angle_rad), -np.sin(rotation_angle_rad), 0],
                      [np.sin(rotation_angle_rad), np.cos(rotation_angle_rad), 0],
                      [0, 0, 1]])

    # Combine the two rotation matrices by matrix multiplication
    rot_matrix = np.dot(rot_z, rot_x)

    # Apply the rotation matrix to the point cloud data
    rotated_point_cloud = np.dot(rot_matrix, point_cloud.T).T

    # Project points onto image plane using the new intrinsic matrix and rotated point cloud data
    # Use perspective or orthographic projection depending on the perspective flag
    if perspective_flag:  # Use perspective projection
        image_points = np.dot(intrinsics_matrix, rotated_point_cloud.T).T
        # Divide by the third coordinate to get the normalized image coordinates
        image_points[:, 0] /= image_points[:, 2]
        image_points[:, 1] /= image_points[:, 2]
    else:  # Use orthographic projection
        image_points = np.dot(intrinsics_matrix[:2, :2], rotated_point_cloud.T[:2, :]).T
        # Add the image center offsets to get the pixel coordinates
        image_points[:, 0] += image_center_x
        image_points[:, 1] += image_center_y

    # Clear the previous image canvas by filling it with zeros
    image.fill(0)

    # Assign color values to image pixels based on projected points
    for row in image_points:
        if perspective_flag:  # Use perspective projection
            x, y, _ = row  # Unpack three values
        else:  # Use orthographic projection
            x, y = row  # Unpack two values
        x, y = int(x), int(y)

        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
            image[y, x] = [255, 255, 255]

    # Display the updated image using OpenCV
    cv2.imshow("Image", image)  # Show the image in a window named "Image"


# Create a main GUI window
root = tk.Tk()
root.title("Image Projection")


def save_img():
    # Ask the user for a filename
    filename = filedialog.asksaveasfilename(defaultextension=".jpg")
    # If the user entered a filename, save the image using OpenCV
    if filename:
        cv2.imwrite(filename, image)


# Create a label for instructions
label = tk.Label(root, text="Use the sliders to change the transformation parameters and update the image")
label.pack()

# Create a slider for focal length
scale_focal = tk.Scale(root, from_=0.01, to=1000.00, resolution=0.01,
                       orient=tk.HORIZONTAL,
                       label="Focal Length",
                       command=lambda x: update())  # Call update function when slider value changes

scale_focal.set(focal_length)  # Set initial value of slider to focal length variable
scale_focal.pack()

# Create a slider for image center x
scale_x = tk.Scale(root, from_=0, to=image_size[0], resolution=1,
                   orient=tk.HORIZONTAL,
                   label="Image Center X",
                   command=lambda x: update())  # Call update function when slider value changes

scale_x.set(image_center_x)  # Set initial value of slider to image center x variable
scale_x.pack()

# Create a slider for image center y
scale_y = tk.Scale(root, from_=0, to=image_size[1], resolution=1,
                   orient=tk.HORIZONTAL,
                   label="Image Center Y",
                   command=lambda x: update())  # Call update function when slider value changes

scale_y.set(image_center_y)  # Set initial value of slider to image center y variable
scale_y.pack()

# Create a slider for view angle
scale_view = tk.Scale(root, from_=-90, to=90, resolution=1,
                      orient=tk.HORIZONTAL,
                      label="View Angle",
                      command=lambda x: update())  # Call update function when slider value changes

scale_view.set(view_angle)  # Set initial value of slider to view angle variable
scale_view.pack()

# Create a slider for rotation angle
scale_rot = tk.Scale(root, from_=-180, to=180, resolution=1,
                     orient=tk.HORIZONTAL,
                     label="Rotation Angle",
                     command=lambda x: update())  # Call update function when slider value changes

scale_rot.set(rotation_angle)  # Set initial value of slider to rotation angle variable
scale_rot.pack()

# Create a checkbox for perspective flag
var_perspective = tk.BooleanVar()  # Create a boolean variable to store the checkbox value
var_perspective.set(perspective_flag)  # Set initial value of checkbox to perspective flag variable

chk_perspective = tk.Checkbutton(root, text="Perspective", variable=var_perspective,
                                 command=lambda: update())  # Call update function when checkbox value changes

chk_perspective.pack()  # Pack the checkbox widget

# Create a button to close the GUI and the image window
button = tk.Button(root, text="Close", command=lambda: (cv2.destroyAllWindows(),
                                                        root.destroy()))  # Call cv2.destroyAllWindows and root.destroy functions when button is clicked
# Create a button that calls the save_image function
button = Button(root, text="Save image", command=save_img)
button.pack()

# Display the initial image using OpenCV
cv2.imshow("Image", image)  # Show the image in a window named "Image"

# Run the main loop of the GUI
root.mainloop()
