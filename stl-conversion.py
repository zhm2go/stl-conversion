import numpy as np
import matplotlib.pyplot as plt
import pydicom
from os import walk

import trimesh as trimesh
from stl import mesh
import plotly.graph_objects as go
import SimpleITK as sitk
import scipy

def resample_image(itk_image, ref_image):
    out_spacing = ref_image.GetSpacing()
    out_size = ref_image.GetSize()
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(ref_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0.0)
    resample.SetInterpolator(sitk.sitkLinear)
    return resample.Execute(itk_image)


def get_itk(image, spacing, origin, direction):
    itkimage = sitk.GetImageFromArray(image)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    itkimage.SetDirection(direction)
    return itkimage

def get_tform(N, position_first, position_last, orientation, spacing):
    # transformation matrix
    M = np.array(
        [
            [orientation[0]*spacing[0],     orientation[3]*spacing[1],  (position_last[0] - position_first[0])/(N-1),   position_first[0]],
            [orientation[1]*spacing[0],     orientation[4]*spacing[1],  (position_last[1] - position_first[1])/(N-1),   position_first[1]],
            [orientation[2]*spacing[0],     orientation[5]*spacing[1],  (position_last[2] - position_first[2])/(N-1),   position_first[2]],
            [0,                             0,                          0,                                              1],
        ]
    )
    return M

def ijk2XYZ(i, j, k, M):

    initial_shape = i.shape

    # pixel_loc = [j ; i ; k ; 1] (already swapped i and j)
    pixel_loc = np.concatenate(
                    [
                        np.reshape(j, [-1, 1]),
                        np.reshape(i, [-1, 1]),
                        np.reshape(k, [-1, 1]),
                        np.ones_like(np.reshape(k, [-1, 1])),
                    ],
                    axis=1
    ).T

    # convert back from homogeneous to regular coords
    rcs_coord = np.dot(M, pixel_loc)
    rcs_coord = np.delete(rcs_coord, -1, axis=0)
    rcs_coord = rcs_coord.T

    X = np.reshape(rcs_coord[:, 0], initial_shape)
    Y = np.reshape(rcs_coord[:, 1], initial_shape)
    Z = np.reshape(rcs_coord[:, 2], initial_shape)

    return X, Y, Z

def stl_to_binarymask():
    stl_file = 'cube.stl'
    stl_mesh = mesh.Mesh.from_file(stl_file)

    # open all DCM files in path
    dcm_path = 't2_ax_0/'
    _, _, filenames = next(walk(dcm_path))
    filenames.sort(reverse=True)
    n_dcm = len(filenames)
    dc = [pydicom.dcmread(dcm_path + f) for f in filenames]

    # read important tags
    slice_position = np.array([d[0x0020, 0x0032].value for d in dc])
    slice_orientation = np.array([d[0x0020, 0x0037].value for d in dc])
    slice_spacing = np.array([d[0x0028, 0x0030].value for d in dc])
    slice_image = np.array([d.pixel_array.astype(np.float32) for d in dc])
    slice_location = np.array([d[0x0020, 0x1041].value for d in dc])

    M = get_tform(n_dcm, slice_position[0], slice_position[-1], slice_orientation[0], slice_spacing[0])
    print(M)
    stl_path = 'cube.stl'
    tri_mesh = trimesh.load_mesh(stl_path)
    assert tri_mesh.is_watertight
    #binary_mask = [[[False for _ in range(256)] for _ in range(256)] for _ in range(n_dcm)]
    binary_mask = []
    frames = []
    min_z, max_z = min(stl_mesh.z.flatten()), max(stl_mesh.z.flatten())
    min_x, max_x = min(stl_mesh.x.flatten()), max(stl_mesh.x.flatten())
    min_y, max_y = min(stl_mesh.y.flatten()), max(stl_mesh.y.flatten())
    min_x_idx, max_x_idx, min_y_idx, max_y_idx = None, None, None, None
    X_coor, Y_coor, Z_coor = [], [], []
    for k, slc_ind in enumerate(range(0, n_dcm, 5)):
        print(k)
        slc_i, slc_j, slc_k = np.meshgrid(np.arange(256), np.arange(256), slc_ind, indexing='ij')
        X, Y, Z = ijk2XYZ(
            slc_i,
            slc_j,
            slc_k,
            M
        )
        X, Y, Z = np.squeeze(X), np.squeeze(Y), np.squeeze(Z)
        X_coor.append(X)
        Y_coor.append(Y)
        Z_coor.append(Z)
        if min_x_idx is None:
            min_x_idx, max_x_idx = 0, 255
            while X[0][min_x_idx] < min_x:
                min_x_idx += 1
            while X[0][max_x_idx] > max_x:
                max_x_idx -= 1
            min_y_idx, max_y_idx = 0, 255
            while Y[min_y_idx][0] < min_y:
                min_y_idx += 1
            while Y[max_y_idx][0] > max_y:
                max_y_idx -= 1
        if min_z <= Z[0][0] <= max_z:
            binary_slice = []
            for i in range(0, 256):
                binary_row = tri_mesh.contains([[X[i][j], Y[i][j], Z[i][j]] for j in range(0, 256)])
                binary_slice.append(binary_row)
            binary_mask.append(binary_slice)
            fig, axes = plt.subplots(1, 3, figsize=(25, 25))
            axes[0].imshow(slice_image[slc_ind])
            axes[1].imshow(binary_mask[k])
            axes[2].imshow(slice_image[slc_ind])
            axes[2].imshow(binary_mask[k], alpha=0.5)
            fig.show()
        else:
            binary_mask.append([[False for _ in range(256)] for _ in range(256)])
            '''for i in range(max(0, min_x_idx - 10), min(255, max_x_idx + 10)):
                for j in range(max(0, min_y_idx - 10), min(255, max_y_idx + 10)):
                    if tri_mesh.contains([[X[i][j], Y[i][j], Z[i][j]]]):
                        binary_mask[slc_ind][i][j] = True'''
        frames.append(
            go.Frame(
                data=go.Surface(
                    x=np.squeeze(X),
                    y=np.squeeze(Y),
                    z=np.squeeze(Z),
                    surfacecolor=binary_mask[k],
                    opacity=1,
                    showscale=False,
                ),
                name=str(k)
            )
        )

    fig = go.Figure(frames=frames)
    slc_i, slc_j, slc_k = np.meshgrid(np.arange(256), np.arange(256), 0, indexing='ij')
    X, Y, Z = ijk2XYZ(
        slc_i,
        slc_j,
        slc_k,
        M
    )

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        x=np.squeeze(X),
        y=np.squeeze(Y),
        z=np.squeeze(Z),
        surfacecolor=binary_mask[0],
        showscale=False
    )
    )

    fig.add_trace(
        go.Scatter3d(
            x=stl_mesh.x.flatten(),
            y=stl_mesh.y.flatten(),
            z=stl_mesh.z.flatten(),
            mode='markers',
            marker=dict(
                size=1,
                opacity=0.5
            )
        )
    )

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.update_layout(
        title='t2_ax_0',
        width=1000,
        height=800,
        scene=dict(
            xaxis=dict(range=[-200, 200], autorange=False),
            yaxis=dict(range=[-200, 200], autorange=False),
            zaxis=dict(range=[-200, 200], autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders
    )
    fig.show()
    return binary_mask, X_coor, Y_coor, Z_coor, (min_z, max_z, min_x_idx, max_x_idx, min_y_idx, max_y_idx)

def binarymask_to_stl(binary_mask, X_coor, Y_coor, Z_coor, scope):
    min_z, max_z, min_x_idx, max_x_idx, min_y_idx, max_y_idx = scope
    contour = []
    num_slice = len(Z_coor)
    is_findpointinside = False
    for k in range(num_slice):
        if min_z <= Z_coor[k][0][0] <= max_z:
            for i in range(0, 256, 1):
                for j in range(0, 256, 1):
                    if binary_mask[k][i][j] and ((k == 0 or not binary_mask[k - 1][i][j]) or (k == num_slice - 1 or not binary_mask[k + 1][i][j])\
                        or (i == 0 or not binary_mask[k][i - 1][j]) or (i == 255 or not binary_mask[k][i + 1][j])\
                        or (j == 0 or not binary_mask[k][i][j - 1]) or (j == 255 or not binary_mask[k][i][j + 1])):
                        contour.append([X_coor[k][i][j], Y_coor[k][i][j], Z_coor[k][i][j]])
    print(len(contour))
    tri = scipy.spatial.ConvexHull(contour)
    len_faces = len(tri.simplices)
    print(len_faces)
    cube = mesh.Mesh(np.zeros(len_faces, dtype=mesh.Mesh.dtype))
    for i, f in enumerate(tri.simplices):
        for j in range(3):
            cube.vectors[i][j] = contour[f[j]]
    cube.save('cube.stl')
    cube_mesh = trimesh.load_mesh('cube.stl')
    print(cube_mesh.is_watertight)

if __name__ == "__main__":
    binary_mask, X_coor, Y_coor, Z_coor, scope = stl_to_binarymask()
    #binarymask_to_stl(binary_mask, X_coor, Y_coor, Z_coor, scope)

