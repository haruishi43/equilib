from typing import List, Dict
import numpy as np

def calculate_tangent_rots(
    subdivision_level: List[int]
    ) -> List[Dict[str, float]]:
    rots = [[] for _ in subdivision_level]
    radius = 10
    for i, sub_lvl in enumerate(subdivision_level):
        faces = init_20_faces(radius)
        for _ in range(0, sub_lvl):
            faces = subdivide_faces(faces, radius)
        angles = calculate_angles(faces)
        for ang in angles:
            rots[i].append(
                {
                    'roll': 0.,
                    'pitch': ang[1],
                    'yaw': ang[0]
                }
            )
    return rots

def calculate_tangent_angles(
    subdivision_level: List[int]
    ) -> List[np.ndarray]:
    angles = [[] for _ in subdivision_level]
    radius = 10
    for i, sub_lvl in enumerate(subdivision_level):
        faces = init_20_faces(radius)
        for _ in range(0, sub_lvl):
            faces = subdivide_faces(faces, radius)
        angles[i] = calculate_angles(faces)
    return angles

def calculate_angles(faces:np.ndarray) -> np.ndarray:
    centroids = calculate_centroids(faces)
    angles = np.zeros((centroids.shape[0], 2))
    for i, c in enumerate(centroids):
        lat = calculate_lat(c, [c[0], c[1], 0])
        lon = calculate_lon(c[:2], [0,1]) 
        angles[i] = [lon, lat]
    angles[:, 1] *= np.sign(centroids[:, 2])
    return angles * -1

def subdivide_faces(faces: np.ndarray, radius:int=10):
    """
        TODO: fix crashes after second subdividing
    """
    new_faces = np.zeros((faces.shape[0]*4,3,3))
    for i, face in enumerate(faces):
        v1,v2,v3 = face

        v1_new = __computeHalfVertex(v1, v2)
        v2_new = __computeHalfVertex(v2, v3)
        v3_new = __computeHalfVertex(v1, v3)

        # add 4 new triangles to vertex array
        new_faces[i*4] = np.array([v1,    v1_new, v3_new])
        new_faces[i*4+1] = np.array([v1_new, v2,    v2_new])
        new_faces[i*4+2] = np.array([v1_new, v2_new, v3_new])
        new_faces[i*4+3] = np.array([v3_new, v2_new, v3])
    return new_faces

def init_icosahedron_vertices(radius:int=10) -> np.ndarray:
    PI, r = np.pi, radius
    H_ANGLE, V_ANGLE = PI / 180 * 72, np.arctan(1.0 / 2)
    vertices = np.zeros((12,3))                         
    hAngle1, hAngle2 = -PI / 2 - H_ANGLE / 2, -PI / 2

    vertices[0] = (0,0,r)
    vertices[-1] = (0,0,-r)
    
    xy,z = np.ones((10,2)), np.ones((10,1))
    xy *= r*np.cos(V_ANGLE)
    z *= r*np.sin(V_ANGLE)
    h1 = (np.arange(0,5) * H_ANGLE) + hAngle1
    h2 = (np.arange(0,5) * H_ANGLE) + hAngle2
    xy[:5,0] = xy[:5,0] * np.cos(h1)
    xy[:5,1] = xy[:5,1] * np.sin(h1)
    xy[5:,0] = xy[5:,0] * np.cos(h2)
    xy[5:,1] = xy[5:,1] * np.sin(h2)
    z[5:] = -z[5:]
    xyz = np.hstack([xy,z])
    
    vertices[1:-1] = xyz
    return vertices

def init_20_faces(radius:int=10):
    vertices = init_icosahedron_vertices(radius)
    faces = np.zeros((20,3,3))
    upper_pent = vertices[1:6]
    down_pent = vertices[-6:]
    for i in range(1, 6):
        faces[i-1] = np.array([vertices[0], upper_pent[i%5], upper_pent[i-1]]).reshape((1,3,3))
        faces[-i] = np.array([vertices[-1], down_pent[i%5], down_pent[i-1]]).reshape((1,3,3))
    # head, middle and tail pointers for triangles
    ptr_h, ptr_m, ptr_t = -1, 0, 0
    for i in range(1, 11):
        if i%2 != 0:
            ptr_h += 1
            ptr_t += 1
            faces[i+4] = np.array([upper_pent[ptr_h], down_pent[ptr_m%5], upper_pent[ptr_t%5]]).reshape((1,3,3))
        if i%2 == 0:
            ptr_m += 1
            faces[i+4] = np.array([down_pent[ptr_h], upper_pent[ptr_m%5], down_pent[ptr_t%5]]).reshape((1,3,3))
    return faces


def calculate_centroids(faces:np.ndarray):
    centroids = np.zeros((faces.shape[0],3))
    for i, face in enumerate(faces):
        centroids[i] = face.sum(axis=0)/3
    return centroids
    
def __computeHalfVertex(v1, v2, radius:int = 10):
    new_v = v1+v2
    scale = radius / np.sqrt(np.sum(new_v**2))
    return new_v * scale
    

def calculate_lat(vec1: np.ndarray, vec2: np.ndarray):
    unit_vector_1 = vec1 / np.linalg.norm(vec1)
    unit_vector_2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(dot_product)

def calculate_lon(vec1: np.ndarray, vec2: np.ndarray):
    dot = np.dot(vec1, vec2)
    det = np.linalg.det(np.vstack([vec1, vec2]))
    return np.arctan2(det, dot)