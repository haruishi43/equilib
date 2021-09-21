from typing import List, Dict
import torch
from torch import tensor
from torch._C import device
from torch_utils import pi

def calculate_tangent_rots(
    subdivision_level: List[int],
    device: torch.device = torch.device("cpu")
    ) -> List[Dict[str, float]]:
    rots = [[] for _ in subdivision_level]
    radius = 10
    for i, sub_lvl in enumerate(subdivision_level):
        faces = init_20_faces(radius, device=device)
        for _ in range(0, sub_lvl):
            faces = subdivide_faces(faces, radius, device=device)
        angles = calculate_angles(faces, device=device)
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
    subdivision_level: List[int],
    device: torch.device = torch.device("cpu")
    ) -> List[torch.Tensor]:
    angles = [[] for _ in subdivision_level]
    radius = 10
    for i, sub_lvl in enumerate(subdivision_level):
        faces = init_20_faces(radius, device=device)
        for _ in range(0, sub_lvl):
            faces = subdivide_faces(faces, radius, device=device)
        angles[i] = calculate_angles(faces, device=device)
    return angles

def calculate_angles(
    faces: torch.Tensor,
    device: torch.device=torch.device("cpu")
    ) -> torch.Tensor:
    centroids = calculate_centroids(faces, device)
    angles = torch.zeros(centroids.shape[0], 2, device=device)
    for i, c in enumerate(centroids):
        lat = calculate_lat(c, tensor([c[0], c[1], 0], dtype=torch.float32, device=device))
        lon = calculate_lon(c[:2], tensor([0,1], dtype=torch.float32, device=device)) 
        angles[i] = tensor([lon, lat])
    angles[:, 1] *= torch.sign(centroids[:, 2])
    return angles * -1

def subdivide_faces(
    faces: torch.Tensor, 
    radius: int=10, 
    device: torch.device = torch.device("cpu")
    ):
    """
        TODO: fix crashes after second subdividing
    """
    new_faces = torch.zeros(faces.shape[0]*4,3,3, device=device)
    for i, face in enumerate(faces):
        v1,v2,v3 = face

        v1_new = __computeHalfVertex(v1, v2)
        v2_new = __computeHalfVertex(v2, v3)
        v3_new = __computeHalfVertex(v1, v3)

        # add 4 new triangles to vertex array
        new_faces[i*4] = torch.vstack([v1,    v1_new, v3_new])
        new_faces[i*4+1] = torch.vstack([v1_new, v2,    v2_new])
        new_faces[i*4+2] = torch.vstack([v1_new, v2_new, v3_new])
        new_faces[i*4+3] = torch.vstack([v3_new, v2_new, v3])
    return new_faces

def init_icosahedron_vertices(
    radius: int = 10,
    device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
    PI, r = pi.to(device), radius
    H_ANGLE = torch.clone(PI) / 180 * 72
    V_ANGLE = torch.arctan(tensor(1.0 / 2, device=device))
    vertices = torch.zeros(12,3, device=device)                        
    hAngle1, hAngle2 = -PI / 2 - H_ANGLE / 2, -PI / 2

    vertices[0] = tensor([0,0,r], device=device)
    vertices[-1] = tensor([0,0,-r], device=device)
    
    xy,z = torch.ones(10,2, device=device), torch.ones(10,1, device=device)
    xy *= r*torch.cos(V_ANGLE)
    z *= r*torch.sin(V_ANGLE)
    h1 = (torch.arange(0,5, device=device) * H_ANGLE) + hAngle1
    h2 = (torch.arange(0,5, device=device) * H_ANGLE) + hAngle2
    xy[:5,0] = xy[:5,0] * torch.cos(h1)
    xy[:5,1] = xy[:5,1] * torch.sin(h1)
    xy[5:,0] = xy[5:,0] * torch.cos(h2)
    xy[5:,1] = xy[5:,1] * torch.sin(h2)
    z[5:] = -z[5:]
    xyz = torch.hstack([xy,z])
    
    vertices[1:-1] = xyz
    return vertices

def init_20_faces(
    radius: int = 10,
    device: torch.device = torch.device("cpu")
    ):
    vertices = init_icosahedron_vertices(radius, device)
    faces = torch.zeros(20,3,3, device=device)
    upper_pent = vertices[1:6]
    down_pent = vertices[-6:]
    for i in range(1, 6):
        faces[i-1] = torch.vstack([vertices[0], upper_pent[i%5], upper_pent[i-1]])
        faces[-i] = torch.vstack([vertices[-1], down_pent[i%5], down_pent[i-1]])
    # head, middle and tail pointers for triangles
    ptr_h, ptr_m, ptr_t = -1, 0, 0
    for i in range(1, 11):
        if i%2 != 0:
            ptr_h += 1
            ptr_t += 1
            faces[i+4] = torch.vstack([upper_pent[ptr_h], down_pent[ptr_m%5], upper_pent[ptr_t%5]]).reshape((1,3,3))
        if i%2 == 0:
            ptr_m += 1
            faces[i+4] = torch.vstack([down_pent[ptr_h], upper_pent[ptr_m%5], down_pent[ptr_t%5]]).reshape((1,3,3))
    return faces


def calculate_centroids(
    faces: torch.Tensor,
    device: torch.device = torch.device("cpu")
    ):
    centroids = torch.zeros((faces.shape[0],3), device=device)
    for i, face in enumerate(faces):
        centroids[i] = face.sum(axis=0)/3
    return centroids
    
def __computeHalfVertex(v1, v2, radius:int = 10):
    new_v = v1+v2
    scale = radius / torch.sqrt(torch.sum(new_v**2))
    return new_v * scale
    

def calculate_lat(
    vec1: torch.Tensor,
    vec2: torch.Tensor, 
    device: torch.device = torch.device("cpu")
    ):
    unit_vector_1 = (vec1 / torch.linalg.norm(vec1)).to(device)
    unit_vector_2 = (vec2 / torch.linalg.norm(vec2)).to(device)
    dot_product = torch.dot(unit_vector_1, unit_vector_2)
    return torch.arccos(dot_product)

def calculate_lon(
    vec1: torch.Tensor, 
    vec2: torch.Tensor
    ):
    dot = torch.dot(vec1, vec2)
    det = torch.linalg.det(torch.vstack([vec1, vec2]))
    return torch.atan2(det, dot)