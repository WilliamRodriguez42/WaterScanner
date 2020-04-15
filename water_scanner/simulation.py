import trimesh
import numpy as np
import pickle
import sys
import trimesh.intersections
import trimesh.repair
import networkx as nx
import shapely

TEST_NUM = 5
NUM_STEPS = 61
NUM_RANDOM_SAMPLES = 500

def slice_mesh_plane_bidirectional(mesh, plane_normal, plane_origin, section):
	# Compute positive side of plane
	dots_cache = np.dot(plane_normal, (mesh.vertices - plane_origin).T)[mesh.faces]
	positive = trimesh.intersections.slice_mesh_plane(mesh,
													 plane_normal=plane_normal,
													 plane_origin=plane_origin,
													 cached_dots=dots_cache)
	
	# Compute negative side of plane
	dots_cache *= -1
	negative = trimesh.intersections.slice_mesh_plane(mesh,
													 plane_normal=-plane_normal,
													 plane_origin=plane_origin,
													 cached_dots=dots_cache)
	
	# Compute the cap
	cap = trimesh.intersections.mesh_plane(mesh,
									   plane_normal=-plane_normal, 
									   plane_origin=plane_origin, 
									   return_faces=False, 
									   cached_dots=dots_cache)
	
	temp_vertices, cap_faces = section.triangulate()
	if temp_vertices.shape[0] == 0:
		positive.show()

	cap_vertices = np.zeros((temp_vertices.shape[0], 3))
	cap_vertices[:, 0] = temp_vertices[:, 0]
	cap_vertices[:, 1] = temp_vertices[:, 1]
	plane = trimesh.Trimesh(cap_vertices, cap_faces)
	plane.apply_transform(section.metadata['to_3D'])
	cap_vertices = plane.vertices
	cap_faces = plane.faces
	# plane.show()

	# # Append cap to positive mesh
	positive_vertices, n_positive_vertices = positive.vertices, len(positive.vertices)
	positive_vertices = np.append(positive_vertices, cap_vertices, axis=0)
	positive_faces = positive.faces
	positive_faces = np.append(positive_faces, np.flip(cap_faces)+n_positive_vertices, axis=0)
	# Construct negative mesh
	positive_mesh = trimesh.Trimesh(positive_vertices, positive_faces)
	# print(positive_mesh.volume)
	# print(positive_mesh.split(only_watertight=False))
	
	# Append cap to negative mesh
	negative_vertices, n_negative_vertices = negative.vertices, len(negative.vertices)
	negative_vertices = np.append(negative_vertices, cap_vertices, axis=0)
	negative_faces = negative.faces
	negative_faces = np.append(negative_faces, cap_faces+n_negative_vertices, axis=0)
	# Construct negative mesh
	negative_mesh = trimesh.Trimesh(negative_vertices, negative_faces)
	# print(negative_mesh.volume)
	return [positive_mesh, negative_mesh]

np.random.seed(0)

mesh = trimesh.load_mesh('tests/{}.stl'.format(TEST_NUM))
print(mesh.volume)
mesh.rezero()

max_distance = np.max(np.sqrt(np.sum(np.square(mesh.bounds - mesh.centroid), axis=1)))
STEP_SIZE = max_distance / NUM_STEPS * 2
print(STEP_SIZE)

heights = np.arange(-max_distance, max_distance, STEP_SIZE)
volumes = np.zeros((NUM_RANDOM_SAMPLES, heights.size))


n = NUM_RANDOM_SAMPLES

golden_angle = np.pi * (3 - np.sqrt(5))
theta = golden_angle * np.arange(n)
z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
radius = np.sqrt(1 - z * z)

normals = np.zeros((n, 3))
normals[:,0] = radius * np.cos(theta)
normals[:,1] = radius * np.sin(theta)
normals[:,2] = z


# normals = np.random.rand(NUM_RANDOM_SAMPLES, 3)
# normals /= np.sqrt(np.sum(np.square(normals), axis=1))[:, None]


# normals[0] = (1, 0, 0)
# normals[1] = (0, 1, 0)
# normals[2] = (0, 0, 1)

# normals = normals[normals[:, 0] >= 0]
# normals = normals[normals[:, 1] >= 0]
# normals = normals[normals[:, 2] >= 0]

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(normals[:, 0], normals[:, 1], normals[:, 2], c='r', marker='o')
# plt.show()
# sys.exit()

for i in range(NUM_RANDOM_SAMPLES):
	sliced_mesh = mesh.section_multiplane(
		plane_origin=mesh.centroid,
		plane_normal=normals[i],
		heights=heights)
	for j, sm in enumerate(sliced_mesh):
		if sm is not None:
			volumes[i, j] = sm.area * STEP_SIZE

	# sections = mesh.section_multiplane(
	# 	plane_origin=mesh.centroid,
	# 	plane_normal=normals[i],
	# 	heights=heights+STEP_SIZE/2)

	# sliced_mesh = mesh
	# # print(sliced_mesh.volume)
	# for j in range(heights.size):
		
	# 	if sections[j] is not None and sections[j].vertices.shape[0] > 0:
	# 		# print(dir(sections[j]))
	# 		height = heights[j]
	# 		sliced_mesh, neg_mesh = slice_mesh_plane_bidirectional(
	# 			sliced_mesh,
	# 			normals[i],
	# 			mesh.centroid+normals[i]*(height+STEP_SIZE/2),
	# 			sections[j])

	# 		volumes[i, j] = neg_mesh.volume
	# 		# print(volumes[i, j])
	# 		# sliced_mesh.show()
	# 		# neg_mesh.show()
	# 	else:
	# 		volumes[i, j] = 0

	print(i)

# volumes *= STEP_SIZE

with open('pickle/{}/volumes.pkl'.format(TEST_NUM), 'wb+') as f:
	pickle.dump(volumes, f)

with open('pickle/{}/normals.pkl'.format(TEST_NUM), 'wb+') as f:
	pickle.dump(normals, f)

with open('pickle/{}/heights.pkl'.format(TEST_NUM), 'wb+') as f:
	pickle.dump(heights, f)

with open('pickle/{}/centroid.pkl'.format(TEST_NUM), 'wb+') as f:
	pickle.dump(mesh.centroid, f)

with open('pickle/{}/STEP_SIZE.pkl'.format(TEST_NUM), 'wb+') as f:
	pickle.dump(STEP_SIZE, f)