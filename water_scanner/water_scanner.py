import numpy as np
import pickle
import open3d as o3d
import math
import multiprocessing
import sys
import ctypes
import scipy.signal
import scipy.ndimage
import random

# ============================================================================================
# Load Data
# ============================================================================================
TEST_NUM = 5

with open('pickle/{}/volumes.pkl'.format(TEST_NUM), 'rb') as f:
	volumes = pickle.load(f)

with open('pickle/{}/normals.pkl'.format(TEST_NUM), 'rb') as f:
	normals = pickle.load(f)

with open('pickle/{}/heights.pkl'.format(TEST_NUM), 'rb') as f:
	heights = pickle.load(f)

with open('pickle/{}/centroid.pkl'.format(TEST_NUM), 'rb') as f:
	centroid = pickle.load(f)

with open('pickle/{}/STEP_SIZE.pkl'.format(TEST_NUM), 'rb') as f:
	STEP_SIZE = pickle.load(f)

RESOLUTION_FACTOR = 2
RESOLVED_STEP_SIZE = STEP_SIZE / RESOLUTION_FACTOR
RESOLVED_VOXEL_VOLUME = RESOLVED_STEP_SIZE ** 3
RESOLVED_NUM_STEPS = heights.size*RESOLUTION_FACTOR

voxels = np.ones((RESOLVED_NUM_STEPS, RESOLVED_NUM_STEPS, RESOLVED_NUM_STEPS), dtype=np.float64)
voxel_indices = np.argwhere(voxels)
voxel_locations = voxel_indices * RESOLVED_STEP_SIZE + heights.min() # This pretty much sets voxel_locations to every permutation of heights
voxel_distances = np.sqrt(np.sum(np.square(voxel_locations), axis=1))
distance_mask = voxel_distances <= heights.max()
voxel_locations = voxel_locations[distance_mask]
voxel_distances = voxel_distances[distance_mask]
voxel_indices = voxel_indices[distance_mask]
shared_a = multiprocessing.Array(ctypes.c_double, voxel_locations.shape[0])
with shared_a.get_lock():
	a = np.frombuffer(shared_a.get_obj())
	a[:] = 0.5

print(RESOLVED_NUM_STEPS)
""" Coordinate System
			z
			|
			|
			|
			|
			+---------- y
		   /
		  /
		 x
"""

def parse_section(args):
	volume = args[0]
	normal = args[1]

	projected_distance = normal.dot(voxel_locations.T)
	projected_distance = projected_distance
	ms = []
	vs = []

	for j in range(heights.size):
		height = heights[j]
		# m = 1 - np.abs(projected_distance - height) / RESOLVED_STEP_SIZE
		# m[m < 0] = 0
		m = np.logical_and(projected_distance <= height + STEP_SIZE/2, projected_distance > height-STEP_SIZE/2)
		#m = projected_distance <= height
		v = volume[j]
		ms.append(m)
		vs.append(v)
	return ms, vs

ms = []
vs = []

# args = list(zip(volumes, normals))
# pool = multiprocessing.Pool(processes=6)
# i = 0
# for m, v in pool.imap_unordered(parse_section, args):
# 	ms.extend(m)
# 	vs.extend(v)
# 	print(i)
# 	i += 1

for i in range(volumes.shape[0]):
	volume = volumes[i]
	normal = normals[i]

	projected_distance = normal.dot(voxel_locations.T)
	projected_distance = projected_distance
	for j in range(heights.size):
		height = heights[j]
		# m = 1 - np.abs(projected_distance - height) / RESOLVED_STEP_SIZE
		# m[m < 0] = 0
		m = np.logical_and(projected_distance <= height + STEP_SIZE/2, projected_distance > height-STEP_SIZE/2)
		#m = projected_distance <= height
		v = volume[j]
		ms.append(m)
		vs.append(v)
	print(i)




# ============================================================================================
# AI Solver
# ============================================================================================

def target():
	global shared_a, ms, vs
	WEATHER_THRESHOLD = 0.1
	lr = 1

	with shared_a.get_lock():
		a = np.frombuffer(shared_a.get_obj())
		working_set = np.ones_like(a).astype(np.bool)

	n = 0
	while working_set.sum() > 0:
		total_error = 0
		for i in range(len(ms)):
			m = np.logical_and(ms[i], working_set)
			# m = ms[i]
			v = vs[i]

			with shared_a.get_lock():
				a = np.frombuffer(shared_a.get_obj())

				if v <= 0:
					working_set[m] = False
					a[m] = 0
					continue

				# _v = a[m].dot(ms[i][m].T) * RESOLVED_VOXEL_VOLUME
				_v = np.sum(a[m] > 0.5) * RESOLVED_VOXEL_VOLUME

			dv = (v - _v) 
			total_error += abs(dv)

			with shared_a.get_lock():
				a = np.frombuffer(shared_a.get_obj())
				a[m] += lr * dv #/ m.sum() #* ms[i][m] # / (np.sqrt(dv_var[m]) + 1e-9)

		lr *= 0.9
		print("\nEpoch {} summary".format(n))
		print("Total error from measurements {}".format(total_error))
		print("Learning rate: {}".format(lr))

		if n % 1 == 0: # and n != 0:
			vi = voxel_indices[working_set]
			voxel_shape = np.zeros(3, dtype=np.uint32)
			voxel_origin = np.zeros(3, dtype=np.uint32)
			for i in range(3):
				voxel_origin[i] = vi[:, i].min()
				voxel_shape[i] = vi[:, i].max() - vi[:, i].min() + 1

			voxels = np.zeros(voxel_shape, dtype=np.float64)
			vi, vj, vk = (vi - voxel_origin).T

			with shared_a.get_lock():
				a = np.frombuffer(shared_a.get_obj())
				# mask = a > 0.5
				# a[mask] = 1
				# a[np.logical_not(mask)] = 0

				# Remove values from the working set that we are confident are a part of the final model
				# Then update the volumes list by removing the volume occupied by these voxels
				# confident_voxels = np.logical_and(a >= 1, working_set)
				# a[confident_voxels] = 1

				# for i in range(len(ms)):
				# 	m = np.logical_and(ms[i], confident_voxels)
				# 	# _v = a[m].dot(ms[i][m].T) * RESOLVED_VOXEL_VOLUME
				# 	_v = np.sum(a[m] > 0.5) * RESOLVED_VOXEL_VOLUME
				# 	vs[i] -= _v

				# Remove values from the working set that we are confident are not a part of the model
				# remove = a < 0.1

				# Denoise
				voxels[vi, vj, vk] = a[working_set] > 0.5
				# print(voxels.sum())
				surround = scipy.ndimage.gaussian_filter(voxels, sigma=1)
				# print(surround[vi, vj, vk].size)
				# print(surround[vi, vj, vk].sum())
				# print((surround[vi, vj, vk] != 0).sum())
				# print((surround[vi, vj, vk] == 0).sum())
				weather = surround[vi, vj, vk] > 0.1
				# print("WEATHER SUM: {}".format(weather.sum()))
				a[working_set] *= weather
				working_set[working_set] = weather
				# a[working_set] = surround[vi, vj, vk] > 0.5

				# print("{} voxels were marked as definitely a part of the model".format(confident_voxels.sum()))
				# print("{} voxels were marked as not a part of the model".format(np.logical_and(remove, working_set).sum()))


				# Update working set
				# working_set[remove] = False
				# working_set[confident_voxels] = False
				# print("{} voxels were left undetermined".format(working_set.sum()))


			# print(voxel_shape)
		# surround = scipy.signal.fftconvolve(voxels, ker, mode='same')
		#remove &= surround >= WEATHER_THRESHOLD

		# voxels = scipy.ndimage.gaussian_filter(voxels, sigma=1)

		# print(shared_a.sum() * RESOLVED_VOXEL_VOLUME)

		n += 1

t = multiprocessing.Process(target=target)
t.start()







# ============================================================================================
# Open3D Viewer
# ============================================================================================

rendering_mode = 0
confidence_level = 0.5
mi = 0
denoise_thresh = 0.1

pcd = o3d.geometry.PointCloud()

# Set first point cloud (all points red)
vl = voxel_locations
vc = np.zeros((vl.shape[0], 3))
vc[:, 0] = 1

pcd.points = o3d.utility.Vector3dVector(vl)
pcd.colors = o3d.utility.Vector3dVector(vc)

def update_voxels():
	global pcd, rendering_mode, default_colors, confidence_level, mi, denoise_thresh
	a = np.frombuffer(shared_a.get_obj())

	mask = a >= confidence_level
	vl = voxel_locations[mask]
	vc = np.zeros((vl.shape[0], 3))

	if vl.shape[0] > 0:

		if rendering_mode == 0:
			b = np.copy(a[mask])
			b -= b.min()
			b /= (b.max() + 1e-9)
			vc[:, 0] = b
		elif rendering_mode == 1 or rendering_mode == 2:
			vd = voxel_distances[mask]
			vd -= vd.min() - 0.5
			vd /= vd.max()

			vx, vy, vz = np.copy(vl).T
			vx -= vx.min()
			vx /= vx.max()
			vy -= vy.min()
			vy /= vy.max()
			vz -= vz.min()
			vz /= vz.max()

			vc[:, 0] = vd
			vc[:, 1] = vz * 0.6
		elif rendering_mode == 3:
			vl = voxel_locations
			vc = np.zeros((vl.shape[0], 3))
			vc[:, 0] = ms[mi]
		elif rendering_mode == 4:
			vi, vj, vk = voxel_indices.T
			voxels[vi, vj, vk] = a > 0.5
			surround = scipy.ndimage.gaussian_filter(voxels, sigma=5)
			b = surround[vi, vj, vk]
			print(b[b > 0].min())
			b -= b.min()
			b /= b.max()
			vc[:, 0] = b[mask]

		if rendering_mode == 2:
			vi, vj, vk = voxel_indices[mask].T
			voxels[:] = 0
			voxels[vi, vj, vk] = a[mask] > 0.5
			surround = scipy.ndimage.gaussian_filter(voxels, sigma=1)
			weather = surround[vi, vj, vk] > denoise_thresh

			vc = vc[weather]
			vl = vl[weather]
			print(denoise_thresh)

	pcd.points = o3d.utility.Vector3dVector(vl)
	pcd.colors = o3d.utility.Vector3dVector(vc)

def update_view(vis):
	update_voxels()
	vis.update_geometry()
	vis.update_renderer()

def decrease_mi(vis):
	global confidence_level, mi, rendering_mode, denoise_thresh
	if rendering_mode == 3:
		mi -= 1
		if mi < 0:
			mi = 0
	elif rendering_mode == 0 or rendering_mode == 1:
		confidence_level -= 0.01
		if confidence_level < 0:
			confidence_level = 0
	elif rendering_mode == 2:
		denoise_thresh -= 0.01

	update_view(vis)

def increase_mi(vis):
	global confidence_level, mi, rendering_mode, denoise_thresh
	if rendering_mode == 3:
		mi += 1
		if mi == len(ms):
			mi = len(ms) - 1
	elif rendering_mode == 0 or rendering_mode == 1:
		confidence_level += 0.01
		if confidence_level >= 1:
			confidence_level = 1
	elif rendering_mode == 2:
		denoise_thresh += 0.01

	update_view(vis)

def render_confidence(vis):
	global rendering_mode
	rendering_mode = 0
	update_view(vis)

def render_depth(vis):
	global rendering_mode
	rendering_mode = 1
	update_view(vis)

def render_denoise(vis):
	global rendering_mode
	rendering_mode = 2
	update_view(vis)

def render_scan(vis):
	global rendering_mode
	rendering_mode = 3
	update_view(vis)

def render_gaussian(vis):
	global rendering_mode
	rendering_mode = 4
	update_view(vis)

key_to_callback = {}
key_to_callback[ord('N')] = decrease_mi
key_to_callback[ord('M')] = increase_mi
key_to_callback[ord(' ')] = update_view
key_to_callback[ord('1')] = render_confidence
key_to_callback[ord('2')] = render_depth
key_to_callback[ord('3')] = render_denoise
key_to_callback[ord('4')] = render_scan
key_to_callback[ord('5')] = render_gaussian

o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

"""
Epoch 200 summary
Total error from measurements 13.696406288463711
654 voxels were marked as definitely a part of the model
338 voxels were marked as not a part of the model
40828 voxels were left undetermined

Epoch 200 summary
Total error from measurements 11.24745551077396
"""