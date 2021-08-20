# based on https://github.com/weiaicunzai/blender_shapenet_render/blob/master/render_depth.py

import os
import sys
import json
import math
import argparse

import bpy

def camera_location(azimuth, elevation, dist):
    """get camera_location (x, y, z)
    you can write your own version of camera_location function
    to return the camera loation in the blender world coordinates
    system
    Args:
        azimuth: azimuth degree(object centered)
        elevation: elevation degree(object centered)
        dist: distance between camera and object(in meter)
    
    Returens:
        return the camera location in world coordinates in meters
    """

    #convert azimuth, elevation degree to radians
    phi = float(elevation) * math.pi / 180 
    theta = float(azimuth) * math.pi / 180
    dist = float(dist)

    x = dist * math.cos(phi) * math.cos(theta)
    y = dist * math.cos(phi) * math.sin(theta)
    z = dist * math.sin(phi)

    return x, y, z

def camera_rot_XYZEuler(azimuth, elevation, tilt):
    """get camera rotaion in XYZEuler
    Args:
        azimuth: azimuth degree(object centerd)
        elevation: elevation degree(object centerd)
        tilt: twist degree(object centerd)
    
    Returns:
        return the camera rotation in Euler angles(XYZ ordered) in radians
    """

    azimuth, elevation, tilt = float(azimuth), float(elevation), float(tilt)
    x, y, z = 90, 0, 90 #set camera at x axis facing towards object

    #twist
    #if tilt > 0:
    #    y = tilt
    #else:
    #    y = 360 + tilt

    #latitude
    x = x - elevation
    #longtitude
    z = z + azimuth

    return x * math.pi / 180, y * math.pi / 180, z * math.pi / 180

def render_one_view(viewpoint):
    # render z pass ; render a object z pass map by a given camera viewpoints

    cam_location = camera_location(viewpoint.azimuth, viewpoint.elevation, viewpoint.distance)
    cam_rot = camera_rot_XYZEuler(viewpoint.azimuth, viewpoint.elevation, viewpoint.tilt)
   
    bpy.data.objects['Camera'].location[0] = cam_location[0]
    bpy.data.objects['Camera'].location[1] = cam_location[1]
    bpy.data.objects['Camera'].location[2] = cam_location[2]

    bpy.data.objects['Camera'].rotation_euler[0] = cam_rot[0]
    bpy.data.objects['Camera'].rotation_euler[1] = cam_rot[1]
    bpy.data.objects['Camera'].rotation_euler[2] = cam_rot[2]

    file_output_node = bpy.context.scene.node_tree.nodes[2]
    file_output_node.file_slots[0].path = 'blender-######.depth.png'

    bpy.ops.render.render(write_still = True)

    bpy.context.scene.frame_set(1 + bpy.context.scene.frame_current)

def init_camera_scene(use_gpu = False, g_engine_type = 'CYCLES', g_resolution_x = 640, g_resolution_y = 480, g_resolution_percentage = 100, g_depth_file_format = 'PNG', g_depth_use_overwrite = True, g_depth_use_file_extension = True, g_depth_clip_start = 0.5, g_depth_clip_end 4.0, g_rotation_mode = 'XYZ', g_hilbert_spiral = 512, g_syn_depth_folder = 'syn_depth'):
    scene = bpy.data.scenes[bpy.context.scene.name]
    scene.render.engine = g_engine_type

    #scene.render.image_settings.color_mode = g_depth_color_mode
    #scene.render.image_settings.color_depth = g_depth_color_depth
    scene.render.image_settings.file_format = g_depth_file_format
    scene.render.use_overwrite = g_depth_use_overwrite
    scene.render.use_file_extension = g_depth_use_file_extension 

    scene.render.resolution_x = g_resolution_x
    scene.render.resolution_y = g_resolution_y
    scene.render.resolution_percentage = g_resolution_percentage

    if use_gpu:
        scene.render.engine = 'CYCLES'
        scene.render.tile_x = g_hilbert_spiral
        scene.render.tile_x = g_hilbert_spiral
        #bpy.context.user_preferences.addons['cycles'].preferences.devices[0].use = True
        #bpy.context.user_preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.types.CyclesRenderSettings.device = 'GPU'
        scene.cycles.device = 'GPU'
    
    bpy.data.cameras['Camera'].clip_start = g_depth_clip_start
    bpy.data.cameras['Camera'].clip_end = g_depth_clip_end
    bpy.data.objects['Camera'].rotation_mode = g_rotation_mode

    
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for node in tree.nodes:
        tree.nodes.remove(node)
    
    render_layer_node = tree.nodes.new('CompositorNodeRLayers')
    map_value_node = tree.nodes.new('CompositorNodeMapValue')
    file_output_node = tree.nodes.new('CompositorNodeOutputFile')

    map_value_node.offset[0] = -g_depth_clip_start
    map_value_node.size[0] = 1 / (g_depth_clip_end - g_depth_clip_start)
    map_value_node.use_min = True
    map_value_node.use_max = True
    map_value_node.min[0] = 0.0
    map_value_node.max[0] = 1.0

    file_output_node.format.color_mode = g_depth_color_mode
    file_output_node.format.color_depth = g_depth_color_depth
    file_output_node.format.file_format = g_depth_file_format 
    file_output_node.base_path = g_syn_depth_folder

    links.new(render_layer_node.outputs[2], map_value_node.inputs[0])
    links.new(map_value_node.outputs[0], file_output_node.inputs[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i', default = 'data/common/pix3d/pix3d.json')
    parser.add_argument('--output-path', '-o', default = 'data/pix3d_renders')
    parser.add_argument('--rot-mat', default = 'rot_mat.json')
    args = parser.parse_args(sys.argv[1 + sys.argv.index('--'):] if '--' in sys.argv else [])

    meta = json.load(open(args.input_path))
    rot_mat = json.load(open(args.rot_mat))
    model_paths = sorted(set(m['model'] for m in meta))
	
	init_camera_scene()

    for i, model_path in enumerate(model_paths):
    	print(i, '/', len(model_paths))
        model_dir = os.path.join(args.output_path, os.path.dirname(model_path))
        category = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
        os.makedirs(model_dir, exist_ok = True)
        rot_mats = rot_mat[category]
        
		bpy.ops.object.select_all(action = 'DESELECT')
		for obj in bpy.data.objects:
			if obj.type == 'MESH':
				obj.select = True
		bpy.ops.object.delete()
        bpy.ops.import_scene.obj(filepath = model_path)

		for viewpoint in rot_mats:
			render_one_view(viewpoint)
