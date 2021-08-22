# Blender 2.93
# blender -noaudio --background --python render_pix3d.py -- -o data/pix3d_renders

# based on:
# https://github.com/weiaicunzai/blender_shapenet_render/blob/master/render_depth.py
# https://github.com/xingyuansun/pix3d/blob/master/demo.py

import os
import sys
import json
import math
import argparse

import bpy
import mathutils

def set_camera_location(azimuth, elevation, distance, tilt):
    # render z pass ; render a object z pass map by a given camera viewpoints
    # args in degrees/meters (object centered)
    azimuth, elevation, distance, tilt = map(float, [azimuth, elevation, distance, tilt])
    camera = bpy.data.objects['Camera']
    
    phi = elevation * math.pi / 180 
    theta = azimuth * math.pi / 180
    x = distance * math.cos(phi) * math.cos(theta)
    y = distance * math.cos(phi) * math.sin(theta)
    z = distance * math.sin(phi)
    camera.location = (x, y, z)

    x, y, z = 90, 0, 90 #set camera at x axis facing towards object
    x = x - elevation   #latitude
    z = z + azimuth     #longtitude
    camera.rotation_euler = (x * math.pi / 180, y * math.pi / 180, z * math.pi / 180)

def init_camera_scene_(w=None, h=None, n_samples=None, xyz=(0, 0, 0), rot_vec_rad=(0, 0, 0), name=None, proj_model='PERSP', f=35, sensor_fit='HORIZONTAL', sensor_width=32, sensor_height=18):
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    
    cam.location = xyz
    cam.rotation_euler = rot_vec_rad
    cam.data.type = proj_model
    cam.data.lens = f
    cam.data.sensor_fit = sensor_fit
    cam.data.sensor_width = sensor_width
    cam.data.sensor_height = sensor_height

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    cycles = scene.cycles

    cycles.use_progressive_refine = True
    if n_samples is not None:
        cycles.samples = n_samples
    cycles.max_bounces = 100
    cycles.min_bounces = 10
    cycles.caustics_reflective = False
    cycles.caustics_refractive = False
    cycles.diffuse_bounces = 10
    cycles.glossy_bounces = 4
    cycles.transmission_bounces = 4
    cycles.volume_bounces = 0
    cycles.transparent_min_bounces = 8
    cycles.transparent_max_bounces = 64

    # Avoid grainy renderings (fireflies)
    world = bpy.data.worlds['World']
    world.cycles.sample_as_light = True
    cycles.blur_glossy = 5
    cycles.sample_clamp_indirect = 5

    # Ensure no background node
    world.use_nodes = True
    try:
        world.node_tree.nodes.remove(world.node_tree.nodes['Background'])
    except KeyError:
        pass

    scene.render.tile_x = 16
    scene.render.tile_y = 16
    if w is not None:
        scene.render.resolution_x = w
    if h is not None:
        scene.render.resolution_y = h
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA' # 'BW'
    scene.render.image_settings.color_depth = '8'

def init_camera_scene(use_gpu = False, resolution_x = 640, resolution_y = 480, clip_start = 0.5, clip_end = 4.0, hilbert_spiral = 512):
    scene = bpy.data.scenes[bpy.context.scene.name]
    scene.render.engine = 'CYCLES'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.use_overwrite = True
    scene.render.use_file_extension = True
    scene.render.resolution_x, scene.render.resolution_y = resolution_x, resolution_y
    scene.render.resolution_percentage = 100

    world = bpy.data.worlds['World']
    world.light_settings.use_ambient_occlusion = True
    world.light_settings.ao_factor = 0.9

    if use_gpu:
        scene.render.engine = 'CYCLES'
        scene.render.tile_x = scene.render.tile_y = hilbert_spiral
        #bpy.context.user_preferences.addons['cycles'].preferences.devices[0].use = True
        #bpy.context.user_preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.types.CyclesRenderSettings.device = 'GPU'
        scene.cycles.device = 'GPU'
    
    camera = bpy.data.cameras['Camera']
    #camera.clip_start = clip_start
    #camera.clip_end = clip_end

    cam = bpy.data.objects['Camera']
    
    cam.location = (0, 0, 0)
    cam.rotation_euler = (0, math.pi, 0)
    cam.rotation_mode = 'XYZ'
    cam.data.type = 'PERSP'
    cam.data.lens = 35
    cam.data.sensor_fit = 'HORIZONTAL'
    cam.data.sensor_width = 32
    cam.data.sensor_height = 18

    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    for node in tree.nodes:
        tree.nodes.remove(node)
    render_layer_node = tree.nodes.new('CompositorNodeRLayers')
    map_value_node = tree.nodes.new('CompositorNodeMapValue')
    file_output_node = tree.nodes.new('CompositorNodeOutputFile')
    #map_value_node.offset[0] = -clip_start
    #map_value_node.size[0] = 1 / (clip_end - clip_start)
    #map_value_node.use_min = True
    #map_value_node.use_max = True
    #map_value_node.min[0] = 0.0
    #map_value_node.max[0] = 1.0
    file_output_node.format.color_mode = 'RGBA'
    file_output_node.format.color_depth = '8'
    file_output_node.format.file_format = 'PNG' 
    links.new(render_layer_node.outputs[2], map_value_node.inputs[0])
    links.new(map_value_node.outputs[0], file_output_node.inputs[0])

    return file_output_node

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i', default = 'data/common/pix3d/pix3d.json')
    parser.add_argument('--output-path', '-o', default = 'data/pix3d_renders')
    parser.add_argument('--rot-mat', default = 'rot_mat.json')
    args = parser.parse_args(sys.argv[1 + sys.argv.index('--'):] if '--' in sys.argv else [])

    meta = json.load(open(args.input_path))
    rot_mat_by_category = json.load(open(args.rot_mat))
    model_paths = sorted(set(m['model'] for m in meta))
    
    file_output_node = init_camera_scene()

    m = meta[0]
    model_paths = [m['model']]
    trans_vec = m['trans_mat']
    rot_mat = m['rot_mat']

    for i, model_path in enumerate(model_paths):
        print(i, '/', len(model_paths))
        model_dir = os.path.join(args.output_path, os.path.dirname(model_path))
        category = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
        os.makedirs(model_dir, exist_ok = True)
        rot_mats = rot_mat_by_category[category]
        
        bpy.ops.object.select_all(action = 'DESELECT')
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                obj.select_set(True)
        bpy.ops.object.delete()

        bpy.ops.import_scene.obj(filepath = os.path.join(os.path.dirname(args.input_path), model_path))
        obj = bpy.context.selected_objects[0]

        #for rot_mat in rot_mats:
        if True:
            trans_4x4 = mathutils.Matrix.Translation(trans_vec)
            rot_4x4 = mathutils.Matrix(rot_mat).to_4x4()
            obj.matrix_world = trans_4x4 @ rot_4x4

            file_output_node.base_path = model_dir
            file_output_node.file_slots[0].path = 'view-######.png'

            bpy.ops.render.render(write_still = False)
            bpy.context.scene.frame_set(1 + bpy.context.scene.frame_current)

            break
        break
