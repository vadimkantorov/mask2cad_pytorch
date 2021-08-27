# based on:
# https://github.com/weiaicunzai/blender_shapenet_render/blob/master/render_depth.py
# https://github.com/xingyuansun/pix3d/blob/master/demo.py

import os
import sys
import json
import math
import random
import argparse
import subprocess

try:
    import numpy as np, scipy.spatial 
except:
    subprocess.call([sys.executable, '-m', 'ensurepip'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy', 'scipy'])
finally:
    import numpy as np, scipy.spatial

import bpy
import mathutils

def delete_mesh_objects():
    bpy.ops.object.select_all(action = 'DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.object.delete()

def configure_camera(camera_obj, lens):
    camera_obj.location = (0, 0, 0)
    camera_obj.rotation_euler = (0, math.pi, 0)
    camera_obj.data.type = 'PERSP'
    camera_obj.data.sensor_width = 32
    camera_obj.data.sensor_height = 18
    camera_obj.data.sensor_fit = 'HORIZONTAL'
    camera_obj.data.lens = lens 

def configure_scene_render(scene_render, resolution_x, resolution_y, tiles, color_mode, color_depth, file_format = 'JPEG'):
    scene_render.engine = 'CYCLES'
    scene_render.image_settings.file_format = file_format
    scene_render.use_overwrite = True
    scene_render.use_file_extension = True
    scene_render.resolution_x = resolution_x
    scene_render.resolution_y = resolution_y
    scene_render.resolution_percentage = 100
    scene_render.tile_x = tiles
    scene_render.tile_y = tiles 
    scene_render.image_settings.color_mode = color_mode
    scene_render.image_settings.color_depth = color_depth
    
def enable_gpu(gpu):
    if gpu:
        #bpy.context.user_preferences.addons['cycles'].preferences.devices[0].use = True
        #bpy.context.user_preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.types.CyclesRenderSettings.device = 'GPU'
        bpy.data.scenes[bpy.context.scene.name].cycles.device = 'GPU'

def init_camera_scene_regular(samples = 5):
    camera_obj = bpy.data.objects['Camera']
    camera_obj.data.clip_end = 1e10
    
    cycles = bpy.context.scene.cycles
    cycles.use_progressive_refine = True
    cycles.samples = samples
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
    cycles.blur_glossy = 5
    cycles.sample_clamp_indirect = 5
    
    world = bpy.data.worlds['World']
    world.cycles.sample_as_light = True
    world.use_nodes = True
    world.node_tree.nodes.remove(world.node_tree.nodes['Background']) if 'Background' in world.node_tree.nodes else None

def init_camera_scene_depth(color_mode, color_depth, clip_start = 0.5, clip_end = 4.0):
    camera = bpy.data.cameras['Camera']
    camera.clip_start = clip_start
    camera.clip_end = clip_end
    
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    for node in tree.nodes:
        tree.nodes.remove(node)
    render_layer_node = tree.nodes.new('CompositorNodeRLayers')
    map_value_node = tree.nodes.new('CompositorNodeMapValue')
    file_output_node = tree.nodes.new('CompositorNodeOutputFile')
    map_value_node.offset[0] = -clip_start
    map_value_node.size[0] = 1 / (clip_end - clip_start)
    map_value_node.use_min = True
    map_value_node.use_max = True
    map_value_node.min[0] = 0.0
    map_value_node.max[0] = 1.0
    file_output_node.format.color_mode = color_mode
    file_output_node.format.color_depth = color_depth
    file_output_node.format.file_format = 'PNG' 
    links.new(render_layer_node.outputs[2], map_value_node.inputs[0])
    links.new(map_value_node.outputs[0], file_output_node.inputs[0])

    return file_output_node

def render_ground_truth_pose(metadata, args, color_mode, color_depth):
    for i, m in enumerate(metadata):
        print(i, '/', len(metadata), m['img'])
        category = m['category']
        if args.category and category not in args.category:
            continue
        
        w, h = m['img_size']
        f = m['focal_length']
        model_path = m['model']
        trans_vec, rot_mat = m['trans_mat'], m['rot_mat']

        frame_path = os.path.join(args.output_path, m['img'])
        frame_dir = os.path.dirname(frame_path)
        os.makedirs(frame_dir, exist_ok = True)
    
        configure_scene_render(bpy.data.scenes[bpy.context.scene.name].render, w, h, args.tiles, color_mode = color_mode, color_depth = color_depth)
        configure_camera(bpy.data.objects['Camera'], f)
        
        delete_mesh_objects()
        bpy.ops.import_scene.obj(filepath = os.path.join(os.path.dirname(args.input_path), model_path), axis_forward='-Z', axis_up='Y')
        obj = bpy.context.selected_objects[0]
        
        obj.matrix_world = mathutils.Matrix.Translation(trans_vec) @ mathutils.Matrix(rot_mat).to_4x4()
    
        bpy.context.scene.render.filepath = frame_path
        bpy.ops.render.render(write_still = True)
        
        print(frame_path)
        break
    
def render_synthetic_views(metadata, args, color_mode, color_repth, viewpoints_by_category):
    w, h = args.wh

    configure_scene_render(bpy.data.scenes[bpy.context.scene.name].render, w, h, args.tiles, color_mode = color_mode, color_depth = color_depth)
    
    sample_trans_vec = lambda category: viewpoints_by_category[category]['trans_vec'][0]
    #sample_trans_vec = lambda category: random.choice(viewpoints_by_category[category]['trans_vec'])

    model_paths = sorted(set(m['model'] for m in metadata))
    for i, model_path in enumerate(model_paths):
        print(i, '/', len(model_paths), model_path)
        category = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
        if args.category and category not in args.category:
            continue
    
        f = args.focal_length[category]
        configure_camera(bpy.data.objects['Camera'], f)

        frame_dir = os.path.join(args.output_path, os.path.dirname(model_path))
        os.makedirs(frame_dir, exist_ok = True)

        delete_mesh_objects()    

        bpy.ops.import_scene.obj(filepath=os.path.join(os.path.dirname(args.input_path), model_path), axis_forward='-Z', axis_up='Y')
        obj = bpy.context.selected_objects[0]
        for k in range(len(viewpoints_by_category[category]['quat'])):
            frame_path = os.path.join(frame_dir, '{:04}.jpg'.format(1 + k))
            trans_vec = sample_trans_vec(category)
            quat_ = viewpoints_by_category[category]['quat'][k]
            
            obj.rotation_quaternion = (quat_[-1], quat_[0], quat_[1], quat_[2])
            obj.location = trans_vec
            
            #file_output_node.base_path = os.path.dirname(frame_path)
            #file_output_node.file_slots[0].path = '####.jpg'
            #bpy.ops.render.render(write_still = False)
    
            bpy.context.scene.render.filepath = frame_path
            bpy.ops.render.render(write_still = True)

            bpy.context.scene.frame_set(1 + bpy.context.scene.frame_current)

            print(frame_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i', default = 'data/common/pix3d/pix3d.json')
    parser.add_argument('--output-path', '-o', default = 'data/pix3d_renders')
    parser.add_argument('--viewpoints-path', default = 'pix3d_clustered_viewpoints.json')
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--gpu', action = 'store_true')
    parser.add_argument('--category', nargs = '*')
    parser.add_argument('--tiles', type = int, default = 16)
    parser.add_argument('--samples', type = int, default = 50)
    parser.add_argument('--wh', type = int, nargs = 2, default = [128, 128])
    parser.add_argument('--render-ground-truth-views', action = 'store_true')
    parser.add_argument('--render-synthetic-views', action = 'store_true')
    parser.add_argument('--focal-length', action = type('', (argparse.Action, ), dict(__call__ = lambda a, p, n, v, o: getattr(n, a.dest).update(dict([v.split('=')])))), default = dict(bed = 50, bookcase = 200, chair = 200, desk = 50, misc = 50, sofa = 50, table = 40, tool = 50, wardrobe = 40) )
    args = parser.parse_args(sys.argv[1 + sys.argv.index('--'):] if '--' in sys.argv else [])

    random.seed(args.seed)

    metadata = json.load(open(args.input_path))
    viewpoints_by_category = json.load(open(args.viewpoints_path))
    
    color_mode, color_depth = 'BW', '8'
    
    bpy.context.scene.render.engine = 'CYCLES'
    world = bpy.data.worlds['World']
    world.light_settings.use_ambient_occlusion = True
    world.light_settings.ao_factor = 0.9
    bpy.context.scene.camera = bpy.data.objects['Camera']
    enable_gpu(args.gpu)

    #file_output_node = init_camera_scene_depth(color_mode = color_mode, color_depth = color_depth)
    init_camera_scene_regular(samples = args.samples)

    if args.render_ground_truth_views:
        render_ground_truth_pose(metadata, args, color_mode, color_depth)
    
    if args.render_synthetic_views:
        render_synthetic_views(metadata, args, color_mode, color_depth, viewpoints_by_category)
