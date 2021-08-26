import sys
import json
import math
import itertools
import argparse

import bpy
import mathutils

def set_camera_location_rotation(azimuth, elevation, distance, tilt):
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

def main(args):
    meta = json.load(open(args.input_path))
    key_model = lambda m: m['model']
    by_model = [list(g) for k, g in itertools.groupby(sorted(meta, key = key_model), key = key_model)]
    views = by_model[args.model_idx]
    model_path = views[0]['model']
    view_slice = slice(None) if args.view_slice is None else slice(args.view_slice[0], 1 + args.view_slice[0]) if len(args.view_slice) == 1 else slice(*args.view_slice)

    bpy.ops.object.select_all(action = 'SELECT')
    bpy.ops.object.delete()

    bpy.ops.import_scene.obj(filepath = model_path, axis_forward = 'Y', axis_up = '-Z')
    obj = bpy.context.selected_objects[0]
    obj.location = (0, 0, 0)

    print(model_path)
    for m in views[view_slice]:
        x, y, z = m['cam_position']
        cam_location = (-x, y, -z)
        bpy.ops.object.camera_add(location = cam_location)
        print(m['cam_position'], m['img'])
        camera_obj = bpy.context.selected_objects[0]
        camera_obj.data.lens = m['focal_length']
        scene = bpy.data.scenes["Scene"]
        
        #euler_y = math.atan2(-z, -x)
        #euler_x = math.asin(y)
        #camera_obj.rotation_euler = (-euler_x, euler_y, 0)

        track_to = bpy.context.object.constraints.new('TRACK_TO')
        track_to.target = obj
        track_to.track_axis = 'TRACK_NEGATIVE_Z'
        track_to.up_axis = 'UP_X'
        track_to.use_target_z = True

    bpy.ops.wm.save_mainfile(filepath = args.output_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i', default = 'data/common/pix3d/pix3d.json')
    parser.add_argument('--output-path', '-o', default = 'test.blend')
    parser.add_argument('--model-idx', type = int, default = 0)
    parser.add_argument('--view-slice', type = int, nargs = '*')
    args = parser.parse_args(sys.argv[1 + sys.argv.index('--'):] if '--' in sys.argv else [])

    main(args)
