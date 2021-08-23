import sys
import json
import itertools
import argparse

import bpy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i', default = 'data/common/pix3d/pix3d.json')
    parser.add_argument('--output-path', '-o', default = 'test.blend')
    args = parser.parse_args(sys.argv[1 + sys.argv.index('--'):] if '--' in sys.argv else [])
    
    meta = json.load(open(args.input_path))
    key_model = lambda m: m['model']
    by_model = [list(g) for k, g in itertools.groupby(sorted(meta, key = key_model), key = key_model)]
    
    idx = 0
    views = by_model[idx]
    model_path = views[0]['model']

    bpy.data.objects['Cube'].select_set(True)
    bpy.data.objects['Camera'].select_set(True)
    bpy.ops.object.delete()
    
    bpy.ops.import_scene.obj(filepath = model_path, axis_forward='-Z', axis_up='Y')
    obj = bpy.context.selected_objects[0]

    for m in views:
        res2 = bpy.ops.object.camera_add(location = m['cam_position'])
        track_to = bpy.context.object.constraints.new('TRACK_TO')
        track_to.target = obj
        track_to.track_axis = 'TRACK_NEGATIVE_Z'
        track_to.up_axis = 'UP_Y'
        

    bpy.ops.wm.save_mainfile(filepath = args.output_path)
