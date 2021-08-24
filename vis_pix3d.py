import sys
import json
import itertools
import argparse

import bpy

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', '-i', default = 'data/common/pix3d/pix3d.json')
parser.add_argument('--output-path', '-o', default = 'test.blend')
parser.add_argument('--model-idx', type = int, default = 0)
parser.add_argument('--view-slice', type = int, nargs = '*')
args = parser.parse_args(sys.argv[1 + sys.argv.index('--'):] if '--' in sys.argv else [])

meta = json.load(open(args.input_path))
key_model = lambda m: m['model']
by_model = [list(g) for k, g in itertools.groupby(sorted(meta, key = key_model), key = key_model)]
views = by_model[args.model_idx]
model_path = views[0]['model']
view_slice = slice(None) if args.view_slice is None else slice(args.view_slice[0], 1 + args.view_slice[0]) if len(args.view_slice) == 1 else slice(*args.view_slice)

bpy.ops.object.select_all(action = 'SELECT')
bpy.ops.object.delete()

bpy.ops.import_scene.obj(filepath = model_path, axis_forward = 'Y', axis_up = 'Z')
obj = bpy.context.selected_objects[0]
obj.location = (0, 0, 0)

print(model_path)
for m in views[view_slice]:
    bpy.ops.object.camera_add(location = m['cam_position'])
    print(m['cam_position'], m['img'])
    camera_obj = bpy.context.selected_objects[0]
    camera_obj.data.lens = m['focal_length'] # 10
    track_to = bpy.context.object.constraints.new('TRACK_TO')
    track_to.target = obj
    track_to.track_axis = 'TRACK_NEGATIVE_Z'
    track_to.up_axis = 'UP_Y'

bpy.ops.wm.save_mainfile(filepath = args.output_path)
