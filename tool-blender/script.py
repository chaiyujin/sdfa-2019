import os
import re
import sys
import bpy
import bmesh
import mathutils

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import meshio
from argparse_blender import ArgumentParserForBlender


def set_output_properties(scene: bpy.types.Scene, image_size: int = 512, output_file_path: str = "") -> None:
    scene.render.resolution_x = image_size
    scene.render.resolution_y = image_size
    scene.render.resolution_percentage = 100

    if output_file_path:
        scene.render.filepath = output_file_path


def set_animation(scene: bpy.types.Scene,
                  fps: int = 60,
                  frame_start: int = 1,
                  frame_end: int = 5,
                  frame_current: int = 1) -> None:
    scene.render.fps = fps
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.frame_current = frame_current


def set_eevee_renderer(scene: bpy.types.Scene,
                       camera_object: bpy.types.Object,
                       num_samples: int,
                       use_motion_blur: bool = False,
                       use_transparent_bg: bool = False,
                       small_size: bool = False) -> None:
    scene.camera = camera_object

    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.use_motion_blur = use_motion_blur
    scene.render.film_transparent = use_transparent_bg
    # video
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "PERC_LOSSLESS" if small_size else "LOSSLESS"

    scene.eevee.taa_render_samples = num_samples


def find_files(directory, pattern, recursive=True, abspath=False):
    regex = re.compile(pattern)
    file_list = []
    for root, _, files in os.walk(directory):
        for f in files:
            if regex.match(f) is not None:
                file_list.append(os.path.join(root, f))
        if not recursive:
            break
    map_func = os.path.abspath if abspath else os.path.relpath
    return list(map(map_func, sorted(file_list)))


if __name__ == "__main__":
    # blender -b  render_flame.blend --python script.py --render-anim -- --output_prefix render_ --image_size 512 --source_dir ../test_out/test-000
    parser = ArgumentParserForBlender()
    parser.add_argument("-S", "--source_dir",     type=str,   required=True)
    parser.add_argument("-O", "--output_prefix",  type=str,   default="./out_")
    parser.add_argument(      "--image_size",     type=int,   default=512)
    parser.add_argument(      "--render_samples", type=int,   default=8)
    parser.add_argument(      "--fps",            type=float, default=60)
    parser.add_argument(      "--smooth_shading", action="store_true")
    parser.add_argument(      "--small_size",    action="store_true")
    args = parser.parse_args()

    # Find all frames
    mesh_files = find_files(args.source_dir, r"\d+\.obj", recursive=False, abspath=False)
    mesh_files = sorted(mesh_files, key=lambda x: int(os.path.basename(os.path.splitext(x)[0])))
    num_frames = len(mesh_files)

    # Setting
    scene = bpy.context.scene
    camera_object = bpy.data.objects["Camera"]
    flame_object = bpy.data.objects["FLAME_sample"]

    # Shading mode: smooth / flat
    if args.smooth_shading:
        # smooth shading
        for poly in flame_object.data.polygons:
            poly.use_smooth = True
        # subsurface division modifier
        m = flame_object.modifiers.new('My SubDiv', 'SUBSURF')
        m.levels = 1
        m.render_levels = 2
        m.quality = 3

    # Set output and render
    set_output_properties(scene, args.image_size, args.output_prefix)
    set_eevee_renderer(scene, camera_object,
                       num_samples=args.render_samples,
                       small_size=args.small_size)

    # Create animation
    action = bpy.data.actions.new("MeshAnimation")
    flame_object.data.animation_data_create()
    flame_object.data.animation_data.action = action
    # Create fcurves for animation
    fcurves_list = []
    for v in flame_object.data.vertices:
        fcurves = [action.fcurves.new(f"vertices[{v.index:d}].co", index = k) for k in range(3)]
        fcurves_list.append(fcurves)
    set_animation(scene, fps=args.fps, frame_start=1, frame_end=num_frames)

    # Insert frames
    def _insert_keyframe(fcurves, frame, values):
        for fcu, val in zip(fcurves, values):
            fcu.keyframe_points.insert(frame, val, options={'FAST'})

    def _mesh_keyframe(obj, fcurves_list, verts, i_frame):
        for i, v in enumerate(obj.data.vertices):
            values = verts[i]
            fcurves = fcurves_list[i]
            for fcu, val in zip(fcurves, values):
                fcu.keyframe_points.insert(i_frame, val, options={'FAST'})

    for i_frame, filename in enumerate(mesh_files):
        print(f"Set keyframe {i_frame}", end='\r')
        verts, _ = meshio.read_obj(filename)
        _mesh_keyframe(flame_object, fcurves_list, verts, i_frame + 1)
