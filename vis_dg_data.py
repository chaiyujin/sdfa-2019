import os
import saber
import pickle
import numpy as np
import saber.data.mesh.io as meshio
import saber.utils.filesystem as fs
import saber.data.audio as saber_audio
from speech_anime import viewer
from saberspeech.datasets import voca

root = "assets/voca-sr8k/dgrad"
offs_root = "assets/voca-sr8k/offsets"
output_root = "assets/voca_videos/"
speakers = ["f0", "m0", "m1", "m2", "m3", "f1",  "f2", "f3", "m4", "f4"]
sample_rate = 8000


def dump_video(template, dg_prefix, of_prefix, output_path):
    if os.path.exists(output_path):
        return

    if not os.path.exists(of_prefix + '_audio'):
        return

    dg_src = dict(
        title="dg",
        audio=None,
        dgrad_3d=[],
        tslist=[],
    )

    with open(dg_prefix + '_audio', "rb") as fp:
        data = pickle.load(fp)
        dg_src["audio"] = data["audio"]
        start_ts = data["start_ts"]

    npy_list = fs.find_files(dg_prefix, r"^\d+\.npy$", False, True)
    for i, npy_file in enumerate(saber.log.tqdm(npy_list, leave=False)):
        frame_id = int(os.path.splitext(os.path.basename(npy_file))[0])
        dg_src["dgrad_3d"].append(np.load(npy_file))
        dg_src["tslist"].append(float(frame_id * 1000.0) / 60.0 - start_ts)

    offs_src = dict(
        title="offsets",
        audio=None,
        verts_off_3d=[],
        tslist=[],
    )

    with open(of_prefix + '_audio', "rb") as fp:
        data = pickle.load(fp)
        offs_src["audio"] = data["audio"]
        start_ts = data["start_ts"]

    npy_list = fs.find_files(of_prefix, r"\d+\.npy", False, True)
    for i, npy_file in enumerate(saber.log.tqdm(npy_list, leave=False)):
        frame_id = int(os.path.splitext(os.path.basename(npy_file))[0])
        offs_src["verts_off_3d"].append(np.load(npy_file))
        offs_src["tslist"].append(float(frame_id * 1000.0) / 60.0 - start_ts)

    sources = [dg_src, offs_src]
    viewer.render_video(sources, 60, sample_rate, save_video=True, video_path=output_path, grid_w=780, grid_h=780)


for spk in speakers:
    print(spk)
    alias = voca.get_speaker_alias(spk)
    template_path = f"saberspeech/datasets/voca/templates/{alias}.ply"
    assert os.path.exists(template_path)
    verts, faces = saber.mesh.read_mesh(template_path)
    saber.mesh.write_obj(os.path.splitext(template_path)[0] + '.obj', verts, faces)
    viewer.set_template_mesh(os.path.splitext(template_path)[0] + '.obj')

    for sid in range(0, 40):
        print(f"- render {sid+1}")
        dg_prefix = os.path.join(root, "data", spk, "neutral", f"{sid:03d}")
        of_prefix = os.path.join(offs_root, "data", spk, "neutral", f"{sid:03d}")
        output_path = os.path.join(output_root, spk, f"sentence{sid+1:02d}.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dump_video(verts, dg_prefix, of_prefix, output_path)
