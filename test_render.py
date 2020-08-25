import cv2
from saber import mesh
from speech_anime import viewer

viewer.set_template_mesh("speech_anime/datasets/vocaset/templates/FaceTalk_170725_00137_TA.ply")
verts, faces = mesh.read_mesh("speech_anime/datasets/vocaset/templates/FaceTalk_170809_00138_TA.ply")
img = viewer.render_frame(verts, "verts_pos_3d")
cv2.imshow('img', img)
cv2.waitKey()
