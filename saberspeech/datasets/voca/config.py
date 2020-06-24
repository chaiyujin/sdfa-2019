import os
from saber import ConfigDict, mesh

# set voca template
_root = os.path.abspath(os.path.dirname(__file__))
_voca_template = os.path.join(_root, "template", "FLAME_sample.ply")
# _voca_template = "saberspeech/datasets/voca/templates/FaceTalk_170728_03272_TA.ply"
_voca_verts, _voca_indices = mesh.read_ply(_voca_template)

hparams: ConfigDict = ConfigDict(template=_voca_verts, tri_indices=_voca_indices)


# according to https://github.com/TimoBolkart/voca
# train: FaceTalk_170728_03272_TA, FaceTalk_170904_00128_TA, FaceTalk_170725_00137_TA, FaceTalk_170915_00223_TA,
#        FaceTalk_170811_03274_TA, FaceTalk_170913_03279_TA, FaceTalk_170904_03276_TA, FaceTalk_170912_03278_TA
# valid: FaceTalk_170811_03275_TA, FaceTalk_170908_03277_TA
# test:  FaceTalk_170809_00138_TA, FaceTalk_170731_00024_TA

speaker_alias_dict = dict(
    # train
    m0 = "FaceTalk_170728_03272_TA",
    f0 = "FaceTalk_170904_00128_TA",
    m1 = "FaceTalk_170725_00137_TA",
    m2 = "FaceTalk_170915_00223_TA",
    f1 = "FaceTalk_170811_03274_TA",
    m3 = "FaceTalk_170913_03279_TA",
    f2 = "FaceTalk_170904_03276_TA",
    f3 = "FaceTalk_170912_03278_TA",
    # valid
    f4 = "FaceTalk_170811_03275_TA",
    m4 = "FaceTalk_170908_03277_TA",
    # test
    m5 = "FaceTalk_170809_00138_TA",
    f5 = "FaceTalk_170731_00024_TA",
)
