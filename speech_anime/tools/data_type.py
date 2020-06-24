from enum import Enum


class FaceDataType(Enum):
    dgrad_3d = "dgrad_3d"
    blend_1d = "blend_1d"
    verts_pos_3d = "verts_pos_3d"
    verts_off_3d = "verts_off_3d"
    marks_pos_2d = "marks_pos_2d"
    marks_off_2d = "marks_off_2d"

    @classmethod
    def valid_types(cls):
        return tuple(cls._member_map_.keys())

    @classmethod
    def is_mesh(cls, data_type):
        if isinstance(data_type, str):
            data_type = FaceDataType[data_type]
        return data_type in [
            FaceDataType.dgrad_3d,
            FaceDataType.blend_1d,
            FaceDataType.verts_pos_3d,
            FaceDataType.verts_off_3d,
        ]

    @classmethod
    def is_landmarks(cls, data_type):
        if isinstance(data_type, str):
            data_type = FaceDataType[data_type]
        return data_type in [
            FaceDataType.marks_pos_2d,
            FaceDataType.marks_off_2d,
        ]


class PredictionType(Enum):
    pca_coeffs = "pca_coeffs"
    pca_normal = "pca_normal"
    face_data  = "face_data"

    @classmethod
    def valid_types(cls):
        return tuple(cls._member_map_.keys())
