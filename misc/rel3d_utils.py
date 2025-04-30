import numpy.typing as npt

def read_single_transform(vector: list[float]) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    centroid = vector[:3]
    rotation = vector[3:6]

    if len(vector) > 9:
        rel_rot = vector[6:79]
        scale = vector[9:12]
        return centroid, rotation, rel_rot, scale
    
    scale = vector[6:9]
    return centroid, rotation, scale


def read_raw_transformation(raw_transform_vector: list[float]):
    subject_transforms = read_single_transform(raw_transform_vector[:12])
    object_transforms =  read_single_transform(raw_transform_vector[12:])

    return subject_transforms, object_transforms