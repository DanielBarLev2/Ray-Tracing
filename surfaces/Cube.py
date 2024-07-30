from surfaces.Object3D import Object3D


class Cube(Object3D):
    def __init__(self, position, scale, material_index):
        super().__init__(material_index)
        self.position = position
        self.scale = scale

    def transform_to_camera(self, view_matrix) -> None:
        """
        Transform the center of the cube using the view matrix.
        :param view_matrix: shifts and rotates the world coordinates to be aligned and centered on the camera.
        :return:
        """
        self.position = super().transform_to_camera_coordinates(self.position, view_matrix)