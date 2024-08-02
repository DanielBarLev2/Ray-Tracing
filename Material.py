from util import *


class Material:
    def __init__(self, diffuse_color, specular_color, reflection_color, shininess, transparency, mat_index):
        self.diffuse_color = np.array(diffuse_color)
        self.specular_color = np.array(specular_color)
        self.reflection_color = np.array(reflection_color)
        self.shininess = np.array(shininess)
        self.transparency = np.array(transparency)
        self.index = mat_index

    def __repr__(self):
        return (f"mat {self.index}:\n"
                f"\t\t diff: {self.diffuse_color}\n"
                f"\t\t spec: {self.specular_color}\n"
                f"\t\t refl: {self.reflection_color}\n"
                f"\t\t shin: {self.shininess}\n"
                f"\t\t tran: {self.transparency}\n"
                )

    def get_diffusive(self):
        return self.diffuse_color

    def get_specular(self):
        return self.specular_color

    def get_reflective(self):
        return self.reflection_color

    def get_shininess(self):
        return self.shininess

    def get_transparency(self):
        return self.transparency
