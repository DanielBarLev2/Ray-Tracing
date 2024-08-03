import numpy as np

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
        return (f"mat {self.index}:"
                f" diff: {self.diffuse_color}"
                f" spec: {self.specular_color}"
                f" refl: {self.reflection_color}"
                f" shin: {self.shininess}"
                f" tran: {self.transparency}"
                )


def get_materials_base_colors(materials: list[Material], material_indices: np.ndarray) -> tuple:
    """
    Get base colors and properties of materials for a given set of material indices.

    This function processes a list of Material objects and an array of material indices
    to generate arrays representing the diffusive colors, specular colors, shininess values,
    reflective colors, and transparency values for each material index.

    :param materials: List of Material objects, each representing the properties of a material.
    :param material_indices: A 2D numpy array of integers, where each integer is an index corresponding
                             to a material in the materials list.

    :return: A tuple containing the following 5 numpy arrays:
             - diffusive_colors: An array of shape (*material_indices.shape, 3) containing the diffusive
                                 colors for each material index.
             - surfaces_specular_colors: An array of shape (*material_indices.shape, 3) containing the
                                         specular colors for each material index.
             - phong: An array of the same shape as material_indices containing the shininess values
                      for each material index.
             - reflective_colors: An array of shape (*material_indices.shape, 3) containing the reflective
                                  colors for each material index.
             - transparency_values: An array of shape (*material_indices.shape, 3) containing the
                                    transparency values for each material index.
    """
    diffusive_colors = np.zeros((*material_indices.shape, 3))
    surfaces_specular_colors = np.zeros((*material_indices.shape, 3))
    phong = np.zeros_like(material_indices)
    reflective_colors = np.zeros((*material_indices.shape, 3))
    transparency_values = np.zeros((*material_indices.shape, 3))

    for idx in np.unique(material_indices):
        material = materials[idx]
        mask = (material_indices == idx)

        diffusive_colors[mask] = material.diffuse_color
        surfaces_specular_colors[mask] = material.specular_color
        phong[mask] = material.shininess
        reflective_colors[mask] = material.reflection_color
        transparency_values[mask] = material.transparency

    return diffusive_colors, surfaces_specular_colors, phong, reflective_colors, transparency_values
