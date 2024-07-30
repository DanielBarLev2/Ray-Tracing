
import numpy as np


class BSPNode:
    def __init__(self, surfaces, left=None, right=None, plane=None):
        self.surfaces = surfaces
        self.left = left
        self.right = right
        self.plane = plane


def build_bsp_tree(surfaces: list, depth: int) -> BSPNode | None:
    """
    Builds BSP tree.
    Divides the space into octant according to a surface appearance it that octant.
    Recursively Divides the space into two separated spaces in respect to one axis at a time.
    :param surfaces: list of Object3D, Spheres or Cubes.
    :param depth: Current depth of the tree.
    :return: BSPNode pointing on the root of the tree.
    """
    if len(surfaces) == 0:
        return None

    if len(surfaces) == 1:
        return BSPNode(surfaces=surfaces)

    axis = depth % 3
    surfaces.sort(key=lambda s: s.position[axis])
    median = len(surfaces) // 2

    left_surfaces = surfaces[:median]
    right_surfaces = surfaces[median:]

    return BSPNode(surfaces=None,
                   left=build_bsp_tree(left_surfaces, depth + 1),
                   right=build_bsp_tree(right_surfaces, depth + 1),
                   plane=axis)


def traverse_bsp_tree(ray_origin: np.ndarray, ray_direction: np.ndarray, bsp_node: BSPNode):
    if bsp_node is None:
        return None, None

    if bsp_node.surfaces is not None:
        # Leaf node, check intersections with surfaces
        closest_t = float('inf')
        closest_surface = None
        for surface in bsp_node.surfaces:
            hit, t = surface.intersect(ray_origin, ray_direction)
            if hit and t < closest_t:
                closest_t = t
                closest_surface = surface
        return closest_surface, closest_t

    axis = bsp_node.plane
    t_split = (bsp_node.surfaces[0].center[axis] - ray_origin[axis]) / ray_direction[axis]

    if ray_origin[axis] < bsp_node.surfaces[0].center[axis]:
        near_node = bsp_node.left
        far_node = bsp_node.right
    else:
        near_node = bsp_node.right
        far_node = bsp_node.left

    closest_surface, closest_t = traverse_bsp_tree(ray_origin, ray_direction, near_node)

    if closest_surface is None or t_split < closest_t:
        far_surface, far_t = traverse_bsp_tree(ray_origin, ray_direction, far_node)
        if far_surface is not None and far_t < closest_t:
            closest_surface = far_surface
            closest_t = far_t

    return closest_surface, closest_t
