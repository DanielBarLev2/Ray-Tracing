import numpy as np

from Ray import Ray


class BSPNode:
    def __init__(self, surfaces, left=None, right=None, plane=None):
        self.surfaces = surfaces
        self.left = left
        self.right = right
        self.plane = plane

    @staticmethod
    def build_bsp_tree(surfaces: list, depth=0):
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

        # depth = 0 : axis = X
        # depth = 1 : axis = Y
        # depth = 2 : axis = Z
        axis = depth % 3
        surfaces.sort(key=lambda s: s.position[axis])
        median = len(surfaces) // 2

        left_surfaces = surfaces[:median]
        right_surfaces = surfaces[median:]

        return BSPNode(surfaces=None,
                       left=BSPNode.build_bsp_tree(left_surfaces, depth + 1),
                       right=BSPNode.build_bsp_tree(right_surfaces, depth + 1),
                       plane=axis)

    def __repr__(self, depth=0):
        indent = "  " * depth  # Create an indent based on the depth of the node
        repr_str = f"{indent}BSPNode(plane={self.plane}, surfaces={self.surfaces})\n"
        if self.left is not None:
            repr_str += self.left.__repr__(depth + 1)
        if self.right is not None:
            repr_str += self.right.__repr__(depth + 1)
        return repr_str

def traverse(ray_source: np.ndarray, ray_directions: np.ndarray, bsp_node: BSPNode):
    if bsp_node.plane is None:
        return

def traverse_bsp_tree(ray: Ray, bsp_node: BSPNode):
    if bsp_node is None:
        return []

    if bsp_node.surfaces is not None:
        # Leaf node, check intersections with surfaces
        hits = []
        for surface in bsp_node.surfaces:
            point = surface.intersect(ray)
            if point != float('inf'):
                hits.append(surface)
        return hits

    axis, plane_value = bsp_node.plane

    if ray.source < plane_value:
        near_node = bsp_node.left
        far_node = bsp_node.right
    else:
        near_node = bsp_node.right
        far_node = bsp_node.left

    hits = traverse_bsp_tree(ray, near_node)
    far_hits = traverse_bsp_tree(ray, far_node)

    hits.extend(far_hits)

    return hits
