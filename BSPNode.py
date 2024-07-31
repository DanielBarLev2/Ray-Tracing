import numpy as np

from Ray import Ray


class BSPNode:
    def __init__(self, surfaces, left=None, right=None, plane=None, cut=None):
        self.surfaces = surfaces
        self.left = left
        self.right = right
        self.plane = plane
        self.cut = cut

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

        cut = {}
        if axis == 0:
            cut = ("x", round(surfaces[median].position[axis], 4))
        elif axis == 1:
            cut = ("y", round(surfaces[median].position[axis], 4))
        elif axis == 2:
            cut = ("z", round(surfaces[median].position[axis], 4))

        left_surfaces = surfaces[:median]
        right_surfaces = surfaces[median:]

        return BSPNode(surfaces=None,
                       left=BSPNode.build_bsp_tree(left_surfaces, depth + 1),
                       right=BSPNode.build_bsp_tree(right_surfaces, depth + 1),
                       plane=axis,
                       cut=cut)

    def __repr__(self, depth=0):
        indent = "  " * depth  # Create an indent based on the depth of the node
        repr_str = f"{indent}BSPNode(plane={self.plane}, cut={self.cut:}, surfaces={self.surfaces})\n"
        if self.left is not None:
            repr_str += self.left.__repr__(depth + 1)
        if self.right is not None:
            repr_str += self.right.__repr__(depth + 1)
        return repr_str


def traverse(ray_source: np.ndarray, ray_directions: np.ndarray, bsp_node: BSPNode, rays_interactions: list) -> list:
    if bsp_node.plane is None:
        return []

    if bsp_node.surfaces is not None:
        for surface in bsp_node.surfaces:
            interaction = surface.intersect("")
            # todo: imp revert to original dimension
            rays_interactions.append((interaction, surface.index))

    if bsp_node.left or bsp_node.right:
        axis = bsp_node.plane
        axis_index = {'x': 0, 'y': 1, 'z': 2}[bsp_node.cut[0]]  # Convert 'x', 'y', 'z' to 0, 1, 2
        median_value = bsp_node.cut[1]


        if bsp_node.left:
            # todo: fix dimensions
            left_ray_directions = ray_directions[:, :, axis_index] < median_value
            left_ray_source = ray_source[left_ray_directions]
            traverse(left_ray_source, left_ray_directions, bsp_node.left, rays_interactions)

        if bsp_node.right:
            # todo: fix dimensions
            right_ray_directions = ray_directions[:, :, axis_index] >= median_value
            right_ray_source = ray_source[right_ray_directions]
            traverse(right_ray_source, right_ray_directions, bsp_node.right, rays_interactions)

    return rays_interactions


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
