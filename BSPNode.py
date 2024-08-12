import numpy as np
from surfaces.Cube import Cube
from surfaces.Object3D import Object3D
from surfaces.SurfaceAbs import SurfaceAbs
from ray_functions import get_closest_hits
from util import *


class SurfaceContainer:
    def __init__(self, surface: Object3D):
        """
        Initializes a container for a surface, which is an object in 3D space.

        :param surface: An object of type Object3D representing the surface to be contained.
        """
        self.surface = surface
        self.intersection_indicator = 0


class BSPNode:
    BSP_COUNTER = 0

    def __init__(self, surfaces: list[SurfaceContainer] | None, left=None, right=None, cut=None,
                 small_bound=None, big_bound=None, inf_plains=None):
        """
        Initializes a node in the BSP (Binary Space Partitioning) tree.

        :param surfaces: A list of SurfaceContainer objects representing the surfaces contained in this node.
        :param left: The left child node of this BSPNode.
        :param right: The right child node of this BSPNode.
        :param cut: A tuple representing the axis and position where the space is divided.
        :param small_bound: The minimum boundary of the space represented by this node.
        :param big_bound: The maximum boundary of the space represented by this node.
        :param inf_plains: A list of infinite planes that are intersected by rays.
        """
        self.left: BSPNode = left
        self.right: BSPNode = right
        self.small_bound = small_bound
        self.big_bound = big_bound
        self.cut = cut
        self.inf_plains: list = inf_plains
        self.surfaces = surfaces

    @staticmethod
    def build_bsp_tree_rec(surfaces: list[SurfaceContainer], min_small_bound=None, max_big_bound=None,
                           inf_plains: list[SurfaceAbs] = None):
        """
        Recursively builds a BSP tree by dividing space based on the positions of surfaces.

        :param surfaces: A list of SurfaceContainer objects to be included in the BSP tree.
        :param min_small_bound: The minimum boundary of the current space.
        :param max_big_bound: The maximum boundary of the current space.
        :param inf_plains: A list of infinite planes that need to be considered during intersection calculations.
        :return: A BSPNode representing the root of the constructed BSP tree.
        """
        diffs = max_big_bound - min_small_bound
        if len(surfaces) <= 10 or np.linalg.norm(diffs) < EPSILON:
            return BSPNode(surfaces=surfaces, small_bound=min_small_bound, big_bound=max_big_bound,
                           inf_plains=inf_plains)

        axis = np.argmax(diffs)  # The axis along which to divide the space
        cut_location = (max_big_bound[axis] + min_small_bound[axis]) / 2
        axis_names = ["x", "y", "z"]
        cut = (axis_names[axis], cut_location)

        left_surfaces = []
        right_surfaces = []
        for s in surfaces:
            small_corner, big_corner = s.surface.get_enclosing_values()
            if big_corner[axis] <= cut_location:
                left_surfaces.append(s)
            elif small_corner[axis] >= cut_location:
                right_surfaces.append(s)
            else:
                left_surfaces.append(s)
                right_surfaces.append(s)

        b = np.full_like(max_big_bound, max_big_bound)
        b[axis] = cut_location
        left_tree = BSPNode.build_bsp_tree_rec(left_surfaces, min_small_bound, b)

        s = np.full_like(min_small_bound, min_small_bound)
        s[axis] = cut_location
        right_tree = BSPNode.build_bsp_tree_rec(right_surfaces, s, max_big_bound)

        return BSPNode(surfaces=None, left=left_tree, right=right_tree, cut=cut, small_bound=min_small_bound,
                       big_bound=max_big_bound, inf_plains=inf_plains)

    @staticmethod
    def build_bsp_tree(surfaces: list[SurfaceAbs], min_small_bound=None, max_big_bound=None):
        """
        Constructs a BSP tree from a list of 3D objects.

        :param surfaces: A list of SurfaceAbs objects representing the surfaces to be included in the BSP tree.
        :param min_small_bound: The minimum boundary for the space containing the surfaces.
        :param max_big_bound: The maximum boundary for the space containing the surfaces.
        :return: A BSPNode representing the root of the constructed BSP tree.
        """

        finite_objects: list[Object3D] = [o for o in surfaces if isinstance(o, Object3D)]
        inf_plains: list[SurfaceAbs] = [o for o in surfaces if not isinstance(o, Object3D)]

        if min_small_bound is None:
            small_bounds = np.array([s.get_enclosing_values()[0] for s in finite_objects])
            min_small_bound = np.min(small_bounds, axis=0)
        if max_big_bound is None:
            big_bounds = np.array([s.get_enclosing_values()[1] for s in finite_objects])
            max_big_bound = np.max(big_bounds, axis=0)

        surfaces_containers = [SurfaceContainer(s) for s in finite_objects]
        return BSPNode.build_bsp_tree_rec(surfaces_containers, min_small_bound, max_big_bound, inf_plains)

    def intersect_vectorize_rec(self, ray_sources, ray_directions):
        """
        Recursively computes intersections of rays with objects in the BSP tree.

        :param ray_sources: A matrix of ray source coordinates.
        :param ray_directions: A matrix of ray direction coordinates.
        :return: A tuple containing the intersection points and the indices of the intersected surfaces.
        """
        # Filter rays that don't intersect with the bounding box
        intersections_mask, _ = Cube.cube_intersection_mask(ray_sources, ray_directions, self.small_bound,
                                                            self.big_bound, outside_hits_only=False)

        relevant = np.where(intersections_mask)[0]

        # Initialize intersections and distances
        intersections = np.full_like(ray_directions, np.nan)
        intersections_indices = np.zeros(ray_directions.shape[0], dtype=int)
        intersections_dists = np.full(ray_directions.shape[0], np.inf)

        # Process infinite planes if they exist
        if self.inf_plains is not None:
            p_intersections, p_intersections_indices = get_closest_hits(ray_sources, ray_directions, self.inf_plains)
            plain_invalid = np.any(np.isnan(p_intersections), axis=-1)
            p_intersections_dists = np.linalg.norm(p_intersections - ray_sources, axis=-1)

            valid_mask = ~plain_invalid
            closer_mask = p_intersections_dists < intersections_dists

            intersections[valid_mask & closer_mask] = p_intersections[valid_mask & closer_mask]
            intersections_indices[valid_mask & closer_mask] = p_intersections_indices[valid_mask & closer_mask]
            intersections_dists[valid_mask & closer_mask] = p_intersections_dists[valid_mask & closer_mask]

        # Return if no relevant intersections found
        if not np.any(relevant):
            return intersections, intersections_indices

        # Process surfaces or recursively process child nodes
        if self.surfaces is None:
            # Process left child
            left_intersections, left_indices = self.left.intersect_vectorize_rec(ray_sources[relevant],
                                                                                 ray_directions[relevant])
            left_dists = np.linalg.norm(left_intersections - ray_sources[relevant], axis=-1)
            left_invalid = np.any(np.isnan(left_intersections), axis=-1)

            left_mask = np.where(~left_invalid & (left_dists < intersections_dists[relevant]))[0]
            intersections[relevant[left_mask]] = left_intersections[left_mask]
            intersections_indices[relevant[left_mask]] = left_indices[left_mask]
            intersections_dists[relevant[left_mask]] = left_dists[left_mask]

            # Process right child
            right_intersections, right_indices = self.right.intersect_vectorize_rec(ray_sources[relevant],
                                                                                    ray_directions[relevant])
            right_dists = np.linalg.norm(right_intersections - ray_sources[relevant], axis=-1)
            right_invalid = np.any(np.isnan(right_intersections), axis=-1)

            right_mask = np.where(~right_invalid & (right_dists < intersections_dists[relevant]))[0]
            intersections[relevant[right_mask]] = right_intersections[right_mask]
            intersections_indices[relevant[right_mask]] = right_indices[right_mask]
            intersections_dists[relevant[right_mask]] = right_dists[right_mask]

        else:
            # Directly intersect with surfaces in this node
            relevant_surfaces = [s.surface for s in self.surfaces]
            surface_intersections, surface_indices = get_closest_hits(ray_sources[relevant], ray_directions[relevant],
                                                                      relevant_surfaces)
            for s in self.surfaces:
                s.last_bsp_intersection = BSPNode.BSP_COUNTER

            surface_dists = np.linalg.norm(surface_intersections - ray_sources[relevant], axis=-1)
            surface_invalid = np.any(np.isnan(surface_intersections), axis=-1)
            surface_mask = np.where(~surface_invalid & (surface_dists < intersections_dists[relevant]))[0]

            intersections[relevant[surface_mask]] = surface_intersections[surface_mask]
            intersections_indices[relevant[surface_mask]] = surface_indices[surface_mask]
            intersections_dists[relevant[surface_mask]] = surface_dists[surface_mask]

        return intersections, intersections_indices

    def intersect_vectorize(self, ray_sources, ray_directions):
        """
        Initializes the ray tracing process for the BSP tree.

        :param ray_sources: A matrix of ray source coordinates.
        :param ray_directions: A matrix of ray direction coordinates.
        :return: A tuple containing the final intersection points and their respective indices.
        """
        BSPNode.BSP_COUNTER += 1
        return self.intersect_vectorize_rec(ray_sources, ray_directions)

    def __repr__(self, depth=0):
        indent = "  " * depth  # Create an indent based on the depth of the node
        surf_len = 0 if self.surfaces is None else len(self.surfaces)
        repr_str = f"{indent}BSPNode(cut={self.cut:}, bounds=({self.small_bound}, {self.big_bound}), {surf_len} surfaces={self.surfaces}\n"
        if self.left is not None:
            repr_str += self.left.__repr__(depth + 1)
        if self.right is not None:
            repr_str += self.right.__repr__(depth + 1)
        return repr_str
