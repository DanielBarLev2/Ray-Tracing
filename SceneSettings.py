class SceneSettings:
    def __init__(self, background_color, root_number_shadow_rays, max_recursions):
        self.background_color = background_color
        self.root_number_shadow_rays = int(root_number_shadow_rays)
        self.max_recursions = int(max_recursions)
