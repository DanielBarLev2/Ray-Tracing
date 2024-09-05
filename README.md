### Ray Tracer

A ray tracer shoots rays from the observer’s eye (the camera) through a screen and into a scence, which contains one or more surfaces.
it calculates the rays intersection with the surfaces, finds the nearest intersection and calculates the color of the surface according to its material and lighting conditions.

# Surfaces
• Spheres. Each sphere is defined by the position of its center and its radius.
• Infinite planes. Each plane is defined by its normal N and an offset c along the normal. A point P on the plane will satisfy the formula P · N = c.
• Cubes. Each cube is defined by the position of its center (x, y, z) and its edge length (scalar). All boxes are axis aligned (meaning no rotations) to make the computation of intersections easier.

# Materials
• Diffuse color (RGB). This is the ”regular” color of a surface. 
• Specular color (RGB). Specularity is the reflection of a light source.
• Phong specularity coefficient (floating point number).
• Reflection color (RGB). Reflections from the surface are multiplied by this value. 
• Transparency (floating point number). This value will be 0 when the material is opaque (not transparent at all) and 1 when the material is completely transparent.

# Scene Definition Format
The scenes are defined in text scene files with the following format:
"cam" = camera settings (there will be only one per scene file)
params[0,1,2] = position (x, y, z) of the camera
params[3,4,5] = look-at position (x, y, z) of the camera
params[6,7,8] = up vector (x, y, z) of the camera
params[9] = screen distance from camera
params[10] = screen width from camera
"set" = general settings for the scene (once per scene file)
params[0,1,2] = background color (r, g, b)
params[3] = root number of shadow rays (N2 rays will be shot)
params[4] = maximum number of recursions
"mtl" = defines a new material
params[0,1,2] = diffuse color (r, g, b)
params[3,4,5] = specular color (r, g, b)
params[6,7,8] = reflection color (r, g, b)
params[9] = phong specularity coefficient (shininess)
params[10] = transparency value between 0 and 1
"sph" = defines a new sphere
params[0,1,2] = position of the sphere center (x, y, z)
params[3] = radius
params[4] = material index (integer). each defined material gets an
automatic material index starting from 1, 2 and so on
"pln" = defines a new plane
params[0,1,2] = normal (x, y, z)
params[3] = offset
params[4] = material index
"box" = defines a new box
params[0,1,2] = position of the box center (x, y, z)
params[3] = scale of the box, length of each edge
params[4] = material index
"lgt" = defines a new light
params[0,1,2] = position of the light (x, y, z)
params[3,4,5] = light color (r, g, b)
params[6] = specular intensity
params[7] = shadow intensity
params[8] = light width / radius (used for soft shadows)


