import torch
from torch import nn

# Cube (remains the same)
def get_cubes(shape=(64, 64, 64)):
    cube = torch.zeros(shape) + 1.33  # Default outer refractive index
    cube[16:48, 16:48, 16:48] = 1.37  # Outer cube with refractive index 1.37
    cube[24:40, 24:40, 24:40] = 1.4   # Inner cube with refractive index 1.4
    cube[28:36, 28:36, 28:36] = 1.5   # Innermost cube with refractive index 1.5
    return cube

# Sphere (remains the same)
def get_sphere(radius, size, inside_ri=1.5):
    sphere = torch.zeros((size, size, size)) + 1.33  # Default outer refractive index
    center = size // 2
    for x in range(size):
        for y in range(size):
            for z in range(size):
                if (x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2 <= radius ** 2:
                    sphere[x, y, z] = inside_ri  # Inside the sphere
    return sphere

# Cylinder (adapted to cube shape)
def get_cylinder(radius, height, size, inside_ri=1.5):
    # Create a cube with size x size x size dimensions
    cylinder = torch.zeros((size, size, size)) + 1.33  # Default outer refractive index
    center = size // 2
    # Adjust cylinder height to fit in the cube
    height = min(height, size)
    
    for x in range(size):
        for y in range(size):
            for z in range(height):
                if (x - center) ** 2 + (y - center) ** 2 <= radius ** 2:
                    cylinder[x, y, z] = inside_ri  # Inside the cylinder
    return cylinder

# **Cone**: A simple cone inside a cube
def get_cone(radius, height, size, inside_ri=1.5):
    cone = torch.zeros((size, size, size)) + 1.33  # Default outer refractive index
    center = size // 2
    for x in range(size):
        for y in range(size):
            for z in range(height):
                # Radius decreases linearly with height
                current_radius = radius * (1 - z / height)
                if (x - center) ** 2 + (y - center) ** 2 <= current_radius ** 2:
                    cone[x, y, z] = inside_ri  # Inside the cone
    return cone

# **Pyramid**: A pyramid inside a cube
def get_pyramid(base_size, height, size, inside_ri=1.5):
    pyramid = torch.zeros((size, size, size)) + 1.33  # Default outer refractive index
    center = size // 2
    for x in range(size):
        for y in range(size):
            for z in range(height):
                # Base size decreases linearly with height (form a pyramid shape)
                current_base = int(base_size * (1 - z / height))
                if abs(x - center) <= current_base and abs(y - center) <= current_base:
                    pyramid[x, y, z] = inside_ri  # Inside the pyramid
    return pyramid

# **Torus**: A ring-shaped torus inside a cube
def get_torus(inner_radius, outer_radius, size, height, inside_ri=1.5):
    torus = torch.zeros((size, size, size)) + 1.33  # Default outer refractive index
    center = size // 2
    for x in range(size):
        for y in range(size):
            for z in range(height):
                # Torus shape
                dist_from_center = ((x - center) ** 2 + (y - center) ** 2) ** 0.5
                if inner_radius <= dist_from_center <= outer_radius:
                    torus[x, y, z] = inside_ri  # Inside the torus
    return torus

# **Ellipsoid**: Ellipsoid with different radii along x, y, and z axes, inside a cube
def get_ellipsoid(rx, ry, rz, size, inside_ri=1.5):
    ellipsoid = torch.zeros((size, size, size)) + 1.33  # Default outer refractive index
    center = size // 2
    for x in range(size):
        for y in range(size):
            for z in range(size):
                if ((x - center) / rx) ** 2 + ((y - center) / ry) ** 2 + ((z - center) / rz) ** 2 <= 1:
                    ellipsoid[x, y, z] = inside_ri  # Inside the ellipsoid
    return ellipsoid

def get_random_objects(shape=(64, 64, 64), n_objects=64, inside_ri_range=(1.34, 1.60)):

    # Create a batch of empty objects with the shape (n_objects, *shape)
    objects = torch.zeros(n_objects, *shape)
    
    # Loop through and randomly generate n_objects
    for i in range(n_objects):
        # Random refractive index between inside_ri_range
        inside_ri = torch.rand(1) * (inside_ri_range[1] - inside_ri_range[0]) + inside_ri_range[0]
        
        # Randomly select an object type (0-6)
        object_type = torch.randint(0, 7, (1,)).item()

        # Generate the object based on the randomly selected type
        if object_type == 0:  # Cube
            objects[i] = get_cubes(shape)
        elif object_type == 1:  # Sphere
            radius = torch.randint(8, 24, (1,)).item()
            objects[i] = get_sphere(radius, shape[0], inside_ri=inside_ri)
        elif object_type == 2:  # Cylinder
            radius = torch.randint(8, 24, (1,)).item()
            height = torch.randint(8, 24, (1,)).item()
            objects[i] = get_cylinder(radius, height, shape[0], inside_ri=inside_ri)
        elif object_type == 3:  # Cone
            radius = torch.randint(8, 24, (1,)).item()
            height = torch.randint(8, 24, (1,)).item()
            objects[i] = get_cone(radius, height, shape[0], inside_ri=inside_ri)
        elif object_type == 4:  # Pyramid
            base_size = torch.randint(8, 24, (1,)).item()
            height = torch.randint(8, 24, (1,)).item()
            objects[i] = get_pyramid(base_size, height, shape[0], inside_ri=inside_ri)
        elif object_type == 5:  # Torus
            inner_radius = torch.randint(8, 24, (1,)).item()
            outer_radius = torch.randint(24, 40, (1,)).item()
            height = torch.randint(8, 24, (1,)).item()
            objects[i] = get_torus(inner_radius, outer_radius, shape[0], height, inside_ri=inside_ri)
        elif object_type == 6:  # Ellipsoid
            rx = torch.randint(8, 24, (1,)).item()
            ry = torch.randint(8, 24, (1,)).item()
            rz = torch.randint(8, 24, (1,)).item()
            objects[i] = get_ellipsoid(rx, ry, rz, shape[0], inside_ri=inside_ri)

    return objects.unsqueeze(1)  # Add channel dimension

if __name__ == "__main__":
    # Test the object generation function
    random_objects = get_random_objects(shape=(64, 64, 64), n_objects=32)
    
    #It shall return a tensor of shape (32, 64, 64, 64)
    print(random_objects.shape)
