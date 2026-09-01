import trimesh
import vhacdx

# 1. Load the high-poly visual mesh
mesh = trimesh.load('obj/drill/drill.obj')

# 2. Compute exact convex decomposition with 50 hulls
print("Slicing mesh into 50 pieces... this might take a minute...")
convex_pieces = vhacdx.compute_vhacd(mesh.vertices, mesh.faces, maxConvexHulls=30)

asset_tags = ""
geom_tags = ""

# 3. Export objects and build XML strings
for i, (vertices, faces) in enumerate(convex_pieces):
    # Save the 3D file
    piece_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    filepath = f'obj/drill/drill_col_{i}.obj'
    piece_mesh.export(filepath)
    
    # Build the XML lines
    asset_tags += f'    <mesh name="drill_col_{i}" file="{filepath}" scale="0.014 0.014 0.014"/>\n'
    geom_tags += f'    <geom type="mesh" mesh="drill_col_{i}" rgba="1 1 1 0" friction="1.5 0.05 0.01" margin="0.001"/>\n'

# 4. Write the include files (MuJoCo requires a <mujoco> root tag in included files)
with open('drill_assets.xml', 'w') as f:
    f.write(f"<mujoco>\n{asset_tags}</mujoco>")

with open('drill_collisions.xml', 'w') as f:
    f.write(f"<mujoco>\n{geom_tags}</mujoco>")

print("Success! Generated 30 collision meshes and 2 XML include files.")