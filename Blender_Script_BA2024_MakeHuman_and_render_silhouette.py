import bpy
import random
import math
import mathutils
import time
import sys
import argparse


######################################
### 00 - ARGPARSE via Command Line ###
######################################

class ArgumentParserForBlender(argparse.ArgumentParser):
    def _get_argv_after_doubledash(self):
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:] # the list after '--'
        except ValueError as e: # '--' not in the list:
            return []
    def parse_args(self):
        return super().parse_args(args=self._get_argv_after_doubledash())

parser = ArgumentParserForBlender()


parser.add_argument("-id", type=int, default=0,
                    help="ID-Number")
parser.add_argument("-v1", "--value1", type=float, default=1.0,
                    help="float for value1")
parser.add_argument("-v2", "--value2", type=float, default=1.0,
                    help="float for value2")
parser.add_argument("-v3", "--value3", type=float, default=1.0,
                    help="float for value3")
parser.add_argument("-v4", "--value4", type=float, default=1.0,
                    help="float for value4")
parser.add_argument("-v5", "--value5", type=float, default=1.0,
                    help="float for value5")
parser.add_argument("-v6", "--value6", type=float, default=1.0,
                    help="float for value6")
                    
args = parser.parse_args()
ID = args.id
VALUE1 = args.value1
VALUE2 = args.value2
VALUE3 = args.value3
VALUE4 = args.value4
VALUE5 = args.value5
VALUE6 = args.value6


######################################
### 01 - Empty Scene               ###
######################################
# Switch to "object mode"
bpy.ops.object.mode_set(mode='OBJECT')
# Select everything
bpy.ops.object.select_all(action='SELECT')
# Delete Everything
bpy.ops.object.delete(use_global=False, confirm=False)


######################################
### 02 - Set Rendering Options     ###
######################################
# Colors will be rendered exactly in the way they are set for materials (e.g. Hex: 000000 = Hex: 000000)
bpy.context.scene.view_settings.view_transform = 'Standard' 
# This disables the aliasing effect 
bpy.context.scene.render.filter_size = 0.0    
# Set File Format to "JPEG"
bpy.context.scene.render.image_settings.file_format = 'PNG'   
bpy.context.scene.render.image_settings.color_mode = 'BW'
           

######################################
### 03 - Create Camera             ###
######################################
# Create Camera
bpy.ops.object.camera_add(enter_editmode=False, location=(0, 1, 5), rotation=(0.0, 0.0, 1.5708), scale=(1, 1, 1))
bpy.context.scene.camera = bpy.context.active_object

######################################
### 04 - Create Plane              ###
######################################
# Create Plane
bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
# Resize Plane [x = 10, y = 10, z = 10]
bpy.ops.transform.resize(value=(10, 10, 10))
# Add Physics to plane
bpy.ops.rigidbody.object_add()
bpy.context.object.rigid_body.type = 'PASSIVE'
# Add Material
active_obj = bpy.context.active_object
# Get material
mat_surface = bpy.data.materials.get("Surface")
if mat_surface is None:
    # Add new Material "Human"
    mat_surface = bpy.data.materials.new(name="Surface")
    # Use nodes
    mat_surface.use_nodes = True
    # Delete standard "Principled BSDF" shader
    mat_surface.node_tree.nodes.remove(mat_surface.node_tree.nodes.get('Principled BSDF'))
    # Select "Material Output" node
    material_output = mat_surface.node_tree.nodes.get('Material Output')
    # Set location of node
    material_output.location = (-400,0)
    # Add emission shader
    emission_shader = mat_surface.node_tree.nodes.new('ShaderNodeEmission')
    # Set location of node
    emission_shader.location = (-600,0)
    # Set default color
    emission_shader.inputs[0].default_value = (1, 1, 1, 1)
    # Link emission shader to output node
    mat_surface.node_tree.links.new(emission_shader.outputs[0], material_output.inputs[0]) 

    # Assign it to object
if active_obj.data.materials:
    # Assign to 1st material slot
    active_obj.data.materials[0] = mat_surface
else:
    # No slots
    active_obj.data.materials.append(mat_surface)


######################################
### 05 - Create Human              ###
######################################
# Create Human
bpy.ops.mpfb.create_human()

######################################
#### 05-1 - Phenotype: Macro Details #
######################################
### Gender (Female = 0.0, Neutral = 0.5, Male =  1.0)
#bpy.context.scene.mpfb_macropanel_gender = random.random() 
bpy.context.scene.mpfb_macropanel_gender = VALUE1
### Age (Baby = 0.0, Child = 0.19, Young = 0.50, Old = 1.0 )
#bpy.context.scene.mpfb_macropanel_age = random.random()/2+0.5
bpy.context.scene.mpfb_macropanel_age = VALUE2
### Muscle (Minimum = 0.0, Average = 0.5, Maximum =  1.0)
#bpy.context.scene.mpfb_macropanel_muscle = random.random()
bpy.context.scene.mpfb_macropanel_muscle = VALUE3
### Weight (Minimum = 0.0, Average = 0.5, Maximum =  1.0)
#bpy.context.scene.mpfb_macropanel_weight = random.random()
bpy.context.scene.mpfb_macropanel_weight = VALUE4
### Height (Short = 0.0, Average = 0.5, Tall =  1.0)
#bpy.context.scene.mpfb_macropanel_height = random.random()
bpy.context.scene.mpfb_macropanel_height = VALUE5
### Proportions (Uncommon = 0.0, Average = 0.5, Ideal =  1.0)
#bpy.context.scene.mpfb_macropanel_proportions = random.random()
bpy.context.scene.mpfb_macropanel_proportions = VALUE6

######################################
#### 05-2 - Add Material           ###
######################################
active_obj = bpy.context.active_object

# Get material
mat_human = bpy.data.materials.get('Human')
if mat_human is None:
    # Add new Material "Human"
    mat_human = bpy.data.materials.new(name='Human')
    # Use nodes
    mat_human.use_nodes = True
    # Delete standard "Principled BSDF" shader
    mat_human.node_tree.nodes.remove(mat_human.node_tree.nodes.get('Principled BSDF'))
    # Select "Material Output" node
    material_output = mat_human.node_tree.nodes.get('Material Output')
    # Set location of node
    material_output.location = (-400,0)
    # Add emission shader
    emission_shader = mat_human.node_tree.nodes.new('ShaderNodeEmission')
    # Set location of node
    emission_shader.location = (-600,0)
    # Set default color
    emission_shader.inputs[0].default_value = (0, 0, 0, 1)
    # Link emission shader to output node
    mat_human.node_tree.links.new(emission_shader.outputs[0], material_output.inputs[0]) 

    # Assign it to object
if active_obj.data.materials:
    # Assign to 1st material slot
    active_obj.data.materials[0] = mat_human
else:
    # No slots
    active_obj.data.materials.append(mat_human)


######################################
#### 05-3 - Add Rig                ###
######################################
# Change standard rig to "game engine" rig
bpy.context.scene.MPFB_ADR_standard_rig = 'game_engine'
# Add rig
bpy.ops.mpfb.add_standard_rig()
# Change Rig Visualisation
bpy.context.object.data.display_type = 'OCTAHEDRAL'


######################################
#### 05-3.1 - Apply Shape Keys     ###
######################################
# Define object called "human" and link to "Human"-Model in Scene
human = bpy.data.objects['Human']
# Deselect everything
bpy.context.view_layer.objects.active = None
# Select Human-Model
bpy.context.view_layer.objects.active = human
# Apply all Shape Keys
bpy.ops.object.shape_key_remove(all=True, apply_mix=True)


##################################################
#### 05-3.2 - Delete unnecessary Vertex Groups ###
##################################################
# Switch to "edit mode"
bpy.ops.object.mode_set(mode='EDIT')
# Deselect everything 
bpy.ops.mesh.select_all(action='DESELECT')
# Show Armature
bpy.context.object.modifiers['Hide helpers'].mode = 'ARMATURE'
# Select "HelperGeometry"
human.vertex_groups.active = human.vertex_groups['HelperGeometry']
# Select and delete vertices
bpy.ops.object.vertex_group_select()
bpy.ops.mesh.delete(type='VERT')
# Select "JointCubes"
human.vertex_groups.active = human.vertex_groups['JointCubes']
# Select and delete vertices
bpy.ops.object.vertex_group_select()
bpy.ops.mesh.delete(type='VERT')
# Switch to "object mode"
bpy.ops.object.mode_set(mode='OBJECT')


######################################
#### 05-3.3 - Reposition Extremities #
######################################
def reposition_Extremities(single_bone, rotX, rotY, rotZ):

    # Deselect everything
    bpy.ops.object.select_all(action="DESELECT")
    # Select "Human.rig"
    bpy.data.objects["Human.rig"].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects["Human.rig"]
    # Switch to "pose mode"
    bpy.ops.object.mode_set(mode="POSE")
    # Random
    randX = 10*(2*random.random()-1)
    randY = 10*(2*random.random()-1)
    randZ = 10*(2*random.random()-1) 
    # Define 45 degree rotation
    rotX_deg = math.radians(rotX+randX)
    rotY_deg = math.radians(rotY+randY)
    rotZ_deg = math.radians(rotZ+randZ)
    # Select Bone
    bpy.context.object.data.bones.active = bpy.data.objects['Human.rig'].pose.bones[single_bone].bone
    bpy.data.objects['Human.rig'].pose.bones[single_bone].bone.select = True
    # Rotate Bone
    bpy.ops.transform.rotate(value=rotX_deg, orient_axis='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL')
    bpy.ops.transform.rotate(value=rotY_deg, orient_axis='Y', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL')
    bpy.ops.transform.rotate(value=rotZ_deg, orient_axis='Z', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL')
    bpy.data.objects['Human.rig'].pose.bones[single_bone].bone.select = False
    
    # Switch to "object mode"
    bpy.ops.object.mode_set(mode="OBJECT")
    
     
reposition_Extremities('upperarm_r',30, 0, -30)
reposition_Extremities('upperarm_l',30, 0 ,30)
reposition_Extremities('thigh_r', 45, 0, 0)
reposition_Extremities('thigh_l', 45, 0, 0)

########################################
#### 05-3.3 - Reposition Body - Part 1 #
########################################

# Deselect everything
bpy.ops.object.select_all(action="DESELECT")
# Select "Human.rig"
bpy.data.objects["Human.rig"].select_set(True)
bpy.context.view_layer.objects.active = bpy.data.objects["Human.rig"]
# Switch to "pose mode"
bpy.ops.object.mode_set(mode="POSE") 
# Define 45 degree rotation
rot_90deg = math.radians(-90)
# Select Bone
bpy.context.object.data.bones.active = bpy.data.objects['Human.rig'].pose.bones["Root"].bone
bpy.data.objects['Human.rig'].pose.bones["Root"].bone.select = True
# Rotate Bone
bpy.ops.transform.rotate(value=rot_90deg, orient_axis='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), mirror=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False, release_confirm=True)
bpy.data.objects['Human.rig'].pose.bones["Root"].bone.select = False
# Switch to "object mode"
bpy.ops.object.mode_set(mode="OBJECT")



#######################################
#### 05-3.4 - Apply new Rest Position #
#######################################

def newRestPos():
    # Switch to "object mode"
    bpy.ops.object.mode_set(mode='OBJECT')
    # Deselect everything
    bpy.ops.object.select_all(action="DESELECT")
    # Deselect everything
    bpy.context.view_layer.objects.active = None
    # Select Human-Model
    bpy.context.view_layer.objects.active = human
    # Duplicate Armature
    bpy.ops.object.modifier_copy(modifier="Armature")
    # Apply Armature
    bpy.ops.object.modifier_apply(modifier="Armature")
    # Deselect everything
    bpy.ops.object.select_all(action="DESELECT")
    # Select "Human.rig"
    bpy.data.objects["Human.rig"].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects["Human.rig"]
    # Switch to "pose mode"
    bpy.ops.object.mode_set(mode='POSE')
    # Apply Pose as Rest Pose
    bpy.ops.pose.armature_apply(selected=False)
    # Switch to "object mode"
    bpy.ops.object.mode_set(mode='OBJECT')
    # Deselect everything
    bpy.ops.object.select_all(action="DESELECT")
    # Deselect everything
    bpy.context.view_layer.objects.active = None
    # Select Human-Model
    bpy.context.view_layer.objects.active = human
    # Change name of Armature Modifier to "Armature"
    bpy.context.object.modifiers["Armature.001"].name = "Armature"

newRestPos()

########################################
#### 05-3.5 - Reposition Body - Part 2 #
########################################

########################################
### Find the lowst Z-Coordinate (minZ) #
########################################
# Deselect everything
bpy.context.view_layer.objects.active = None
# Select Human-Model
bpy.context.view_layer.objects.active = human
# Active object's world matrix
worldMatrix = bpy.context.object.matrix_world      
# Global coordinates of vertices
glob_vertex_coordinates = [ worldMatrix @ v.co for v in bpy.context.object.data.vertices ] 
# Find the lowest Z value amongst the object's verts
minZ = min( [ co.z for co in glob_vertex_coordinates ] ) 

########################################
#### Reposition Body - Part 2          #
########################################
# Deselect everything
bpy.ops.object.select_all(action="DESELECT")
# Select "Human.rig"
bpy.data.objects["Human.rig"].select_set(True)
bpy.context.view_layer.objects.active = bpy.data.objects["Human.rig"]
# Switch to "pose mode"
bpy.ops.object.mode_set(mode="POSE") 
# Define Absolute "z" Value
minZAbs = abs(minZ)
# Add "0.01" Units to create a set-off to the plane
newZ = minZAbs + 0.01 
# Select Bone
bpy.context.object.data.bones.active = bpy.data.objects['Human.rig'].pose.bones["Root"].bone
bpy.data.objects['Human.rig'].pose.bones["Root"].bone.select = True
# Translate Bone
bpy.ops.transform.translate(value=(0, 0, newZ), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL')
bpy.data.objects['Human.rig'].pose.bones["Root"].bone.select = False
# Switch to "object mode"
bpy.ops.object.mode_set(mode="OBJECT")
bpy.data.objects['Human.rig'].pose.bones["Root"].bone.select = False

#######################################
#### Apply new Rest Position          #
#######################################
newRestPos()


######################################
#### 05-3.6 - Make bones Parent    ###
######################################
# Deselect everything
bpy.ops.object.select_all(action="DESELECT")
# Select "Human.rig"
bpy.data.objects["Human.rig"].select_set(True)
bpy.context.view_layer.objects.active = bpy.data.objects["Human.rig"]
# Switch to "edit mode"
bpy.ops.object.mode_set(mode="EDIT") 

def make_BoneParent(bone_name):

    bpy.ops.armature.select_all(action='DESELECT')
    bpy.context.object.data.edit_bones[bone_name].select = True
    bpy.ops.armature.parent_clear(type='CLEAR')
    bpy.context.object.data.edit_bones[bone_name].select = False
    bpy.ops.armature.select_all(action='DESELECT')
    
make_BoneParent("pelvis")
make_BoneParent("spine_01")
make_BoneParent("spine_02")
make_BoneParent("spine_03")
make_BoneParent("neck_01")
make_BoneParent("head")
make_BoneParent("clavicle_l")
make_BoneParent("clavicle_r")
make_BoneParent("upperarm_r")
make_BoneParent("upperarm_l")
make_BoneParent("lowerarm_r")
make_BoneParent("lowerarm_l")
make_BoneParent("hand_r")
make_BoneParent("hand_l")
make_BoneParent("index_01_r")
make_BoneParent("index_02_r")
make_BoneParent("index_03_r")
make_BoneParent("middle_01_r")
make_BoneParent("middle_02_r")
make_BoneParent("middle_03_r")
make_BoneParent("pinky_01_r")
make_BoneParent("pinky_02_r")
make_BoneParent("pinky_03_r")
make_BoneParent("ring_01_r")
make_BoneParent("ring_02_r")
make_BoneParent("ring_03_r")
make_BoneParent("thumb_01_r")
make_BoneParent("thumb_02_r") 
make_BoneParent("thumb_03_r")
make_BoneParent("index_01_l")
make_BoneParent("index_02_l")
make_BoneParent("index_03_l")
make_BoneParent("middle_01_l")
make_BoneParent("middle_02_l")
make_BoneParent("middle_03_l")
make_BoneParent("pinky_01_l")
make_BoneParent("pinky_02_l")
make_BoneParent("pinky_03_l")
make_BoneParent("ring_01_l")
make_BoneParent("ring_02_l")
make_BoneParent("ring_03_l")
make_BoneParent("thumb_01_l")
make_BoneParent("thumb_02_l") 
make_BoneParent("thumb_03_l")
make_BoneParent("thigh_r")
make_BoneParent("thigh_l")
make_BoneParent("calf_r")
make_BoneParent("calf_l")
make_BoneParent("foot_r")
make_BoneParent("ball_r")
make_BoneParent("foot_l")
make_BoneParent("ball_l")                  
# Switch to "object mode"
bpy.ops.object.mode_set(mode="OBJECT") 



######################################
#### 05-4 - Add Ragdoll            ###
######################################

#################################################
#### 05-4.1 - Create Meshes from Extremities  ###
#################################################
# Deselect everything
bpy.context.view_layer.objects.active = None
# Select Human-Model
bpy.context.view_layer.objects.active = human


def make_rigdollPart(body_parts_and_weights, new_name):
    bpy.ops.object.mode_set(mode="EDIT") 
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for body_part, weight in body_parts_and_weights.items():
        for vertex in human.data.vertices:
            if human.vertex_groups[body_part].index in list(map(lambda g: g.group, vertex.groups)):
                if human.vertex_groups[body_part].weight(vertex.index) > weight:
                    vertex.select = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.duplicate()
    bpy.ops.mesh.separate(type="SELECTED")
    bpy.data.objects["Human.001"].name = new_name
    
#make_rigdollPart({"head" : 0.0, "neck_01" : 0.0, "clavicle_l" : 0.6, "clavicle_r" : 0.6, "pelvis" : 0.6, "spine_01" : 0.1, "spine_02" : 0.5, "spine_03":   0.5,}, "Torso")
make_rigdollPart({"head" : 0.0, "neck_01" : 0.0, "clavicle_l" : 0.4, "clavicle_r" : 0.4, "pelvis" : 0.6, "spine_01" : 0.1, "spine_02" : 0.0, "spine_03":   0.3,}, "Torso")
#make_rigdollPart({"upperarm_r" : 0.7}, "Upperarm_R")
make_rigdollPart({"upperarm_r" : 0.5}, "Upperarm_R")
#make_rigdollPart({"upperarm_l" : 0.7}, "Upperarm_L")
make_rigdollPart({"upperarm_l" : 0.5}, "Upperarm_L")
make_rigdollPart({"lowerarm_r" : 0.5}, "Lowerarm_R")
make_rigdollPart({"lowerarm_l" : 0.5}, "Lowerarm_L")
make_rigdollPart({"hand_r" : 0.5, "index_01_r" : 0.0, "index_02_r" : 0.0, "index_03_r" : 0.0, "middle_01_r": 0.0, "middle_02_r" : 0.0, "middle_03_r" : 0.0, "pinky_01_r" : 0.0, "pinky_02_r" : 0.0, "pinky_03_r" : 0.0, "ring_01_r" : 0.0, "ring_02_r" : 0.0, "ring_03_r" : 0.0, "thumb_01_r" : 0.0, "thumb_02_r" : 0.0, "thumb_03_r" : 0.0}, "Hand_R")
make_rigdollPart({"hand_l" : 0.5, "index_01_l" : 0.0, "index_02_l" : 0.0, "index_03_l" : 0.0, "middle_01_l": 0.0, "middle_02_l" : 0.0, "middle_03_l" : 0.0, "pinky_01_l" : 0.0, "pinky_02_l" : 0.0, "pinky_03_l" : 0.0, "ring_01_l" : 0.0, "ring_02_l" : 0.0, "ring_03_l" : 0.0, "thumb_01_l" : 0.0, "thumb_02_l" : 0.0, "thumb_03_l" : 0.0}, "Hand_L")
make_rigdollPart({"thigh_r" : 0.5}, "Thigh_R")
make_rigdollPart({"thigh_l" : 0.5}, "Thigh_L")
make_rigdollPart({"calf_r" : 0.45}, "Calf_R")
make_rigdollPart({"calf_l" : 0.45}, "Calf_L")
make_rigdollPart({"foot_r" : 0.5, "ball_r" : 0.0}, "Foot_R")
make_rigdollPart({"foot_l" : 0.5, "ball_l" : 0.0}, "Foot_L")


###############################################
#### 05-4.2 - Delete Modifier from Meshes   ###
###############################################

def delete_modifiers(body_part):

    # Switch to "object mode" 
    bpy.ops.object.mode_set(mode='OBJECT')    
    # Deselect everything
    bpy.ops.object.select_all(action="DESELECT")
    # Define "bodyPart"
    bodyPart = bpy.data.objects[body_part]
    # Select body part
    bodyPart.select_set(True)
    bpy.context.view_layer.objects.active = bodyPart
    # Delete modifier
    bpy.ops.object.modifier_remove(modifier="Armature")
    bpy.ops.object.modifier_remove(modifier="Hide helpers")
    # Delete Parent Relation
    bpy.context.object.parent = None
    # Delete material
    bpy.ops.object.material_slot_remove()

delete_modifiers("Torso")
delete_modifiers("Upperarm_R")
delete_modifiers("Upperarm_L")
delete_modifiers("Lowerarm_R")
delete_modifiers("Lowerarm_L")
delete_modifiers("Hand_R")
delete_modifiers("Hand_L")
delete_modifiers("Thigh_R")
delete_modifiers("Thigh_L")
delete_modifiers("Calf_R")
delete_modifiers("Calf_L")
delete_modifiers("Foot_R")
delete_modifiers("Foot_L")

########################################
#### 05-4.3 - Create Collider Meshes ###
########################################

def cleanRagdollMesh(body_part):
    bodyPartName = body_part
    bodyPart = bpy.data.objects[bodyPartName]
    # Deselect everything
    bpy.ops.object.select_all(action="DESELECT")
    # Select body part
    bodyPart.select_set(True)
    bpy.context.view_layer.objects.active = bodyPart
    # Switch to "edit mode"
    bpy.ops.object.mode_set(mode="EDIT")
    # Select all vertices
    bpy.ops.mesh.select_all(action='SELECT')
    # Clean Mesh
    bpy.ops.mesh.delete_loose()
    # Select all vertices
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.fill_holes(sides=100)
    bpy.ops.mesh.face_make_planar(repeat=20)
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.mesh.tris_convert_to_quads()
    # Switch to "object mode"
    bpy.ops.object.mode_set(mode='OBJECT')
    # Set Origin of Mesh to "Human Centre"
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')

cleanRagdollMesh('Torso')
cleanRagdollMesh('Upperarm_R')
cleanRagdollMesh('Upperarm_L')
cleanRagdollMesh('Lowerarm_R')
cleanRagdollMesh('Lowerarm_L')
cleanRagdollMesh('Hand_R')
cleanRagdollMesh('Hand_L')
cleanRagdollMesh('Thigh_R')
cleanRagdollMesh('Thigh_L')
cleanRagdollMesh('Calf_R')
cleanRagdollMesh('Calf_L')
cleanRagdollMesh('Foot_R')
cleanRagdollMesh('Foot_L')


######################################
#### 05-4.4 - Group  Meshes        ###
######################################

# Switch to "object mode" 
bpy.ops.object.mode_set(mode='OBJECT')    
# Deselect everything
bpy.ops.object.select_all(action="DESELECT")
# Select Body Part
bpy.data.objects['Torso'].select_set(True)
bpy.data.objects['Upperarm_R'].select_set(True)
bpy.data.objects['Upperarm_L'].select_set(True)
bpy.data.objects['Lowerarm_R'].select_set(True)
bpy.data.objects['Lowerarm_L'].select_set(True)
bpy.data.objects['Hand_R'].select_set(True)
bpy.data.objects['Hand_L'].select_set(True)
bpy.data.objects['Thigh_R'].select_set(True)
bpy.data.objects['Thigh_L'].select_set(True)
bpy.data.objects['Calf_R'].select_set(True)
bpy.data.objects['Calf_L'].select_set(True)
bpy.data.objects['Foot_R'].select_set(True)
bpy.data.objects['Foot_L'].select_set(True)
# Move Body Parts to new Collection called "BodyParts"
bpy.ops.object.move_to_collection(collection_index=0, is_new=True, new_collection_name="BodyParts")




######################################
#### 05-4.5 - Adjust Ragdool Origin ##
######################################

def adjust_ragdollOrigin(body_part, single_bone):
    bodyPartName = body_part
    singlebone = single_bone
    bodyPart = bpy.data.objects[bodyPartName]
    # Deselect everything
    bpy.ops.object.select_all(action="DESELECT")
    # Select Armature
    bpy.context.view_layer.objects.active = bpy.data.objects["Human.rig"]
    # Switch to "edit mode"
    bpy.ops.object.mode_set(mode="EDIT")
    # Set Active bone
    bpy.context.object.data.edit_bones[single_bone].select = True
    bpy.context.object.data.edit_bones[single_bone].select_head = True
    # Get Coordinates
    boneName_head = bpy.data.objects["Human.rig"].data.edit_bones[single_bone].head
    boneName_tail = bpy.data.objects["Human.rig"].data.edit_bones[single_bone].tail
    boneName_roll = bpy.data.objects["Human.rig"].data.edit_bones[single_bone].roll
    
    print(bpy.data.objects["Human.rig"].data.edit_bones[single_bone])
    a = mathutils.Vector((0.0,1.0,0.0))
    b = mathutils.Vector(boneName_tail-boneName_head).normalized()
    
    #####################################################
    # Moved this line upwards:                          #
    #####################################################
    # Set 3D cursor to Bone head location
    bpy.context.scene.cursor.location = (boneName_head[0],boneName_head[1],boneName_head[2])
    #####################################################

    # Switch to "object mode"
    bpy.ops.object.mode_set(mode="OBJECT")
    # Deselect everything
    bpy.ops.object.select_all(action="DESELECT")
    # Select body part
    bodyPart.select_set(True)
    bpy.context.view_layer.objects.active = bodyPart
    bpy.context.scene.tool_settings.use_transform_data_origin = True
    # Set origin to 3D cursor
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
    
    bpy.ops.transform.rotate(value=(-1*mathutils.Vector.angle(a,b)),orient_axis="X",orient_matrix=(mathutils.Vector.cross(a,b).normalized(),(0,0,0),(0,0,0)))
    bpy.ops.transform.rotate(value=(-1*boneName_roll), orient_axis='Y', orient_type='LOCAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='LOCAL')
    
    
    # Deselect everything
    bpy.ops.object.select_all(action="DESELECT")
    # Select Armature
    bpy.context.view_layer.objects.active = bpy.data.objects["Human.rig"]
    bpy.context.scene.tool_settings.use_transform_data_origin = False
    # Switch to "edit mode"
    bpy.ops.object.mode_set(mode="EDIT")
    # Set Active bone to False
    bpy.context.object.data.edit_bones[single_bone].select = False
    bpy.context.object.data.edit_bones[single_bone].select_head = False
    # Switch to "object mode"
    bpy.ops.object.mode_set(mode="OBJECT")


adjust_ragdollOrigin("Torso", "pelvis")
adjust_ragdollOrigin("Upperarm_R", "upperarm_r")
adjust_ragdollOrigin("Upperarm_L", "upperarm_l")
adjust_ragdollOrigin("Lowerarm_R", "lowerarm_r")
adjust_ragdollOrigin("Lowerarm_L", "lowerarm_l")
adjust_ragdollOrigin("Hand_R", "hand_r")
adjust_ragdollOrigin("Hand_L", "hand_l")
adjust_ragdollOrigin("Thigh_R", "thigh_r")
adjust_ragdollOrigin("Thigh_L", "thigh_l")
adjust_ragdollOrigin("Calf_R", "calf_r")
adjust_ragdollOrigin("Calf_L", "calf_l") 
adjust_ragdollOrigin("Foot_R", "foot_r") 
adjust_ragdollOrigin("Foot_L", "foot_l")


######################################
#### 05-4.6 - Build Ragdool        ###
######################################

def addRigidBodyPhysics(body_part):
    bodyPartName = body_part
    bodyPart = bpy.data.objects[bodyPartName]
    # Deselect everything
    bpy.ops.object.select_all(action="DESELECT")
    # Select body part
    bodyPart.select_set(True)
    bpy.context.view_layer.objects.active = bodyPart
    bpy.ops.rigidbody.object_add()

addRigidBodyPhysics('Torso')
addRigidBodyPhysics('Upperarm_R')
addRigidBodyPhysics('Upperarm_L')
addRigidBodyPhysics('Lowerarm_R')
addRigidBodyPhysics('Lowerarm_L')
addRigidBodyPhysics('Hand_R')
addRigidBodyPhysics('Hand_L')
addRigidBodyPhysics('Thigh_R')
addRigidBodyPhysics('Thigh_L')
addRigidBodyPhysics('Calf_R')
addRigidBodyPhysics('Calf_L')
addRigidBodyPhysics('Foot_R')
addRigidBodyPhysics('Foot_L')

######################################
#### 05-5 - Add Physics            ###
######################################

def addRigidBodyConstraints(body_part_1, body_part_2, x_angle_low, x_angle_up, y_angle_low, y_angle_up, z_angle_low, z_angle_up):
    bodyPartName1 = body_part_1
    bodyPartName2 = body_part_2
    bodyPart1 = bpy.data.objects[bodyPartName1]
    bodyPart2 = bpy.data.objects[bodyPartName2]
    # Deselect everything
    bpy.ops.object.select_all(action="DESELECT")
    # Select body part
    bodyPart1.select_set(True)
    bpy.context.view_layer.objects.active = bodyPart1
    # Add Rigid Body Constaint
    bpy.ops.rigidbody.constraint_add()
    # Choose Type "Generic" 
    bpy.context.object.rigid_body_constraint.type = 'GENERIC'
    # Connect Neighbouring Rigid Bodies
    bpy.context.object.rigid_body_constraint.object1 = bpy.data.objects[bodyPartName1]
    bpy.context.object.rigid_body_constraint.object2 = bpy.data.objects[bodyPartName2]
    # Disable Collisions
    bpy.context.object.rigid_body_constraint.disable_collisions = True
    # Limits: Angular "X"
    
    bpy.context.object.rigid_body_constraint.use_limit_ang_x = True
    bpy.context.object.rigid_body_constraint.limit_ang_x_lower = math.radians(x_angle_low)
    bpy.context.object.rigid_body_constraint.limit_ang_x_upper = math.radians(x_angle_up)
     # Limits: Angular "Y"
    bpy.context.object.rigid_body_constraint.use_limit_ang_y = True
    bpy.context.object.rigid_body_constraint.limit_ang_y_lower = math.radians(y_angle_low)
    bpy.context.object.rigid_body_constraint.limit_ang_y_upper = math.radians(y_angle_up)
     # Limits: Angular "Z"
    bpy.context.object.rigid_body_constraint.use_limit_ang_z = True
    bpy.context.object.rigid_body_constraint.limit_ang_z_lower = math.radians(z_angle_low)
    bpy.context.object.rigid_body_constraint.limit_ang_z_upper = math.radians(z_angle_up)
    
    # Limits: Linear "X" 
    bpy.context.object.rigid_body_constraint.use_limit_lin_x = True
    bpy.context.object.rigid_body_constraint.limit_lin_x_lower = 0
    bpy.context.object.rigid_body_constraint.limit_lin_x_upper = 0
    # Limits: Linear "Y"
    bpy.context.object.rigid_body_constraint.use_limit_lin_y = True
    bpy.context.object.rigid_body_constraint.limit_lin_y_lower = 0
    bpy.context.object.rigid_body_constraint.limit_lin_y_upper = 0
    # Limits: Linear "Z"
    bpy.context.object.rigid_body_constraint.use_limit_lin_z = True
    bpy.context.object.rigid_body_constraint.limit_lin_z_lower = 0
    bpy.context.object.rigid_body_constraint.limit_lin_z_upper = 0
    
addRigidBodyConstraints('Hand_R', 'Lowerarm_R', -45,45,-5,5,-45,45)
addRigidBodyConstraints('Hand_L', 'Lowerarm_L', -45,45,-5,5,-45,45)
addRigidBodyConstraints('Upperarm_R', 'Torso', -90, 0, -70, 0,0,45) #-90,110,-45,45,-10,90)
addRigidBodyConstraints('Upperarm_L', 'Torso', -90, 0, 0, 70,-45,0) #-90,110,-45,45,-10,90)
addRigidBodyConstraints('Lowerarm_L', 'Upperarm_L', -15,15,-45,45,0,0) # 0,150,-45,45,0,0
addRigidBodyConstraints('Lowerarm_R', 'Upperarm_R', -15,15,-45,45,0,0) # 0,150,-45,45,0,0
addRigidBodyConstraints('Foot_R', 'Calf_R', -10,90,-5,5,-30,30)
addRigidBodyConstraints('Foot_L', 'Calf_L', -10,90,-5,5,-30,30)
addRigidBodyConstraints('Calf_R', 'Thigh_R', 0,100,0,0,0,0)
addRigidBodyConstraints('Calf_L', 'Thigh_L', 0,100,0,0,0,0)
addRigidBodyConstraints('Thigh_R', 'Torso', -100,100,-25,25,-25,25)
addRigidBodyConstraints('Thigh_L', 'Torso', -100,100,-25,25,-25,25)


# Deselect everything
bpy.ops.object.select_all(action="DESELECT")   


######################################
#### 05-6 - Add Armature to Rigdoll ##
######################################

# Deselect everything
bpy.ops.object.select_all(action="DESELECT")
# Select "Human.rig"
bpy.data.objects["Human.rig"].select_set(True)
bpy.context.view_layer.objects.active = bpy.data.objects["Human.rig"]
# Switch to "pose mode"
bpy.ops.object.mode_set(mode="POSE") 

def add_bonesToRigdoll(single_bone, body_part):
    bpy.context.object.data.bones.active = bpy.data.objects['Human.rig'].pose.bones[single_bone].bone
    bpy.data.objects['Human.rig'].pose.bones[single_bone].bone.select = True
    bpy.ops.pose.constraint_add(type='CHILD_OF')
    bpy.context.object.pose.bones[single_bone].constraints["Child Of"].target = bpy.data.objects[body_part]


add_bonesToRigdoll("pelvis", "Torso")
add_bonesToRigdoll("spine_01", "Torso")
add_bonesToRigdoll("spine_02", "Torso")
add_bonesToRigdoll("spine_03", "Torso")
add_bonesToRigdoll("clavicle_l", "Torso")
add_bonesToRigdoll("clavicle_r", "Torso")
add_bonesToRigdoll("neck_01", "Torso")
add_bonesToRigdoll("head", "Torso")
add_bonesToRigdoll("upperarm_l", "Upperarm_L")
add_bonesToRigdoll("lowerarm_l", "Lowerarm_L")
add_bonesToRigdoll("hand_l", "Hand_L")
add_bonesToRigdoll("index_01_l", "Hand_L")
add_bonesToRigdoll("index_02_l", "Hand_L")
add_bonesToRigdoll("index_03_l", "Hand_L")
add_bonesToRigdoll("middle_01_l", "Hand_L")
add_bonesToRigdoll("middle_02_l", "Hand_L")
add_bonesToRigdoll("middle_03_l", "Hand_L")
add_bonesToRigdoll("pinky_01_l", "Hand_L")
add_bonesToRigdoll("pinky_02_l", "Hand_L")
add_bonesToRigdoll("pinky_03_l", "Hand_L")
add_bonesToRigdoll("ring_01_l", "Hand_L")
add_bonesToRigdoll("ring_02_l", "Hand_L")
add_bonesToRigdoll("ring_03_l", "Hand_L")
add_bonesToRigdoll("thumb_01_l", "Hand_L")
add_bonesToRigdoll("thumb_02_l", "Hand_L")
add_bonesToRigdoll("thumb_03_l", "Hand_L")
add_bonesToRigdoll("thigh_l", "Thigh_L")
add_bonesToRigdoll("calf_l", "Calf_L")
add_bonesToRigdoll("foot_l", "Foot_L")
add_bonesToRigdoll("ball_l", "Foot_L")
add_bonesToRigdoll("upperarm_r", "Upperarm_R")
add_bonesToRigdoll("lowerarm_r", "Lowerarm_R")
add_bonesToRigdoll("hand_r", "Hand_R")
add_bonesToRigdoll("index_01_r", "Hand_R")
add_bonesToRigdoll("index_02_r", "Hand_R")
add_bonesToRigdoll("index_03_r", "Hand_R")
add_bonesToRigdoll("middle_01_r", "Hand_R")
add_bonesToRigdoll("middle_02_r", "Hand_R")
add_bonesToRigdoll("middle_03_r", "Hand_R")
add_bonesToRigdoll("pinky_01_r", "Hand_R")
add_bonesToRigdoll("pinky_02_r", "Hand_R")
add_bonesToRigdoll("pinky_03_r", "Hand_R")
add_bonesToRigdoll("ring_01_r", "Hand_R")
add_bonesToRigdoll("ring_02_r", "Hand_R")
add_bonesToRigdoll("ring_03_r", "Hand_R")
add_bonesToRigdoll("thumb_01_r", "Hand_R")
add_bonesToRigdoll("thumb_02_r", "Hand_R")
add_bonesToRigdoll("thumb_03_r", "Hand_R")
add_bonesToRigdoll("thigh_r", "Thigh_R")
add_bonesToRigdoll("calf_r", "Calf_R")
add_bonesToRigdoll("foot_r", "Foot_R")
add_bonesToRigdoll("ball_r", "Foot_R") 

# Switch to "object mode"
bpy.ops.object.mode_set(mode='OBJECT')


######################################
#### 06 - Rendering                 ##
######################################

# Camera tracking
# Deselect everything
bpy.ops.object.select_all(action="DESELECT")
# Select "Human.rig"
bpy.data.objects["Camera"].select_set(True)
bpy.context.view_layer.objects.active = bpy.data.objects["Camera"]
bpy.ops.object.constraint_add(type='COPY_LOCATION')
bpy.context.object.constraints["Copy Location"].name = "Copy Location: Torso"
bpy.context.object.constraints["Copy Location: Torso"].use_z = False
bpy.context.object.constraints["Copy Location: Torso"].use_y = False
bpy.context.object.constraints["Copy Location: Torso"].target = bpy.data.objects["Torso"]

bpy.ops.object.constraint_add(type='COPY_LOCATION')
bpy.context.object.constraints["Copy Location"].name = "Copy Location: Thigh_R"
bpy.context.object.constraints["Copy Location: Thigh_R"].use_z = False
bpy.context.object.constraints["Copy Location: Thigh_R"].use_x = False
bpy.context.object.constraints["Copy Location: Thigh_R"].target = bpy.data.objects["Thigh_R"]

# Hide Body Parts
bpy.data.collections["BodyParts"].hide_render = True


######################################
#### 07 - Animation                 ##
######################################

for i in range(251):
    bpy.context.scene.frame_set(i)

