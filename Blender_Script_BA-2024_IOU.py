import bpy
import sys
import argparse
import bmesh
import csv
import os
from mathutils import Vector


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

    # overrides superclass
    def parse_args(self):
         return super().parse_args(args=self._get_argv_after_doubledash())

parser = ArgumentParserForBlender()


parser.add_argument("-id", type=int, default=0,
                    help="ID-Number")
parser.add_argument("-v1", "--value1", type=float, default=0.9,
                    help="float for value1")
parser.add_argument("-v2", "--value2", type=float, default=0.9,
                    help="float for value2")
parser.add_argument("-v3", "--value3", type=float, default=1.0,
                    help="float for value3")
parser.add_argument("-v4", "--value4", type=float, default=1.0,
                    help="float for value4")
parser.add_argument("-v5", "--value5", type=float, default=0.9,
                    help="float for value5")
parser.add_argument("-v6", "--value6", type=float, default=1.0,
                    help="float for value6")
                    
parser.add_argument("-p1", "--prediction1", type=float, default=0.0,
                    help="float for prediction1")
parser.add_argument("-p2", "--prediction2", type=float, default=1.0,
                    help="float for prediction2")
parser.add_argument("-p3", "--prediction3", type=float, default=0.85,
                    help="float for prediction3")
parser.add_argument("-p4", "--prediction4", type=float, default=0.85,
                    help="float for prediction4")
parser.add_argument("-p5", "--prediction5", type=float, default=1.0,
                    help="float for prediction5")
parser.add_argument("-p6", "--prediction6", type=float, default=1.0,
                    help="float for prediction6")
                    
args = parser.parse_args()
ID = args.id
VALUE1 = args.value1
VALUE2 = args.value2
VALUE3 = args.value3
VALUE4 = args.value4
VALUE5 = args.value5
VALUE6 = args.value6
PREDICTION1 = args.prediction1
PREDICTION2 = args.prediction2
PREDICTION3 = args.prediction3
PREDICTION4 = args.prediction4
PREDICTION5 = args.prediction5
PREDICTION6 = args.prediction6

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
### 04 - Create Human              ###
######################################

#def generateHuman(oldModelName, newModelName, humanName, parameter1,parameter2,parameter3,parameter4,parameter5,parameter6):
def generateHuman(newModelName, humanName, parameter1,parameter2,parameter3,parameter4,parameter5,parameter6):

    # Create Human
    bpy.ops.mpfb.create_human()

    ######################################
    #### 04-1 - Phenotype: Macro Details #
    ######################################
    bpy.context.scene.mpfb_macropanel_gender = parameter1 #z.B. VALUE1
    bpy.context.scene.mpfb_macropanel_age = parameter2 #z.B. VALUE2
    bpy.context.scene.mpfb_macropanel_muscle = parameter3 #z.B. VALUE3
    bpy.context.scene.mpfb_macropanel_weight = parameter4 #z.B. VALUE4
    bpy.context.scene.mpfb_macropanel_height = parameter5 #z.B. VALUE5
    bpy.context.scene.mpfb_macropanel_proportions = parameter6 #z.B. VALUE6
    
    ######################################
    #### 04-3 - Apply Shape Keys     ###
    ######################################
    # Define object called "human" and link to "Human"-Model in Scene
    human = bpy.data.objects[humanName]
    # Deselect everything
    bpy.context.view_layer.objects.active = None
    # Select Human-Model
    bpy.context.view_layer.objects.active = human
    # Apply all Shape Keys
    bpy.ops.object.shape_key_remove(all=True, apply_mix=True)

    ##################################################
    #### 04-4 - Delete unnecessary Vertex Groups ###
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

    ############################################
    #### 04-5 - Change the Name of the Model and delete existing modifiers ###
    ############################################
    bpy.data.objects[humanName].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects[humanName]
    # Delete existing modifiers hide helpers"
    bpy.ops.object.modifier_remove(modifier="Hide helpers")
    # Change name of the Model
    bpy.context.object.name = newModelName
    # Deselect everything
    bpy.ops.object.select_all(action="DESELECT")
    
generateHuman("Input", "Human",  VALUE1,VALUE2,VALUE3,VALUE4,VALUE5,VALUE6)
generateHuman("Output", "Human", PREDICTION1,PREDICTION2,PREDICTION3,PREDICTION4,PREDICTION5,PREDICTION6)

####################################################
### 05 - Intersection over Union (Jaccard Index) ###
####################################################

    #################################
    #### 05-1 - Create Duplicates ###
    #################################
    
def createDuplicates(modelToCopy, oldObjectName, newObjectName):

    bpy.data.objects[modelToCopy].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects[modelToCopy]
    bpy.ops.object.duplicate() 
    bpy.data.objects[oldObjectName].name = newObjectName
    bpy.ops.object.select_all(action="DESELECT")

createDuplicates("Input", "Input.001", "InputUnion")
createDuplicates("Input", "Input.001", "InputIntersection")

    ####################################
    #### 05-2 - Add Boolean Modifier ###
    ####################################

def addboolean(modelForOperation, secondModel, operation):
    
    bpy.data.objects[modelForOperation].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects[modelForOperation]
    bpy.ops.object.modifier_add(type='BOOLEAN')
    bpy.context.object.modifiers["Boolean"].operation = operation
    bpy.context.object.modifiers["Boolean"].object = bpy.data.objects[secondModel]
    bpy.ops.object.modifier_apply(modifier="Boolean")
    bpy.ops.object.select_all(action="DESELECT")

addboolean("InputIntersection", "Output", 'INTERSECT')
addboolean("InputUnion", "Output", 'UNION')

    ##############################
    #### 05-3 - Measure Volume ###
    ##############################

'''    
# Alternative Option to calculate the Volume
# Measurements deviate slightly

def measureVolume2(modelName):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects[modelName].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects[modelName]   
    bpy.ops.rigidbody.object_add()
    bpy.ops.rigidbody.mass_calculate(material='Custom', density=1) # density in kg/m^3
    return bpy.context.view_layer.objects.active.rigid_body.mass

measureVolume2("InputUnion")
measureVolume2("InputIntersection")    
''' 


def measureVolume(modelName):
    bpy.data.objects[modelName].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects[modelName]
    obj = bpy.context.active_object
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.transform(obj.matrix_world)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    volume = 0
    for f in bm.faces:
        v1 = f.verts[0].co
        v2 = f.verts[1].co
        v3 = f.verts[2].co
        volume += v1.dot(v2.cross(v3)) / 6
    bm.free()
    bpy.ops.object.select_all(action="DESELECT")
    return volume

measureUnion = measureVolume("InputUnion")
measureIntersection = measureVolume("InputIntersection")

    #############################
    #### 05-4 - Calculate IOU ###
    #############################
    # Je näher der Jaccard-Koeffizient an 1 liegt, desto größer ist die 
    # Ähnlichkeit der Mengen. Der minimale Wert des Jaccard-Koeffizienten ist 0
    
IOU = (measureIntersection / measureUnion)

    ##########################################
    #### 05-5 - Save and Export IOU-Result ###
    ##########################################

# Get the path to the current Blender file
blend_file_path = bpy.data.filepath

# Get the directory of the Blender file
blend_dir = os.path.dirname(blend_file_path)

# Construct the absolute path to the "data.csv" file
data_file_path = os.path.join(blend_dir, "data.csv")


def append_to_csv(filepath, IOU, VALUE1, VALUE2, VALUE3, VALUE4, VALUE5, VALUE6, PREDICTION1, PREDICTION2, PREDICTION3, PREDICTION4, PREDICTION5, PREDICTION6):
    # Check if the file exists
    file_exists = os.path.isfile(filepath)

    # Open the file in append mode
    with open(filepath, 'a', newline='') as file:
        writer = csv.writer(file)

        # If the file doesn't exist, write the header
        if not file_exists:
            writer.writerow(['IOU', 'VALUE1', 'VALUE2', 'VALUE3', 'VALUE4', 'VALUE5', 'VALUE6', 'PREDICTION1', 'PREDICTION2', 'PREDICTION3', 'PREDICTION4', 'PREDICTION5', 'PREDICTION6'])

        # Write the values to the file
        writer.writerow([IOU, VALUE1, VALUE2, VALUE3, VALUE4, VALUE5, VALUE6, PREDICTION1, PREDICTION2, PREDICTION3, PREDICTION4, PREDICTION5, PREDICTION6])
        
append_to_csv(data_file_path,IOU, VALUE1, VALUE2, VALUE3, VALUE4, VALUE5, VALUE6, PREDICTION1, PREDICTION2, PREDICTION3, PREDICTION4, PREDICTION5, PREDICTION6)