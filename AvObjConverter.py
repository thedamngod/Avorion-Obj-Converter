import os
import xml.etree.ElementTree

from sys import argv
from typing import List

import numpy as np
import sys

import time

from waveobjparser import Face
from waveobjparser import Normal
from waveobjparser import ObjParser
from waveobjparser import ObjWriter
from waveobjparser import Scene
from waveobjparser import Text_Coord
from waveobjparser import Vertex
from waveobjparser import WaveObject

DECIMALS = 8

COLOR = 'color'
UP = 'up'
LOOK = 'look'
MATERIAL = 'material'
INDEX = 'index'
UPPER_Z = 'upperZ'
LOWER_Z = 'lowerZ'
UPPER_Y = 'upperY'
LOWER_Y = 'lowerY'
UPPER_X = 'upperX'
LOWER_X = 'lowerX'
BLOCK = 'block'
ITEM = 'item'


class Block:
    def __init__(self, index: int, x_min: float, x_max: float, y_min: float, y_max: float, z_min: float, z_max: float,
                 type: int, material: int, look: int, up: int, color: str):
        self.material = float(material)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.index = int(index)
        self.look = int(look)
        self.up = int(up)
        self.color = color
        self.x_min = float(x_min)
        self.type = int(type)


def generateObject(vertices: List[Vertex], faces: List[Face],
                   vertex_normals: List[Normal], name: str, uvs: List[Text_Coord] = None, smoothing: bool = False):
    obj = WaveObject()
    obj.setName(name)
    obj.setSmoothing(smoothing)
    obj.addMaterial('Standard')

    for vertex in vertices:  # Add the strings for the vertices
        obj.addVertex(vertex)

    if uvs is not None:  # Add UV coordinates to the object
        for uv in uvs:
            obj.addTextureCoord(uv)

    if vertex_normals is not None:  # Add the Strings for the vertex normals
        for normal in vertex_normals:
            obj.addNormal(normal)

    if faces is not None:  # Add the strings for the faces of the object
        for face in faces:
            obj.addFace(face)
    return obj


def convertBlockToObj(block):
    print("Converting block with index {}, type = {}, look = {} and up = {}".format(
        block.index, block.type, block.look, block.up))

    if block.type in [10]:  # Hangar Block -- TODO: Replace Placeholder for hangar blocks
        file = "Cube.obj"

    elif block.type in [100, 104, 151, 171, 185, 191, 511]:  # Edge with 6 vertices
        file = "Edge.obj"

    elif block.type in [103, 107, 154, 174, 188, 194, 514]:  # Corner with 5 vertices
        file = "5 Vertex Corner"

    elif block.type in [101, 105, 152, 172, 186, 192, 512]:  # Corner with 4 vertices
        file = "4 Vertex Corner.obj"

    elif block.type in [102, 106, 153, 173, 187, 193, 513]:  # Edge with 7 vertices
        file = "7 vertex corner.obj"
    else:
        file = "Cube.obj"

    parser = ObjParser("./objects/{}".format(file))
    scene = parser.parseFile()
    obj = scene.objects[0]
    return transformObject(obj, block.look, block.up, block)


class AvToObjConverter:
    def __init__(self, blocks=None):
        self.blocks = blocks
        self.scene = Scene()

    def convertToObj(self, inputFile, outputFile):
        writer = ObjWriter(outputFile)
        self.scene.setMtllib("Avorion_Materials")

        if self.blocks is None:
            self.blocks = readSourceFile(inputFile)

        start_time = time.time()
        for block in self.blocks:
            self.scene.addObject(convertBlockToObj(block))
        writer.writeScene(self.scene)
        print("Conversion process took {} seconds".format(time.time() - start_time))
        print("Converted {} blocks".format(len(self.blocks)))


def getSizes(block):
    """
    Calculates the maximum length of the block in the 3 directions.

    :param block: The block which should be measured
    :return: The measurements as a list [x,y,z]
    """
    sizes = [0, 0, 0]
    if block is not None:
        sizes[0] = (block.x_max - block.x_min) / 2
        sizes[1] = (block.y_max - block.y_min) / 2
        sizes[2] = (block.z_max - block.z_min) / 2
    return sizes


def normalize(vector: Vertex):
    length = 0
    for element in vector:
        length += element ** 2
    length = np.math.sqrt(length)
    vector = tuple(entry / length for entry in vector)
    return vector


def readSourceFile(file):
    """
    Read the source file and return representations of the blocks as a list.

    :param file: Path of the xml file which should be processed
    :return: A list of blocks extracted from the file
    """
    try:
        tree = xml.etree.ElementTree.parse(file).getroot()
    except FileNotFoundError:
        print("The input file was not found.\n Please make sure the file exists")
        sys.exit()
    blocks = []
    for item in tree.findall(ITEM):
        for block in item.findall(BLOCK):
            index = item.get(INDEX)
            x_min = block.get(LOWER_X)
            x_max = block.get(UPPER_X)
            y_min = block.get(LOWER_Y)
            y_max = block.get(UPPER_Y)
            z_min = block.get(LOWER_Z)
            z_max = block.get(UPPER_Z)
            type = block.get(INDEX)
            material = block.get(MATERIAL)
            look = block.get(LOOK)
            up = block.get(UP)
            color = block.get(COLOR)
            blocks.append(Block(index, x_min, x_max, y_min, y_max, z_min, z_max, type, material, look, up, color))
    return blocks


def rotateVectors(vertices: List[Vertex], look: int = 5, up: int = 3):
    """
    Rotates the given vectors to correspond with the given values for look and up.

    :param vertices: List of vectors to rotate
    :param look: Look value to orient the vectors to (Determines the face to the back)
    :param up: Up value to orient the vecturs to (Determines the face facing up)
    :return: The rotated vectors
    """
    return_vectors = []
    for vector in vertices:
        temp_result = list(vector)
        temp_result.append(1)
        temp_result = np.matrix(temp_result).transpose()
        if look == 5:  #
            if up == 0:
                temp_result = rotate(temp_result, z_degrees=90)
            elif up == 2:
                temp_result = rotate(temp_result, z_degrees=180)
            elif up == 1:
                temp_result = rotate(temp_result, z_degrees=270)

        elif look == 0:  #
            if up == 2:
                temp_result = rotate(temp_result, y_degrees=90)
                temp_result = rotate(temp_result, x_degrees=180)
            elif up == 3:
                temp_result = rotate(temp_result, y_degrees=90)
            elif up == 4:
                temp_result = rotate(temp_result, x_degrees=270)
                temp_result = rotate(temp_result, z_degrees=90)
            elif up == 5:
                temp_result = rotate(temp_result, x_degrees=90)
                temp_result = rotate(temp_result, z_degrees=270)

        elif look == 1:  #
            temp_result = rotate(temp_result, y_degrees=270)
            if up == 2:
                temp_result = rotate(temp_result, x_degrees=180)
            elif up == 4:
                temp_result = rotate(temp_result, x_degrees=270)
            elif up == 5:
                temp_result = rotate(temp_result, x_degrees=90)

        elif look == 2:  #
            temp_result = rotate(temp_result, x_degrees=90)
            if up == 0:
                temp_result = rotate(temp_result, y_degrees=90)
            elif up == 1:
                temp_result = rotate(temp_result, y_degrees=270)
            elif up == 4:
                temp_result = rotate(temp_result, y_degrees=180)

        elif look == 3:  #
            temp_result = rotate(temp_result, x_degrees=270)
            if up == 0:
                temp_result = rotate(temp_result, y_degrees=270)
            elif up == 1:
                temp_result = rotate(temp_result, y_degrees=90)
            elif up == 5:
                temp_result = rotate(temp_result, y_degrees=180)

        elif look == 4:  #
            temp_result = rotate(temp_result, y_degrees=180)
            if up == 0:
                temp_result = rotate(temp_result, z_degrees=90)
            elif up == 1:
                temp_result = rotate(temp_result, z_degrees=270)
            elif up == 2:
                temp_result = rotate(temp_result, z_degrees=180)
        temp_result = temp_result.tolist()
        return_vectors.append(
            (np.round(temp_result[0][0], DECIMALS), np.round(temp_result[1][0], DECIMALS),
             np.round(temp_result[2][0], DECIMALS)))
        # del (vector[-1])
    return return_vectors


def transformObject(obj: WaveObject, look: int = 5, up: int = 3, block=None):
    """
    Transforms the given object

    :param obj: Wavefront Object which should be transformed
    :param look: Index of the look vector
    :param up: Index of the up vector
    :param block: Block object, which is the base of the vertices and normals
    :return: The transformed vertices and normal vectors
    """
    vertices = obj.vertices.copy()
    normals = obj.vertex_normals.copy()
    translation_vector = [0, 0, 0]
    scale_amounts = [1, 1, 1]

    if block is not None:
        translation_vector[0] = (block.x_max + block.x_min) / 2
        translation_vector[1] = (block.y_max + block.y_min) / 2
        translation_vector[2] = (block.z_max + block.z_min) / 2
        scale_amounts = getSizes(block)

    vertices = rotateVectors(vertices, look, up)
    normals = rotateVectors(normals, look, up)

    output_vertices = [transformVector(vertex, scale_amounts, translation_vector) for vertex in vertices]

    scale_amounts = [1 / amount for amount in scale_amounts]
    output_normals = [transformVector(normal, scale_amounts, [0, 0, 0]) for normal in normals]

    out_obj = generateObject(output_vertices, obj.faces, output_normals, obj.name, obj.texture_coords, obj.smoothing)

    return out_obj


def transformVector(vector: Vertex, scales: List[float], translations: List[float]):
    """
    Scale the given vertex using the values from scales and afterwards translate the vertex using the translations vector.

    :param vector: Vertex to transform
    :param scales: Specifies the scale factors for the x, y, and z axis
    :param translations: Specifies the translation amount in x, y, and z direction
    :return: The transformed vector as Vertex
    """

    temp_vector = list(vector)
    temp_vector.append(1)
    temp_result = np.matrix(temp_vector).transpose()

    temp_result = scale(temp_result, scales[0], scales[1], scales[2])
    temp_result = translate(temp_result, translations[0], translations[1], translations[2])

    temp_result = temp_result.tolist()
    result = (np.round(temp_result[0][0], DECIMALS), np.round(temp_result[1][0], DECIMALS),
              np.round(temp_result[2][0], DECIMALS))
    # del (vector[-1])
    return result


def translate(vector: np.ndarray, x: float, y: float, z: float):
    """
    Translates the vector by the given amounts in x, y, and z direction

    :param vector: The vector to translate
    :param x: Translation distance in x direction
    :param y: Translation distance in y direction
    :param z: Translation distance in z direction
    :return: The translated vector
    """
    out_vector = _getTranslationMatrix(x, y, z).dot(vector)
    return out_vector


def _getTranslationMatrix(x: float, y: float, z: float):
    return np.matrix([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


def scale(vector: np.ndarray, x_amount: float = 1, y_amount: float = 1, z_amount: float = 1, dim=4):
    """
    Scales the given vector in the given amounts

    :param vector: The vector which will be scaled
    :param x_amount: Scale in the x direction
    :param y_amount: Scale in the y direction
    :param z_amount: Scale in the z direction
    :param dim: dimension of the vector
    :return: The scaled vector
    """
    matrix = np.identity(dim)
    matrix[0][0] = x_amount
    matrix[1][1] = y_amount
    matrix[2][2] = z_amount
    result = matrix.dot(vector)
    return result


def rotate(vector: np.ndarray, x_degrees: float = 0, y_degrees: float = 0, z_degrees: float = 0):
    """
    Rotates the given vector by the given amounts in the order z -> y -> x

    :param vector: Vector to rotate
    :param x_degrees: Rotation degree around the x-axis
    :param y_degrees: Degrees around the y-axis
    :param z_degrees: Degrees around the z-axis
    :return: The rotated vector
    """
    xMatrix, yMatrix, zMatrix = np.identity(4), np.identity(4), np.identity(4)
    if x_degrees != 0:
        c, s = np.cos(np.radians(x_degrees)), np.sin(np.radians(x_degrees))
        xMatrix = np.matrix(
            '{} {} {} {}; {} {} {} {}; {} {} {} {}; {} {} {} {}'.format(1, 0, 0, 0,
                                                                        0, c, -s, 0,
                                                                        0, s, c, 0,
                                                                        0, 0, 0, 1))
    if y_degrees != 0:
        c, s = np.cos(np.radians(y_degrees)), np.sin(np.radians(y_degrees))
        yMatrix = np.matrix(
            '{} {} {} {}; {} {} {} {}; {} {} {} {}; {} {} {} {}'.format(c, 0, -s, 0,
                                                                        0, 1, 0, 0,
                                                                        s, 0, c, 0,
                                                                        0, 0, 0, 1))
    if z_degrees != 0:
        c, s = np.cos(np.radians(z_degrees)), np.sin(np.radians(z_degrees))
        zMatrix = np.matrix(
            '{} {} {} {}; {} {} {} {}; {} {} {} {}; {} {} {} {}'.format(c, -s, 0, 0,
                                                                        s, c, 0, 0,
                                                                        0, 0, 1, 0,
                                                                        0, 0, 0, 1))
    result = zMatrix.dot(vector)
    result = yMatrix.dot(result)
    result = xMatrix.dot(result)
    return result


if __name__ == "__main__":
    path = os.path.join(os.getenv('APPDATA'), 'Avorion', 'ships')
    if len(argv) < 2:
        print("Please enter the name of the file you want to convert (without the file ending):")
        ship_name = input("Name: ")
    else:
        ship_name = argv[1]
        if len(argv) == 3:
            path = argv[2]

    converter = AvToObjConverter()
    converter.convertToObj(os.path.join(path, ship_name + '.xml'), './out/' + ship_name + '.obj')
    # parser = ObjParser("./objects/Cube.obj")
    # scene = parser.parseFile()
    # print(scene.objects[0])
