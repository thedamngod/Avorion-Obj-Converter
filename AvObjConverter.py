import os
import xml.etree.ElementTree

import itertools
from sys import argv
from typing import List

import numpy as np
import sys

import time

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

Vertex = List[float]
Face = List[int]
Normal = List[float]
UV = List[float]


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


class AvToObjConverter:
    def __init__(self, blocks=None):
        self.normal_count = 0
        self.vc = 0
        self.blocks = blocks

    def convertToObj(self, inputFile, outputFile):
        if self.blocks is None:
            self.blocks = readSourceFile(inputFile)
        try:
            os.makedirs(os.path.dirname(outputFile), exist_ok=True)
            if os.path.exists(outputFile) and os.path.getsize(outputFile) > 0:
                os.remove(outputFile)
                print(outputFile)
            start_time = time.time()
            file = open(outputFile, 'a')
            file.write("mtllib Avorion_Materials.mtl\n")
            for block in self.blocks:
                file.write(self.convertBlockToObj(block))
            file.close()
            print("Conversion process took {} seconds".format(time.time() - start_time))
            print("Converted {} blocks".format(len(self.blocks)))
        except FileNotFoundError:
            print('Could not open output file')

    def convertBlockToObj(self, block):
        out_string = ""
        name = "Block"
        print("Converting block with index {}, type = {}, look = {} and up = {}".format(
            block.index, block.type, block.look, block.up))
        faces = None
        v_indices = []
        f_indices = []

        if block.type in [10]:  # Hangar Block -- TODO: Replace Placeholder for hangar blocks
            v_indices = [1, 2, 3, 4, 5, 6, 7, 8]
            f_indices = [0, 1, 2, 3, 4, 5]

        elif block.type in range(1, 16) or block.type in range(50, 56) \
                or block.type in [150, 170, 190, 510]:  # Cuboid blocks
            v_indices = [1, 2, 3, 4, 5, 6, 7, 8]
            f_indices = [0, 1, 2, 3, 4, 5]

        elif block.type in [100, 104, 151, 171, 185, 191, 511]:  # Edge with 6 vertices
            name = "Edge"
            v_indices = [1, 2, 5, 6, 7, 8]
            f_indices = [2, 4, 6, 7, 16]

        elif block.type in [103, 107, 154, 174, 188, 194, 514]:  # Corner with 5 vertices
            name = "5 Vertex Corner"
            v_indices = [1, 2, 5, 6, 7]
            f_indices = [2, 7, 10, 13, 14]

        elif block.type in [101, 105, 152, 172, 186, 192, 512]:  # Corner with 4 vertices
            name = "4 Vertex Corner"
            v_indices = [1, 5, 6, 7]
            f_indices = [7, 8, 10, 15]

        elif block.type in [102, 106, 153, 173, 187, 193, 513]:  # Edge with 7 vertices
            name = "7 Vertex Edge"
            v_indices = [1, 2, 3, 5, 6, 7, 8]
            f_indices = [1, 2, 4, 6, 9, 11, 12]
            faces = [[1, 4, 6, 3],  # Left
                     [4, 5, 7, 6],  # Back
                     [1, 2, 5, 4],  # Bottom
                     [3, 6, 7],  # Top
                     [1, 2, 3],  # Front
                     [2, 3, 7],  # Diagonal Missing corner
                     [2, 5, 7]]  # Right

        vertices, genfaces, vertex_normals = self.generate(v_indices, f_indices)
        vertices, vertex_normals = transformBlock(vertices, vertex_normals, block.look, block.up, block)
        if faces is None:
            faces = genfaces
        out_string += self.generateObjectString(vertices, faces, vertex_normals, "{} {}".format(name, block.index))
        return out_string

    def generate(self, v_indices: List[int], f_indices: List[int]):

        vertices = [
            [1, -1, -1],  # 1 Bottom left front
            [-1, -1, -1],  # 2 Bottom right front
            [1, 1, -1],  # 3 Top left front
            [-1, 1, -1],  # 4 Top right front
            [1, -1, 1],  # 5 Bottom left back
            [-1, -1, 1],  # 6 Bottom right back
            [1, 1, 1],  # 7 Top left back
            [-1, 1, 1]]  # 8 Top right back

        faces = [
            [2, 6, 8, 4],  # 0 Right
            [1, 5, 7, 3],  # 1 Left -
            [1, 2, 6, 5],  # 2 Bottom -
            # [3, 4, 8, 7],  # 3 Top
            [7, 8, 4, 3],  # 3 Top
            [5, 6, 8, 7],  # 4 Back -
            [1, 2, 4, 3],  # 5 Front
            [2, 6, 8],  # 6 Diagonal Right -
            [1, 5, 7],  # 7 Diagonal Left
            [1, 6, 5],  # 8 Diagonal Bottom
            [3, 7, 8],  # 9 Diagonal Top -
            [5, 6, 7],  # 10 Diagonal Back
            [1, 2, 3],  # 11 Diagonal Front -
            [2, 3, 8],  # 12 Triangle 7 Verts -
            [1, 2, 7],  # 13 Triangle Front Bottom to Back Top Left
            [2, 6, 7],  # 14 Triangle Right Bottom to Back Top Left
            [1, 6, 7],  # 15 Triangle 4 Verts
            [1, 2, 8, 7]  # 16 Slope for Edge Block
        ]

        vertex_normals = [
            [-1, 0, 0],  # 0 Right
            [1, 0, 0],  # 1 Left
            [0, -1, 0],  # 2 Top
            [0, 1, 0],  # 3 Bottom
            [0, 0, 1],  # 4 Back
            [0, 0, -1],  # 5 Front
            [-1, 0, 0],  # 6 Diagonal Right
            [1, 0, 0],  # 7 Diagonal Left
            [0, -1, 0],  # 8 Diagonal Bottom
            [0, 1, 0],  # 9 Diagonal Top
            [0, 0, 1],  # 10 Diagonal Back
            [0, 0, -1],  # 11 Diagonal Front
            [-1, 1, -1],  # 12 Diagonal missing corner 4
            [0, 1, -1],  # 13 Triangle Front Bottom to Back Top Left
            [-1, 1, 0],  # 14 Triangle Right Bottom to Back Top Left
            [-1, 1, -1],  # 15 Triangle 4 Verts
            [0, 1, -1]  # 16 Slope
        ]

        out_vertices = [vertices[index - 1] for index in range(1, 9) if index in v_indices]  # Select Vertices
        out_faces = [faces[index] for index in range(len(faces)) if index in f_indices]  # Select Faces
        out_normals = []
        for face in out_faces:  # Select Normals
            index = faces.index(face)
            # out_normals.append(normalize(vertex_normals[index]))
            out_normals.append(vertex_normals[index])

        # Map the vertex indices to the correct number for all faces
        for i in range(len(v_indices)):
            if v_indices[i] != i + 1:
                mapping = lambda x: i + 1 if x == v_indices[i] else x
                for index in range(len(out_faces)):
                    out_faces[index] = list(map(mapping, out_faces[index]))

        # Construct UV Maps -- TODO

        return out_vertices, out_faces, out_normals

    def generateObjectString(self, vertices: List[Vertex], faces: List[Face],
                             vertex_normals: List[Normal], name: str, uvs: List[UV] = None, smoothing: bool = False):
        out_string = "o {}\n".format(name)
        smoothing = "on" if smoothing else "off"
        for vertex in vertices:  # Add the strings for the vertices
            out_string += "v {} {} {}\n".format(vertex[0], vertex[1], vertex[2])

        if uvs is not None:  # Add UV coordinates to the object
            for uv in uvs:
                out_string += "vt "
                for coord in uv:
                    out_string += "{} ".format(coord)
                out_string += "\n"

        if vertex_normals is not None:  # Add the Strings for the vertex normals
            for normal in vertex_normals:
                out_string += "vn "
                out_string += "{} {} {}".format(normal[0], normal[1], normal[2])
                out_string += "\n"

        out_string += "usemtl Standard\n"  # Add strign for the used material
        out_string += "s {}\n".format(smoothing)  # Add string for the enabling/disabling of smoothing
        if faces is not None:  # Add the strings for the faces of the object
            for face in faces:
                out_string += "f "
                for vertex in face:
                    if vertex_normals is not None:
                        normal_index = faces.index(face) + 1
                        xstr = lambda s: str(s) if s is not None else ''
                        out_string += "{}/{}/{} ".format(self.vc + vertex, xstr(''), self.normal_count + normal_index)
                    else:
                        out_string += "{} ".format(self.vc + vertex)
                out_string += "\n"
        self.vc += len(vertices)
        if vertex_normals is not None:
            self.normal_count += len(vertex_normals)
        return out_string


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
    vector = [entry / length for entry in vector]
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
        vector.append(1)

        temp_result = np.matrix(vector).transpose()
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
            [np.round(temp_result[0][0], DECIMALS), np.round(temp_result[1][0], DECIMALS),
             np.round(temp_result[2][0], DECIMALS)])
        del (vector[-1])
    return return_vectors


def transformBlock(verts: List[Vertex], vertex_normals: List[Normal] = None, look: int = 5, up: int = 3, block=None):
    """
    Transforms the given vertices

    :param verts: Vertices which will be transformed
    :param vertex_normals: Normal vectors which will be transformed
    :param look: Index of the look vector
    :param up: Index of the up vector
    :param block: Block object, which is the base of the vertices and normals
    :return: The transformed vertices and normal vectors
    """
    vertices = verts.copy()
    normals = vertex_normals.copy() if vertex_normals is not None else []
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

    return output_vertices, output_normals


def transformVector(vector: Vertex, scales: List[float], translations: List[float]):
    """
    Scale the given vertex using the values from scales and afterwards translate the vertex using the translations vector.

    :param vector: Vertex to transform
    :param scales: Specifies the scale factors for the x, y, and z axis
    :param translations: Specifies the translation amount in x, y, and z direction
    :return: The transformed vector as Vertex
    """
    temp_vector = vector.copy()
    temp_vector.append(1)
    temp_result = np.matrix(temp_vector).transpose()

    temp_result = scale(temp_result, scales[0], scales[1], scales[2])
    temp_result = translate(temp_result, translations[0], translations[1], translations[2])

    temp_result = temp_result.tolist()
    result = [np.round(temp_result[0][0], DECIMALS), np.round(temp_result[1][0], DECIMALS),
              np.round(temp_result[2][0], DECIMALS)]
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
            '{} {} {} {}; {} {} {} {}; {} {} {} {}; {} {} {} {}'.format(1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0,
                                                                        1))
    if y_degrees != 0:
        c, s = np.cos(np.radians(y_degrees)), np.sin(np.radians(y_degrees))
        yMatrix = np.matrix(
            '{} {} {} {}; {} {} {} {}; {} {} {} {}; {} {} {} {}'.format(c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0,
                                                                        1))
    if z_degrees != 0:
        c, s = np.cos(np.radians(z_degrees)), np.sin(np.radians(z_degrees))
        zMatrix = np.matrix(
            '{} {} {} {}; {} {} {} {}; {} {} {} {}; {} {} {} {}'.format(c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                                                        1))
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
