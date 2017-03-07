import os
import xml.etree.ElementTree

import itertools
from sys import argv

import numpy as np
import sys

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
    def __init__(self, index, x_min, x_max, y_min, y_max, z_min, z_max, type, material, look, up, color):
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
            while os.path.exists(outputFile) and os.path.getsize(outputFile) > 0:
                # outputFile = outputFile[:-4] + '1' + outputFile[-4:]
                os.remove(outputFile)
                print(outputFile)

            file = open(outputFile, 'a')
            file.write("mtllib Avorion_Materials.mtl\n")
            for block in self.blocks:
                file.write(self.convertBlockToObj(block))

            file.close()
        except FileNotFoundError:
            print('Could not open output file')

    def convertBlockToObj(self, block):
        out_string = ""
        name = "Block"
        print("Converting block with index {}, type = {}, look = {} and up = {}".format(block.index, block.type,
                                                                                        block.look, block.up))
        faces = None
        # vertex_normals = None
        v_indices = []
        f_indices = []

        if block.type in range(1, 16) or block.type in range(50, 56) \
                or block.type in [150, 170, 190, 510]:  # Non-diagonal blocks
            v_indices = [1, 2, 3, 4, 5, 6, 7, 8]
            f_indices = [0, 1, 2, 3, 4, 5]

        elif block.type in [100, 104, 151, 171, 185, 191, 511]:  # Edge with 6 vertices
            name = "Edge"
            v_indices = [1, 2, 5, 6, 7, 8]
            f_indices = [2, 4, 6, 7, 16]

        elif block.type in [103, 107, 154, 174, 188, 194, 514]:  # Edge with 5 vertices
            name = "5 Vertex Edge"
            v_indices = [1, 2, 5, 6, 7]
            f_indices = [2, 7, 10, 13, 14]

        elif block.type in [101, 105, 152, 172, 186, 192, 512]:  # Edge with 4 vertices
            name = "4 Vertex Edge"
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
        # vertex_normals = rotateVectors(vertex_normals, block.look, block.up)
        if faces is None:
            faces = genfaces
        if faces is not None:
            out_string += self.generateObjectString(vertices, faces, vertex_normals, "{} {}".format(name, block.index))
        return out_string

    def generate(self, v_indices, f_indices):

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
            [4, 8, 7],  # 9 Diagonal Top -
            [5, 6, 7],  # 10 Diagonal Back
            [1, 2, 4],  # 11 Diagonal Front -
            [2, 8, 4],  # 12 Triangle 7 Verts -
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
            [1, 0, 0],  # Diagonal Left
            [0, -1, 0],  # Diagonal Bottom
            [0, 1, 0],  # Diagonal Top
            [0, 0, 1],  # Diagonal Back
            [0, 0, -1],  # Diagonal Front
            [-1, -1, -1],  # Diagonal missing corner 4
            [0, -1, -1],  # Triangle Front Bottom to Back Top Left
            [-1, -1, 0],  # Triangle Right Bottom to Back Top Left
            [-1, -1, -1],  # Triangle 4 Verts
            [0, -1, -1]  # Slope
        ]

        out_vertices = [vertices[index - 1] for index in range(1, 9) if index in v_indices]  # Select Vertices
        out_faces = [faces[index] for index in range(len(faces)) if index in f_indices]  # Select Faces
        out_normals = []
        for face in out_faces:  # Select Normals
            index = faces.index(face)
            out_normals.append(normalize(vertex_normals[index]))

        # Map the vertex indices to the correct number for all faces
        for i in range(len(v_indices)):
            if v_indices[i] != i + 1:
                mapping = lambda x: i + 1 if x == v_indices[i] else x
                for index in range(len(out_faces)):
                    out_faces[index] = list(map(mapping, out_faces[index]))

        # Construct UV Maps
        # for face in out_faces:
        #     for vert in face:
        #         pass

        return out_vertices, out_faces, out_normals

    def generateObjectString(self, vertices, faces, vertex_normals, name, uvs=None):
        out_string = "o {}\n".format(name)

        for vertex in vertices:
            out_string += "v {} {} {}\n".format(vertex[0], vertex[1], vertex[2])

        if vertex_normals is not None:
            for normal in vertex_normals:
                out_string += "vn "
                for value in normal:
                    out_string += "{} ".format(value)
                out_string += "\n"
        out_string += "usemtl Standard\n"
        out_string += "s off\n"
        if faces is not None:
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


def normalize(vector):
    length = 0
    for element in vector:
        length += element ** 2
    length = np.math.sqrt(length)
    vector = [entry/length for entry in vector]
    return vector


def readSourceFile(file):
    tree = xml.etree.ElementTree.parse(file).getroot()
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


def rotateVectors(vectors, look, up):
    return_vectors = []
    for vector in vectors:
        vector.append(1)

        temp_result = np.matrix(vector).transpose()
        if look == 5:  # Vorne
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
            [np.round(temp_result[0][0], 15), np.round(temp_result[1][0], 15), np.round(temp_result[2][0], 15)])
    return return_vectors


def transformBlock(verts, vert_normals, look, up, block=None):
    vertices = verts.copy()
    normals = vert_normals.copy()
    output_vertices = []
    output_normals = []
    translation_vector = [0, 0, 0]
    scale_amounts = [1, 1, 1]

    if block is not None:
        translation_vector[0] = (block.x_max + block.x_min) / 2
        translation_vector[1] = (block.y_max + block.y_min) / 2
        translation_vector[2] = (block.z_max + block.z_min) / 2
        scale_amounts[0] = (block.x_max - block.x_min) / 2
        scale_amounts[1] = (block.y_max - block.y_min) / 2
        scale_amounts[2] = (block.z_max - block.z_min) / 2

    vertices = rotateVectors(vertices, look, up)
    normals = rotateVectors(normals, look, up)

    for vertex in vertices:
        vertex.append(1)

        temp_result = np.matrix(vertex).transpose()
        temp_result = scale(temp_result, scale_amounts[0], scale_amounts[1], scale_amounts[2])
        temp_result = getTranslationMatrix(translation_vector[0], translation_vector[1],
                                           translation_vector[2]) * temp_result

        temp_result = temp_result.tolist()
        result = [np.round(temp_result[0][0], 15), np.round(temp_result[1][0], 15), np.round(temp_result[2][0], 15)]
        output_vertices.append(result)

    for normal in normals:
        normal.append(0)

        temp_result = np.matrix(normal).transpose()
        temp_result = scale(temp_result, scale_amounts[0], scale_amounts[1], scale_amounts[2])
        temp_result = temp_result.tolist()
        result = [np.round(temp_result[0][0], 15), np.round(temp_result[1][0], 15), np.round(temp_result[2][0], 15)]
        output_normals.append(normalize(result))

    return output_vertices, output_normals


def getTranslationMatrix(x, y, z):
    return np.matrix([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


def scale(vector, x_amount=0, y_amount=0, z_amount=0):
    xMatrix, yMatrix, zMatrix = np.identity(4), np.identity(4), np.identity(4)
    xMatrix[0][0] = x_amount
    yMatrix[1][1] = y_amount
    yMatrix[2][2] = z_amount
    result = xMatrix.dot(yMatrix.dot(zMatrix.dot(vector)))
    return result


def rotate(vector, x_degrees=0, y_degrees=0, z_degrees=0):
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


path = os.getenv('APPDATA') + '/Avorion/ships/'
if len(argv) < 2:
    print('Usage: AvObjConverter.py <file_name> [<path_to_folder_containing_file_name>]')
    sys.exit()
elif len(argv) == 3:
    path = argv[2]
ship_name = argv[1]

conv = AvToObjConverter()
conv.convertToObj(path + ship_name + '.xml', './out/' + ship_name + '.obj')

scale(np.matrix([1, 1, 1, 0]).transpose(), 5, 6, 7)
