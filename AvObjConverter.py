import os
import xml.etree.ElementTree
import numpy as np

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
        self.vc = 0
        self.blocks = blocks

    def convertToObj(self, inputFile, outputFile):
        if self.blocks is None:
            self.blocks = readSourceFile(inputFile)
        try:
            while os.path.exists(outputFile) and os.path.getsize(outputFile) > 0:
                outputFile = outputFile[:-4] + '1' + outputFile[-4:]
                print(outputFile)

            file = open(outputFile, 'a')

            for block in self.blocks:
                file.write(self.convertBlockToObj(block))

            file.close()
        except FileNotFoundError:
            print('Could not open output file')

    def convertBlockToObj(self, block):
        out_string = ""
        name = "Block"
        # out_string += "o Block {}\n".format(block.index)

        vertices = [
            [block.x_max, block.y_min, block.z_min],  # 2 Bottom right front
            [block.x_min, block.y_min, block.z_min],  # 1 Bottom left front
            [block.x_max, block.y_max, block.z_min],  # 4 Top right front
            [block.x_min, block.y_max, block.z_min],  # 3 Top left front
            [block.x_max, block.y_min, block.z_max],  # 6 Bottom right back
            [block.x_min, block.y_min, block.z_max],  # 5 Bottom left back
            [block.x_max, block.y_max, block.z_max],  # 8 Top right back
            [block.x_min, block.y_max, block.z_max]]  # 7 Top left back

        faces = None

        if block.type in range(1, 16) or block.type in range(50, 56) or block.type in [150, 170, 190, 510]:  # Non-diagonal blocks
            # In forward flying direction
            vertices = transformBlock(vertices, 5, 3, block)
            faces = [[1, 2, 4, 3],
                     [2, 6, 8, 4],
                     [5, 6, 8, 7],
                     [1, 5, 7, 3],
                     [1, 2, 6, 5],
                     [3, 4, 8, 7]]

        elif block.type in [100, 104, 151, 171, 185, 191, 511]:  # Edge with 6 vertices
            # vertices = [
            #             [block.x_max, block.y_min, block.z_min],  # 2 Bottom right front
            #             [block.x_min, block.y_min, block.z_min],  # 1 Bottom left front
            #             [block.x_max, block.y_min, block.z_max],  # 6 Bottom right back -> 4
            #             [block.x_min, block.y_min, block.z_max],  # 5 Bottom left back -> 3
            #             [block.x_max, block.y_max, block.z_max],  # 8 Top right back -> 6
            #             [block.x_min, block.y_max, block.z_max]]  # 7 Top left back -> 5
            del vertices[2:4]

            # vertices = transformBlock(vertices, 5, 3, block)
            vertices = transformBlock(vertices, block.look, block.up, block)
            faces = [[1, 3, 5],
                     [2, 4, 6],
                     [1, 2, 4, 3],
                     [1, 2, 6, 5],
                     [3, 4, 6, 5]]

        elif block.type in [103, 107, 154, 174, 188, 194, 514]:  # Edge with 5 vertices
            name = "5 Corner Edge"
            # vertices = [
            #             [block.x_max, block.y_min, block.z_min],  # 2 Bottom right front
            #             [block.x_min, block.y_min, block.z_min],  # 1 Bottom left front
            #             [block.x_max, block.y_min, block.z_max],  # 6 Bottom right back -> 4
            #             [block.x_min, block.y_min, block.z_max],  # 5 Bottom left back -> 3
            #             [block.x_min, block.y_max, block.z_max]]  # 7 Top left back -> 5
            del vertices[7]
            del vertices[2:4]
            vertices = transformBlock(vertices, block.look, block.up, block)
            faces = [[1, 3, 5],
                     [1, 2, 5],
                     [2, 4, 5],
                     [3, 4, 5],
                     [3, 4, 1]]

        elif block.type in [101, 105, 152, 172, 186, 192, 512]:  # Edge with 4 vertices
            name = "4 Corner Edge"
            # vertices = [[block.x_min, block.y_min, block.z_min],  # 1 Bottom left front
            #             [block.x_max, block.y_min, block.z_max],  # 6 Bottom right back -> 3
            #             [block.x_min, block.y_min, block.z_max],  # 5 Bottom left back -> 2
            #             [block.x_min, block.y_max, block.z_max]]  # 7 Top left back -> 4
            del vertices[7]
            del vertices[1:4]
            vertices = transformBlock(vertices, block.look, block.up, block)
            faces = [[1, 2, 4],
                     [2, 3, 4],
                     [2, 3, 1],
                     [1, 3, 4]]

        elif block.type in [102, 106, 153, 173, 187, 193, 513]:  # Edge with 7 vertices
            name = "7 Corner Edge"
            # vertices = [
            #             [block.x_max, block.y_min, block.z_min],  # 2 Bottom right front
            #             [block.x_min, block.y_min, block.z_min],  # 1 Bottom left front
            #             [block.x_max, block.y_max, block.z_min],  # 3 Top left front
            #             [block.x_max, block.y_min, block.z_max],  # 6 Bottom right back -> 5
            #             [block.x_min, block.y_min, block.z_max],  # 5 Bottom left back -> 4
            #             [block.x_max, block.y_max, block.z_max],  # 8 Top right back -> 7
            #             [block.x_min, block.y_max, block.z_max]]  # 7 Top left back -> 6
            del vertices[3]
            vertices = transformBlock(vertices, block.look, block.up, block)
            faces = [[1, 4, 6, 3],
                     [4, 5, 7, 6],
                     [1, 2, 5, 4],
                     [3, 6, 7],
                     [1, 2, 3],
                     [2, 3, 7],
                     [2, 5, 7]]

        if faces is not None:
            out_string += self.generateObjectString(vertices, faces, "{} {}".format(name, block.index))
        return out_string

    def generateObjectString(self, vertices, faces, name):
        out_string = "o {}\n".format(name)

        for vertex in vertices:
            out_string += "v {} {} {}\n".format(vertex[0], vertex[1], vertex[2])

        for face in faces:
            out_string += "f "
            for vertex in face:
                out_string += "{} ".format(self.vc + vertex)
            out_string += "\n"
        self.vc += len(vertices)
        return out_string


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


def transformBlock(verts, look, up, block=None):
    vertices = verts.copy()
    output_vertices = []
    translation_vector = [0, 0, 0]

    if block is not None:
        translation_vector[0] = (block.x_max + block.x_min) / 2
        translation_vector[1] = (block.y_max + block.y_min) / 2
        translation_vector[2] = (block.z_max + block.z_min) / 2

    for vertex in vertices:
        vertex.append(1)

        trafo_matrix = getTranslationMatrix(-translation_vector[0], -translation_vector[1], -translation_vector[2])
        temp_result = trafo_matrix * np.matrix(vertex).transpose()
        if look == 5:  # Vorne
            if up == 1:
                temp_result = getRotationMatrix(z=True, degrees=270) * temp_result
            elif up == 2:
                temp_result = getRotationMatrix(z=True, degrees=180) * temp_result  # um z 180°
            elif up == 0:
                temp_result = getRotationMatrix(z=True, degrees=90) * temp_result

        elif look == 0:  # Links
            # rotation um y 90° counter, Rotation nach vorne
            temp_result = getRotationMatrix(y=True, degrees=90) * temp_result
            if up == 2:
                temp_result = getRotationMatrix(z=True, degrees=180) * temp_result  # 180° um z, nach der ersten
            elif up == 4:
                temp_result = getRotationMatrix(z=True, degrees=270) * temp_result  # 90 counter z
            elif up == 5:
                temp_result = getRotationMatrix(z=True, degrees=90) * temp_result

        elif look == 1:  # Rechts
            # rotation um y 90° counter, Rotation nach vorne
            temp_result = getRotationMatrix(y=True, degrees=-90) * temp_result
            if up == 2:
                temp_result = getRotationMatrix(z=True, degrees=-180) * temp_result  # 180° um z, nach der ersten
            elif up == 4:
                temp_result = getRotationMatrix(z=True, degrees=270) * temp_result  # 90 counter z
            elif up == 5:
                temp_result = getRotationMatrix(z=True, degrees=-90) * temp_result

        elif look == 2:  # Unten
            temp_result = getRotationMatrix(x=True, degrees=-270) * temp_result  # Rotation nach vorne
            if up == 1:
                temp_result = getRotationMatrix(z=True, degrees=270) * temp_result
            elif up == 4:
                temp_result = getRotationMatrix(z=True, degrees=180) * temp_result
            elif up == 0:
                temp_result = getRotationMatrix(z=True, degrees=90) * temp_result

        elif look == 3:  # Oben
            temp_result = getRotationMatrix(x=True, degrees=-90) * temp_result  # Rotation nach vorne
            if up == 1:
                temp_result = getRotationMatrix(z=True, degrees=270) * temp_result
            elif up == 0:
                temp_result = getRotationMatrix(z=True, degrees=90) * temp_result
            elif up == 5:
                temp_result = getRotationMatrix(z=True, degrees=180) * temp_result

        elif look == 4:  # Hinten
            temp_result = getRotationMatrix(x=True, degrees=180) * temp_result  # Rotation nach vorne
            if up == 1:
                temp_result = getRotationMatrix(z=True, degrees=270) * temp_result
            elif up == 0:
                temp_result = getRotationMatrix(z=True, degrees=90) * temp_result
            elif up == 3:
                temp_result = getRotationMatrix(z=True, degrees=180) * temp_result

        temp_result = getTranslationMatrix(translation_vector[0], translation_vector[1],
                                           translation_vector[2]) * temp_result

        temp_result = temp_result.tolist()
        result = [np.round(temp_result[0][0], 10), np.round(temp_result[1][0], 10), np.round(temp_result[2][0], 10)]
        output_vertices.append(result)

    return output_vertices


def getTranslationMatrix(x, y, z):
    return np.matrix([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


def getRotationMatrix(x=False, y=False, z=False, degrees=90):
    xMatrix, yMatrix, zMatrix = np.identity(4), np.identity(4), np.identity(4)
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    if x:
        xMatrix = np.matrix(
            '{} {} {} {}; {} {} {} {}; {} {} {} {}; {} {} {} {}'.format(1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0,
                                                                        1))
    if y:
        yMatrix = np.matrix(
            '{} {} {} {}; {} {} {} {}; {} {} {} {}; {} {} {} {}'.format(c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1))
    if z:
        zMatrix = np.matrix(
            '{} {} {} {}; {} {} {} {}; {} {} {} {}; {} {} {} {}'.format(c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                                                        1))
    return xMatrix * yMatrix * zMatrix


conv = AvToObjConverter()
appdata_path = 'C:/Users/Michael/AppData/Roaming/Avorion/ships/'
# ship_name = 'rotationtest'
# ship_name = 'Iron Slope Front'
# ship_name = 'RotationThruster'
# ship_name = 'long slope'
# ship_name = 'block'
ship_name = 'corners'
# ship_name = 'blocktypes'
# ship_name = 'Hyperion II'
conv.convertToObj(appdata_path + ship_name + '.xml', './out/' + ship_name + '.obj')
