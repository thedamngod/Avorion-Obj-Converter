import operator
import os
import xml.etree.ElementTree
from functools import reduce
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

        if block.type in [1, 2, 3, 5, 6, 7, 8, 9, 50, 51, 52, 53, 54, 55]:  # Non-diagonal blocks
            out_string += "o Block {}\n".format(block.index)
            # out_string += "v {} {} {}\n".format(block.x_min, -block.z_min, block.y_min)  # Add vertex lower front left corner
            # out_string += "v {} {} {}\n".format(block.x_min, -block.z_min, block.y_max)  # Add vertex lower back left corner
            # out_string += "v {} {} {}\n".format(block.x_min, -block.z_max, block.y_min)  # Add vertex upper front left corner
            # out_string += "v {} {} {}\n".format(block.x_min, -block.z_max, block.y_max)  # Add vertex upper back left corner
            # out_string += "v {} {} {}\n".format(block.x_max, -block.z_min, block.y_min)  # Add vertex lower front right corner
            # out_string += "v {} {} {}\n".format(block.x_max, -block.z_min, block.y_max)  # Add vertex lower back left corner
            # out_string += "v {} {} {}\n".format(block.x_max, -block.z_max, block.y_min)  # Add vertex upper front right corner
            # out_string += "v {} {} {}\n".format(block.x_max, -block.z_max, block.y_max)  # Add vertex upper back right corner
            out_string += "v {0} {2} {1}\n".format(block.x_min, block.z_min, block.y_min)  # Add vertex lower front left corner
            out_string += "v {0} {2} {1}\n".format(block.x_min, block.z_min, block.y_max)  # Add vertex lower back left corner
            out_string += "v {0} {2} {1}\n".format(block.x_min, block.z_max, block.y_min)  # Add vertex upper front left corner
            out_string += "v {0} {2} {1}\n".format(block.x_min, block.z_max, block.y_max)  # Add vertex upper back left corner
            out_string += "v {0} {2} {1}\n".format(block.x_max, block.z_min, block.y_min)  # Add vertex lower front right corner
            out_string += "v {0} {2} {1}\n".format(block.x_max, block.z_min, block.y_max)  # Add vertex lower back right corner
            out_string += "v {0} {2} {1}\n".format(block.x_max, block.z_max, block.y_min)  # Add vertex upper front right corner
            out_string += "v {0} {2} {1}\n".format(block.x_max, block.z_max, block.y_max)  # Add vertex upper back right corner
            out_string += "f {} {} {} {}\n".format(self.vc+1, self.vc+2, self.vc+4, self.vc+3)
            out_string += "f {} {} {} {}\n".format(self.vc+5, self.vc+6, self.vc+8, self.vc+7)
            out_string += "f {} {} {} {}\n".format(self.vc+1, self.vc+2, self.vc+6, self.vc+5)
            out_string += "f {} {} {} {}\n".format(self.vc+1, self.vc+5, self.vc+7, self.vc+3)
            out_string += "f {} {} {} {}\n".format(self.vc+2, self.vc+6, self.vc+8, self.vc+4)
            out_string += "f {} {} {} {}\n".format(self.vc+3, self.vc+7, self.vc+8, self.vc+4)
            self.vc += 8
        elif block.type == 100:  # Without rotation
            out_string += "o Block {}\n".format(block.index)
            # out_string += "v {0} {2} {1}\n".format(block.x_min, -block.z_min, block.y_min)  # Add vertex lower front left corner
            # out_string += "v {0} {2} {1}\n".format(block.x_min, -block.z_min, block.y_max)  # Add vertex lower back left corner
            # out_string += "v {0} {2} {1}\n".format(block.x_min, -block.z_max, block.y_max)  # Add vertex upper back left corner
            # out_string += "v {0} {2} {1}\n".format(block.x_max, -block.z_min, block.y_min)  # Add vertex lower front right corner
            # out_string += "v {0} {2} {1}\n".format(block.x_max, -block.z_min, block.y_max)  # Add vertex lower back right corner
            # out_string += "v {0} {2} {1}\n".format(block.x_max, -block.z_max, block.y_max)  # Add vertex upper back right corner
            vertices = [[block.x_min, block.y_min, block.z_min], [block.x_max, block.y_min, block.z_min], [block.x_max, block.y_max, block.z_min],
                       [block.x_min, block.y_max, block.z_min], [block.x_max, block.y_min, block.z_max], [block.x_min, block.y_min, block.z_max]]
            out_string += "v {} {} {}\n".format(block.x_min, block.y_min, block.z_min)  # Add vertex lower front left corner
            out_string += "v {} {} {}\n".format(block.x_max, block.y_min, block.z_min)  # Add vertex lower front right corner
            out_string += "v {} {} {}\n".format(block.x_max, block.y_max, block.z_min)  # Add vertex upper back left corner
            out_string += "v {} {} {}\n".format(block.x_min, block.y_max, block.z_min)  # Add vertex lower front right corner
            out_string += "v {} {} {}\n".format(block.x_max, block.y_min, block.z_max)  # Add vertex lower back right corner
            out_string += "v {} {} {}\n".format(block.x_min, block.y_min, block.z_max)  # Add vertex upper back right corner
            # out_string += "f {} {} {} \n".format(self.vc+1, self.vc+2, self.vc+3)
            # out_string += "f {} {} {} \n".format(self.vc+4, self.vc+5, self.vc+6)
            # out_string += "f {} {} {} {}\n".format(self.vc+1, self.vc+2, self.vc+5, self.vc+4)
            # out_string += "f {} {} {} {}\n".format(self.vc+2, self.vc+3, self.vc+6, self.vc+5)
            # out_string += "f {} {} {} {}\n".format(self.vc+1, self.vc+3, self.vc+6, self.vc+4)
            out_string += "f {} {} {} \n".format(self.vc+5, self.vc+2, self.vc+3)
            out_string += "f {} {} {} \n".format(self.vc+4, self.vc+6, self.vc+1)
            out_string += "f {} {} {} {}\n".format(self.vc+1, self.vc+2, self.vc+3, self.vc+4)
            out_string += "f {} {} {} {}\n".format(self.vc+1, self.vc+2, self.vc+5, self.vc+6)
            out_string += "f {} {} {} {}\n".format(self.vc+5, self.vc+6, self.vc+4, self.vc+3)
            self.vc += 6
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




conv = AvToObjConverter()
appdata_path = 'C:/Users/Michael/AppData/Roaming/Avorion/ships/'
ship_name = 'rotationstest'
# ship_name = 'Iron Slope Front'
# ship_name = 'long slope'
conv.convertToObj(appdata_path + ship_name + '.xml', './out/' + ship_name + '.obj')
