from waveobjparser import WaveObject
from .Scene import Scene


class ObjParser:
    def __init__(self, source=None):
        self.source = source
        self.scene = Scene()

    def parseFile(self):
        with open(self.source) as file:
            lines = file.readlines()
            starts = []
            for index in range(len(lines)):
                if lines[index].startswith('o'):
                    starts.append(index)
                if lines[index].startswith('mtllib'):
                    self.scene.setMtllib(' '.split(lines[index])[-1])

            for index in range(len(starts)):
                if index != len(starts) - 1:
                    self.parseObject(lines[starts[index]:starts[index + 1]])
                else:
                    self.parseObject(lines[starts[index]:])
        return self.scene

    def parseObject(self, lines):
        obj = WaveObject()
        for line in lines:
            line = line[:-1].split(' ')
            if len(line) < 1:  # No information in the line
                return
            elif line[0] == 'o':
                obj.setName(line[-1])
            elif line[0] == "vn":
                obj.addNormal(tuple(map(float, line[1:])))
            elif line[0] == "vt":
                obj.addTextureCoord(tuple(map(float, line[1:])))
            elif line[0] == 'v':
                obj.addVertex(tuple(map(float, line[1:])))
            elif line[0] == 'f':
                face = []
                del line[0]
                for vert in line:
                    face.append(tuple(vert.split('/')))
                obj.addFace(face)
            elif line[0] == 'mtllib':
                obj.addMaterial(line[1:])
        self.scene.addObject(obj)
