import os
from typing import List

from waveobjparser import Face
from waveobjparser import Normal
from waveobjparser import Scene
from waveobjparser import Text_Coord
from waveobjparser import Vertex
from waveobjparser import WaveObject


def writeName(file, name: str):
    file.write("o {}\n".format(name))


def writeVertices(file, vertices: List[Vertex]):
    for vertex in vertices:
        file.write("v {v[0]} {v[1]} {v[2]}\n".format(v=vertex))


def writeTextureCoordinates(file, texture_coords: List[Text_Coord]):
    for coord in texture_coords:
        file.write("vt {t[0]} {t[1]}\n".format(t=coord))


def writeNormals(file, vertex_normals: List[Normal]):
    for normal in vertex_normals:
        file.write("vn {n[0]} {n[1]} {n[2]}\n".format(n=normal))


def writeMaterial(file, material):
    file.write("usemtl None\n")  # TODO Implement


def writeSmoothing(file, smoothing: bool):
    string = "on" if smoothing else "off"
    file.write("s {}\n".format(string))


class ObjWriter:
    def __init__(self, outputFile):
        self.vertex_count = 0
        self.uv_count = 0
        self.normals_count = 0
        self.output_file = outputFile

    def writeScene(self, scene: Scene):
        if type(self.output_file) is os.path:
            path = os.path.join(self.output_file)
        else:
            path = self.output_file

        print("Starting output on file ", path)
        with open(path, "w") as file:
            file.write("mtllib {}\n".format(scene.mtllib))
            for object in scene.objects:
                writeName(file, object.name)
                writeVertices(file, object.vertices)
                writeTextureCoordinates(file, object.texture_coords)
                writeNormals(file, object.vertex_normals)
                writeMaterial(file, object.material)
                writeSmoothing(file, object.smoothing)
                self.writeFaces(file, object.faces)
                self.vertex_count += len(object.vertices)
                self.normals_count += len(object.vertex_normals)
                self.uv_count += len(object.texture_coords)

    def writeFaces(self, file, faces: List[Face]):
        for face in faces:
            file.write("f ")
            for vertex in face:
                if vertex[2] != '':
                    if vertex[1] != '':
                        file.write("{}/{}/{} ".format(int(vertex[0]) + self.vertex_count,
                                                      int(vertex[1]) + self.uv_count,
                                                      int(vertex[2]) + self.normals_count))
                    else:
                        file.write("{}//{} ".format(int(vertex[0]) + self.vertex_count,
                                                    int(vertex[2]) + self.normals_count))
                else:
                    file.write("{} ".format(int(vertex[0] + self.vertex_count)))
            file.write("\n")
