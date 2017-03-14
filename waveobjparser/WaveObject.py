from typing import List, Tuple

Vertex = Tuple[float, float, float]
Normal = Tuple[float, float, float]
FaceVertex = Tuple[int, int, int]
Face = List[FaceVertex]
Text_Coord = Tuple[float, float]


class WaveObject:
    def __init__(self):
        self.vertices = []  # List[Vertex]
        self.faces = []  # List[Face]
        self.vertex_normals = []  # List[Normal]
        self.texture_coords = []  # List[Text_Coord]
        self.material = None  # str
        self.smoothing = False  # bool
        self.name = None  # str

    def addVertex(self, vertex: Vertex):
        self.vertices.append(vertex)

    def addNormal(self, normal: Normal):
        self.vertex_normals.append(normal)

    def addTextureCoord(self, coordinate: Text_Coord):
        self.texture_coords.append(coordinate)

    def addMaterial(self, material: str):
        self.material = material  # TODO: Implement materials

    def setSmoothing(self, smoothing: bool):
        self.smoothing = smoothing

    def addFace(self, face: Face):
        # Check if the face is correct:
        # for vertex in face:
        #     for entry in vertex:
        #         if entry > len(self.vertices) + 1 or entry
        self.faces.append(face)

    def setName(self, name: str):
        self.name = name

    def __str__(self):
        string = "WaveObject \n"
        string += "Name: {}\n".format(self.name)
        string += "Vertices: \n{}\n".format(self.vertices)
        string += "Vertex Normals: \n{}\n".format(self.vertex_normals)
        string += "Texture Coordinates: \n{}\n".format(self.texture_coords)
        string += "Faces: \n{}\n".format(self.faces)
        return string
