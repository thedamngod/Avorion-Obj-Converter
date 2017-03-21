from waveobjparser import WaveObject


class Scene:
    def __init__(self):
        self.objects = []
        self.mtllib = None

    def setMtllib(self, mtllib):
        self.mtllib = mtllib

    def addObject(self, object: WaveObject):
        self.objects.append(object)

    def removeObject(self, object: WaveObject):
        if object in self.objects:
            self.objects.remove(object)
