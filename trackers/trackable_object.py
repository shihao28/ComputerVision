class TrackableObject:
    def __init__(self, objectID, bbox, attrb1, attrb2):
        self.objectID = objectID
        self.bboxes = [bbox]      
        self.attrb1 = [attrb1]
        self.attrb2 = [attrb2]

    def update(self, bbox, attrb1, attrb2):

        self.bboxes.append(bbox)
        self.attrb1.append(attrb1)
        self.attrb2.append(attrb2)
        return None
