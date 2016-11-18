import numpy as np
from operator import itemgetter
from sklearn.cluster import Birch, MiniBatchKMeans


class OBLocalization:
    def __init__(self, zDistance, xDistance, updateRadius):
        # Must be in the order [ul, ur, dl, dr] where l and r are
        # considered when facing the collection bin from inside the arena
        self.corners = np.array([])
        self.zDistance = zDistance
        self.xDistance = xDistance
        self.updateRadius = updateRadius

    def getRotationMatrix(self):
        ul, ur, dl, dr = self.corners
        xVector = dl - dr
        zVector = ur - dr
        yVector = np.cross(zVector, xVector)

        bases = np.vstack((xVector, yVector, zVector))
        desired = np.diag([self.xDistance, self.xDistance * self.zDistance, self.zDistance])

        t1 = desired.dot(np.linalg.pinv(bases).T)

        return t1 / np.linalg.norm(t1, axis=0)

    # No idea if this is right
    def getTheta(self):
        ul, ur, dl, dr = self.corners
        xVector = dl - dr
        zVector = ur - dr
        yVector = np.cross(zVector, xVector)

        return np.arctan2(xVector, yVector)[1]

    # old should be [ul, ur, dl, dr]
    # data should be a numpy array of points where a point is [x,y,z]
    # returns ul,ur,dl,dr
    def updateCorners(self, data):
        old = self.corners
        ulPoints = data.take(np.linalg.norm(data - old[0], axis=1) < self.updateRadius)
        urPoints = data.take(np.linalg.norm(data - old[1], axis=1) < self.updateRadius)
        dlPoints = data.take(np.linalg.norm(data - old[2], axis=1) < self.updateRadius)
        drPoints = data.take(np.linalg.norm(data - old[3], axis=1) < self.updateRadius)
        self.corners = np.array([np.mean(ulPoints, axis=0), np.mean(urPoints, axis=0), np.mean(dlPoints, axis=0),
                                 np.mean(drPoints, axis=0)])

    def initializeCorners(self, oranges):
        birch = Birch(threshold=0.1, n_clusters=4).fit(oranges)
        # kmeans = MiniBatchKMeans(n_clusters=4).fit(oranges)
        labels = birch.labels_
        # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # print('Estimated number of clusters: %d' % n_clusters_)

        # points might need to be tuples instead of lists? when making this array
        # print(np.mean(oranges[labels == 0], axis=0))
        # print(np.mean(oranges[labels == 1], axis=0))
        # print(np.mean(oranges[labels == 2], axis=0))
        # print(np.mean(oranges[labels == 3], axis=0))

        self.corners = np.vstack(
            (np.mean(oranges.take(labels == 0), axis=0), np.mean(oranges.take(labels == 1), axis=0),
             np.mean(oranges.take(labels == 2), axis=0), np.mean(oranges.take(labels == 3), axis=0)))
        self.orderCorners()

    def orderCorners(self):
        # self.corners.sort(axis=0) # this is where stuff is going wrong
        self.corners = np.array(sorted(self.corners, key=itemgetter(0, 1)))
        # print(self.corners)
        minx = self.corners[:2]
        minx_theta = np.arctan2((minx[0][1] + minx[1][1]) / 2, (minx[0][0] + minx[1][0]) / 2)
        maxx = self.corners[2:]
        maxx_theta = np.arctan2((maxx[0][1] + maxx[1][1]) / 2, (maxx[0][0] + maxx[1][0]) / 2)
        if minx_theta < 0:
            minx_theta += 2 * np.pi
        if maxx_theta < 0:
            maxx_theta += 2 * np.pi

        if maxx_theta < minx_theta:
            i = 1 if minx[0][2] < minx[1][2] else 0
            j = 1 if maxx[0][2] < maxx[1][2] else 0
            ul = minx[i]
            ur = maxx[j]
            dl = minx[1 - i]
            dr = maxx[1 - j]
            self.corners = np.array([ul, ur, dl, dr])
        elif maxx_theta > minx_theta:
            j = 1 if minx[0][2] < minx[1][2] else 0
            i = 1 if maxx[0][2] < maxx[1][2] else 0
            ul = maxx[i]
            ur = minx[j]
            dl = maxx[1 - i]
            dr = minx[1 - j]
            self.corners = np.array([ul, ur, dl, dr])

    def update(self, newData):
        self.updateCorners(newData)
        self.orderCorners()


if __name__ == '__main__':
    # ul = np.array([3, -3.5, 0.5])
    # ur = np.array([3, -2, 0.5])
    # dl = np.array([3, -3.5, 0])
    # dr = np.array([3, -2, 0])
    # l = OBLocalization(0.5, 1.5, 0.2)
    # l.corners = [ul, ur, dl, dr]
    # print(l.getRotationMatrix())
    # print(l.getTheta())
    # l.initializeCorners(np.array([[3, -3.5, 0.5], [3, -2, 0.5], [3, -3.5, 0], [3, -2, 0], [3, -2, 0.05]]))
    # l.orderCorners()
    # print(l.corners)
    l = OBLocalization(0.5, 1.5, 0.2)
    e = np.array([[3, -2, 0.5], [3, -3.5, 0.5], [3, -2, 0], [3, -3.5, 0]])
    l.corners = np.array([[3, -2, 0.5], [3, -3.5, 0.5], [3, -2, 0], [3, -3.5, 0]])
    while (l.corners == e).all():
        np.random.shuffle(l.corners)
    # print(l.corners)
    l.orderCorners()
    print(l.corners)
