import math
import numpy as np
import pygame
from sklearn.preprocessing import normalize
import cProfile, pstats, io
from pstats import SortKey

width = 800
height = 800
a_ratio = height / width
fov = 90
disp = pygame.display.set_mode((width, height))
whArray = np.array([width, height])
disp.set_alpha(None)
clock = pygame.time.Clock()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

yRot = 0


# matrix functions----------------------------------------
def o_normalize(a):
    return a / np.sqrt(np.sum(a ** 2))


# this is needed because numpy is terribly slow with small arrays
def rotateVectorY(a, angle):
    first = a[0] * math.cos(angle) + a[2] * -math.sin(angle)
    second = a[1]
    third = a[0] * math.sin(angle) + math.cos(angle) * a[2]
    matrix = np.array([first, second, third])
    return matrix


def dotProd(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


# ---------------------------------------------------------


# Transforms a point from 0,0,0 coords to rotated coordinate system, pos is current position, target it where you want to look, up is up direction of new coord system
# takes numpy arrays as input, returns numpy arrays
# is a transformation matrix
# again, this is only used with a 3D vector, because of which using numpy multiplication will be much slower
def PointAtMatrix(pos, target, up):
    newForward = target - pos
    newForward = newForward / math.sqrt(np.sum(newForward ** 2))

    a = newForward * np.dot(up, newForward)
    newUp = up - a
    newUp = newUp / np.sqrt(np.sum(newUp ** 2))

    newRight = np.cross(newUp, newForward)

    matrix = [[newRight[0], newRight[1], newRight[2], 0],
              [newUp[0], newUp[1], newUp[2], 0],
              [newForward[0], newForward[1], newForward[2], 0],
              [pos[0], pos[1], pos[2], 1]]
    return matrix


# takes in the transformation matrix, and returns its inverse
def LookAtMatrix(a):
    t1 = np.dot(a[0][:3], a[3][:3])
    t2 = np.dot(a[1][:3], a[3][:3])
    t3 = np.dot(a[2][:3], a[3][:3])
    matrix = [[a[0][0], a[1][0], a[2][0], 0],
              [a[0][1], a[1][1], a[2][1], 0],
              [a[0][2], a[1][2], a[2][2], 0],
              [-t1, -t2, -t3, 1]]
    return matrix


# --------------------------------------------------------

# READING TEXTURE DATA IS BROKEN ATM
# reading in vertex data, and creating an n x 4 array of triangles
readVertices = []
readTextureCoords = []
triangleIndices = []
textureIndices = []
readLines = []
colors = {}
# contains a list of material names, like [red,red,red,red,blue,blue,blue,blue]
triangleColorInfo = []
# CHANGE THIS FILE NAME TO LOAD CUSTOM FILES
with open('objectData.obj', 'r') as f:
    readLines = f.readlines()

for line in readLines:
    if line not in ('#', 'o', 'off', 'mtllib'):
        line = line.strip().split(' ')
        if 'v' in line:
            readVertices.append([float(i) for i in line[1:]])
        elif 'f' in line:
            nums = line[1:]
            tI = []
            teI = []
            for pair in nums:
                n = pair.split('/')
                tI.append(int(n[0]) - 1)
                # teI.append(int(n[1])-1)
            triangleIndices.append(tI)
            # textureIndices.append(teI)

mat = ''
for line in readLines:
    if line not in ("#", "o", 'off', 'mtllib'):
        if 'usemtl' in line:
            mat = line.split(' ')[1]
        elif 'f' in line:
            triangleColorInfo.append(mat)
f.close()

mtlInfo = []
# with open('objectData.mtl', 'r') as f:
# mtlInfo = f.readlines()
f.close()
currColor = ''
for line in mtlInfo:
    if 'newmtl' in line:
        currColor = line.strip().split(' ')[1]
    elif 'Kd' in line:
        colors[currColor] = [float(i) for i in line.strip().split(' ')[1:]]

triangles = []
uvTriangles = []
for tri in triangleIndices:
    v1 = [readVertices[tri[0]][0], readVertices[tri[0]][1], readVertices[tri[0]][2], 1]
    v2 = [readVertices[tri[1]][0], readVertices[tri[1]][1], readVertices[tri[1]][2], 1]
    v3 = [readVertices[tri[2]][0], readVertices[tri[2]][1], readVertices[tri[2]][2], 1]
    triangles.append(np.array(v1))
    triangles.append(np.array(v2))
    triangles.append(np.array(v3))

for tri in textureIndices:
    v1 = [readTextureCoords[tri[0]][0], readTextureCoords[tri[0]][1]]
    v2 = [readTextureCoords[tri[1]][0], readTextureCoords[tri[1]][1]]
    v3 = [readTextureCoords[tri[2]][0], readTextureCoords[tri[2]][1]]
    uvTriangles.append(v1)
    uvTriangles.append(v2)
    uvTriangles.append(v3)
# triangles now consists of a list of vertices, such that i, i+1, i+2 are one triangle
# data is stored in such a way to allow for vectorization using numpy
# its usually a bad idea to use np.array() on small arrays, but this code is only executed once so it doesnt matter
triangles = np.array(triangles)
print(f'Vertices: {len(readVertices)}, Triangles: {len(triangles) // 3}')
# matrix constants -----------------------------------
# projection matrix
f = 1 / math.tan(math.radians(fov / 2))
z_F = 1000
z_N = 0.3
q = z_F / (z_F - z_N)
t_matrix = [[a_ratio * f, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, q, 1],
            [0, 0, -1 * z_N * q, 0]]
t_matrix = np.array([np.array(i) for i in t_matrix])
# ----------------------------------------------------

camera = np.array([0, 0, 0], dtype=float)
cameraLook = np.array([0, 0, 1])
# constants for manipulation
zTransform = np.array([np.array([0, 0, 20, 0]) for i in range(len(triangles))])
upDir = np.array([0, 1, 0])
forwardDir = np.array([0, 0, 1], dtype=float)
targetVector = np.array([0, 0, 1])

# creating vector constants here to avoid repeated initializations
v = triangles[0]
# for z clipping
forwardPlaneVector = np.array([0, 0, 0.5])
forwardNormal = np.array([0, 0, 1])
dotProductForwardPlaneVectorNormal = np.dot(forwardNormal, forwardPlaneVector)

# for x clipping
leftPlanePoint = np.array([0, 0, 0])
leftPlaneNormal = np.array([1, 0, 0])
dotProductLeftPlaneVectorNormal = np.dot(leftPlaneNormal, leftPlanePoint)

# for top clipping
topPlanePoint = np.array([0, 0, 0])
topPlaneNormal = np.array([0, 1, 0])
dotProductTopPlaneNormal = np.dot(topPlaneNormal, topPlanePoint)

# for right clipping
rightPlanePoint = np.array([width, 0, 0])
rightPlaneNormal = np.array([-1, 0, 0])
dotProductRightPlaneNormal = np.dot(rightPlaneNormal, rightPlanePoint)

# for bottom clipping
bottomPlanePoint = np.array([0, height, 0])
bottomPlaneNormal = np.array([0, -1, 0])
dotProductBottomPlaneNormal = np.dot(bottomPlaneNormal, bottomPlanePoint)

# creating an array for clipped triangles now to avoid appending. maximum number of new clipped triangles are 2, so it has a size of verts length * 2
# these will be filled up later. statically declared array functionality is emulated
clippedVerticesEmpty = np.empty((len(triangles) * 2, 4))
clippedLightInfo = np.empty((1, (len(triangles) * 2) // 3)).squeeze()
clippedNormalInfo = np.empty((1, (len(triangles) * 2) // 3)).squeeze()

totalClippedTris = 0


def renderTris(verts, light_info, normalsAlignment):
    # projecting from 3d to 2d
    projectedMatrix = np.matmul(verts, t_matrix.T)
    # 0 in this means 0 in verts, 1 in this means 3 in verts
    sortIndices = np.argsort(projectedMatrix[0::3, 2])[::-1]
    # getting an array containing all z values
    zs = projectedMatrix[::, 2:3]
    # dividing all values by the z value
    a = projectedMatrix[::, 3:4].reshape(len(verts), )
    projectedMatrix[::, 0:2] /= a[:, None]
    # normalizing between 0 and 1
    projectedMatrix = (projectedMatrix + 1) * 0.5
    # fitting into screen
    projectedMatrix[::, 0:2] *= whArray
    # normalsAlignment = np.sum(normals * (verts[0::3, 0:3] - cameraPos), axis=1)
    drawn = []
    normals = []
    # drawing the triangles
    for i in sortIndices:
        normals.append(f'{i} : {normalsAlignment[i]}')
        if normalsAlignment[i] > 0:
            # face is NOT facing the camera
            continue
        else:
            drawn.append(i)
            light = light_info[i]
            if light < 0.5:
                color = 100 * light + 100
            else:
                color = round(light * 255 * 0.98)
            if color > 255:
                color = 255
            elif color < 0:
                color = 0
            color = (color, color, color)

            # clipping to screen sides
            # this is done to avoid trying to draw triangles of length approaching infinity

            trisToRender = [projectedMatrix[3 * i: 3 * i + 3, 0:3]]
            numNewTris = 1

            for j in range(4):
                clippedTris = []
                while numNewTris > 0:
                    test = trisToRender.pop(0)
                    numNewTris -= 1

                    if j == 0:
                        dists = np.sum(test * leftPlaneNormal, axis=1) - dotProductLeftPlaneVectorNormal
                        clippedTris = triangleClipIndividual(leftPlanePoint, leftPlaneNormal, test, dists)
                    if j == 1:
                        dists = np.sum(test * topPlaneNormal, axis=1) - dotProductTopPlaneNormal
                        clippedTris = triangleClipIndividual(topPlanePoint, topPlaneNormal, test, dists)
                    if j == 2:
                        dists = np.sum(test * rightPlaneNormal, axis=1) - dotProductRightPlaneNormal
                        clippedTris = triangleClipIndividual(rightPlanePoint, rightPlaneNormal, test, dists)
                    if j == 3:
                        dists = np.sum(test * bottomPlaneNormal, axis=1) - dotProductBottomPlaneNormal
                        clippedTris = triangleClipIndividual(bottomPlanePoint, bottomPlaneNormal, test, dists)
                    for t in clippedTris:
                        trisToRender.append(t)
                numNewTris = len(trisToRender)
            for triangle in trisToRender:
                pygame.draw.polygon(disp, color,
                                    [(triangle[0][0], triangle[0][1]), (triangle[1][0], triangle[1][1]),
                                     (triangle[2][0], triangle[2][1])])

                linewidth = 0
                if linewidth > 0:
                    v1 = triangle[0]
                    v2 = triangle[1]
                    v3 = triangle[2]
                    pygame.draw.line(disp, BLACK, (v1[0], v1[1]), (v2[0], v2[1]), width=linewidth)
                    pygame.draw.line(disp, BLACK, (v1[0], v1[1]), (v3[0], v3[1]), width=linewidth)
                    pygame.draw.line(disp, BLACK, (v2[0], v2[1]), (v3[0], v3[1]), width=linewidth)

            # pygame.draw.polygon(disp, color, [(v1[0], v1[1]), (v2[0], v2[1]), (v3[0], v3[1])])


# finds the point at which a line intersects a plane
def planeIntersectPoint(plane_p, plane_n, line_s, line_e):
    plane_n = o_normalize(plane_n)
    plane_d = -1 * dotProd(plane_n, plane_p)
    ad = dotProd(line_s, plane_n)
    bd = dotProd(line_e, plane_n)
    if bd - ad != 0:
        t = (-plane_d - ad) / (bd - ad)
        lineStartToEnd = [a - b for a, b in zip(line_e, line_s)]
        lineIntersect = [i * t for i in lineStartToEnd]
        returnV = [a + b for a, b in zip(line_s, lineIntersect)]
        return np.array([returnV[0], returnV[1], returnV[2], 1])
    else:
        return np.array([0, 0, 0, 1])


def planeIntersectPointIndividual(plane_p, plane_n, line_s, line_e):
    plane_n = o_normalize(plane_n)
    plane_d = -1 * dotProd(plane_n, plane_p)
    ad = dotProd(line_s, plane_n)
    bd = dotProd(line_e, plane_n)
    if bd - ad != 0:
        t = (-plane_d - ad) / (bd - ad)
        lineStartToEnd = [a - b for a, b in zip(line_e, line_s)]
        lineIntersect = [i * t for i in lineStartToEnd]
        returnV = [a + b for a, b in zip(line_s, lineIntersect)]
        return np.array([returnV[0], returnV[1], returnV[2]])
    else:
        return np.array([0, 0, 0])


def triangleClip(plane_p, plane_n, verts, dists, light_info, normal_info, toPrint=False):
    global clippedVerticesEmpty, clippedNormalInfo, clippedLightInfo
    # to keep track of the index to assign new vertices to. this is to avoid using an apped, which copies the entire array
    emptyListIndex = 0
    lightindex = 0
    normalindex = 0
    isInner = dists > 0
    numInner = 0
    numOuter = 0
    for i in range(0, len(verts), 3):
        # contains truth or false values
        isInnerSlice = isInner[i:i + 3]
        numInner = np.sum(isInnerSlice)
        numOuter = 3 - numInner
        innerPoints = []
        outerPoints = []
        for j in range(3):
            if isInnerSlice[j]:
                innerPoints.append(verts[i + j])
            else:
                outerPoints.append(verts[i + j])
        if numInner == 0:
            continue
        if numInner == 3:
            clippedVerticesEmpty[emptyListIndex] = verts[i]
            clippedVerticesEmpty[emptyListIndex + 1] = verts[i + 1]
            clippedVerticesEmpty[emptyListIndex + 2] = verts[i + 2]
            # assigning light info on a per triangle basis
            clippedLightInfo[lightindex] = light_info[i // 3]
            clippedNormalInfo[normalindex] = normal_info[i // 3]
            normalindex += 1
            lightindex += 1
            emptyListIndex += 3
        if numInner == 1 and numOuter == 2:
            clippedVerticesEmpty[emptyListIndex] = innerPoints[0]
            clippedVerticesEmpty[emptyListIndex + 1] = planeIntersectPoint(plane_p, plane_n, innerPoints[0],
                                                                           outerPoints[0])

            clippedVerticesEmpty[emptyListIndex + 2] = planeIntersectPoint(plane_p, plane_n, innerPoints[0],
                                                                           outerPoints[1])
            clippedLightInfo[lightindex] = light_info[i // 3]
            clippedNormalInfo[normalindex] = normal_info[i // 3]
            normalindex += 1
            lightindex += 1
            emptyListIndex += 3
        if numInner == 2 and numOuter == 1:
            # first triangle
            clippedVerticesEmpty[emptyListIndex] = innerPoints[0]
            clippedVerticesEmpty[emptyListIndex + 1] = innerPoints[1]
            clippedVerticesEmpty[emptyListIndex + 2] = planeIntersectPoint(plane_p, plane_n, innerPoints[0],
                                                                           outerPoints[0])

            clippedLightInfo[lightindex] = light_info[i // 3]
            clippedNormalInfo[normalindex] = normal_info[i // 3]
            normalindex += 1
            lightindex += 1
            emptyListIndex += 3
            # second triangle
            clippedVerticesEmpty[emptyListIndex] = innerPoints[1]
            clippedVerticesEmpty[emptyListIndex + 1] = clippedVerticesEmpty[emptyListIndex - 1]
            clippedVerticesEmpty[emptyListIndex + 2] = planeIntersectPoint(plane_p, plane_n, innerPoints[1],
                                                                           outerPoints[0])

            clippedLightInfo[lightindex] = light_info[i // 3]
            clippedNormalInfo[normalindex] = normal_info[i // 3]
            normalindex += 1
            lightindex += 1
            emptyListIndex += 3
    return clippedVerticesEmpty[0:emptyListIndex]


# for clipping single triangles (sides)
# returns 2d array
def triangleClipIndividual(plane_p, plane_n, verts, dists):
    numInner = 0
    numOuter = 0

    innerPoints = [1, 2, 3]
    outerPoints = [1, 2, 3]
    for i in range(3):
        if dists[i] > 0:
            innerPoints[numInner] = verts[i]
            numInner += 1
        else:
            outerPoints[numOuter] = verts[i]
            numOuter += 1

    if numInner == 0:
        return []
    if numInner == 3:
        return [verts]
    if numInner == 1 and numOuter == 2:
        v1 = innerPoints[0]
        v2 = planeIntersectPointIndividual(plane_p, plane_n, innerPoints[0], outerPoints[0])
        v3 = planeIntersectPointIndividual(plane_p, plane_n, innerPoints[0], outerPoints[1])

        return [[v1, v2, v3]]
    if numInner == 2 and numOuter == 1:
        v1 = innerPoints[0]
        v2 = innerPoints[1]
        v3 = planeIntersectPointIndividual(plane_p, plane_n, innerPoints[0], outerPoints[0])

        v4 = innerPoints[1]
        v5 = np.copy(v3)
        v6 = planeIntersectPointIndividual(plane_p, plane_n, innerPoints[1], outerPoints[0])

        return [[v1, v2, v3], [v4, v5, v6]]


def get2DProjection(verts):
    rotatedTarget = o_normalize(rotateVectorY(targetVector, yRot))
    global cameraLook
    global camera
    cameraLook = rotatedTarget
    target = camera + cameraLook
    matCam = PointAtMatrix(camera, target, upDir)
    matView = LookAtMatrix(matCam)

    # transforming every vertex away from the camera
    transformedVertices = verts + zTransform
    # creating lines for finding normals using row by row array slicing
    # a[2::3] gives us the third vertex of each tri, a[1::3] gives us the second, and a[0::3] gives the first
    # so we dont perform slicing twice
    firstPointOfTriangles = transformedVertices[0::3, 0:3]
    firstLines = transformedVertices[1::3, 0:3] - firstPointOfTriangles
    secondLines = transformedVertices[2::3, 0:3] - firstPointOfTriangles

    # calculating normals for each triangle. this will have length of verts/3
    normals = np.cross(firstLines, secondLines)
    normals = normalize(normals, axis=1)

    light = np.array([-cameraLook[0], -cameraLook[1], -cameraLook[2]])
    light = o_normalize(light)
    # calculating the row wise dot product, of the normals and the light, has the same length as the normals, can be considered to be indexed along with the first point in each triangle
    light_alignments = np.sum(normals * light, axis=1)
    normalsAlignment = np.sum(normals * (transformedVertices[0::3, 0:3] - camera), axis=1)
    # performing matrix calculations on every single vertex, to avoid looping. try looping?
    transformedViewVertices = np.matmul(transformedVertices, matView)

    # setting up clipping for the forward z axis by storing the distance of each vertex from the plane (signed)
    zClipDistances = np.sum(transformedViewVertices[::, 0:3] * forwardNormal,
                            axis=1) - dotProductForwardPlaneVectorNormal
    clippedVertices = triangleClip(forwardPlaneVector, forwardNormal, transformedViewVertices, zClipDistances,
                                   light_alignments, normalsAlignment)
    renderTris(clippedVertices, clippedLightInfo, clippedNormalInfo)


run = True
pr = cProfile.Profile()
pr.enable()
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        camera[1] += 0.01 * clock.get_time()
    if keys[pygame.K_LCTRL]:
        camera[1] -= 0.01 * clock.get_time()
    forwardDir = cameraLook * 0.01 * clock.get_time()
    if keys[pygame.K_w]:
        camera += forwardDir
    if keys[pygame.K_s]:
        camera -= forwardDir
    rightDir = np.cross(forwardDir, [0, 1, 0]) * clock.get_time() * 0.09
    if keys[pygame.K_a]:
        camera = camera - rightDir
    if keys[pygame.K_d]:
        camera = camera + rightDir
    if keys[pygame.K_RIGHT]:
        yRot += 0.002 * clock.get_time()
    if keys[pygame.K_LEFT]:
        yRot -= 0.002 * clock.get_time()
    disp.fill(BLACK)
    get2DProjection(triangles)
    pygame.display.update()
    clock.tick()
    pygame.display.set_caption(f'FPS: {clock.get_fps()}')
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
