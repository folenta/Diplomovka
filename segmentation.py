import cv2
import numpy as np
from collections import Counter
from bresenham import bresenham


def savePalmprintSegmented(image):
    cv2.imwrite('segmented.bmp', image)


def getWhitePixelRatio(block, totalPixels):
    """ Ziska percento (nie 50% ale 0.5) bielych pixelov v danom bloku """
    limit = 127  # TODO - urcite dajak upravit

    x, y = (block > limit).nonzero()
    values = block[x, y]

    wpp = len(values) / totalPixels

    return wpp


def segmentationSmoothing(blockImage, blocks):
    """ Vyhladi to napriklad vnutro odltacku kde mozu byt dajake bloky patriace do pozadia """
    blockRows, blockCols, _, _ = blockImage.shape

    for row in range(0, blockRows):
        for col in range(0, blockCols):
            if blocks[row][col]["background"]:
                topRow = max(0, row - 1)
                bottomRow = min(row + 1, blockRows - 1)
                leftCol = max(0, col - 1)
                rightCol = min(col + 1, blockCols - 1)

                filterValue = blocks[topRow][leftCol]["background"] + 2 * blocks[topRow][col]["background"] + \
                              blocks[topRow][rightCol]["background"] + 2 * blocks[row][leftCol]["background"] + \
                              2 * blocks[row][rightCol]["background"] + blocks[bottomRow][leftCol]["background"] + \
                              2 * blocks[bottomRow][col]["background"] + blocks[bottomRow][rightCol]["background"]

                """ Ked z 8 susednych blokov je aspon 5 blokov odtlacku, tak aj aktualny blok bude v odtlacku """
                if filterValue < 6:
                    blocks[row][col]["background"] = 0

    return blocks


def fillNeighborsWithIndex(foregroundRegions, currentBlock, index, blockImage, blocks):
    blockRows, blockCols, _, _ = blockImage.shape
    toSearch = [currentBlock]

    """ Vyplni vsetky susedne regiony (stale sa pozera na 4 susedov) rovnakym indexom """
    while toSearch:
        currentRow, currentCol = toSearch.pop(0)

        topRow = max(0, currentRow - 1)
        bottomRow = min(currentRow + 1, blockRows - 1)
        leftCol = max(0, currentCol - 1)
        rightCol = min(currentCol + 1, blockCols - 1)
        
        if blocks[topRow][currentCol]["background"] == 0:
            if foregroundRegions[topRow][currentCol] == 0:
                foregroundRegions[topRow][currentCol] = index
                toSearch.append((topRow, currentCol))

        if blocks[currentRow][rightCol]["background"] == 0:
            if foregroundRegions[currentRow][rightCol] == 0:
                foregroundRegions[currentRow][rightCol] = index
                toSearch.append((currentRow, rightCol))

        if blocks[bottomRow][currentCol]["background"] == 0:
            if foregroundRegions[bottomRow][currentCol] == 0:
                foregroundRegions[bottomRow][currentCol] = index
                toSearch.append((bottomRow, currentCol))

        if blocks[currentRow][leftCol]["background"] == 0:
            if foregroundRegions[currentRow][leftCol] == 0:
                foregroundRegions[currentRow][leftCol] = index
                toSearch.append((currentRow, leftCol))

    return foregroundRegions


def findRegionWithMaxSize(foregroundRegions, index):
    """ Najde region s najvacsim poctom blokov """
    maxSize = 0
    regionWithMaxSize = 0
    for indexRegion in range(1, index):
        regionSize = (foregroundRegions == indexRegion).sum()
        if regionSize > maxSize:
            maxSize = regionSize
            regionWithMaxSize = indexRegion

    return regionWithMaxSize


def putSmallerRegionsIntoBackground(blockRows, blockCols, foregroundRegions, palmprintRegion, blocks):
    """ Vsetky bloky ktore nepatria do najvacsieho regionu sa nastavia ako pozadie """
    for row in range(blockRows):
        for col in range(blockCols):
            if foregroundRegions[row][col] != 0 and foregroundRegions[row][col] != palmprintRegion:
                blocks[row][col]["background"] = 1

    return blocks


def checkFingerTip(fingerTips, point, lastPalmprintRow):
    x = point[1]
    y = point[0]
    alreadyFound = False

    if x < lastPalmprintRow - 10:
        if len(fingerTips) == 0:
            fingerTips.append((x, y))
        else:
            for fingerTip in fingerTips:
                xx = fingerTip[0]
                yy = fingerTip[1]

                distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

                if distance <= 10:
                    alreadyFound = True

            if not alreadyFound:
                fingerTips.append((x, y))

    return fingerTips


def checkPointBetweenFingers(pointsBetweenFingers, point, lastPalmprintRow):
    x = point[1]
    y = point[0]

    if x < lastPalmprintRow - 10:
        pointsBetweenFingers.append((x, y))

    return pointsBetweenFingers


def findLastRowOfMask(palmprintMask):
    rows, cols = palmprintMask.shape
    lastRow = 0

    for row in range(rows):
        for col in range(cols):
            if palmprintMask[row][col] == 1:
                lastRow = row
                break

    return lastRow


def findBorderBetweenFingers(palmprintBorder, startingPoint, blocks, palmprintSegment):
    startingPointIndex = palmprintBorder.index(startingPoint)
    pointsBetweenFingers = []

    """ Hlada v lavom smere """
    currentPoint = startingPoint
    currentPointRow, currentPointCol = currentPoint
    pointsBetweenFingers.append(currentPoint)
    blocks[currentPointRow][currentPointCol]["palmprintSegment"] = palmprintSegment

    for distanceFromStartingPoint in range(1, len(palmprintBorder)):
        nextBorderPoint = palmprintBorder[startingPointIndex + distanceFromStartingPoint]
        nextBorderPointRow, nextBorderPointCol = nextBorderPoint
        if nextBorderPointCol >= currentPointCol:
            if nextBorderPointRow < currentPointRow:
                break
        pointsBetweenFingers.append(nextBorderPoint)
        blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = palmprintSegment
        currentPointRow, currentPointCol = nextBorderPoint

    """ Hlada v pravom smere """
    currentPoint = startingPoint
    currentPointRow, currentPointCol = currentPoint
    for distanceFromStartingPoint in range(1, len(palmprintBorder)):
        nextBorderPoint = palmprintBorder[startingPointIndex - distanceFromStartingPoint]
        nextBorderPointRow, nextBorderPointCol = nextBorderPoint
        if nextBorderPointCol <= currentPointCol:
            if nextBorderPointRow < currentPointRow:
                break
        pointsBetweenFingers.append(nextBorderPoint)
        blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = palmprintSegment
        currentPointRow, currentPointCol = nextBorderPoint

    pointsBetweenFingers.sort(key=lambda x: x[1])

    return blocks, pointsBetweenFingers


def cutFinger(startingPoint, endingPoint, palmprintBorder, palmprintMask, blockImage, blocks):
    blockRows, blockCols, _, _ = blockImage.shape

    startingPointIndex = palmprintBorder.index(startingPoint)
    endingPointIndex = palmprintBorder.index(endingPoint)
    toSearch = [palmprintBorder[startingPointIndex + 1], palmprintBorder[endingPointIndex - 1]]

    newFingerBorder = list(bresenham(startingPoint[0], startingPoint[1], endingPoint[0], endingPoint[1]))

    if startingPointIndex > endingPointIndex:
        del palmprintBorder[startingPointIndex:len(palmprintBorder)]
        del palmprintBorder[0:endingPointIndex + 1]
        startingPointIndex = 0
    else:
        del palmprintBorder[startingPointIndex:endingPointIndex + 1]

    distanceFromStartingPoint = 0

    for borderPoint in newFingerBorder:
        #blockImage[borderPoint[0]][borderPoint[1]] = 0
        palmprintMask[borderPoint[0]][borderPoint[1]] = 0
        palmprintBorder.insert(startingPointIndex + distanceFromStartingPoint, borderPoint)
        distanceFromStartingPoint += 1

    """ Vyplni vsetky susedne regiony (stale sa pozera na 4 susedov) rovnakym indexom """
    while toSearch:
        currentRow, currentCol = toSearch.pop(0)

        topRow = max(0, currentRow - 1)
        bottomRow = min(currentRow + 1, blockRows - 1)
        leftCol = max(0, currentCol - 1)
        rightCol = min(currentCol + 1, blockCols - 1)

        if palmprintMask[topRow][currentCol] == 1:
            toSearch.append((topRow, currentCol))
            #blockImage[topRow][currentCol] = 255
            blocks[topRow][currentCol]["background"] = 1
            palmprintMask[topRow][currentCol] = 0

        if palmprintMask[currentRow][rightCol] == 1:
            toSearch.append((currentRow, rightCol))
            #blockImage[currentRow][rightCol] = 255
            blocks[currentRow][rightCol]["background"] = 1
            palmprintMask[currentRow][rightCol] = 0

        if palmprintMask[bottomRow][currentCol] == 1:
            toSearch.append((bottomRow, currentCol))
            #blockImage[bottomRow][currentCol] = 255
            blocks[bottomRow][currentCol]["background"] = 1
            palmprintMask[bottomRow][currentCol] = 0

        if palmprintMask[currentRow][leftCol] == 1:
            toSearch.append((currentRow, leftCol))
            #blockImage[currentRow][leftCol] = 255
            blocks[currentRow][leftCol]["background"] = 1
            palmprintMask[currentRow][leftCol] = 0

    return blocks, palmprintBorder, palmprintMask


def findFingerTips(palmprintMask, palmprintBorder, blockImage, contour):
    lastPalmprintRow = findLastRowOfMask(palmprintMask)

    fingerTips = []
    pointsBetweenFingers = []

    hull = cv2.convexHull(contour, returnPoints=False)

    defects = cv2.convexityDefects(contour, hull)

    for i in range(defects.shape[0]):  # calculate the angle
        s, e, f, d = defects[i][0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
        if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
            fingerTips = checkFingerTip(fingerTips, start, lastPalmprintRow)
            fingerTips = checkFingerTip(fingerTips, end, lastPalmprintRow)
            pointsBetweenFingers = checkPointBetweenFingers(pointsBetweenFingers, far, lastPalmprintRow)

    # cv2.drawContours(convexImg, [hull], -1, (0, 255, 255), 2)

    fingerTips.sort(key=lambda x: x[1])
    pointsBetweenFingers.sort(key=lambda x: x[1])

    if len(fingerTips) == 5:
        maxValue = max(fingerTips, key=lambda t: t[0])
        maxValueIndex = fingerTips.index(maxValue)

        if maxValueIndex == 0:
            print("RIGHT")
            if len(pointsBetweenFingers) == 4:
                startingPoint = pointsBetweenFingers[1]
                startingPointIndex = palmprintBorder.index(startingPoint)

                between23 = []
                currentPoint = startingPoint
                currentPointRow, currentPointCol = currentPoint
                between23.append(currentPoint)
                for distanceFromStartingPoint in range(1, len(palmprintBorder)):
                    nextBorderPoint = palmprintBorder[startingPointIndex + distanceFromStartingPoint]
                    nextBorderPointRow, nextBorderPointCol = nextBorderPoint
                    if nextBorderPointCol >= currentPointCol:
                        if nextBorderPointRow < currentPointRow:
                            break
                    between23.append(nextBorderPoint)
                    blockImage[nextBorderPointRow][nextBorderPointCol] = 0
                    currentPointRow, currentPointCol = nextBorderPoint

                between23.sort(key=lambda x: x[1])
                startingPoint2 = between23[0]
                startingPointIndex2 = palmprintBorder.index(startingPoint2)
                startingPoint2Row, startingPoint2Col = startingPoint2

                minDistance = 100
                minDistanceIndex = 0

                for distanceFromStartingPoint in range(10, len(palmprintBorder)):
                    nextBorderPoint = palmprintBorder[startingPointIndex2 + distanceFromStartingPoint]
                    nextBorderPointRow, nextBorderPointCol = nextBorderPoint
                    if startingPoint2Row - 5 <= nextBorderPointRow <= startingPoint2Row + 5:
                        distance = np.sqrt((nextBorderPointRow - startingPoint2Row) ** 2 + (nextBorderPointCol -
                                                                                            startingPoint2Col) ** 2)

                        if distance < minDistance:
                            minDistance = distance
                            minDistanceIndex = distanceFromStartingPoint

                    if nextBorderPointRow > startingPoint2Row + 5:
                        break

                nextPoint = palmprintBorder[startingPointIndex2 + minDistanceIndex]
                blockImage[nextPoint[0]][nextPoint[1]] = 0

                newBorder = list(bresenham(startingPoint2Row, startingPoint2Col, nextPoint[0], nextPoint[1]))

                for borderPoint in newBorder:
                    blockImage[borderPoint[0]][borderPoint[1]] = 0
                    palmprintMask[borderPoint[0]][borderPoint[1]] = 0

                palmprintMask[startingPoint2Row][startingPoint2Col] = 0
                palmprintMask[nextPoint[0]][nextPoint[1]] = 0

                blockImage = cutFinger(palmprintBorder[startingPointIndex2 + 1], palmprintMask, blockImage)


                currentPoint = startingPoint
                currentPointRow, currentPointCol = currentPoint
                for distanceFromStartingPoint in range(1, len(palmprintBorder)):
                    nextBorderPoint = palmprintBorder[startingPointIndex - distanceFromStartingPoint]
                    nextBorderPointRow, nextBorderPointCol = nextBorderPoint
                    if nextBorderPointCol <= currentPointCol:
                        if nextBorderPointRow < currentPointRow:
                            break
                    blockImage[nextBorderPointRow][nextBorderPointCol] = 0
                    currentPointRow, currentPointCol = nextBorderPoint

                blockImage = findBorderBetweenFingers(palmprintBorder, pointsBetweenFingers[2], blockImage)
                blockImage = findBorderBetweenFingers(palmprintBorder, pointsBetweenFingers[3], blockImage)

        if maxValueIndex == 4:
            print("LEFT")

    for fingerTip in fingerTips:
        blockImage[fingerTip[0]][fingerTip[1]] = 0

    for pointBetweenFingers in pointsBetweenFingers:
        blockImage[pointBetweenFingers[0]][pointBetweenFingers[1]] = 0

    image = mergeBlocksToImage(blockImage)

    """ Zmensenie vystupu aby sa zmestil na obrazovku """
    scale_percent = 15  # percent of original size
    imageHeight, imageWidth = image.shape

    height = int(imageHeight * scale_percent / 100)
    width = int(imageWidth * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('image', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    exit(2)


def findPalmprintBorder(palmprintMask, blockImage):
    contours, _ = cv2.findContours(palmprintMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    palmprintBorder = []

    contour = contours[0]

    for point in contour:
        p = point[0]
        y, x = p
        #blockImage[x][y] = 100
        palmprintBorder.append((x, y))

    return palmprintBorder, blockImage, contour


def findFingerPoints(palmprintMask, contour):
    lastPalmprintRow = findLastRowOfMask(palmprintMask)

    fingerTips = []
    pointsBetweenFingers = []

    hull = cv2.convexHull(contour, returnPoints=False)

    defects = cv2.convexityDefects(contour, hull)

    for i in range(defects.shape[0]):  # calculate the angle
        s, e, f, d = defects[i][0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
        if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
            fingerTips = checkFingerTip(fingerTips, start, lastPalmprintRow)
            fingerTips = checkFingerTip(fingerTips, end, lastPalmprintRow)
            pointsBetweenFingers = checkPointBetweenFingers(pointsBetweenFingers, far, lastPalmprintRow)

    fingerTips.sort(key=lambda x: x[1])
    pointsBetweenFingers.sort(key=lambda x: x[1])

    return fingerTips, pointsBetweenFingers


def findHandOrientation(fingerTips):
    if len(fingerTips) == 5:
        maxValue = max(fingerTips, key=lambda t: t[0])
        maxValueIndex = fingerTips.index(maxValue)

        if maxValueIndex == 0:
            return "Right"

        if maxValueIndex == 4:
            return "Left"

    return "None"


def findPointOnTheOtherSideOfFinger(startingPoint, palmprintBorder, blocks, reverse, palmprintSegment):
    startingPointIndex = palmprintBorder.index(startingPoint)
    startingPointRow, startingPointCol = startingPoint

    minDistance = 100
    minDistanceIndex = 0

    for distanceFromStartingPoint in range(10, len(palmprintBorder)):
        if not reverse:
            nextBorderPoint = palmprintBorder[startingPointIndex + distanceFromStartingPoint]
        else:
            nextBorderPoint = palmprintBorder[startingPointIndex - distanceFromStartingPoint]
        nextBorderPointRow, nextBorderPointCol = nextBorderPoint
        if startingPointRow - 5 <= nextBorderPointRow <= startingPointRow + 5:
            distance = np.sqrt((nextBorderPointRow - startingPointRow) ** 2 + (nextBorderPointCol -
                                                                               startingPointCol) ** 2)

            if distance < minDistance:
                minDistance = distance
                minDistanceIndex = distanceFromStartingPoint

        if nextBorderPointRow > startingPointRow + 5:
            break

    if not reverse:
        endingPoint = palmprintBorder[startingPointIndex + minDistanceIndex]
    else:
        endingPoint = palmprintBorder[startingPointIndex - minDistanceIndex]

    blocks[endingPoint[0]][endingPoint[1]]["palmprintSegment"] = palmprintSegment

    return endingPoint, blocks


def findPointsBetweenThumb(startingPoint, palmprintBorder, blockImage, blocks, reverse, palmprintSegment):
    startingPointIndex = palmprintBorder.index(startingPoint)
    pointsBetweenFingers = []

    """ Hlada v lavom smere """
    currentPoint = startingPoint
    currentPointRow, currentPointCol = currentPoint
    pointsBetweenFingers.append(currentPoint)

    for distanceFromStartingPoint in range(1, len(palmprintBorder)):
        nextBorderPoint = palmprintBorder[startingPointIndex + distanceFromStartingPoint]
        nextBorderPointRow, nextBorderPointCol = nextBorderPoint
        if nextBorderPointRow < currentPointRow:
            break
        pointsBetweenFingers.append(nextBorderPoint)
        #blockImage[nextBorderPointRow][nextBorderPointCol] = 0
        blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = palmprintSegment
        currentPointRow, currentPointCol = nextBorderPoint

    startingPoint = pointsBetweenFingers[-1]
    startingPointIndex = palmprintBorder.index(startingPoint)
    startingPointRow, startingPointCol = startingPoint

    minDistance = 100
    minDistanceIndex = 0

    for distanceFromStartingPoint in range(10, len(palmprintBorder)):
        if not reverse:
            nextBorderPoint = palmprintBorder[startingPointIndex + distanceFromStartingPoint]
        else:
            nextBorderPoint = palmprintBorder[startingPointIndex - distanceFromStartingPoint]
        nextBorderPointRow, nextBorderPointCol = nextBorderPoint
        if startingPointCol - 5 <= nextBorderPointCol <= startingPointCol + 5:
            distance = np.sqrt((nextBorderPointRow - startingPointRow) ** 2 + (nextBorderPointCol -
                                                                               startingPointCol) ** 2)

            if distance < minDistance:
                minDistance = distance
                minDistanceIndex = distanceFromStartingPoint

        if nextBorderPointCol > startingPointCol + 5:
            break

    if not reverse:
        endingPoint = palmprintBorder[startingPointIndex + minDistanceIndex]
    else:
        endingPoint = palmprintBorder[startingPointIndex - minDistanceIndex]

    #blockImage[endingPoint[0]][endingPoint[1]] = 0

    return endingPoint, pointsBetweenFingers, blockImage, blocks


def detectPalmprint(blockImage, blocks, image):
    blockRows, blockCols, _, _ = blockImage.shape

    foregroundRegions = np.zeros((blockRows, blockCols))

    index = 1

    """ Postupne prechadza vsetky bloky a ked najde blok patriaci do popredia tak ho oznaci indexom a vyplni rovnakym
        indexom vsetky susedne bloky v popredi - takym sposobom vznikne niekolko regionov """
    for row in range(blockRows):
        for col in range(blockCols):
            if blocks[row][col]["background"] == 0:
                if foregroundRegions[row][col] == 0:
                    foregroundRegions[row][col] = index
                    foregroundRegions = fillNeighborsWithIndex(foregroundRegions, (row, col), index, blockImage, blocks)
                    index += 1

    """ Ked su aspon 2 najdene regiony (tym nultym regionom je pozadie), tak najde najvacsi a zvysne bloky da do
        pozadia """
    if index > 2:
        palmprintRegion = findRegionWithMaxSize(foregroundRegions, index)
        blocks = putSmallerRegionsIntoBackground(blockRows, blockCols, foregroundRegions, palmprintRegion, blocks)

    """ Najde vsetky hranice pozadia a odtlacku """
    palmprintMask = np.zeros((blockRows, blockCols), dtype=np.uint8)

    """ Najde masku odtlacku dlane """
    for row in range(blockRows):
        for col in range(blockCols):
            if blocks[row][col]["background"] == 0:
                palmprintMask[row][col] = 1

    """ Spodok dlane to oreze, aby jej sirka bola apson 25 pixelov """
    for row in range(blockRows):
        length = 0
        firstFound = False
        secondFound = False
        for col in range(blockCols):
            if palmprintMask[blockRows - row - 1][col] == 0 and firstFound:
                break
            if palmprintMask[blockRows - row - 1][col] == 1:
                firstFound = True
                length += 1
        if firstFound and length <= 25:
            for col in range(blockCols):
                palmprintMask[blockRows - row - 1][col] = 0
        if firstFound and length >= 25:
            break


    """ Najde to kontury odtlacku (pozadie vo vnutri odtlacku ignoruje) """
    palmprintBorder, blockImage, contour = findPalmprintBorder(palmprintMask, blockImage)

    """for borderPoint in palmprintBorder:
        blockImage[borderPoint[0]][borderPoint[1]] = 0

    image = mergeBlocksToImage(blockImage)

    scale_percent = 15  # percent of original size
    imageHeight, imageWidth = image.shape

    height = int(imageHeight * scale_percent / 100)
    width = int(imageWidth * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('image', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    exit(2)"""

    """ Najde konceky prstov a body medzi prstami """
    fingerTips, pointsBetweenFingers = findFingerPoints(palmprintMask, contour)

    handOrientation = findHandOrientation(fingerTips)


    if handOrientation == "Right":
        if len(pointsBetweenFingers) == 4:
            """ Najdenie hranice medzi prstom 2 a 3 (ukazovak a prostrednik), najdenie bodu na druhej strane 
                ukazovaka a odstrihnutie ukazovaka"""
            blocks, between23 = findBorderBetweenFingers(palmprintBorder, pointsBetweenFingers[1], blocks, 11)
            leftPointOfFinger2, blocks = findPointOnTheOtherSideOfFinger(between23[0], palmprintBorder, blocks, False, 13)
            blocks, palmprintBorder, palmprintMask = cutFinger(between23[0], leftPointOfFinger2, palmprintBorder,
                                                                   palmprintMask, blockImage, blocks)

            """ Najdenie hranice medzi prstom 3 a 4 (prostrednik a prstenik), a odstrihnutie prostrednika """
            blocks, between34 = findBorderBetweenFingers(palmprintBorder, pointsBetweenFingers[2], blocks, 9)
            blocks, palmprintBorder, palmprintMask = cutFinger(between34[0], between23[-1], palmprintBorder,
                                                                   palmprintMask, blockImage, blocks)

            """ Najdenie hranice medzi prstom 4 a 5 (prstenik a malicek), a odstrihnutie prstenika """
            blocks, between45 = findBorderBetweenFingers(palmprintBorder, pointsBetweenFingers[3], blocks, 7)
            blocks, palmprintBorder, palmprintMask = cutFinger(between45[0], between34[-1], palmprintBorder,
                                                                   palmprintMask, blockImage, blocks)

            """ Najdenie druhej strany malicka a odstrihnutie malicka """
            rightPointOfFinger5, blocks = findPointOnTheOtherSideOfFinger(between45[-1], palmprintBorder,
                                                                              blocks, True, 5)
            blocks, palmprintBorder, palmprintMask = cutFinger(rightPointOfFinger5, between45[-1], palmprintBorder,
                                                                   palmprintMask, blockImage, blocks)

            """ Najdenie oboch stran palca a jeho odstrihnutie """
            bottomPointOfThumb, between12, blockImage, blocks = findPointsBetweenThumb(leftPointOfFinger2, palmprintBorder,
                                                                               blockImage, blocks, False, 13)
            blocks, palmprintBorder, palmprintMask = cutFinger(between12[-1], bottomPointOfThumb, palmprintBorder,
                                                                   palmprintMask, blockImage, blocks)

            """ Vyplnenie medzi dolnym bodom palca a spodkom dlane """
            startingPointIndex = palmprintBorder.index(bottomPointOfThumb)
            startingPointRow, startingPointCol = bottomPointOfThumb
            for distanceFromStartingPoint in range(1, 22):
                nextBorderPoint = palmprintBorder[startingPointIndex + distanceFromStartingPoint]
                nextBorderPointRow, nextBorderPointCol = nextBorderPoint
                pointsBetweenFingers.append(nextBorderPoint)
                # blockImage[nextBorderPointRow][nextBorderPointCol] = 0
                blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = 1
                currentPointRow, currentPointCol = nextBorderPoint

            """ Vyplnenie medzi pravym bodom malicka a spodkom dlane """
            startingPointIndex = palmprintBorder.index(rightPointOfFinger5)
            startingPointRow, startingPointCol = rightPointOfFinger5
            for distanceFromStartingPoint in range(1, 38):
                nextBorderPoint = palmprintBorder[startingPointIndex - distanceFromStartingPoint]
                nextBorderPointRow, nextBorderPointCol = nextBorderPoint
                pointsBetweenFingers.append(nextBorderPoint)
                # blockImage[nextBorderPointRow][nextBorderPointCol] = 0
                blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = 5
                currentPointRow, currentPointCol = nextBorderPoint

    for borderPoint in palmprintBorder:
        blocks[borderPoint[0]][borderPoint[1]]["background"] = 2

    """image = mergeBlocksToImage(blockImage)

    scale_percent = 15  # percent of original size
    imageHeight, imageWidth = image.shape

    height = int(imageHeight * scale_percent / 100)
    width = int(imageWidth * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('image', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    exit(2)"""
    #findFingerTips(palmprintMask, palmprintBorder, blockImage, contour)

    return blocks, blockImage, leftPointOfFinger2, between23, between34, between45, rightPointOfFinger5


def mergeBlocksToImage(blockImage):
    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    imageHeight = blockRows * blockHeight
    imageWidth = blockCols * blockWidth

    """ Spojenie blokov do celkoveho obrazu"""
    image = blockImage.reshape(imageHeight // blockHeight, imageWidth // blockWidth, blockHeight, blockWidth)
    image = image.swapaxes(1, 2).reshape(imageHeight, imageWidth)

    return image


def showSegmentedPalmprint(blockImage, blocks):
    blockRows, blockCols, _, _ = blockImage.shape

    """ Vsetkym pixelom v pozadi nastavi hodnotu 0 """
    for row in range(blockRows):
        for col in range(blockCols):
            if blocks[row][col]["background"] == 1:
                pass
                #blockImage[row][col] = 0
            if blocks[row][col]["background"] == 2:
                if blocks[row][col]["palmprintSegment"] > 0:
                    #pass
                    blockImage[row][col] = 0
                else:
                    #pass
                    blockImage[row][col] = 128

    image = mergeBlocksToImage(blockImage)

    savePalmprintSegmented(image)

    """ Zmensenie vystupu aby sa zmestil na obrazovku """
    scale_percent = 15  # percent of original size
    imageHeight, imageWidth = image.shape

    height = int(imageHeight * scale_percent / 100)
    width = int(imageWidth * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('image', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def segmentationSmoothingFingers(blockImage, blocks):
    """ Snazi sa spojit clanky prstov s odtlackom """
    blockRows, blockCols, _, _ = blockImage.shape

    for row in range(0, blockRows):
        for col in range(0, blockCols):
            if blocks[row][col]["background"]:
                topRow = max(0, row - 1)
                bottomRow = min(row + 1, blockRows - 1)
                belowBottom = min(row + 2, blockRows - 1)

                if blocks[topRow][col]["background"] == 0 and (blocks[bottomRow][col]["background"] == 0 or blocks[belowBottom][col]["background"] == 0):
                    blocks[row][col]["background"] = 0

    return blocks


def segmentation(blockImage, blocks, image):
    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    totalPixels = blockHeight * blockWidth

    """ Prechadza postupne vsetky bloky """
    for row in range(blockRows):
        for col in range(blockCols):
            block = blockImage[row][col]
            wpp = getWhitePixelRatio(block, totalPixels)
            var = np.var(block)
            v = wpp - 0.001 * var  # Pozriet sa na koeficienty

            if v > 0.95:
                blocks[row][col]["background"] = 1

    """ Vyhladenie prebehne 3-krat po sebe (mozno sa na to neskor pozriet) --> pre novy dataset zatial netreba """
    #for segmentationPhase in range(3):
    blocks = segmentationSmoothingFingers(blockImage, blocks)
    #blocks = segmentationSmoothing(blockImage, blocks)

    """ Tu to funguje tak ze ked poslem blockImage do funkcie a v nej to prefarbim ta aj ked blockImage nevratim 
        z funkcie tak sa farba ulozi...takze potom zmenit (ked poslem segmentedImage ta OK)"""
    # segmentedImage = blockImage.copy()
    blocks, blockImage, leftOfFinger2, between23, between34, between45, rightOfFinger5 = detectPalmprint(blockImage, blocks, image)

    showSegmentedPalmprint(blockImage, blocks)

    return blocks, leftOfFinger2, between23, between34, between45, rightOfFinger5
