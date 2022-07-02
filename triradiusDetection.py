#! /usr/bin/env python

"""

Implementacia detekcie triradii.

Autor: Bc. Jan Folenta
Email: xfolen00@stud.fit.vutbr.cz
Rok: 2022

"""

import copy
import math
import numpy as np
from collections import Counter

from orientation import findRadonTransform, findPixelsForRadonTransform
from segmentation import findExtremePalmprintPoints


def poincareIndex(blocks, possibleTriradiusBlocks, row, col, angles):
    # Funkcia najde moznost vyskytu triradia pomocou algoritmu Poincare Index

    # Orientacie v susednych pixeloch (orientacie su ukladane v smere hodinovych ruciciek)
    neighboringOrientations = [blocks[row - 1][col - 1]["orientation"], blocks[row - 1][col]["orientation"],
                               blocks[row - 1][col + 1]["orientation"], blocks[row][col + 1]["orientation"],
                               blocks[row + 1][col + 1]["orientation"], blocks[row + 1][col]["orientation"],
                               blocks[row + 1][col - 1]["orientation"], blocks[row][col - 1]["orientation"],
                               blocks[row - 1][col - 1]["orientation"]]

    possibleTriradius = True
    index = 0

    for k in range(8):
        orientation = neighboringOrientations[k]
        nextOrientation = neighboringOrientations[k + 1]

        if orientation == -1 or nextOrientation == -1:
            possibleTriradius = False
            break

        orientationDifference = math.degrees(angles[orientation]) - math.degrees(angles[nextOrientation])

        if orientationDifference - 5 < -90:
            orientationDifference += 180

        if orientationDifference + 5 > 90:
            orientationDifference -= 180

        index += orientationDifference

    if possibleTriradius:
        if -180 - 20 <= index <= -180 + 20:
            possibleTriradiusBlocks.append((row, col))

            return possibleTriradiusBlocks

    return possibleTriradiusBlocks


def findOrientationsInTriradiusBlocks(blockHeight, blockWidth, blockImage, angles, smallBlockRows, smallBlockCols,
                                      firstRow, firstCol, pixelsForRadonTransform):
    # Funkcia najde orientacie v mensich blokov (ktore vznikli rozdelenim povodnych blokov)

    rows, cols = (smallBlockRows, smallBlockCols)
    triradiusRegion = [[-1 for _ in range(cols)] for _ in range(rows)]

    for blockRow in range(smallBlockRows):
        for blockCol in range(smallBlockCols):
            blockImageRow = firstRow + (blockRow * 25) // blockHeight
            blockImageCol = firstCol + (blockCol * 25) // blockWidth
            startRow = (blockRow * 25) % blockHeight
            endRow = ((blockRow + 1) * 25) % blockHeight
            if endRow == 0:
                endRow = blockHeight
            startCol = (blockCol * 25) % blockWidth
            endCol = ((blockCol + 1) * 25) % blockWidth
            if endCol == 0:
                endCol = blockWidth

            block = blockImage[blockImageRow][blockImageCol][startRow:endRow, startCol:endCol]

            pixelOrientations = []
            mean = np.mean(block)

            for row in range(25):
                for col in range(25):
                    value = block[row][col]
                    if value < mean - 10:
                        r = findRadonTransform(row, col, block, angles, pixelsForRadonTransform)
                        pixelOrientations.append(r)

            try:
                finalOrientation = Counter(pixelOrientations).most_common()[0][0]
            except IndexError:
                finalOrientation = -1

            triradiusRegion[blockRow][blockCol] = finalOrientation

    return triradiusRegion


def localizeTriradius(triradiusRegion, smallBlockRows, smallBlockCols, firstRow, firstCol, blockImage, triradiusBlocksMask):
    # Funckia presnejsie lokalizuje triradius

    _, _, blockHeight, blockWidth = blockImage.shape
    maxRow = 0
    maxCol = 0
    maxVal = 0

    # Postupne su prechadzane
    for blockRow in range(smallBlockRows - 1):
        for blockCol in range(smallBlockCols - 1):
            topLeft = triradiusRegion[blockRow][blockCol]
            topRight = triradiusRegion[blockRow][blockCol + 1]
            bottomRight = triradiusRegion[blockRow + 1][blockCol + 1]
            bottomLeft = triradiusRegion[blockRow + 1][blockCol]
            if triradiusBlocksMask[blockRow][blockCol] == 0 or triradiusBlocksMask[blockRow][blockCol + 1] == 0 or \
                    triradiusBlocksMask[blockRow + 1][blockCol + 1] == 0 or triradiusBlocksMask[blockRow + 1][blockCol] == 0:
                continue

            # Najdenie zmeny orientacii v smere hodinovych ruciciek
            diff = 0
            diff += min(abs(topLeft - topRight), abs((topLeft + 12) - topRight))
            diff += min(abs(topRight - bottomRight), abs((topRight + 12) - bottomRight))
            diff += min(abs(bottomRight - bottomLeft), abs((bottomRight + 12) - bottomLeft))
            diff += min(abs(bottomLeft - topLeft), abs((bottomLeft + 12) - topLeft))

            if diff > maxVal:
                maxVal = diff
                maxRow = blockRow
                maxCol = blockCol

    # Najdenie suradnic triradia pre povodny obraz
    blockImageRow = firstRow + (maxRow * 25) // blockHeight
    blockImageCol = firstCol + (maxCol * 25) // blockWidth
    endRow = ((maxRow + 1) * 25) % blockHeight
    if endRow == 0:
        endRow = blockHeight
    endCol = ((maxCol + 1) * 25) % blockWidth
    if endCol == 0:
        endCol = blockWidth

    triradiusX = (blockImageRow * blockHeight) + endRow + 1
    triradiusY = (blockImageCol * blockWidth) + endCol + 1
    triradius = (triradiusX, triradiusY)

    return triradius


def findTriradiusPosition(possibleTriradiusBlocks, blockImage, angles, pixelsForRadonTransform):
    # Funkcia najde najpravdepodobnejsiu poziciu triradia.

    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    possibleTriradiusBlocks = getMostProbableTriradiusBlocks(possibleTriradiusBlocks, blockRows, blockCols, blockHeight,
                                                             blockWidth)

    firstRow = min(possibleTriradiusBlocks, key=lambda t: t[0])[0]
    lastRow = max(possibleTriradiusBlocks, key=lambda t: t[0])[0]
    firstCol = min(possibleTriradiusBlocks, key=lambda t: t[1])[1]
    lastCol = max(possibleTriradiusBlocks, key=lambda t: t[1])[1]

    blocksInHeight = lastRow - firstRow + 1
    blocksInWidth = lastCol - firstCol + 1

    smallBlockRows = blocksInHeight * blockHeight // 25
    smallBlockCols = blocksInWidth * blockWidth // 25

    triradiusBlocksMask = [[0 for i in range(smallBlockCols)] for j in range(smallBlockRows)]

    for row in range(blocksInWidth):
        for col in range(blocksInWidth):
            if (firstRow + row, firstCol + col) in possibleTriradiusBlocks:
                triradiusBlocksMask[2 * row][2 * col] = 1
                triradiusBlocksMask[2 * row + 1][2 * col] = 1
                triradiusBlocksMask[2 * row][2 * col + 1] = 1
                triradiusBlocksMask[2 * row + 1][2 * col + 1] = 1

    triradiusRegion = findOrientationsInTriradiusBlocks(blockHeight, blockWidth, blockImage, angles, smallBlockRows,
                                                        smallBlockCols, firstRow, firstCol, pixelsForRadonTransform)

    triradius = localizeTriradius(triradiusRegion, smallBlockRows, smallBlockCols, firstRow, firstCol, blockImage,
                                  triradiusBlocksMask)

    return triradius


def getMostProbableTriradiusBlocks(possibleTriradiusBlocks, blockRows, blockCols, blockHeight, blockWidth):
    # Funkcia zuzi pocet kandidatov (blokov), v ktorych sa moze vyskytovat triradius

    possibleTriradiusParts = []

    while possibleTriradiusBlocks:
        newTriradiusPart = []
        toSearch = []
        possTriX, possTriY = possibleTriradiusBlocks.pop(0)
        newTriradiusPart.append((possTriX, possTriY))
        if (possTriX, possTriY + 1) in possibleTriradiusBlocks:
            toSearch.append((possTriX, possTriY + 1))
        if (possTriX + 1, possTriY) in possibleTriradiusBlocks:
            toSearch.append((possTriX + 1, possTriY))
        while toSearch:
            tri = toSearch.pop(0)
            triIdx = possibleTriradiusBlocks.index(tri)
            del possibleTriradiusBlocks[triIdx]
            triX, triY = tri
            newTriradiusPart.append((triX, triY))

            if (triX - 1, triY) in possibleTriradiusBlocks:
                if (triX - 1, triY) not in toSearch:
                    toSearch.append((triX - 1, triY))
            if (triX, triY + 1) in possibleTriradiusBlocks:
                if (triX, triY + 1) not in toSearch:
                    toSearch.append((triX, triY + 1))
            if (triX + 1, triY) in possibleTriradiusBlocks:
                if (triX + 1, triY) not in toSearch:
                    toSearch.append((triX + 1, triY))
            if (triX, triY - 1) in possibleTriradiusBlocks:
                if (triX, triY - 1) not in toSearch:
                    toSearch.append((triX, triY - 1))

        possibleTriradiusParts.append(newTriradiusPart)

    if len(possibleTriradiusParts) == 1:
        triradiusPart = possibleTriradiusParts[0]
        possibleTriradiusBlocks = copy.deepcopy(triradiusPart)
        if len(triradiusPart) == 1:
            triradiusBlock = triradiusPart[0]
            triradiusX = (triradiusBlock[0] * blockRows) + (blockHeight // 2)
            triradiusY = (triradiusBlock[1] * blockCols) + (blockWidth // 2)

            # return triradiusX, triradiusY
        else:
            for blockInTriradiusPart in triradiusPart:
                pointX, pointY = blockInTriradiusPart
                if (pointX, pointY + 1) in triradiusPart and (pointX + 1, pointY + 1) in triradiusPart and \
                        (pointX + 1, pointY) in triradiusPart:
                    possibleTriradiusBlocks = [(pointX, pointY), (pointX, pointY + 1), (pointX + 1, pointY),
                                               (pointX + 1, pointY + 1)]
                    break

    else:
        for triradiusPart in possibleTriradiusParts:
            foundSquare = False
            for blockInTriradiusPart in triradiusPart:
                pointX, pointY = blockInTriradiusPart
                if (pointX, pointY + 1) in triradiusPart and (pointX + 1, pointY + 1) in triradiusPart and \
                        (pointX + 1, pointY) in triradiusPart:
                    possibleTriradiusBlocks = [(pointX, pointY), (pointX, pointY + 1), (pointX + 1, pointY),
                                               (pointX + 1, pointY + 1)]
                    foundSquare = True
                    break
            if foundSquare:
                break

        if not foundSquare:
            biggestTriradiusRegion = max(possibleTriradiusParts, key=len)
            possibleTriradiusBlocks = copy.deepcopy(biggestTriradiusRegion)

    return possibleTriradiusBlocks


def saveTriradiusRegion(blocks, triradius, blockHeight, blockWidth):
    # Funkcia ulozi okolie bloku, v ktorom bol detekovany triradius ako triradiovy region (v nom nasledne neprebehne
    # vyhladenie orientacii)

    triradiusBlockRow = triradius[0] // blockHeight
    triradiusBlockCol = triradius[1] // blockWidth

    # Triradiovy region ma velkost 5 x 5 blokov
    for row in range(triradiusBlockRow - 2, triradiusBlockRow + 3):
        for col in range(triradiusBlockCol - 2, triradiusBlockCol + 3):
            blocks[row][col]["triradiusRegion"] = 1

    return blocks


def detectTriradiusA(blocks, blockImage, angles, leftOfFinger2, between23, pixelsForRadonTransform):
    # Funkcia najde triradius A.

    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    possibleTriradiusBlocks = []
    leftOfFingerRow, leftOfFingerCol = leftOfFinger2
    rightOfFingerRow, rightOfFingerCol = between23[(len(between23) // 2) - 1]

    searchRow = max(leftOfFingerRow, rightOfFingerRow)

    for row in range(searchRow + 1, searchRow + 8):
        for col in range(leftOfFingerCol, rightOfFingerCol):
            if blocks[row][col]["orientation"] != -1:
                possibleTriradiusBlocks = poincareIndex(blocks, possibleTriradiusBlocks, row, col, angles)

    if len(possibleTriradiusBlocks) == 0:
        return (0, 0), blocks

    triradius = findTriradiusPosition(possibleTriradiusBlocks, blockImage, angles, pixelsForRadonTransform)
    blocks = saveTriradiusRegion(blocks, triradius, blockHeight, blockWidth)

    return triradius, blocks


def detectTriradiusB(blocks, blockImage, angles, between23, between34, pixelsForRadonTransform):
    # Funkcia najde triradius B.

    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    possibleTriradiusBlocks = []
    leftOfFingerRow, leftOfFingerCol = between23[(len(between23) // 2) - 1]
    rightOfFingerRow, rightOfFingerCol = between34[(len(between34) // 2) + 1]

    for row in range(leftOfFingerRow - 1,  leftOfFingerRow + 7):
        for col in range(leftOfFingerCol, rightOfFingerCol):
            if blocks[row][col]["orientation"] != -1:
                possibleTriradiusBlocks = poincareIndex(blocks, possibleTriradiusBlocks, row, col, angles)

    if len(possibleTriradiusBlocks) == 0:
        return (0, 0), blocks

    triradius = findTriradiusPosition(possibleTriradiusBlocks, blockImage, angles, pixelsForRadonTransform)
    blocks = saveTriradiusRegion(blocks, triradius, blockHeight, blockWidth)

    return triradius, blocks


def detectTriradiusC(blocks, blockImage, angles, between34, between45, pixelsForRadonTransform):
    # Funkcia najde triradius C.

    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    possibleTriradiusBlocks = []
    leftOfFingerRow, leftOfFingerCol = between34[(len(between34) // 2) - 1]
    rightOfFingerRow, rightOfFingerCol = between45[1]

    for row in range(leftOfFingerRow - 1, leftOfFingerRow + 7):
        for col in range(leftOfFingerCol, rightOfFingerCol):
            if blocks[row][col]["orientation"] != -1:
                possibleTriradiusBlocks = poincareIndex(blocks, possibleTriradiusBlocks, row, col, angles)

    if len(possibleTriradiusBlocks) == 0:
        for row in range(rightOfFingerRow - 2, rightOfFingerRow + 3):
            for col in range(leftOfFingerCol, rightOfFingerCol):
                if blocks[row][col]["orientation"] != -1:
                    possibleTriradiusBlocks = poincareIndex(blocks, possibleTriradiusBlocks, row, col, angles)

    if len(possibleTriradiusBlocks) == 0:
        return (0, 0), blocks

    triradius = findTriradiusPosition(possibleTriradiusBlocks, blockImage, angles, pixelsForRadonTransform)
    blocks = saveTriradiusRegion(blocks, triradius, blockHeight, blockWidth)

    return triradius, blocks


def detectTriradiusD(blocks, blockImage, angles, between45, rightOfFinger5, pixelsForRadonTransform):
    # Funkcia najde triradius D.

    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    possibleTriradiusBlocks = []
    leftOfFingerRow, leftOfFingerCol = between45[0]
    rightOfFingerRow, rightOfFingerCol = rightOfFinger5

    for row in range(leftOfFingerRow + 2, leftOfFingerRow + 10):
        for col in range(leftOfFingerCol, rightOfFingerCol):
            if blocks[row][col]["orientation"] != -1:
                possibleTriradiusBlocks = poincareIndex(blocks, possibleTriradiusBlocks, row, col, angles)

    if len(possibleTriradiusBlocks) == 0:
        for row in range(rightOfFingerRow, rightOfFingerRow + 8):
            for col in range(leftOfFingerCol, rightOfFingerCol):
                if blocks[row][col]["orientation"] != -1:
                    possibleTriradiusBlocks = poincareIndex(blocks, possibleTriradiusBlocks, row, col, angles)

    if len(possibleTriradiusBlocks) == 0:
        return (0, 0), blocks

    triradius = findTriradiusPosition(possibleTriradiusBlocks, blockImage, angles, pixelsForRadonTransform)
    blocks = saveTriradiusRegion(blocks, triradius, blockHeight, blockWidth)

    return triradius, blocks


def detectTriradiusT(blocks, blockImage, angles, pixelsForRadonTransform):
    # Funkcia najde triradius T.

    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    firstRow, lastRow, firstCol, lastCol = findExtremePalmprintPoints(blocks, blockImage)

    # Urcenie okna, v ktorom bude detekcia triradia T prebiehat.
    middleCol = (firstCol + lastCol) // 2
    middleRow = (firstRow + lastRow) // 2
    firstSearchCol = middleCol
    lastSearchCol = ((middleCol + lastCol) // 2) + 2
    firstSearchRow = ((middleRow + lastRow) // 2)
    lastSearchRow = lastRow

    possibleTriradiusBlocks = []

    for row in range(firstSearchRow, lastSearchRow):
        for col in range(firstSearchCol, lastSearchCol):
            if blocks[row][col]["orientation"] != -1:
                possibleTriradiusBlocks = poincareIndex(blocks, possibleTriradiusBlocks, row, col, angles)

    if len(possibleTriradiusBlocks) == 0:
        return (0, 0), blocks

    triradius = findTriradiusPosition(possibleTriradiusBlocks, blockImage, angles, pixelsForRadonTransform)
    blocks = saveTriradiusRegion(blocks, triradius, blockHeight, blockWidth)

    return triradius, blocks


def findTriradius(blocks, blockImage, angles, betweenFingersAreas):
    # Funkcia najde triradia nachadzajuce sa na odtlacku dlane.

    # Pre presnejsiu lokalizaciu triradia bude potrebne urcenie orientacie v mensich blokoch. Pouzita je polovica
    # velkosti bloku.
    _, _, blockHeight, blockWidth = blockImage.shape
    triradiusBlockHeight, triradiusBlockWidth = (blockHeight // 2, blockWidth // 2)

    pixelsForRadonTransform = findPixelsForRadonTransform(triradiusBlockHeight, triradiusBlockWidth, angles)

    # Region, v ktorom bude triradius hladany je pre kazdy triradius urceny samostatne.
    triradiusA, blocks = detectTriradiusA(blocks, blockImage, angles, betweenFingersAreas[0], betweenFingersAreas[1],
                                          pixelsForRadonTransform)
    triradiusB, blocks = detectTriradiusB(blocks, blockImage, angles, betweenFingersAreas[1], betweenFingersAreas[2],
                                          pixelsForRadonTransform)
    triradiusC, blocks = detectTriradiusC(blocks, blockImage, angles, betweenFingersAreas[2], betweenFingersAreas[3],
                                          pixelsForRadonTransform)
    triradiusD, blocks = detectTriradiusD(blocks, blockImage, angles, betweenFingersAreas[3], betweenFingersAreas[4],
                                          pixelsForRadonTransform)
    triradiusT, blocks = detectTriradiusT(blocks, blockImage, angles, pixelsForRadonTransform)

    triradius = [triradiusA, triradiusB, triradiusC, triradiusD, triradiusT]

    return blocks, triradius
