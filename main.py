# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
from typing import List
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import os

import cv2
import numpy as np
from scipy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter
from skimage.filters import window
from scipy import signal
from skimage.morphology import flood
from collections import Counter
import copy

from orientation import orientationField, findPixelsForRadonTransform, findRadonTransform, calculateCosOfAngles, \
    calculateSinOfAngles, splitIntoParts
from segmentation import segmentation, showSegmentedPalmprint

""" TODO
- pozriet sa na Otsu prahovanie pri vypocte poctu bielych pixelov
- (to mozno skor ku koncu) osetrit rozdielnu velkost bloku
- urcite sa pozriet na rychlost radonovej transformacie - pouzit DFT pre dobre definovane bloky
- vykreslenie orientacie v jednotlivych blokoch
- pomerne casto pouzivam prahovanie ta ci by nebolo lepsie vytvorit masku pre vsetky pixely
- pri vyplnovani flekcnych ryh je priestor pre zrychlenie
- uprvait stlpce pri hladani colCandidates (aby to nebralo do uvahy aj hranice odtlacku)
"""


def saveSmoothingPalmprint(image):
    cv2.imwrite('afterSmoothing.bmp', image)


def saveOrientationsImage(image, blockImage, blocks, angles):
    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape

    for row in range(blockRows):
        for col in range(blockCols):
            if blocks[row][col]["orientationConfidence"] != 100:
                blockImage[row][col] = 0

    image = mergeBlocksToImage(blockImage)

    drawBlock = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for row in range(blockRows):
        for col in range(blockCols):
            if blocks[row][col]["orientation"] != -1:
                angle = angles[blocks[row][col]["orientation"]]

                centerXX = blockWidth // 2
                centerYY = blockHeight // 2

                centerY = (row * blockWidth) + centerXX
                centerX = (col * blockWidth) + centerYY

                x = int(centerX + (10 * math.cos(angle)))
                y = int(centerY + (10 * math.sin(angle)))

                x2 = int(centerX + (10 * math.cos(angle - math.pi)))
                y2 = int(centerY + (10 * math.sin(angle - math.pi)))

                drawBlock = cv2.line(drawBlock, (x2, y), (x, y2), (0, 0, 255), 2)

    saveSmoothingPalmprint(drawBlock)


    """ Vsetkym pixelom v pozadi nastavi hodnotu 0 """
    for row in range(blockRows):
        for col in range(blockCols):
            if blocks[row][col]["background"] == 1:
                pass
                # blockImage[row][col] = 0
            if blocks[row][col]["background"] == 2:
                if blocks[row][col]["palmprintSegment"] > 0:
                    if blocks[row][col]["palmprintSegment"] == 4:
                        blockImage[row][col] = 100
                    # pass
                    else:
                        blockImage[row][col] = 0
                else:
                    # pass
                    blockImage[row][col] = 128

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


def saveSegmentedPalmprint(blockImage, blocks, directoryName):
    blockRows, blockCols, _, _ = blockImage.shape

    """ Vsetkym pixelom v pozadi nastavi hodnotu 0 """
    for row in range(blockRows):
        for col in range(blockCols):
            if blocks[row][col]["background"] == 1:
                pass
                # blockImage[row][col] = 0
            if blocks[row][col]["background"] == 2:
                if blocks[row][col]["palmprintSegment"] != 0:
                    if blocks[row][col]["palmprintSegment"] == 4:
                        blockImage[row][col] = 128
                    elif blocks[row][col]["palmprintSegment"] == 55:
                        blockImage[row][col] = 128
                    elif blocks[row][col]["palmprintSegment"] == 1313:
                        blockImage[row][col] = 128
                    # pass
                    else:
                        blockImage[row][col] = 0
                else:
                    # pass
                    blockImage[row][col] = 128

    image = mergeBlocksToImage(blockImage)

    cv2.imwrite(f'{directoryName}/segmented.bmp', image)


def saveOrientationAfterSmoothing(blockImage, blocks, angles, directoryName):
    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    image = mergeBlocksToImage(blockImage)

    orientationImage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for row in range(blockRows):
        for col in range(blockCols):
            if blocks[row][col]["orientation"] != -1:
                angle = angles[blocks[row][col]["orientation"]]

                centerXX = blockWidth // 2
                centerYY = blockHeight // 2

                centerY = (row * blockWidth) + centerXX
                centerX = (col * blockWidth) + centerYY

                x = int(centerX + (10 * math.cos(angle)))
                y = int(centerY + (10 * math.sin(angle)))

                x2 = int(centerX + (10 * math.cos(angle - math.pi)))
                y2 = int(centerY + (10 * math.sin(angle - math.pi)))

                orientationImage = cv2.line(orientationImage, (x2, y), (x, y2), (0, 0, 255), 2)

    cv2.imwrite(f'{directoryName}/afterSmoothing.bmp', orientationImage)


def loadPalmprint(fileName):
    image = cv2.imread(fileName, 0)
    return image


def showPalmprint(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def savePalmprint(image):
    cv2.imwrite('newPalmprint.bmp', image)


def saveBlock(image):
    cv2.imwrite('block.bmp', image)


def saveTriradiusPalmprint(blockImage, triradiusA, triradiusB, triradiusC, triradiusD, triradiusT, mainLines, directoryName):
    image = mergeBlocksToImage(blockImage)
    triradiusImage = copy.deepcopy(image)

    triradiusImage = cv2.circle(triradiusImage, (triradiusA[1], triradiusA[0]), 10, (0, 0, 255), 3)
    triradiusImage = cv2.circle(triradiusImage, (triradiusB[1], triradiusB[0]), 10, (0, 0, 255), 3)
    triradiusImage = cv2.circle(triradiusImage, (triradiusC[1], triradiusC[0]), 10, (0, 0, 255), 3)
    triradiusImage = cv2.circle(triradiusImage, (triradiusD[1], triradiusD[0]), 10, (0, 0, 255), 3)
    triradiusImage = cv2.circle(triradiusImage, (triradiusT[1], triradiusT[0]), 10, (0, 0, 255), 3)

    for mainLine in mainLines:
        for mainLinePoint in range(len(mainLine) - 1):
            triradiusImage = cv2.line(triradiusImage, mainLine[mainLinePoint], mainLine[mainLinePoint + 1], (0, 0, 255), 3)

    cv2.imwrite(f'{directoryName}/triradius.bmp', triradiusImage)


def initializeBlocks(blockRows, blockColumns):
    """ Vytvorenie struktury blokov """
    blocks = {}

    for row in range(blockRows):
        blocks[row] = {}
        for col in range(blockColumns):
            blocks[row][col] = {}
            blocks[row][col]["background"] = 0
            blocks[row][col]["orientation"] = -1
            blocks[row][col]["orientationConfidence"] = -1
            blocks[row][col]["palmprintSegment"] = 0
            blocks[row][col]["triradiusRegion"] = 0
            blocks[row][col]["frequency"] = 0
            blocks[row][col]["ridgeWidth"] = 0

    return blocks


def splitImageToBlocks(image, blockWidth, blockHeight):
    imageHeight, imageWidth = image.shape

    """ Rozdelenie na bloky - rychlejsie ako cez loopy. 
        blockImage[pocet blokov (riadky), pocet blokov(stlpce), vyska bloku, sirka bloku]"""
    blockImage = image.reshape(imageHeight // blockHeight, blockHeight, imageWidth // blockWidth, blockWidth)
    blockImage = blockImage.swapaxes(1, 2)

    blockRows, blockCols, _, _ = blockImage.shape

    blocks = initializeBlocks(blockRows, blockCols)

    return blockImage, blocks


def mergeBlocksToImage(blockImage):
    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    imageHeight = blockRows * blockHeight
    imageWidth = blockCols * blockWidth

    """ Spojenie blokov do celkoveho obrazu"""
    image = blockImage.reshape(imageHeight // blockHeight, imageWidth // blockWidth, blockHeight, blockWidth)
    image = image.swapaxes(1, 2).reshape(imageHeight, imageWidth)

    return image


def getStartEndOfPalmprint(palmprintBlocksInRow, palmprintBlocksInColumn):
    firstRow = np.nonzero(palmprintBlocksInRow)[0][0]
    lastRow = np.nonzero(palmprintBlocksInRow)[0][-1]

    firstCol = np.nonzero(palmprintBlocksInColumn)[0][0]
    lastCol = np.nonzero(palmprintBlocksInColumn)[0][-1]

    return firstRow, lastRow, firstCol, lastCol


def getNumberOfPalmprintBlocks(blocks):
    """ Pre kazdy stlpec/riadok najde pocet blokov, ktore patria do odtlacku """
    palmprintBlocksInColumn = [0] * len(blocks)
    palmprintBlocksInRow = [0] * len(blocks)

    for row in blocks:
        for col in blocks[row]:
            if blocks[row][col]["background"] == 0:
                palmprintBlocksInColumn[col] += 1
                palmprintBlocksInRow[row] += 1

    return palmprintBlocksInRow, palmprintBlocksInColumn


def findAveragePalmprintHeightOrWidth(palmprintBlocksInColumnOrRow):
    """ Najde priemer poctu blokov patriacich do odtlacku v jednotlivych stlpcoch/riadkoch. Priemer je pocitany iba
        zo stlpcov alebo riadkov je viac jak 10 blokov patriacich do odtlacku """
    palmprintBlocksInColumnOrRow = np.array(palmprintBlocksInColumnOrRow)
    # TODO hodnota 10 sa moze zmenit v zavislosti od velkosti bloku
    palmprintBlocksInColumnOrRow = palmprintBlocksInColumnOrRow[palmprintBlocksInColumnOrRow >= 10]
    average = int(np.mean(palmprintBlocksInColumnOrRow))

    return average


def checkColCandidate(colCandidate, blocks):
    firstFound = False
    firstBlockRow = 0
    lastBlockRow = 0

    """ Najdem prvy a posledny riadok bloku, ktory patri do odtlacku v danom stlpci """
    for row in blocks:
        if blocks[row][colCandidate]["background"] == 0:
            if not firstFound:
                firstBlockRow = row
                firstFound = True
            lastBlockRow = row

    middleBlockRow = (firstBlockRow + lastBlockRow) // 2

    valid = True

    """ Ked sa medzi prvym a strednym riadkom bloku nachadza blok patriaci do pozadia tak je stlpec nevalidny pre 
        pouzitie """
    for row in range(firstBlockRow, middleBlockRow):
        if blocks[row][colCandidate]["background"]:
            valid = False
            break

    return valid


def getColCandidateForProjection(palmprintBlocksInColumn, firstCol, lastCol, averagePalmprintHeight, blocks, left):
    colCandidates: list[int] = []
    middleCol = (lastCol + firstCol) // 2

    if left:
        """ Aby mohol by stlpec kandidatom musi mat urcitu vysku (aspon 3/4 priemernej vysky odtlacku) """
        for index in range(firstCol, middleCol):
            if palmprintBlocksInColumn[index] > (averagePalmprintHeight * 0.75):
                colCandidates.append(index)

    else:
        """ Aby mohol by stlpec kandidatom musi mat urcitu vysku (aspon 3/4 priemernej vysky odtlacku) """
        for index in range(middleCol, lastCol):
            if palmprintBlocksInColumn[index] > (averagePalmprintHeight * 0.75):
                colCandidates.append(index)

        """ Pre pravu stranu odtlacku sa poradie stlpcov otoci """
        colCandidates.reverse()

    checkedColCandidates = []

    for colCandidate in colCandidates:
        ok = checkColCandidate(colCandidate, blocks)
        if ok:
            checkedColCandidates.append(colCandidate)

    return checkedColCandidates


def getColCandidateForProjection2(topPointLeft, bottomPointLeft, topPointRight, bottomPointRight, blocks, left, blockImage):
    colCandidates: list[int] = []
    topRowLeft, topColLeft = topPointLeft
    bottomRowLeft, bottomColLeft = bottomPointLeft

    topRowRight, topColRight = topPointRight
    bottomRowRight, bottomColRight = bottomPointRight

    middleCol = (topColLeft + topColRight) // 2

    if left:
        firstCol = 0
        middleRowLeft = (topRowLeft + bottomRowLeft) // 2
        for col in range(topColLeft + 1, middleCol):
            valid = True
            firstCol = col
            for row in range(topRowLeft + 8, middleRowLeft - 3):
                #blockImage[row][col] = 100
                if blocks[row][col]["background"] != 0:
                    valid = False
                    break
            if valid:
                break

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

        for col in range(firstCol, middleCol):
            colCandidates.append(col)

        return colCandidates

    else:
        lastCol = 0
        middleRowRight = (topRowRight + bottomRowRight) // 2
        i = 0
        for col in range(middleCol, topColRight):
            valid = True
            lastCol = topColRight - i
            i += 1
            for row in range(topRowRight + 8, middleRowRight - 3):
                #blockImage[row][topColRight - i] = 100
                if blocks[row][topColRight - i]["background"] != 0:
                    valid = False
                    #break
            if valid:
                break

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

        for col in range(middleCol, lastCol):
            colCandidates.append(col)

        colCandidates.reverse()
        return colCandidates

    #middleCol = (lastCol + firstCol) // 2

    #if left:
        """ Aby mohol by stlpec kandidatom musi mat urcitu vysku (aspon 3/4 priemernej vysky odtlacku) """
        #for index in range(firstCol, middleCol):
            #if palmprintBlocksInColumn[index] > (averagePalmprintHeight * 0.75):
               # colCandidates.append(index)

    #else:
        """ Aby mohol by stlpec kandidatom musi mat urcitu vysku (aspon 3/4 priemernej vysky odtlacku) """
        #for index in range(middleCol, lastCol):
            #if palmprintBlocksInColumn[index] > (averagePalmprintHeight * 0.75):
                #colCandidates.append(index)

        """ Pre pravu stranu odtlacku sa poradie stlpcov otoci """
        #colCandidates.reverse()

    """checkedColCandidates = []

    for colCandidate in colCandidates:
        ok = checkColCandidate(colCandidate, blocks)
        if ok:
            checkedColCandidates.append(colCandidate)

    return checkedColCandidates"""


def projectionSmoothing(sumIntensityInRow):
    step = 5
    smoothenedSumIntensityInRow = [0] * len(sumIntensityInRow)

    for row in range(len(sumIntensityInRow)):
        rowsInSmoothing = 0
        for rowIndex in range(max(0, row - step), min(len(sumIntensityInRow), row + step)):
            smoothenedSumIntensityInRow[row] += sumIntensityInRow[rowIndex]
            rowsInSmoothing += 1

        smoothenedSumIntensityInRow[row] = smoothenedSumIntensityInRow[row] // rowsInSmoothing

    return smoothenedSumIntensityInRow


def fillPrincipleLine(seed, blockImage, blocks, principleLinePoints):
    for row in range(blockImage.shape[0]):
        for col in range(blockImage.shape[1]):
            if blocks[row][col]["background"]:
                blockImage[row][col] = 0

    image = mergeBlocksToImage(blockImage)

    whitePixelsMask = flood(image, seed, tolerance=30)

    toSearch = [seed]
    alreadySearched = []

    i = 0
    while toSearch:
        i += 1
        if i == 30000:
            break
        seedRow, seedCol = toSearch.pop(0)

        if (seedRow, seedCol) in alreadySearched:
            continue

        regionSum = sum(sum(whitePixelsMask[seedRow - 3:seedRow + 4, seedCol - 3:seedCol + 4]))
        if regionSum > 46:
            principleLinePoints[seedRow - 3:seedRow + 4, seedCol - 3:seedCol + 4] = True
            toSearch.append((seedRow - 3, seedCol))
            toSearch.append((seedRow - 3, seedCol + 3))
            toSearch.append((seedRow, seedCol + 3))
            toSearch.append((seedRow + 3, seedCol + 3))
            toSearch.append((seedRow + 3, seedCol))
            toSearch.append((seedRow + 3, seedCol - 3))
            toSearch.append((seedRow, seedCol - 3))
            toSearch.append((seedRow - 3, seedCol - 3))

        alreadySearched.append((seedRow, seedCol))

    """colorImage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    colorImage[whitePixelsMask] = (0, 255, 0)
    colorImage[principleLineMask] = (255, 0, 0)

    colorImage[880][40] = (255, 0, 0)

    savePalmprint(colorImage)"""

    return principleLinePoints


def findFirstAndLastBlockRow(blocks, colCandidate):
    firstFound = False
    firstBlockRow = 0
    lastBlockRow = 0

    for row in blocks:
        if blocks[row][colCandidate]["background"] == 0:
            if not firstFound:
                firstBlockRow = row
                firstFound = True
            lastBlockRow = row

    return firstBlockRow, lastBlockRow


def orientedProjection(image, blocks, blockImage, colCandidate, topPoint, bottomPoint):
    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape

    firstBlockRow = topPoint[0]
    lastBlockRow = bottomPoint[0]
    """ Najdem prvy a posledny blok, ktory lezi v odtlacku v danom stlpci """
    #firstBlockRow, lastBlockRow = findFirstAndLastBlockRow(blocks, colCandidate)

    """ Vypocitam stred stlpca """
    middleBlockRow = (firstBlockRow + lastBlockRow) // 2

    middleSearchRow = (middleBlockRow + firstBlockRow) // 2
    quarterDistanceFromMiddleSearchRow = (middleSearchRow - firstBlockRow) // 2

    """ Najdem prvy a posledny riadok smerovej projekcie """
    #firstSearchRow = (firstBlockRow + 8) * blockHeight
    #lastSearchRow = (middleBlockRow - 5) * blockHeight

    firstSearchRow = (middleSearchRow - quarterDistanceFromMiddleSearchRow) * blockHeight
    lastSearchRow = (middleSearchRow + quarterDistanceFromMiddleSearchRow) * blockHeight

    """ Najdem prvy a posledny stlpec smerovej projekcie """
    firstSearchCol = colCandidate * blockWidth
    lastSearchCol = firstSearchCol + blockWidth

    sumIntensityInRow = [0] * (lastSearchRow - firstSearchRow)
    rowIndex = 0

    """ Vypocitam sumu intenzit pixelov pozdlz kazdeho bodu stlpca """
    for row in range(firstSearchRow, lastSearchRow):
        for col in range(firstSearchCol, lastSearchCol):
            sumIntensityInRow[rowIndex] += image[row, col]
        rowIndex += 1

    """ Vyhladenie projekcie """
    sumIntensityInRow = projectionSmoothing(sumIntensityInRow)

    """ Maximalna hodnota znamena bod na flekcnej ryhe - jeho index potom pre najdenie suradnice """
    index = np.argmax(sumIntensityInRow)

    """ Najdenie suradnic """
    pointRowOnPrincipleLine = firstSearchRow + index
    pointColOnPrincipleLine = (firstSearchCol + lastSearchCol) // 2

    return pointRowOnPrincipleLine, pointColOnPrincipleLine


def orientedProjectionLeft(image, blocks, blockImage, palmprintBlocksInColumn, firstCol, lastCol,
                           averagePalmprintHeight, principleLinesPoints, colCandidatesForProjection, leftOfFinger2, bottomPointOfSegment1):
    #left = True

    """ Vyberie stlpce v ktorych moze prebehnut smerova projekcia -  """
    #colCandidatesForProjection = getColCandidateForProjection(palmprintBlocksInColumn, firstCol, lastCol,
     #                                                         averagePalmprintHeight, blocks, left)

    """ Vyberie druheho kandidata (prvy moze byt na hranici odtlacku) """
    #colCandidate = colCandidatesForProjection[1]

    """ Vykonanie smerovej projekcie v danom stlpci """
    #pointOnPrincipleLine = orientedProjection(image, blocks, blockImage, colCandidate)

    """ Vyplnenie flekcnej ryhy """
    #principleLinesPoints = fillPrincipleLine(pointOnPrincipleLine, blockImage, blocks, principleLinePoints)

    #return principleLinesPoints

    _, _, blockHeight, blockWidth = blockImage.shape

    principalPointSeeds = []
    colorImage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for ind in range(0, min(len(colCandidatesForProjection), 11)):
        """ Vyberie druheho kandidata (prvy moze byt na hranici odtlacku) """
        colCandidate = colCandidatesForProjection[ind]

        """ Vykonanie smerovej projekcie v danom stlpci """
        pointOnPrincipleLine = orientedProjection(image, blocks, blockImage, colCandidate, leftOfFinger2, bottomPointOfSegment1)
        principalPointSeeds.append(pointOnPrincipleLine)

        for i in range(-5, 5):
            for j in range(-5, 5):
                colorImage[pointOnPrincipleLine[0] + i][pointOnPrincipleLine[1] + j] = (255, 0, 0)

    pointRow = 0

    found = False
    for point in principalPointSeeds:
        inOneLine = 0
        pointRow = point[0]
        for i in range(len(principalPointSeeds)):
            if pointRow - 100 <= principalPointSeeds[i][0] <= pointRow + 200:
                inOneLine += 1
        if inOneLine >= 5:
            found = True
            break

    if found:
        principalLineRow = pointRow // blockHeight

    else:
        firstBlockRow = leftOfFinger2[0]
        lastBlockRow = bottomPointOfSegment1[0]

        """ Vypocitam stred stlpca """
        middleBlockRow = (firstBlockRow + lastBlockRow) // 2

        principalLineRow = (middleBlockRow + firstBlockRow) // 2

    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=100).fit(principalPointSeeds)
    print(clustering.labels_)

    savePalmprint(colorImage)

    #savePalmprint(colorImage)
    #exit(2)

    #pointOnPrincipleLine = principalPointSeeds[5]

    #for pointOnPrincipleLine in principalPointSeeds:
        #""" Ked uz bod lezi v detekovanej flekcnej ryhe, nie je potrebne ju zase vyplnovat """
        #if not principleLinesPoints[pointOnPrincipleLine[0]][pointOnPrincipleLine[1]]:
            #""" Vyplnenie flekcnej ryhy """
            #principleLinesPoints = fillPrincipleLine(pointOnPrincipleLine, blockImage, blocks, principleLinesPoints)

    return principalLineRow


def orientedProjectionRight(image, blocks, blockImage, palmprintBlocksInColumn, firstCol, lastCol,
                            averagePalmprintHeight, principleLinesPoints, colCandidatesForProjection, rightOfFinger5,
                            bottomPointOfSegment3):
    _, _, blockHeight, blockWidth = blockImage.shape
    #left = False

    """ Vyberie stlpce v ktorych moze prebehnut smerova projekcia -  """
    #colCandidatesForProjection = getColCandidateForProjection(palmprintBlocksInColumn, firstCol, lastCol,
     #                                                         averagePalmprintHeight, blocks, left)

    principalPointSeeds = []
    colorImage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for ind in range(1, min(len(colCandidatesForProjection), 11)):
        """ Vyberie druheho kandidata (prvy moze byt na hranici odtlacku) """
        colCandidate = colCandidatesForProjection[ind]

        """ Vykonanie smerovej projekcie v danom stlpci """
        pointOnPrincipleLine = orientedProjection(image, blocks, blockImage, colCandidate, rightOfFinger5, bottomPointOfSegment3)
        principalPointSeeds.append(pointOnPrincipleLine)

        for i in range(-5, 5):
            for j in range(-5, 5):
                colorImage[pointOnPrincipleLine[0] + i][pointOnPrincipleLine[1] + j] = (255, 0, 0)

    pointRow = 0

    found = False
    for point in principalPointSeeds:
        inOneLine = 0
        pointRow = point[0]
        for i in range(len(principalPointSeeds)):
            if pointRow - 200 <= principalPointSeeds[i][0] <= pointRow + 100:
                inOneLine += 1
        if inOneLine >= 5:
            found = True
            break

    if found:
        principalLineRow = (pointRow // blockHeight) + 1

    else:
        firstBlockRow = rightOfFinger5[0]
        lastBlockRow = bottomPointOfSegment3[0]

        """ Vypocitam stred stlpca """
        middleBlockRow = (firstBlockRow + lastBlockRow) // 2

        principalLineRow = (middleBlockRow + firstBlockRow) // 2

    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=200).fit(principalPointSeeds)
    print(clustering.labels_)

    #savePalmprint(colorImage)
    #exit(2)

    #pointOnPrincipleLine = principalPointSeeds[5]

    #for pointOnPrincipleLine in principalPointSeeds:
        #""" Ked uz bod lezi v detekovanej flekcnej ryhe, nie je potrebne ju zase vyplnovat """
        #if not principleLinesPoints[pointOnPrincipleLine[0]][pointOnPrincipleLine[1]]:
            #""" Vyplnenie flekcnej ryhy """
            #principleLinesPoints = fillPrincipleLine(pointOnPrincipleLine, blockImage, blocks, principleLinesPoints)

    return principalLineRow


def checkRowCandidate(rowCandidate, blocks):
    firstFound = False
    firstBlockCol = 0
    lastBlockCol = 0

    """ Najdem prvy a posledny stlpec bloku, ktory patri do odtlacku v danom riadku """
    for col in range(len(blocks[rowCandidate])):
        if blocks[rowCandidate][col]["background"] == 0:
            if not firstFound:
                firstBlockCol = col
                firstFound = True
            lastBlockCol = col

    """ Hladat sa bude od 1/4 do 3/4 """
    middleBlockCol = (firstBlockCol + lastBlockCol) // 2
    startBlockCol = (firstBlockCol + middleBlockCol) // 2
    endBlockCol = (middleBlockCol + lastBlockCol) // 2

    valid = True

    """ Ked sa medzi prvym a strednym riadkom bloku nachadza blok patriaci do pozadia tak je stlpec nevalidny pre 
        pouzitie """
    for col in range(startBlockCol, endBlockCol):
        if blocks[rowCandidate][col]["background"]:
            valid = False
            break

    return valid


def getRowCandidateForProjection(palmprintBlocksInRow, firstRow, lastRow, averagePalmprintHeight, blocks):
    rowCandidates: list[int] = []
    middleRow = (lastRow + firstRow) // 2

    """ Aby mohol by stlpec kandidatom musi mat urcitu vysku (aspon 3/4 priemernej vysky odtlacku) - hladam v dolenj 
        polovici odtlacku, preto od middleRow do lastRow """
    for index in range(middleRow, lastRow):
        if palmprintBlocksInRow[index] > (averagePalmprintHeight * 0.75):
            rowCandidates.append(index)

    """ Pre pravu stranu odtlacku sa poradie stlpcov otoci """
    rowCandidates.reverse()

    checkedRowCandidates = []

    for rowCandidate in rowCandidates:
        ok = checkRowCandidate(rowCandidate, blocks)
        if ok:
            checkedRowCandidates.append(rowCandidate)

    return checkedRowCandidates


def findFirstAndLastBlockCol(blocks, rowCandidate):
    firstFound = False
    firstBlockCol = 0
    lastBlockCol = 0

    for col in range(len(blocks[rowCandidate])):
        if blocks[rowCandidate][col]["background"] == 0:
            if not firstFound:
                firstBlockCol = col
                firstFound = True
            lastBlockCol = col

    return firstBlockCol, lastBlockCol


def orientedProjection90(image, blocks, blockImage, rowCandidate):
    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape

    """ Najdem prvy a posledny blok, ktory lezi v odtlacku v danom stlpci """
    firstBlockCol, lastBlockCol = findFirstAndLastBlockCol(blocks, rowCandidate)

    """ Vypocitam stred stlpca """
    middleBlockCol = (firstBlockCol + lastBlockCol) // 2
    startBlockCol = (firstBlockCol + middleBlockCol) // 2
    endBlockCol = (middleBlockCol + lastBlockCol) // 2

    """ Najdem prvy a posledny riadok smerovej projekcie """
    firstSearchCol = startBlockCol * blockHeight
    lastSearchCol = endBlockCol * blockHeight

    """ Najdem prvy a posledny stlpec smerovej projekcie """
    firstSearchRow = rowCandidate * blockWidth
    lastSearchRow = firstSearchRow + blockWidth

    sumIntensityInCol = [0] * (lastSearchCol - firstSearchCol)
    rowIndex = 0

    """ Vypocitam sumu intenzit pixelov pozdlz kazdeho bodu stlpca """
    for col in range(firstSearchCol, lastSearchCol):
        for row in range(firstSearchRow, lastSearchRow):
            sumIntensityInCol[rowIndex] += image[row, col]
        rowIndex += 1

    """ Vyhladenie projekcie """
    sumIntensityInCol = projectionSmoothing(sumIntensityInCol)

    """ Maximalna hodnota znamena bod na flekcnej ryhe - jeho index potom pre najdenie suradnice """
    index = np.argmax(sumIntensityInCol)

    """ Najdenie suradnic """
    pointRowOnPrincipleLine = (firstSearchRow + lastSearchRow) // 2
    pointColOnPrincipleLine = firstSearchCol + index

    return pointRowOnPrincipleLine, pointColOnPrincipleLine


def orientedProjectionBottom(image, blocks, blockImage, palmprintBlocksInRow, firstCol, lastCol,
                             averagePalmprintHeight, principleLinesPoints):
    """ Vyberie riadky,v ktorych moze prebehnut smerova projekcia -  """
    rowCandidatesForProjection = getRowCandidateForProjection(palmprintBlocksInRow, firstCol, lastCol,
                                                              averagePalmprintHeight, blocks)

    """ Vyberie druheho kandidata (prvy moze byt na hranici odtlacku) """
    rowCandidate = rowCandidatesForProjection[1]

    """ Vykonanie smerovej projekcie v danom stlpci """
    pointOnPrincipleLine = orientedProjection90(image, blocks, blockImage, rowCandidate)

    """ Ked uz bod lezi v detekovanej flekcnej ryhe, nie je potrebne ju zase vyplnovat """
    if not principleLinesPoints[pointOnPrincipleLine[0]][pointOnPrincipleLine[1]]:
        """ Vyplnenie flekcnej ryhy """
        principleLinesPoints = fillPrincipleLine(pointOnPrincipleLine, blockImage, blocks, principleLinesPoints)

    return principleLinesPoints


def principleLines(image, blocks, blockImage, leftOfFinger2, bottomPointOfSegment1, rightOfFinger5, bottomPointOfSegment3, palmprintBorder, ):
    """ Pre kazdy stlpec/riadok najde pocet blokov, ktore patria do odtlacku """
    palmprintBlocksInRow, palmprintBlocksInColumn = getNumberOfPalmprintBlocks(blocks)

    """ Priemer poctu blokov pre vysku a sirku """
    averagePalmprintHeight = findAveragePalmprintHeightOrWidth(palmprintBlocksInColumn)
    averagePalmprintWidth = findAveragePalmprintHeightOrWidth(palmprintBlocksInRow)

    """ Prve a posledne rady a stlpce, kde zacina odtlacok """
    firstRow, lastRow, firstCol, lastCol = getStartEndOfPalmprint(palmprintBlocksInRow, palmprintBlocksInColumn)

    """ Ciernobiely na farebny """
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    principleLinesPoints = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    principleLinesPoints = np.bool_(principleLinesPoints)

    left = True

    """ Vyberie stlpce v ktorych moze prebehnut smerova projekcia -  """
    colCandidatesForProjection = getColCandidateForProjection2(leftOfFinger2, bottomPointOfSegment1, rightOfFinger5,
                                                               bottomPointOfSegment3, blocks, left, blockImage)

    """ Smerova projekcia v lavej hornej polovici odtlacku """
    principleLineRow = orientedProjectionLeft(image, blocks, blockImage, palmprintBlocksInColumn, firstCol, lastCol,
                                              averagePalmprintHeight, principleLinesPoints, colCandidatesForProjection,
                                              leftOfFinger2, bottomPointOfSegment1)

    startingPointIndex = palmprintBorder.index(leftOfFinger2)
    for distanceFromStartingPoint in range(1, len(palmprintBorder)):
        nextBorderPoint = palmprintBorder[startingPointIndex + distanceFromStartingPoint]
        nextBorderPointRow, nextBorderPointCol = nextBorderPoint
        if blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] != 13:
            break
        if nextBorderPointRow == principleLineRow:
            blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = 1313
        if nextBorderPointRow < principleLineRow:
            blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = "13\'"
        if nextBorderPointRow > principleLineRow:
            blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = "13\""

    left = False

    """ Vyberie stlpce v ktorych moze prebehnut smerova projekcia -  """
    colCandidatesForProjection = getColCandidateForProjection2(leftOfFinger2, bottomPointOfSegment1, rightOfFinger5,
                                                               bottomPointOfSegment3, blocks, left, blockImage)

    """ Smerova projekcia v pravej hornej polovici odtlacku """
    principleLineRow = orientedProjectionRight(image, blocks, blockImage, palmprintBlocksInColumn, firstCol,
                                                   lastCol, averagePalmprintHeight, principleLinesPoints, colCandidatesForProjection,
                                                   rightOfFinger5, bottomPointOfSegment3)

    startingPointIndex = palmprintBorder.index(rightOfFinger5)
    for distanceFromStartingPoint in range(1, len(palmprintBorder)):
        nextBorderPoint = palmprintBorder[startingPointIndex - distanceFromStartingPoint]
        nextBorderPointRow, nextBorderPointCol = nextBorderPoint
        if nextBorderPointRow == principleLineRow:
            blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = 55
        if blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] == 4:
            break
        if nextBorderPointRow < principleLineRow:
            blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = "5\""
        if nextBorderPointRow > principleLineRow:
            blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = "5\'"
    """ Smerova projekcia v strednej dolnej polovici odtlacku """
    """principleLinesPoints = orientedProjectionBottom(image, blocks, blockImage, palmprintBlocksInRow, firstRow,
                                                    lastRow, averagePalmprintWidth, principleLinesPoints)"""

    """colorImage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    colorImage[principleLinesPoints] = (255, 0, 0)

    savePalmprint(colorImage)
    exit(2)"""
    return blocks


def poincareIndex(blocks, possibleTriradiusBlocks, row, col, angles):
    orientations = []
    topLeft = blocks[row - 1][col - 1]["orientation"]
    orientations.append(topLeft)
    top = blocks[row - 1][col]["orientation"]
    orientations.append(top)
    topRight = blocks[row - 1][col + 1]["orientation"]
    orientations.append(topRight)
    right = blocks[row][col + 1]["orientation"]
    orientations.append(right)
    bottomRight = blocks[row + 1][col + 1]["orientation"]
    orientations.append(bottomRight)
    bottom = blocks[row + 1][col]["orientation"]
    orientations.append(bottom)
    bottomLeft = blocks[row + 1][col - 1]["orientation"]
    orientations.append(bottomLeft)
    left = blocks[row][col - 1]["orientation"]
    orientations.append(left)
    orientations.append(topLeft)

    found = True
    index = 0
    a = 0

    for k in range(8):
        a1 = orientations[k]
        a2 = orientations[k + 1]

        if a1 == -1 or a2 == -1:
            found = False
            break

        diff = math.degrees(angles[a1]) - math.degrees(angles[a2])

        if diff - 5 < -90:
            diff += 180

        if diff + 5 > 90:
            diff -= 180

        index += diff

    if found:
        if -180 - 20 <= index <= -180 + 20:
            found = True
            a = 1
        elif 180 - 20 <= index <= 180 + 20:
            found = True
            a = 2
        else:
            found = False

    if found:
        if a == 1:
            # blockImage[row][col] = 0
            possibleTriradiusBlocks.append((row, col))
        else:
            pass
            #blockImage[row][col] = 255
            # possibleTriradiusBlocks.append((row, col))

    return possibleTriradiusBlocks


def findOrientationsInTriradiusBlocks(blockHeight, blockWidth, blockImage, angles, smallBlockRows, smallBlockCols,
                                      firstRow, firstCol):
    rows, cols = (smallBlockRows, smallBlockCols)
    triradiusRegion = [[-1 for i in range(cols)] for j in range(rows)]

    pixelsForRadonTransform = findPixelsForRadonTransform(25, 25, angles)

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
            print(f"Mean: {mean}")

            for row in range(25):
                for col in range(25):
                    value = block[row][col]
                    if value < mean - 10:
                        r = findRadonTransform(row, col, block, angles, pixelsForRadonTransform)
                        pixelOrientations.append(r)

            print(Counter(pixelOrientations))
            try:
                finalOrientation = Counter(pixelOrientations).most_common()[0][0]
            except IndexError:
                finalOrientation = -1

            # showPalmprint(block)

            triradiusRegion[blockRow][blockCol] = finalOrientation
            print(f"{blockRow}:{blockCol} --> {finalOrientation}")

    return triradiusRegion


def localizeTriradius(triradiusRegion, smallBlockRows, smallBlockCols, firstRow, firstCol, blockImage, triradiusBlocksMask):
    _, _, blockHeight, blockWidth = blockImage.shape
    maxRow = 0
    maxCol = 0
    maxVal = 0

    for blockRow in range(smallBlockRows - 1):
        for blockCol in range(smallBlockCols - 1):
            start = triradiusRegion[blockRow][blockCol]
            right = triradiusRegion[blockRow][blockCol + 1]
            bottom = triradiusRegion[blockRow + 1][blockCol + 1]
            left = triradiusRegion[blockRow + 1][blockCol]
            if triradiusBlocksMask[blockRow][blockCol] == 0 or triradiusBlocksMask[blockRow][blockCol + 1] == 0 or \
                    triradiusBlocksMask[blockRow + 1][blockCol + 1] == 0 or triradiusBlocksMask[blockRow + 1][blockCol] == 0:
                continue

            diff = 0
            diff += min(abs(start - right), abs((start + 12) - right))
            diff += min(abs(right - bottom), abs((right + 12) - bottom))
            diff += min(abs(bottom - left), abs((bottom + 12) - left))
            diff += min(abs(left - start), abs((left + 12) - start))

            if diff > maxVal:
                maxVal = diff
                maxRow = blockRow
                maxCol = blockCol

    blockImageRow = firstRow + (maxRow * 25) // blockHeight
    blockImageCol = firstCol + (maxCol * 25) // blockWidth
    startRow = (maxRow * 25) % blockHeight
    endRow = ((maxRow + 1) * 25) % blockHeight
    if endRow == 0:
        endRow = blockHeight
    startCol = (maxCol * 25) % blockWidth
    endCol = ((maxCol + 1) * 25) % blockWidth
    if endCol == 0:
        endCol = blockWidth

    #blockImage[blockImageRow][blockImageCol][startRow:endRow, startCol:endCol] = 0
    triradiusX = (blockImageRow * blockHeight) + endRow + 1
    triradiusY = (blockImageCol * blockWidth) + endCol + 1
    triradius = (triradiusX, triradiusY)

    return triradius


def getMostProbableTriradiusRegion(possibleTriradiusBlocks, blockRows, blockCols, blockHeight, blockWidth):
    """ Tota cast kodu rozdeli mozne triradiove bloky na spojite zhluky """
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

    """ Tota cast kodu skusi najst stvorec o velkosti 2x2 """
    if len(possibleTriradiusParts) == 1:
        triradiusPart = possibleTriradiusParts[0]
        possibleTriradiusBlocks = copy.deepcopy(triradiusPart)
        if len(triradiusPart) == 1:
            triradiusBlock = triradiusPart[0]
            triradiusX = (triradiusBlock[0] * blockRows) + (blockHeight // 2)
            triradiusY = (triradiusBlock[1] * blockCols) + (blockWidth // 2)

            # return triradiusX, triradiusY
        else:
            foundSquare = False
            for blockInTriradiusPart in triradiusPart:
                pointX, pointY = blockInTriradiusPart
                if (pointX, pointY + 1) in triradiusPart and (pointX + 1, pointY + 1) in triradiusPart and \
                        (pointX + 1, pointY) in triradiusPart:
                    possibleTriradiusBlocks = [(pointX, pointY), (pointX, pointY + 1), (pointX + 1, pointY),
                                               (pointX + 1, pointY + 1)]
                    foundSquare = True
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
            #biggestTriradiusRegion = max(len(triradiusPart) for triradiusPart in possibleTriradiusParts)
            biggestTriradiusRegion = max(possibleTriradiusParts, key=len)
            possibleTriradiusBlocks = copy.deepcopy(biggestTriradiusRegion)

    return possibleTriradiusBlocks


def detectTriradiusA(blocks, blockImage, angles, leftOfFinger2, between23):
    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    possibleTriradiusBlocks = []
    leftOfFinger2Row, leftOfFinger2Col = leftOfFinger2
    rightOfFinger2Row, rightOfFinger2Col = between23[(len(between23) // 2) - 1]

    for row in range(leftOfFinger2Row + 1, leftOfFinger2Row + 8):
        for col in range(leftOfFinger2Col, rightOfFinger2Col):
            #blockImage[row][col] = 0
            if blocks[row][col]["orientation"] != -1:
                possibleTriradiusBlocks = poincareIndex(blocks, possibleTriradiusBlocks, row, col, angles)

    if len(possibleTriradiusBlocks) == 0:
        return (0, 0), blocks

    """ Tota cast kodu vykresli mozne bloky v ktorych moze byt triradius bielou farbou """
    """triradiusBlocks = copy.deepcopy(blockImage)

    for a in possibleTriradiusBlocks:
        triradiusBlocks[a[0]][a[1]] = 255

    image = mergeBlocksToImage(triradiusBlocks)

    scale_percent = 15  # percent of original size
    imageHeight, imageWidth = image.shape

    height = int(imageHeight * scale_percent / 100)
    width = int(imageWidth * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('image', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    possibleTriradiusBlocks = getMostProbableTriradiusRegion(possibleTriradiusBlocks, blockRows, blockCols, blockHeight, blockWidth)

    """ Tota cast kodu ulozi obraz s moznymi blokmi v ktorych moze byt triradius bielou farbou """
    triradiusBlocks = copy.deepcopy(blockImage)

    for a in possibleTriradiusBlocks:
        triradiusBlocks[a[0]][a[1]] = 255

    image = mergeBlocksToImage(triradiusBlocks)

    cv2.imwrite('triradiusPalmprint1.bmp', image)

    firstRow = min(possibleTriradiusBlocks, key=lambda t: t[0])[0]
    lastRow = max(possibleTriradiusBlocks, key=lambda t: t[0])[0]
    firstCol = min(possibleTriradiusBlocks, key=lambda t: t[1])[1]
    lastCol = max(possibleTriradiusBlocks, key=lambda t: t[1])[1]

    for row in range(firstRow - 2, lastRow + 2):
        for col in range(firstCol - 2, lastCol + 2):
            blocks[row][col]["triradiusRegion"] = 1

    blocksInHeight = lastRow - firstRow + 1
    blocksInWidth = lastCol - firstCol + 1

    smallBlockRows = blocksInHeight * blockHeight // 25
    smallBlockCols = blocksInWidth * blockWidth // 25

    triradiusBlocksMask = [[0 for i in range(smallBlockCols)] for j in range(smallBlockRows)]

    for row in range(blocksInWidth):
        for col in range(blocksInWidth):
            if (firstRow + row, firstCol + col) in possibleTriradiusBlocks:
                triradiusBlocksMask[2*row][2*col] = 1
                triradiusBlocksMask[2*row + 1][2*col] = 1
                triradiusBlocksMask[2*row][2*col + 1] = 1
                triradiusBlocksMask[2*row + 1][2*col + 1] = 1

    triradiusRegion = findOrientationsInTriradiusBlocks(blockHeight, blockWidth, blockImage, angles, smallBlockRows,
                                                        smallBlockCols, firstRow, firstCol)

    #blockImage = localizeTriradius(triradiusRegion, smallBlockRows, smallBlockCols, firstRow, firstCol, blockImage)
    triradius = localizeTriradius(triradiusRegion, smallBlockRows, smallBlockCols, firstRow, firstCol, blockImage, triradiusBlocksMask)

    #return blockImage
    return triradius, blocks


def detectTriradiusB(blocks, blockImage, angles, between23, between34):
    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    possibleTriradiusBlocks = []
    leftOfFinger2Row, leftOfFinger2Col = between23[(len(between23) // 2) - 1]
    rightOfFinger2Row, rightOfFinger2Col = between34[(len(between34) // 2) + 1]

    for row in range(leftOfFinger2Row - 1, leftOfFinger2Row + 7):
        for col in range(leftOfFinger2Col, rightOfFinger2Col):
            #blockImage[row][col] = 0
            if blocks[row][col]["orientation"] != -1:
                possibleTriradiusBlocks = poincareIndex(blocks, possibleTriradiusBlocks, row, col, angles)

    if len(possibleTriradiusBlocks) == 0:
        return (0, 0), blocks

    possibleTriradiusBlocks = getMostProbableTriradiusRegion(possibleTriradiusBlocks, blockRows, blockCols, blockHeight,
                                                             blockWidth)

    triradiusBlocks = copy.deepcopy(blockImage)

    """for a in possibleTriradiusBlocks:
        triradiusBlocks[a[0]][a[1]] = 255

    image = mergeBlocksToImage(triradiusBlocks)

    cv2.imwrite('triradiusPalmprint2.bmp', image)

    scale_percent = 15  # percent of original size
    imageHeight, imageWidth = image.shape

    height = int(imageHeight * scale_percent / 100)
    width = int(imageWidth * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('image', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    firstRow = min(possibleTriradiusBlocks, key=lambda t: t[0])[0]
    lastRow = max(possibleTriradiusBlocks, key=lambda t: t[0])[0]
    firstCol = min(possibleTriradiusBlocks, key=lambda t: t[1])[1]
    lastCol = max(possibleTriradiusBlocks, key=lambda t: t[1])[1]

    for row in range(firstRow - 2, lastRow + 2):
        for col in range(firstCol - 2, lastCol + 2):
            blocks[row][col]["triradiusRegion"] = 1

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
                                                        smallBlockCols, firstRow, firstCol)

    # blockImage = localizeTriradius(triradiusRegion, smallBlockRows, smallBlockCols, firstRow, firstCol, blockImage)
    triradius = localizeTriradius(triradiusRegion, smallBlockRows, smallBlockCols, firstRow, firstCol, blockImage, triradiusBlocksMask)

    # return blockImage
    return triradius, blocks


def detectTriradiusD(blocks, blockImage, angles, between45, rightOfFinger5):
    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    possibleTriradiusBlocks = []
    leftOfFinger2Row, leftOfFinger2Col = between45[0]
    rightOfFinger2Row, rightOfFinger2Col = rightOfFinger5

    for row in range(leftOfFinger2Row + 2, leftOfFinger2Row + 10):
        for col in range(leftOfFinger2Col, rightOfFinger2Col):
            #blockImage[row][col] = 0
            if blocks[row][col]["orientation"] != -1:
                possibleTriradiusBlocks = poincareIndex(blocks, possibleTriradiusBlocks, row, col, angles)

    if len(possibleTriradiusBlocks) == 0:
        return (0, 0), blocks

    possibleTriradiusBlocks = getMostProbableTriradiusRegion(possibleTriradiusBlocks, blockRows, blockCols, blockHeight,
                                                             blockWidth)

    """triradiusBlocks = copy.deepcopy(blockImage)

    for a in possibleTriradiusBlocks:
        triradiusBlocks[a[0]][a[1]] = 255

    image = mergeBlocksToImage(triradiusBlocks)

    cv2.imwrite('triradiusPalmprint3.bmp', image)"""

    firstRow = min(possibleTriradiusBlocks, key=lambda t: t[0])[0]
    lastRow = max(possibleTriradiusBlocks, key=lambda t: t[0])[0]
    firstCol = min(possibleTriradiusBlocks, key=lambda t: t[1])[1]
    lastCol = max(possibleTriradiusBlocks, key=lambda t: t[1])[1]

    for row in range(firstRow - 2, lastRow + 2):
        for col in range(firstCol - 2, lastCol + 2):
            blocks[row][col]["triradiusRegion"] = 1

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
                                                        smallBlockCols, firstRow, firstCol)

    # blockImage = localizeTriradius(triradiusRegion, smallBlockRows, smallBlockCols, firstRow, firstCol, blockImage)
    triradius = localizeTriradius(triradiusRegion, smallBlockRows, smallBlockCols, firstRow, firstCol, blockImage, triradiusBlocksMask)

    # return blockImage
    return triradius, blocks


def detectTriradiusT(blocks, blockImage, angles):
    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    lastRow = 0
    firstCol = 0
    lastCol = 0

    """ Najde posledny riadok odtlacku """
    foundLastRow = False
    for row in range(blockRows):
        for col in range(blockCols):
            if blocks[blockRows - row - 1][col]["background"] != 1:
                lastRow = blockRows - row - 1
                foundLastRow = True
                break
        if foundLastRow:
            break

    foundFirstCol = False
    for col in range(blockCols):
        if not foundFirstCol:
            if blocks[lastRow][col]["background"] != 1:
                firstCol = col
                foundFirstCol = True
        else:
            if blocks[lastRow][col]["background"] != 1:
                lastCol = col

    middleCol = (firstCol + lastCol) // 2
    firstSearchCol = middleCol
    lastSearchCol = ((middleCol + lastCol) // 2) + 2
    firstSearchRow = lastRow - 12
    lastSearchRow = lastRow

    """for row in range(firstSearchRow, lastSearchRow):
            for col in range(firstSearchCol, lastSearchCol):
                blockImage[row][col] = 0

        image = mergeBlocksToImage(blockImage)

        savePalmprint(image)
        exit(2)"""

    possibleTriradiusBlocks = []

    for row in range(firstSearchRow, lastSearchRow):
        for col in range(firstSearchCol, lastSearchCol):
            #blockImage[row][col] = 0
            if blocks[row][col]["orientation"] != -1:
                possibleTriradiusBlocks = poincareIndex(blocks, possibleTriradiusBlocks, row, col, angles)

    if len(possibleTriradiusBlocks) == 0:
        return (0, 0), blocks

    """ Tota cast kodu vykresli mozne bloky v ktorych moze byt triradius bielou farbou """
    """triradiusBlocks = copy.deepcopy(blockImage)

    for a in possibleTriradiusBlocks:
        triradiusBlocks[a[0]][a[1]] = 255

    image = mergeBlocksToImage(triradiusBlocks)

    scale_percent = 15  # percent of original size
    imageHeight, imageWidth = image.shape

    height = int(imageHeight * scale_percent / 100)
    width = int(imageWidth * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('image', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    possibleTriradiusBlocks = getMostProbableTriradiusRegion(possibleTriradiusBlocks, blockRows, blockCols, blockHeight, blockWidth)

    """ Tota cast kodu ulozi obraz s moznymi blokmi v ktorych moze byt triradius bielou farbou """
    triradiusBlocks = copy.deepcopy(blockImage)

    for a in possibleTriradiusBlocks:
        triradiusBlocks[a[0]][a[1]] = 255

    image = mergeBlocksToImage(triradiusBlocks)

    cv2.imwrite('triradiusPalmprintT.bmp', image)

    firstRow = min(possibleTriradiusBlocks, key=lambda t: t[0])[0]
    lastRow = max(possibleTriradiusBlocks, key=lambda t: t[0])[0]
    firstCol = min(possibleTriradiusBlocks, key=lambda t: t[1])[1]
    lastCol = max(possibleTriradiusBlocks, key=lambda t: t[1])[1]

    for row in range(firstRow - 2, lastRow + 2):
        for col in range(firstCol - 2, lastCol + 2):
            blocks[row][col]["triradiusRegion"] = 1

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
                                                        smallBlockCols, firstRow, firstCol)

    # blockImage = localizeTriradius(triradiusRegion, smallBlockRows, smallBlockCols, firstRow, firstCol, blockImage)
    triradius = localizeTriradius(triradiusRegion, smallBlockRows, smallBlockCols, firstRow, firstCol, blockImage,
                                  triradiusBlocksMask)

    # return blockImage
    return triradius, blocks


def comingFromWhere(x, y, nextX, nextY):
    if nextX > x:
        if nextY > y:
            comingFromNew = 0
        elif nextY == y:
            comingFromNew = 1
        else:
            comingFromNew = 2

    elif nextX == x:
        if nextY < y:
            comingFromNew = 3
        else:
            comingFromNew = 7

    else:
        if nextY < y:
            comingFromNew = 4
        elif nextY == y:
            comingFromNew = 5
        else:
            comingFromNew = 6

    return comingFromNew


def calculateSinOfAngles5(angles):
    sinOfAngles = []
    fiveDegrees = math.radians(5)

    for angle in angles:
        sinOfAngles.append(math.sin(angle - fiveDegrees))

    return sinOfAngles


def calculateCosOfAngles5(angles):
    cosOfAngles = []
    fiveDegrees = math.radians(5)

    for angle in angles:
        cosOfAngles.append(math.cos(angle - fiveDegrees))

    return cosOfAngles


def getNextPoint2(previousOrientation, orientation, sinOfAngles, cosOfAngles, x, y):
    """ Orientation bude stale 0-11, previousOrietnation moze byt 0-23 """
    """ Najprv by mi trebalo urcit novu orientation """
    if abs(previousOrientation - orientation) > 11:
        orientation += 12

    if previousOrientation - orientation > 3:
        orientation = previousOrientation - 3

    if previousOrientation - orientation < -3:
        orientation = previousOrientation + 3

    if orientation > 11:
        nextX = int(x + (25 * sinOfAngles[orientation - 12]))
        nextY = int(y - (25 * cosOfAngles[orientation - 12]))
    else:
        nextX = int(x - (25 * sinOfAngles[orientation]))
        nextY = int(y + (25 * cosOfAngles[orientation]))

    return nextX, nextY, orientation


def getNextPoint(comingFrom, previousOrientation, orientation, sinOfAngles, cosOfAngles, x, y, lengthOfLine):
    if lengthOfLine > 30:
        orientationDifference = min(abs(orientation - previousOrientation), abs((orientation + 12) - previousOrientation),
                                    abs(orientation - (previousOrientation + 12)))
        if orientationDifference > 3:
            orientation = previousOrientation

    if comingFrom <= 3:
        nextX = int(x + (25 * sinOfAngles[orientation]))
        nextY = int(y - (25 * cosOfAngles[orientation]))

        comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 0 and (2 <= comingFromNew <= 5):
            nextX = int(x - (25 * sinOfAngles[orientation]))
            nextY = int(y + (25 * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 1 and (4 <= comingFromNew <= 6):
            nextX = int(x - (25 * sinOfAngles[orientation]))
            nextY = int(y + (25 * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if (comingFrom == 2 or comingFrom == 3) and (comingFromNew == 0 or comingFromNew >= 6):
            nextX = int(x - (25 * sinOfAngles[orientation]))
            nextY = int(y + (25 * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

    else:
        nextX = int(x - (25 * sinOfAngles[orientation]))
        nextY = int(y + (25 * cosOfAngles[orientation]))

        comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if 8 <= previousOrientation <= 10 and 0 <= orientation <= 1:
            nextX = int(x + (25 * sinOfAngles[orientation]))
            nextY = int(y - (25 * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

            return nextX, nextY, comingFromNew, orientation

        if comingFrom == 6 and (0 <= comingFromNew <= 3):
            nextX = int(x + (25 * sinOfAngles[orientation]))
            nextY = int(y - (25 * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 7 and (2 <= comingFromNew <= 4):
            nextX = int(x + (25 * sinOfAngles[orientation]))
            nextY = int(y - (25 * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if (comingFrom == 4 or comingFrom == 5) and (comingFromNew == 7 or comingFromNew <= 2):
            nextX = int(x + (25 * sinOfAngles[orientation]))
            nextY = int(y - (25 * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

    return nextX, nextY, comingFromNew, orientation


def getNextPointMainLineB(previousOrientation, orientation, sinOfAngles, cosOfAngles, x, y, alreadyWentLeft, alreadyAddedPoints):
    """ Povolene orientacie 0-5 a 17-23 (ked som uz raz isiel dolava)
        - povolene ist dolava mam iba raz (a to iba v prvych 5 bodoch)
        - ked bude orientacia od 11 do 17 a pocet bodov je aspon 5 tak idem natvrdo doprava
        - ked je menej jak 5 bodov tak ked pojdem dolava tak jedine ked je uhol od 14 do 16 - inak idem priamo dole
        - ked je viac jak 5 bodov idem natvrdo vzdy doprava (ale obmedzim velkost zmeny)"""
    if 0 <= previousOrientation <= 1 and 9 <= orientation <= 11:
        orientation += 12

    if previousOrientation > 11:
        if previousOrientation >= 20 and orientation <= 2:
            pass
        else:
            orientation += 12

    if alreadyAddedPoints <= 8:
        if alreadyWentLeft and 12 <= orientation <= 16:
            orientation = 17

        if not alreadyWentLeft and 12 <= orientation <= 16:
            alreadyWentLeft = True

        """if previousOrientation - orientation > 4:
            orientation = previousOrientation - 4

        if previousOrientation - orientation < -4:
            orientation = previousOrientation + 4"""

    else:
        if alreadyAddedPoints > 30 and 8 > abs(previousOrientation - orientation) > 3:
            orientation = previousOrientation

        else:
            if 12 <= orientation <= 16:
                orientation = orientation - 12


        """if previousOrientation - orientation > 4:
            orientation = previousOrientation - 4

        if previousOrientation - orientation < -4:
            orientation = previousOrientation + 4"""

    """if alreadyWentLeft and 6 <= orientation <= 16 and 12 <= previousOrientation <= 22:
        orientation = 17"""

    if orientation > 11:
        nextX = int(x + (25 * sinOfAngles[orientation - 12]))
        nextY = int(y - (25 * cosOfAngles[orientation - 12]))
    else:
        nextX = int(x - (25 * sinOfAngles[orientation]))
        nextY = int(y + (25 * cosOfAngles[orientation]))

    """if 6 <= orientation <= 16:
        alreadyWentLeft = True"""

    return nextX, nextY, orientation, alreadyWentLeft


def getNextPointMainLineC(previousOrientation, orientation, sinOfAngles, cosOfAngles, x, y, comingFrom):
    if comingFrom <= 3:
        nextX = int(x + (25 * sinOfAngles[orientation]))
        nextY = int(y - (25 * cosOfAngles[orientation]))

        comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 0 and (2 <= comingFromNew <= 5):
            nextX = int(x - (25 * sinOfAngles[orientation]))
            nextY = int(y + (25 * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 1 and (4 <= comingFromNew <= 6):
            nextX = int(x - (25 * sinOfAngles[orientation]))
            nextY = int(y + (25 * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if (comingFrom == 2 or comingFrom == 3) and (comingFromNew == 0 or comingFromNew >= 6):
            if 3 <= previousOrientation <= 4 and 5 <= orientation <= 7:
                pass
            else:
                nextX = int(x - (25 * sinOfAngles[orientation]))
                nextY = int(y + (25 * cosOfAngles[orientation]))

                comingFromNew = comingFromWhere(x, y, nextX, nextY)

    else:
        nextX = int(x - (25 * sinOfAngles[orientation]))
        nextY = int(y + (25 * cosOfAngles[orientation]))

        comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 6 and (0 <= comingFromNew <= 3):
            nextX = int(x + (25 * sinOfAngles[orientation]))
            nextY = int(y - (25 * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 7 and (2 <= comingFromNew <= 4):
            nextX = int(x + (25 * sinOfAngles[orientation]))
            nextY = int(y - (25 * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if (comingFrom == 4 or comingFrom == 5) and (comingFromNew == 7 or comingFromNew <= 2):

            nextX = int(x + (25 * sinOfAngles[orientation]))
            nextY = int(y - (25 * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 4 and comingFromNew == 6:
            if 8 <= previousOrientation <= 10 and 0 <= orientation <= 2:
                nextX = int(x + (25 * sinOfAngles[orientation]))
                nextY = int(y - (25 * cosOfAngles[orientation]))

                comingFromNew = comingFromWhere(x, y, nextX, nextY)

    return nextX, nextY, comingFromNew, orientation


def getNextPointMainLineD(previousOrientation, orientation, sinOfAngles, cosOfAngles, x, y, comingFrom):
    orientationDifference = min(abs(orientation - previousOrientation), abs((orientation + 12) - previousOrientation),
                                abs(orientation - (previousOrientation + 12)))

    if comingFrom <= 3:
        nextX = int(x + (25 * sinOfAngles[orientation]))
        nextY = int(y - (25 * cosOfAngles[orientation]))

        comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 0 and (2 <= comingFromNew <= 5):
            nextX = int(x - (25 * sinOfAngles[orientation]))
            nextY = int(y + (25 * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 1 and (4 <= comingFromNew <= 6):
            nextX = int(x - (25 * sinOfAngles[orientation]))
            nextY = int(y + (25 * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if (comingFrom == 2 or comingFrom == 3) and (comingFromNew == 0 or comingFromNew >= 6):
            nextX = int(x - (25 * sinOfAngles[orientation]))
            nextY = int(y + (25 * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

    else:
        nextX = int(x - (25 * sinOfAngles[orientation]))
        nextY = int(y + (25 * cosOfAngles[orientation]))

        comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 6 and (0 <= comingFromNew <= 3):
            nextX = int(x + (25 * sinOfAngles[orientation]))
            nextY = int(y - (25 * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 7 and (2 <= comingFromNew <= 4):
            nextX = int(x + (25 * sinOfAngles[orientation]))
            nextY = int(y - (25 * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 4:
            a = 3

        if (comingFrom == 4 or comingFrom == 5) and (comingFromNew == 7 or comingFromNew <= 2):

            nextX = int(x + (25 * sinOfAngles[orientation]))
            nextY = int(y - (25 * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 4 and comingFromNew == 6:
            if 8 <= previousOrientation <= 11 and 0 <= orientation <= 2:
                nextX = int(x + (25 * sinOfAngles[orientation]))
                nextY = int(y - (25 * cosOfAngles[orientation]))

                comingFromNew = comingFromWhere(x, y, nextX, nextY)

    return nextX, nextY, comingFromNew, orientation


def findMainLineA(triradius, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks):
    points = [(triradius[1], triradius[0])]

    actualBlockRow = triradius[0] // blockHeight
    # rowInBlock = triradiusB[0] % blockHeight
    rowInBlock = 0
    actualBlockCol = triradius[1] // blockWidth
    colInBlock = triradius[1] % blockWidth

    currentBlockRow = actualBlockRow + 1
    currentBlockCol = actualBlockCol + 1
    previousOrientation = blocks[currentBlockRow][currentBlockCol]["orientation"]
    orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

    pointX = (currentBlockRow * blockHeight) + rowInBlock
    pointY = (currentBlockCol * blockWidth) + colInBlock
    points.append((pointY, pointX))

    currentX = int(pointX + (25 * sinOfAngles[orientation]))
    currentY = int(pointY - (25 * cosOfAngles[orientation]))

    points.append((currentY, currentX))

    comingFrom = 0

    i = 0
    while blocks[currentBlockRow][currentBlockCol]["background"] == 0:
        orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

        currentX, currentY, comingFrom, orientation = getNextPoint(comingFrom, previousOrientation, orientation, sinOfAngles,
                                                      cosOfAngles, currentX, currentY, len(points))
        currentBlockRow = currentX // blockHeight
        currentBlockCol = currentY // blockWidth
        previousOrientation = orientation
        points.append((currentY, currentX))
        i += 1

    return points


def findMainLineB(triradius, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks):
    points = [(triradius[1], triradius[0])]

    actualBlockRow = triradius[0] // blockHeight
    # rowInBlock = triradiusB[0] % blockHeight
    rowInBlock = 0
    actualBlockCol = triradius[1] // blockWidth
    colInBlock = triradius[1] % blockWidth

    currentBlockRow = actualBlockRow + 1
    currentBlockCol = actualBlockCol
    previousOrientation = blocks[currentBlockRow][currentBlockCol]["orientation"] + 12
    orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

    pointX = (currentBlockRow * blockHeight) + rowInBlock
    pointY = (currentBlockCol * blockWidth) + colInBlock
    points.append((pointY, pointX))

    currentX = int(pointX + (25 * sinOfAngles[orientation]))
    currentY = int(pointY - (25 * cosOfAngles[orientation]))

    points.append((currentY, currentX))

    comingFrom = 1
    alreadyWentLeft = False

    i = 0
    #while blocks[currentBlockRow][currentBlockCol]["palmprintSegment"] == 0:
    #while i < 35:
    while blocks[currentBlockRow][currentBlockCol]["background"] == 0:
        if i == 25:
            a = 3
        orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

        """currentX, currentY, comingFrom = getNextPoint(comingFrom, previousOrientation, orientation, sinOfAngles, cosOfAngles, currentX,
                                                      currentY)"""
        currentX, currentY, orientation, alreadyWentLeft = getNextPointMainLineB(previousOrientation, orientation, sinOfAngles, cosOfAngles, currentX, currentY, alreadyWentLeft, len(points))
        currentBlockRow = currentX // blockHeight
        currentBlockCol = currentY // blockWidth
        previousOrientation = orientation
        points.append((currentY, currentX))
        i += 1

    return points


def findMainLineC(triradius, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks):
    points = [(triradius[1], triradius[0])]

    actualBlockRow = triradius[0] // blockHeight
    rowInBlock = triradius[0] % blockHeight
    #rowInBlock = 0
    actualBlockCol = triradius[1] // blockWidth
    colInBlock = triradius[1] % blockWidth

    currentBlockRow = actualBlockRow + 1
    if colInBlock > 1:
        colInBlock -= 25
        currentBlockCol = actualBlockCol
    else:
        colInBlock += 25
        currentBlockCol = actualBlockCol - 1
    previousOrientation = blocks[currentBlockRow][currentBlockCol]["orientation"] + 12
    orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

    pointX = (currentBlockRow * blockHeight) + rowInBlock
    pointY = (currentBlockCol * blockWidth) + colInBlock
    points.append((pointY, pointX))

    currentX = int(pointX + (25 * sinOfAngles[orientation]))
    currentY = int(pointY - (25 * cosOfAngles[orientation]))

    points.append((currentY, currentX))

    comingFrom = 2

    i = 0
    #while blocks[currentBlockRow][currentBlockCol]["palmprintSegment"] == 0:
    #while i < 50:
    while blocks[currentBlockRow][currentBlockCol]["background"] == 0:
        orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

        """currentX, currentY, comingFrom = getNextPoint(comingFrom, previousOrientation, orientation, sinOfAngles, cosOfAngles, currentX,
                                                      currentY)"""
        currentX, currentY, comingFrom, orientation = getNextPointMainLineC(previousOrientation, orientation, sinOfAngles, cosOfAngles, currentX, currentY, comingFrom)
        currentBlockRow = currentX // blockHeight
        currentBlockCol = currentY // blockWidth
        previousOrientation = orientation
        points.append((currentY, currentX))
        i += 1

    return points


def findMainLineD(triradius, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks):
    points = [(triradius[1], triradius[0])]

    actualBlockRow = (triradius[0] + 10) // blockHeight
    rowInBlock = (triradius[0] + 10) % blockHeight
    #rowInBlock = 0
    actualBlockCol = triradius[1] // blockWidth
    colInBlock = triradius[1] % blockWidth

    #currentBlockRow = actualBlockRow + 1
    currentBlockRow = actualBlockRow
    currentBlockCol = actualBlockCol - 1
    previousOrientation = blocks[currentBlockRow][currentBlockCol]["orientation"]
    orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

    pointX = (currentBlockRow * blockHeight) + rowInBlock
    pointY = (currentBlockCol * blockWidth) + colInBlock
    points.append((pointY, pointX))

    currentX = int(pointX + (25 * sinOfAngles[orientation]))
    currentY = int(pointY - (25 * cosOfAngles[orientation]))

    points.append((currentY, currentX))

    comingFrom = 2

    i = 0
    while blocks[currentBlockRow][currentBlockCol]["background"] == 0:
    #while blocks[currentBlockRow][currentBlockCol]["palmprintSegment"] == 0:
    #while i < 20:
        if i == 4:
            a = 2
        orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

        """currentX, currentY, comingFrom = getNextPoint(comingFrom, previousOrientation, orientation, sinOfAngles, cosOfAngles, currentX,
                                                      currentY)"""
        currentX, currentY, comingFrom, orientation = getNextPointMainLineD(previousOrientation, orientation, sinOfAngles, cosOfAngles,
                                                            currentX, currentY, comingFrom)
        currentBlockRow = currentX // blockHeight
        currentBlockCol = currentY // blockWidth
        previousOrientation = orientation
        points.append((currentY, currentX))
        i += 1

    return points


def findMainLineT(triradius, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks):
    points = [(triradius[1], triradius[0])]

    actualBlockRow = triradius[0] // blockHeight
    # rowInBlock = triradiusB[0] % blockHeight
    rowInBlock = 0
    actualBlockCol = triradius[1] // blockWidth
    colInBlock = triradius[1] % blockWidth

    currentBlockRow = actualBlockRow - 1
    currentBlockCol = actualBlockCol
    previousOrientation = blocks[currentBlockRow][currentBlockCol]["orientation"]
    orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

    pointX = (currentBlockRow * blockHeight) + rowInBlock
    pointY = (currentBlockCol * blockWidth) + colInBlock
    points.append((pointY, pointX))

    currentX = int(pointX - (25 * sinOfAngles[orientation]))
    currentY = int(pointY + (25 * cosOfAngles[orientation]))

    points.append((currentY, currentX))

    comingFrom = 5

    i = 0
    while blocks[currentBlockRow][currentBlockCol]["background"] == 0:
    #while blocks[currentBlockRow][currentBlockCol]["palmprintSegment"] == 0:
    #while i < 85:
        if i == 82:
            a = 3
        orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

        currentX, currentY, comingFrom, orientation = getNextPoint(comingFrom, previousOrientation, orientation, sinOfAngles, cosOfAngles, currentX,
                                                      currentY, len(points))
        currentBlockRow = currentX // blockHeight
        currentBlockCol = currentY // blockWidth
        previousOrientation = orientation
        points.append((currentY, currentX))
        i += 1

    return points


def findMainLineOutSegment(point, blocks, blockHeight, blockWidth, edgePointsOfSegments):
    currentBlockRow = point[1] // blockHeight
    currentBlockCol = point[0] // blockWidth

    if blocks[currentBlockRow][currentBlockCol]["palmprintSegment"] != 0:
        return blocks[currentBlockRow][currentBlockCol]["palmprintSegment"]

    else:
        minDistance = 1000
        minDistanceIndex = 0
        for i in range(len(edgePointsOfSegments)):
            edgePointRow, edgePointCol = edgePointsOfSegments[i]
            distance = math.hypot(edgePointRow - currentBlockRow, edgePointCol - currentBlockCol)
            if distance < minDistance:
                minDistance = distance
                minDistanceIndex = i

        if minDistanceIndex % 2 == 0:
            minDistanceIndex += 1

        return minDistanceIndex

    #print("NOT FOUND PALMPRINT SEGMENT!!!")
    #return 0


def singularPoints(blocks, blockImage, angles, leftOfFinger2, between23, between34, between45, rightOfFinger5,
                   directoryName, edgePointsOfSegments):
    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    possibleTriradiusBlocks = []
    sinOfAngles = calculateSinOfAngles(angles)
    cosOfAngles = calculateCosOfAngles(angles)
    sinOfAngles5 = calculateSinOfAngles5(angles)
    cosOfAngles5 = calculateCosOfAngles5(angles)

    triradiusA, blocks = detectTriradiusA(blocks, blockImage, angles, leftOfFinger2, between23)
    triradiusB, blocks = detectTriradiusB(blocks, blockImage, angles, between23, between34)
    triradiusC, blocks = detectTriradiusB(blocks, blockImage, angles, between34, between45)
    triradiusD, blocks = detectTriradiusD(blocks, blockImage, angles, between45, rightOfFinger5)
    """triradiusA = (0, 0)
    triradiusB = (0, 0)
    triradiusC = (0, 0)
    triradiusD = (0, 0)"""
    triradiusT, blocks = detectTriradiusT(blocks, blockImage, angles)

    #mainLines = []

    #saveTriradiusPalmprint(blockImage, triradiusA, triradiusB, triradiusC, triradiusD, triradiusT, mainLines)

    """ Inicializuje masku orientacii """
    orientationMask = np.zeros((blockRows, blockCols), dtype=np.uint8)

    """ Najde v ktorych blokoch bola urcena orientacia (mimo regionov kde su triradie) """
    for row in range(blockRows):
        for col in range(blockCols):
            if blocks[row][col]["background"] == 0:
                orientationMask[row][col] = 1
            if blocks[row][col]["triradiusRegion"] == 1:
                orientationMask[row][col] = 0

    """ Chcem najst region o velkosti 5x5 blokov, kde je confidence vsetkych orientacii 100 """
    suitable = False
    seeds = []
    for row in range(blockRows):
        for col in range(blockCols):
            if orientationMask[row][col] == 1:
                suitable = True
                for row2 in range(row, row + 5):
                    for col2 in range(col, col + 5):
                        if orientationMask[row2][col2] == 0 or blocks[row2][col2]["orientationConfidence"] != 100:
                            suitable = False
                            break
                    if not suitable:
                        break

                if suitable:
                    for row2 in range(row, row + 5):
                        for col2 in range(col, col + 5):
                            seeds.append((row2, col2))
                            orientationMask[row2][col2] = 2
                    break
        if suitable:
            break
    """ Mam najdeny region o velkosti 5x5 - vsetky jeho bloky som v orientationMask nastavil na 2 (uz spracovane) a 
    pridal som ich do seeds (bloky, ktorych susedia budu prehladavany) """

    """ Postupne zvacsujem region """
    while seeds:
        seedX, seedY = seeds.pop(0)
        """ Vyberiem si prvy blok zo seeds a pozeram sa na jeho susedov """
        for row in range(seedX - 1, seedX + 2):
            for col in range(seedY - 1, seedY + 2):
                """ Ked to neni odtlacok alebo uz je spracovany, nerobim nic """
                if row == 86 and col == 72:
                    a = 2
                if orientationMask[row][col] == 0 or orientationMask[row][col] == 2:
                    continue
                else:
                    """ Ked je jeho confindence 100 tak ho nastavim ako spracovany, pridam ho do seeds a nerobim nic """
                    if blocks[row][col]["orientationConfidence"] == 100:
                        seeds.append((row, col))
                        orientationMask[row][col] = 2
                        continue
                    else:
                        """ V opacnom pripade sa pokusim dany blok vyhladit """
                        neighbourOrientations = []
                        """ Najden 8 orientacii susednych blokov, ktore uz boli spracovane alebo ktorych confidence 
                        je 100 """
                        for r in range(row - 1, row + 2):
                            for c in range(col - 1, col + 2):
                                """ Okrem aktualne bloku, ktory spracovanam ofc """
                                if r == row and c == col:
                                    continue
                                else:
                                    if orientationMask[r][c] == 2 or blocks[r][c]["orientationConfidence"] == 100:
                                        neighbourOrientations.append(blocks[r][c]["orientation"])

                        """ Najprv pripad, ze ked su vsetky susedne rovnake tak ho nastavim tiez na rovnaky """
                        if len(neighbourOrientations) == 8:
                            myOrientation = blocks[row][col]["orientation"]
                            allSame = True
                            firstOrientation = neighbourOrientations[0]
                            for neighbour in neighbourOrientations:
                                if neighbour != firstOrientation:
                                    allSame = False
                                    break
                            if allSame:
                                blocks[row][col]["orientation"] = firstOrientation
                            else:
                                """allSimilar = True
                                for neighbour in neighbourOrientations:
                                    if abs(neighbour - firstOrientation) > 1:
                                        allSimilar = False
                                        break
                                if allSimilar:
                                    if abs(myOrientation - firstOrientation) > 1:
                                        blocks[row][col]["orientation"] = firstOrientation"""
                                minOrientation = min(neighbourOrientations)
                                maxOrientation = max(neighbourOrientations)
                                orientationDifference = min(abs(maxOrientation - minOrientation),
                                                            abs(maxOrientation - (minOrientation + 12)))
                                if orientationDifference <= 2:
                                    difference = min(abs(myOrientation - firstOrientation),
                                                     abs((myOrientation + 12) - firstOrientation),
                                                     abs(myOrientation - (firstOrientation + 12)))
                                    if difference > 1:
                                        # Napr v pripade ze maxOrientation je 11 a min je 0
                                        if maxOrientation - minOrientation > 2:
                                            blocks[row][col]["orientation"] = minOrientation
                                        else:
                                            blocks[row][col]["orientation"] = (minOrientation + maxOrientation) // 2

                        else:
                            if len(neighbourOrientations) > 0:
                                myOrientation = blocks[row][col]["orientation"]
                                firstOrientation = neighbourOrientations[0]
                                """allSimilar = True
                                for neighbour in neighbourOrientations:
                                    if abs(neighbour - firstOrientation) > 1:
                                        allSimilar = False
                                        break
                                if allSimilar:
                                    if abs(myOrientation - firstOrientation) > 1:
                                        blocks[row][col]["orientation"] = firstOrientation"""
                                minOrientation = min(neighbourOrientations)
                                maxOrientation = max(neighbourOrientations)
                                orientationDifference = min(abs(maxOrientation - minOrientation),
                                                            abs(maxOrientation - (minOrientation + 12)))
                                if orientationDifference <= 2:
                                    difference = min(abs(myOrientation - firstOrientation),
                                                     abs((myOrientation + 12) - firstOrientation),
                                                     abs(myOrientation - (firstOrientation + 12)))
                                    if difference > 1:
                                        # Napr v pripade ze maxOrientation je 11 a min je 0
                                        if maxOrientation - minOrientation > 2:
                                            blocks[row][col]["orientation"] = minOrientation
                                        else:
                                            blocks[row][col]["orientation"] = (minOrientation + maxOrientation) // 2

                        seeds.append((row, col))
                        orientationMask[row][col] = 2

    for row in range(blockRows):
        for col in range(blockCols):
            if row == 58 and col == 63:
                b = 2
            if orientationMask[row][col] == 2 and blocks[row][col]["orientationConfidence"] != 100:
                orientationsInNeighbourhood = [0] * len(angles)
                for r in range(row - 1, row + 2):
                    for c in range(col - 1, col + 2):
                        
                        if r == row and c == col:
                            continue
                        else:
                            if orientationMask[r][c] != 0:
                                orientationsInNeighbourhood[blocks[r][c]["orientation"]] += 1
                if len(orientationsInNeighbourhood) < 2:
                    continue

                maxOrientations = max(orientationsInNeighbourhood)
                maxOrientationsValue = orientationsInNeighbourhood.index(maxOrientations)
                if maxOrientationsValue == 0:
                    numberOfNeighboursWithSimilarOrientations = maxOrientations + orientationsInNeighbourhood[11] + \
                                                                orientationsInNeighbourhood[maxOrientationsValue + 1]
                elif maxOrientationsValue == 11:
                    numberOfNeighboursWithSimilarOrientations = maxOrientations + orientationsInNeighbourhood[maxOrientationsValue - 1] + \
                                                                orientationsInNeighbourhood[0]
                else:
                    numberOfNeighboursWithSimilarOrientations = maxOrientations + orientationsInNeighbourhood[maxOrientationsValue - 1] + orientationsInNeighbourhood[maxOrientationsValue + 1]

                if numberOfNeighboursWithSimilarOrientations >= 5:
                    myOrientation = blocks[row][col]["orientation"]
                    difference = min(abs(myOrientation - maxOrientationsValue), abs((myOrientation + 12) - maxOrientationsValue),
                                     abs(myOrientation - (maxOrientationsValue + 12)))
                    if difference > 2:
                        blocks[row][col]["orientation"] = maxOrientationsValue

    saveOrientationAfterSmoothing(blockImage, blocks, angles, directoryName)
    """image = mergeBlocksToImage(blockImage)
    saveOrientationsImage(image, blockImage, blocks, angles)"""

    """for row in range(blockRows):
        for col in range(blockCols):
            if blocks[row][col]["triradiusRegion"] == 1:
                blockImage[row][col] = 255

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

    savePalmprint(image)
    exit(2)"""

    mainLines = []

    triradiusAout = 0
    if triradiusA != (0, 0):
        mainLineA = findMainLineA(triradiusA, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks)
        mainLines.append(mainLineA)
        triradiusAout = findMainLineOutSegment(mainLineA[-1], blocks, blockHeight, blockWidth, edgePointsOfSegments)
        if triradiusAout == 55:
            triradiusAout = "5\""
        print(f"A - {triradiusAout}")

    triradiusBout = 0
    if triradiusB != (0, 0):
        mainLineB = findMainLineB(triradiusB, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks)
        mainLines.append(mainLineB)
        triradiusBout = findMainLineOutSegment(mainLineB[-1], blocks, blockHeight, blockWidth, edgePointsOfSegments)
        if triradiusBout == 55:
            triradiusBout = "5\""
        print(f"B - {triradiusBout}")

    triradiusCout = 0
    if triradiusC != (0, 0):
        mainLineC = findMainLineC(triradiusC, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks)
        mainLines.append(mainLineC)
        triradiusCout = findMainLineOutSegment(mainLineC[-1], blocks, blockHeight, blockWidth, edgePointsOfSegments)
        if triradiusCout == 55:
            triradiusCout = "5\""
        print(f"C - {triradiusCout}")

    triradiusDout = 0
    if triradiusD != (0, 0):
        mainLineD = findMainLineD(triradiusD, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks)
        triradiusDout = findMainLineOutSegment(mainLineD[-1], blocks, blockHeight, blockWidth, edgePointsOfSegments)
        if triradiusDout == 55:
            triradiusDout = "5\""
        if (triradiusCout == "5\'" or triradiusCout == "5\"") and triradiusDout == 9:
            mainLineD = findMainLineD(triradiusD, blockHeight, blockWidth, sinOfAngles5, cosOfAngles5, blocks)
            triradiusDout = findMainLineOutSegment(mainLineD[-1], blocks, blockHeight, blockWidth, edgePointsOfSegments)
        if (triradiusBout == "5\'" or triradiusBout == "5\"") and triradiusDout == 11:
            mainLineD = findMainLineD(triradiusD, blockHeight, blockWidth, sinOfAngles5, cosOfAngles5, blocks)
            triradiusDout = findMainLineOutSegment(mainLineD[-1], blocks, blockHeight, blockWidth, edgePointsOfSegments)
        mainLines.append(mainLineD)
        print(f"D - {triradiusDout}")

    triradiusTout = 0
    if triradiusT != (0, 0):
        mainLineT = findMainLineT(triradiusT, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks)
        mainLines.append(mainLineT)
        triradiusTout = findMainLineOutSegment(mainLineT[-1], blocks, blockHeight, blockWidth, edgePointsOfSegments)
        if triradiusTout == 1313:
            triradiusTout = "13\'"
        print(f"T - {triradiusTout}")

    saveTriradiusPalmprint(blockImage, triradiusA, triradiusB, triradiusC, triradiusD, triradiusT, mainLines, directoryName)

    #block = blockImage[60][50]

    """for r in range(blockRows):
        for c in range(blockCols):
            if blocks[r][c]["orientationConfidence"] == 100:
                saveBlock(blockImage[r][c])"""

    #return blockImage


def main():
    blockHeight = 50
    blockWidth = 50
    start = timer()

    directoryName = "NORM_8_M_L_CP_001"

    if not os.path.exists(directoryName):
        os.mkdir(directoryName)

    image = loadPalmprint(f'dlane/muzi/{directoryName}.tif')
    image = image[:-2]
    blockImage, blocks = splitImageToBlocks(image, blockHeight, blockWidth)

    """saveBlock(blockImage[61][64])
    exit(3)"""

    """ --------------------------------------------------- """

    """blockImage[65][52] = 0

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

    savePalmprint(image)
    exit(2)"""

    #showPalmprint(blockImage[50][50])

    image, blockImage, blocks, leftOfFinger2, between23, between34, between45, rightOfFinger5, bottomPointOfSegment1, \
    bottomPointOfSegment3, palmprintBorder, edgePointsOfSegments = segmentation(blockImage, blocks, image, directoryName)

    blocks = principleLines(image, blocks, blockImage, leftOfFinger2, bottomPointOfSegment1, rightOfFinger5,
                            bottomPointOfSegment3, palmprintBorder)

    saveSegmentedPalmprint(blockImage, blocks, directoryName)
    blocks, angles = orientationField(blockImage, blocks, image, directoryName)

    """blockRows, blockCols, _, _ = blockImage.shape
    for row in range(blockRows):
        for col in range(blockCols):
            if blocks[row][col]["orientationConfidence"] == 100:
                blockImage[row][col] = 0

    image = mergeBlocksToImage(blockImage)

    savePalmprint(image)
    exit(2)"""

    sinOfAngles = calculateSinOfAngles(angles)
    cosOfAngles = calculateCosOfAngles(angles)

    blockRows, blockCols, _, _ = blockImage.shape

    for row in range(blockRows):
        for col in range(blockCols):
            if blocks[row][col]["orientationConfidence"] != 100:
                continue
            block = blockImage[row][col]
            orient = blocks[row][col]["orientation"]
            print(orient)
            i = 25
            j = 25
            windowHeight = 50
            windowWidth = 25

            orienta = (orient + 5) % 12

            blur = cv2.GaussianBlur(block, (5, 5), 0)
            ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            saveBlock(block)

            blur2 = cv2.GaussianBlur(th3, (5, 5), 0)

            sum = [0] * windowHeight
            for k in range(windowHeight):
                sum[k] = 0
                numberOfAdded = 0
                for d in range(windowWidth):
                    u = i + (d - windowWidth / 2) * cosOfAngles[orienta] + (k - windowHeight / 2) * sinOfAngles[orienta]
                    v = j + (d - windowWidth / 2) * sinOfAngles[orienta] + (windowHeight / 2 - k) * cosOfAngles[orienta]
                    u = round(u)
                    v = round(v)
                    if 0 <= u <= 49 and 0 <= v <= 49:
                        sum[k] += blur[u][v]
                        numberOfAdded += 1
                        # block[u][v] = 0
                sum[k] = int(sum[k] / numberOfAdded)

            sumSmoothened = [0] * windowHeight
            sumSmoothened[0] = sum[0]
            sumSmoothened[-1] = sum[-1]
            for aaa in range(1, len(sum) - 1):
                sumSmoothened[aaa] = int((sum[aaa - 1] + sum[aaa] + sum[aaa + 1]) / 3)

            priemer = np.mean(sum)
            hranica = [priemer] * 50

            a = []
            b = []
            if sum[0] > priemer:
                nadHranicou = True
                podHranicou = False
            else:
                nadHranicou = False
                podHranicou = True

            for k in range(1, windowHeight):
                if sum[k] > priemer and podHranicou:
                    a.append(k)
                    nadHranicou = True
                    podHranicou = False
                if sum[k] <= priemer and nadHranicou:
                    b.append(k)
                    nadHranicou = False
                    podHranicou = True

            if len(a) < 3 or len(b) < 3:
                break

            else:
                distanceA = 0
                for bottomUpIndex in range(len(a) - 1):
                    distanceA += a[bottomUpIndex + 1] - a[bottomUpIndex]
                averageDistanceA = distanceA / (len(a) - 1)
                validA = True
                for i in range(len(a) - 1):
                    distance = a[i + 1] - a[i]
                    if abs(distance - averageDistanceA) > 5:
                        validA = False
                        break
                if not validA:
                    print(f"{a} - {averageDistanceA} => {1 / averageDistanceA} --- NOT VALID")
                else:
                    print(f"{a} - {averageDistanceA} => {1 / averageDistanceA} --- OK")

                distanceB = 0
                for upBottomIndex in range(len(b) - 1):
                    distanceB += b[upBottomIndex + 1] - b[upBottomIndex]
                averageDistanceB = distanceB / (len(b) - 1)
                validB = True
                for i in range(len(b) - 1):
                    distance = b[i + 1] - b[i]
                    if abs(distance - averageDistanceB) > 5:
                        validB = False
                        break
                if not validB:
                    print(f"{b} - {averageDistanceB} => {1 / averageDistanceB} --- NOT VALID")
                else:
                    print(f"{b} - {averageDistanceB} => {1 / averageDistanceB} --- OK")

                if validA and validB:
                    averageDistance = (averageDistanceA + averageDistanceB) // 2
                    blockFrequency = 1 / averageDistance
                    blocks[row][col]["frequency"] = blockFrequency
                    print(f"Frekvencia - {blockFrequency}")

                    if a[0] > b[0]:
                        b.pop(0)

                    if len(a) > len(b):
                        numberOfWidthCount = len(a) - 2
                    else:
                        numberOfWidthCount = len(a) - 1

                    width = 0
                    i = 0
                    middleOfValleys = copy.deepcopy(hranica)
                    for crossingIndex in range(numberOfWidthCount):
                        i += 1
                        firstBottomUp = a[crossingIndex]
                        secondBottomUp = a[crossingIndex + 1]
                        firstUpBottom = b[crossingIndex]
                        secondUpBottom = b[crossingIndex + 1]
                        middleOfValley1 = (firstUpBottom + firstBottomUp) / 2
                        middleOfValley2 = (secondUpBottom + secondBottomUp) / 2
                        middleOfValleys[int(middleOfValley1)] = 255
                        middleOfValleys[int(middleOfValley2)] = 255
                        width += middleOfValley2 - middleOfValley1

                    blocks[row][col]["ridgeWidth"] = width / i
                    print(f"Width - {width / i}")


            # print(sum)

            """plt.plot(sum)
            plt.plot(hranica)
            plt.plot(sumSmoothened)
            if validA and validB:
                plt.plot(middleOfValleys)
            plt.savefig('threshBez.png')

            # function to show the plot
            plt.show()"""

            # saveBlock(block)

    frequency = 0
    frequencyCount = 0
    width = 0
    widthCount = 0
    for row in range(blockRows):
        for col in range(blockCols):
            if blocks[row][col]["frequency"] != 0:
                frequency += blocks[row][col]["frequency"]
                frequencyCount += 1
            if blocks[row][col]["ridgeWidth"] != 0:
                width += blocks[row][col]["ridgeWidth"]
                widthCount += 1


    f = open(f"{directoryName}/widthAndFrequency.txt", "a")
    f.write(f"Frequency - {frequency / frequencyCount}\n")
    f.write(f"Ridge width - {width / widthCount}")
    f.close()
    #orientationField(blockImage[20][20])

    singularPoints(blocks, blockImage, angles, leftOfFinger2, between23, between34, between45, rightOfFinger5,
                   directoryName, edgePointsOfSegments)

    #image = mergeBlocksToImage(blockImage)

    #savePalmprint(image)

    end = timer()
    print(f"Celkovo: {end - start}")


if __name__ == '__main__':
    main()
