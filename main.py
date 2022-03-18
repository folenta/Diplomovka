# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
from typing import List
from timeit import default_timer as timer

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


def loadPalmprint(fileName):
    image = cv2.imread(fileName, 0)
    return image


def showPalmprint(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def savePalmprint(image):
    cv2.imwrite('newPalmprint.bmp', image)


def saveTriradiusPalmprint(blockImage, triradiusA, triradiusB, triradiusC, triradiusD, mainLines):
    image = mergeBlocksToImage(blockImage)

    image = cv2.circle(image, (triradiusA[1], triradiusA[0]), 10, (0, 0, 255), 3)
    image = cv2.circle(image, (triradiusB[1], triradiusB[0]), 10, (0, 0, 255), 3)
    image = cv2.circle(image, (triradiusC[1], triradiusC[0]), 10, (0, 0, 255), 3)
    image = cv2.circle(image, (triradiusD[1], triradiusD[0]), 10, (0, 0, 255), 3)

    for mainLine in mainLines:
        for mainLinePoint in range(len(mainLine) - 1):
            image = cv2.line(image, mainLine[mainLinePoint], mainLine[mainLinePoint + 1], (0, 0, 255), 3)

    cv2.imwrite('triradiusPalmprint.bmp', image)


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

    whitePixelsMask = flood(image, seed, tolerance=40)

    toSearch = [seed]
    alreadySearched = []

    while toSearch:
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


def orientedProjection(image, blocks, blockImage, colCandidate):
    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape

    """ Najdem prvy a posledny blok, ktory lezi v odtlacku v danom stlpci """
    firstBlockRow, lastBlockRow = findFirstAndLastBlockRow(blocks, colCandidate)

    """ Vypocitam stred stlpca """
    middleBlockRow = (firstBlockRow + lastBlockRow) // 2

    """ Najdem prvy a posledny riadok smerovej projekcie """
    firstSearchRow = firstBlockRow * blockHeight
    lastSearchRow = middleBlockRow * blockHeight

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
                           averagePalmprintHeight, principleLinePoints):
    left = True

    """ Vyberie stlpce v ktorych moze prebehnut smerova projekcia -  """
    colCandidatesForProjection = getColCandidateForProjection(palmprintBlocksInColumn, firstCol, lastCol,
                                                              averagePalmprintHeight, blocks, left)

    """ Vyberie druheho kandidata (prvy moze byt na hranici odtlacku) """
    colCandidate = colCandidatesForProjection[1]

    """ Vykonanie smerovej projekcie v danom stlpci """
    pointOnPrincipleLine = orientedProjection(image, blocks, blockImage, colCandidate)

    """ Vyplnenie flekcnej ryhy """
    principleLinesPoints = fillPrincipleLine(pointOnPrincipleLine, blockImage, blocks, principleLinePoints)

    return principleLinesPoints


def orientedProjectionRight(image, blocks, blockImage, palmprintBlocksInColumn, firstCol, lastCol,
                            averagePalmprintHeight, principleLinesPoints):
    left = False

    """ Vyberie stlpce v ktorych moze prebehnut smerova projekcia -  """
    colCandidatesForProjection = getColCandidateForProjection(palmprintBlocksInColumn, firstCol, lastCol,
                                                              averagePalmprintHeight, blocks, left)

    """ Vyberie druheho kandidata (prvy moze byt na hranici odtlacku) """
    colCandidate = colCandidatesForProjection[1]

    """ Vykonanie smerovej projekcie v danom stlpci """
    pointOnPrincipleLine = orientedProjection(image, blocks, blockImage, colCandidate)

    """ Ked uz bod lezi v detekovanej flekcnej ryhe, nie je potrebne ju zase vyplnovat """
    if not principleLinesPoints[pointOnPrincipleLine[0]][pointOnPrincipleLine[1]]:
        """ Vyplnenie flekcnej ryhy """
        principleLinesPoints = fillPrincipleLine(pointOnPrincipleLine, blockImage, blocks, principleLinesPoints)

    return principleLinesPoints


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


def principleLines(image, blocks, blockImage):
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

    """ Smerova projekcia v lavej hornej polovici odtlacku """
    principleLinesPoints = orientedProjectionLeft(image, blocks, blockImage, palmprintBlocksInColumn, firstCol, lastCol,
                                                  averagePalmprintHeight, principleLinesPoints)

    """ Smerova projekcia v pravej hornej polovici odtlacku """
    principleLinesPoints = orientedProjectionRight(image, blocks, blockImage, palmprintBlocksInColumn, firstCol,
                                                   lastCol, averagePalmprintHeight, principleLinesPoints)

    """ Smerova projekcia v strednej dolnej polovici odtlacku """
    principleLinesPoints = orientedProjectionBottom(image, blocks, blockImage, palmprintBlocksInRow, firstRow,
                                                    lastRow, averagePalmprintWidth, principleLinesPoints)

    colorImage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    colorImage[principleLinesPoints] = (255, 0, 0)

    savePalmprint(colorImage)


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


def localizeTriradius(triradiusRegion, smallBlockRows, smallBlockCols, firstRow, firstCol, blockImage):
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

    """ Tota cast kodu vykresli mozne bloky v ktorych moze byt triradius bielou farbou """
    triradiusBlocks = copy.deepcopy(blockImage)

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
    cv2.destroyAllWindows()

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


    """ Tota cast kodu ulozi obraz s moznymi blokmi v ktorych moze byt triradius bielou farbou """
    triradiusBlocks = copy.deepcopy(blockImage)

    for a in possibleTriradiusBlocks:
        triradiusBlocks[a[0]][a[1]] = 255

    image = mergeBlocksToImage(triradiusBlocks)

    cv2.imwrite('triradiusPalmprint1.bmp', image)

    return (0, 0)

    # TODO najst iba jeden possible region

    firstRow = possibleTriradiusBlocks[0][0]
    lastRow = possibleTriradiusBlocks[-1][0]
    firstCol = min(possibleTriradiusBlocks, key=lambda t: t[1])[1]
    lastCol = max(possibleTriradiusBlocks, key=lambda t: t[1])[1]

    blocksInHeight = lastRow - firstRow + 1
    blocksInWidth = lastCol - firstCol + 1

    smallBlockRows = blocksInHeight * blockHeight // 25
    smallBlockCols = blocksInWidth * blockWidth // 25

    triradiusRegion = findOrientationsInTriradiusBlocks(blockHeight, blockWidth, blockImage, angles, smallBlockRows,
                                                        smallBlockCols, firstRow, firstCol)

    #blockImage = localizeTriradius(triradiusRegion, smallBlockRows, smallBlockCols, firstRow, firstCol, blockImage)
    triradius = localizeTriradius(triradiusRegion, smallBlockRows, smallBlockCols, firstRow, firstCol, blockImage)

    #return blockImage
    return triradius


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

    triradiusBlocks = copy.deepcopy(blockImage)

    for a in possibleTriradiusBlocks:
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
    cv2.destroyAllWindows()

    #showPalmprint(image)



    return (0, 0)

    # TODO najst iba jeden possible region

    firstRow = possibleTriradiusBlocks[0][0]
    lastRow = possibleTriradiusBlocks[-1][0]
    firstCol = min(possibleTriradiusBlocks, key=lambda t: t[1])[1]
    lastCol = max(possibleTriradiusBlocks, key=lambda t: t[1])[1]

    blocksInHeight = lastRow - firstRow + 1
    blocksInWidth = lastCol - firstCol + 1

    smallBlockRows = blocksInHeight * blockHeight // 25
    smallBlockCols = blocksInWidth * blockWidth // 25

    triradiusRegion = findOrientationsInTriradiusBlocks(blockHeight, blockWidth, blockImage, angles, smallBlockRows,
                                                        smallBlockCols, firstRow, firstCol)

    # blockImage = localizeTriradius(triradiusRegion, smallBlockRows, smallBlockCols, firstRow, firstCol, blockImage)
    triradius = localizeTriradius(triradiusRegion, smallBlockRows, smallBlockCols, firstRow, firstCol, blockImage)

    # return blockImage
    return triradius


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

    triradiusBlocks = copy.deepcopy(blockImage)

    for a in possibleTriradiusBlocks:
        triradiusBlocks[a[0]][a[1]] = 255

    image = mergeBlocksToImage(triradiusBlocks)

    cv2.imwrite('triradiusPalmprint3.bmp', image)

    return (0, 0)

    # TODO najst iba jeden possible region

    firstRow = possibleTriradiusBlocks[0][0]
    lastRow = possibleTriradiusBlocks[-1][0]
    firstCol = min(possibleTriradiusBlocks, key=lambda t: t[1])[1]
    lastCol = max(possibleTriradiusBlocks, key=lambda t: t[1])[1]

    blocksInHeight = lastRow - firstRow + 1
    blocksInWidth = lastCol - firstCol + 1

    smallBlockRows = blocksInHeight * blockHeight // 25
    smallBlockCols = blocksInWidth * blockWidth // 25

    triradiusRegion = findOrientationsInTriradiusBlocks(blockHeight, blockWidth, blockImage, angles, smallBlockRows,
                                                        smallBlockCols, firstRow, firstCol)

    # blockImage = localizeTriradius(triradiusRegion, smallBlockRows, smallBlockCols, firstRow, firstCol, blockImage)
    triradius = localizeTriradius(triradiusRegion, smallBlockRows, smallBlockCols, firstRow, firstCol, blockImage)

    # return blockImage
    return triradius


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


def getNextPoint(comingFrom, orientation, sinOfAngles, cosOfAngles, x, y):
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

        if (comingFrom == 4 or comingFrom == 5) and (comingFromNew == 7 or comingFromNew <= 2):
            nextX = int(x + (25 * sinOfAngles[orientation]))
            nextY = int(y - (25 * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

    return nextX, nextY, comingFromNew


def findMainLineA(triradius, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks):
    points = [(triradius[1], triradius[0])]

    actualBlockRow = triradius[0] // blockHeight
    # rowInBlock = triradiusB[0] % blockHeight
    rowInBlock = 0
    actualBlockCol = triradius[1] // blockWidth
    colInBlock = triradius[1] % blockWidth

    currentBlockRow = actualBlockRow + 1
    currentBlockCol = actualBlockCol + 1
    orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

    pointX = (currentBlockRow * blockHeight) + rowInBlock
    pointY = (currentBlockCol * blockWidth) + colInBlock
    points.append((pointY, pointX))

    currentX = int(pointX + (25 * sinOfAngles[orientation]))
    currentY = int(pointY - (25 * cosOfAngles[orientation]))

    points.append((currentY, currentX))

    comingFrom = 0

    i = 0
    while i < 100:
        orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

        currentX, currentY, comingFrom = getNextPoint(comingFrom, orientation, sinOfAngles, cosOfAngles, currentX,
                                                      currentY)
        currentBlockRow = currentX // blockHeight
        currentBlockCol = currentY // blockWidth
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
    orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

    pointX = (currentBlockRow * blockHeight) + rowInBlock
    pointY = (currentBlockCol * blockWidth) + colInBlock
    points.append((pointY, pointX))

    currentX = int(pointX + (25 * sinOfAngles[orientation]))
    currentY = int(pointY - (25 * cosOfAngles[orientation]))

    points.append((currentY, currentX))

    comingFrom = 1

    i = 0
    while blocks[currentBlockRow][currentBlockCol]["palmprintSegment"] == 0:
    #while i < 100:
        orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

        currentX, currentY, comingFrom = getNextPoint(comingFrom, orientation, sinOfAngles, cosOfAngles, currentX,
                                                      currentY)
        currentBlockRow = currentX // blockHeight
        currentBlockCol = currentY // blockWidth
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
    orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

    pointX = (currentBlockRow * blockHeight) + rowInBlock
    pointY = (currentBlockCol * blockWidth) + colInBlock
    points.append((pointY, pointX))

    currentX = int(pointX + (25 * sinOfAngles[orientation]))
    currentY = int(pointY - (25 * cosOfAngles[orientation]))

    points.append((currentY, currentX))

    comingFrom = 2

    i = 0
    while blocks[currentBlockRow][currentBlockCol]["palmprintSegment"] == 0:
    #while i < 100:
        orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

        currentX, currentY, comingFrom = getNextPoint(comingFrom, orientation, sinOfAngles, cosOfAngles, currentX,
                                                      currentY)
        currentBlockRow = currentX // blockHeight
        currentBlockCol = currentY // blockWidth
        points.append((currentY, currentX))
        i += 1

    return points


def findMainLineD(triradius, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks):
    points = [(triradius[1], triradius[0])]

    actualBlockRow = triradius[0] // blockHeight
    rowInBlock = triradius[0] % blockHeight
    #rowInBlock = 0
    actualBlockCol = triradius[1] // blockWidth
    colInBlock = triradius[1] % blockWidth

    currentBlockRow = actualBlockRow + 1
    currentBlockCol = actualBlockCol - 1
    orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

    pointX = (currentBlockRow * blockHeight) + rowInBlock
    pointY = (currentBlockCol * blockWidth) + colInBlock
    points.append((pointY, pointX))

    currentX = int(pointX + (25 * sinOfAngles[orientation]))
    currentY = int(pointY - (25 * cosOfAngles[orientation]))

    points.append((currentY, currentX))

    comingFrom = 2

    i = 0
    while blocks[currentBlockRow][currentBlockCol]["palmprintSegment"] == 0:
    #while i < 100:
        orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

        currentX, currentY, comingFrom = getNextPoint(comingFrom, orientation, sinOfAngles, cosOfAngles, currentX,
                                                      currentY)
        currentBlockRow = currentX // blockHeight
        currentBlockCol = currentY // blockWidth
        points.append((currentY, currentX))
        i += 1

    return points


def singularPoints(blocks, blockImage, angles, leftOfFinger2, between23, between34, between45, rightOfFinger5):
    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    possibleTriradiusBlocks = []
    sinOfAngles = calculateSinOfAngles(angles)
    cosOfAngles = calculateCosOfAngles(angles)

    triradiusA = detectTriradiusA(blocks, blockImage, angles, leftOfFinger2, between23)
    triradiusB = detectTriradiusB(blocks, blockImage, angles, between23, between34)
    triradiusC = detectTriradiusB(blocks, blockImage, angles, between34, between45)
    triradiusD = detectTriradiusD(blocks, blockImage, angles, between45, rightOfFinger5)

    exit(2)

    mainLines = []

    #mainLineA = findMainLineA(triradiusA, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks)
    #mainLineB = findMainLineB(triradiusB, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks)
    #mainLineC = findMainLineC(triradiusC, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks)
    #mainLineD = findMainLineD(triradiusD, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks)

    #mainLines.append(mainLineA)
    #mainLines.append(mainLineB)
    #mainLines.append(mainLineC)
    #mainLines.append(mainLineD)

    saveTriradiusPalmprint(blockImage, triradiusA, triradiusB, triradiusC, triradiusD, mainLines)

    #exit(2)

    #blockImage = detectTriradiusC(blocks, blockImage, angles, leftOfFinger2, between23)
    #blockImage = detectTriradiusD(blocks, blockImage, angles, leftOfFinger2, between23)

    #return blockImage


def main():
    blockHeight = 50
    blockWidth = 50
    start = timer()

    image = loadPalmprint('dlane/muzi/2019_M_0021_HR02.tif')
    image = image[:-2]
    blockImage, blocks = splitImageToBlocks(image, blockHeight, blockWidth)

    #showPalmprint(blockImage[50][50])

    blocks, leftOfFinger2, between23, between34, between45, rightOfFinger5 = segmentation(blockImage, blocks, image)

    """angles = splitIntoParts(math.pi, 12)
    sinOfAngles = calculateSinOfAngles(angles)
    cosOfAngles = calculateCosOfAngles(angles)

    startingPoint = (2700, 2876)
    pointX, pointY = startingPoint
    orientation = 5
    points = []
    points.append((startingPoint[1], startingPoint[0]))

    x = int(pointX - (25 * sinOfAngles[orientation]))
    y = int(pointY + (25 * cosOfAngles[orientation]))

    points.append((y, x))

    image = mergeBlocksToImage(blockImage)

    for point in range(len(points) - 1):
        image = cv2.line(image, points[point], points[point + 1], (0, 0, 255), 3)

    cv2.imwrite('triradiusPalmprint.bmp', image)

    exit(2)"""

    blocks, angles = orientationField(blockImage, blocks, image)

    #orientationField(blockImage[20][20])

    #principleLines(image, blocks, blockImage)

    singularPoints(blocks, blockImage, angles, leftOfFinger2, between23, between34, between45, rightOfFinger5)

    #image = mergeBlocksToImage(blockImage)

    #savePalmprint(image)

    end = timer()
    print(f"Celkovo: {end - start}")


if __name__ == '__main__':
    main()
