#! /usr/bin/env python

"""

Funkcie pre najdenie orientacii v jednotlivych blokoch na odtlacku dlane.

Autor: Bc. Jan Folenta
Email: xfolen00@stud.fit.vutbr.cz
Rok: 2022

"""

import numpy as np
import math
from collections import Counter
from numpy import sqrt, ogrid, mean


def orientationSmoothing(blocks, blockImage, angles):
    # Implementacia vyhladenia orientacii

    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape

    # Inicializuje masku orientacii
    orientationMask = np.zeros((blockRows, blockCols), dtype=np.uint8)

    # Najde, v ktorych blokoch bola urcena orientacia (mimo regionov kde su triradie)
    for row in range(blockRows):
        for col in range(blockCols):
            if blocks[row][col]["background"] == 0:
                orientationMask[row][col] = 1
            if blocks[row][col]["triradiusRegion"] == 1:
                orientationMask[row][col] = 0

    # Najde sa region o velkosti 5x5 blokov, kde je confidence vsetkych orientacii 100
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

                # V najdenom regione sa vsetky bloky nastavia ako spracovane (orientationMask = 2) a vlozia do pola
                # seeds (bloky, ktorych susedia budu prehladavany)
                if suitable:
                    for row2 in range(row, row + 5):
                        for col2 in range(col, col + 5):
                            seeds.append((row2, col2))
                            orientationMask[row2][col2] = 2
                    break
        if suitable:
            break

    # Prva faza vyhladenia ----------------------------------------------------------------------
    while seeds:
        seedX, seedY = seeds.pop(0)
        # Vyberiem si prvy blok zo seeds a pozeram sa na jeho susedov
        for row in range(seedX - 1, seedX + 2):
            for col in range(seedY - 1, seedY + 2):
                # Ked to neni odtlacok alebo uz je spracovany, nerobim nic
                if orientationMask[row][col] == 0 or orientationMask[row][col] == 2:
                    continue
                else:
                    # Ked je jeho confindence 100 tak ho nastavim ako spracovany, pridam ho do seeds
                    if blocks[row][col]["orientationConfidence"] == 100:
                        seeds.append((row, col))
                        orientationMask[row][col] = 2
                        continue
                    else:
                        # V opacnom pripade sa pokusim dany blok vyhladit
                        neighbourOrientations = []
                        # Z osmich susednych blokov najdem bloky, ktore uz boli spracovane alebo ktorych confidence
                        # je 100
                        for r in range(row - 1, row + 2):
                            for c in range(col - 1, col + 2):
                                if r == row and c == col:
                                    continue
                                else:
                                    if orientationMask[r][c] == 2 or blocks[r][c]["orientationConfidence"] == 100:
                                        neighbourOrientations.append(blocks[r][c]["orientation"])

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
                                minOrientation = min(neighbourOrientations)
                                maxOrientation = max(neighbourOrientations)
                                orientationDifference = min(abs(maxOrientation - minOrientation),
                                                            abs(maxOrientation - (minOrientation + 12)))
                                # Ked su vsetky susedne bloky navzajom podobne a aktualny sa od nich vyrazne lisi,
                                # tak sa aktualny zmeni na priemer okolitych
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

    # Druha faza vyhladenia ----------------------------------------------------------------------
    i = 0
    while i < 1:
        i += 1
        for row in range(blockRows):
            for col in range(blockCols):
                if orientationMask[row][col] == 2 and blocks[row][col]["orientationConfidence"] != 100:
                    orientationsInNeighbourhood = [0] * len(angles)
                    # Prehladavam 8 susednych blokov
                    for r in range(row - 1, row + 2):
                        for c in range(col - 1, col + 2):

                            if r == row and c == col:
                                continue
                            else:
                                if orientationMask[r][c] != 0:
                                    orientationsInNeighbourhood[blocks[r][c]["orientation"]] += 1
                    if len(orientationsInNeighbourhood) < 2:
                        continue

                    # Najdem ktora orientacia je medzi orientaciami v susedstve najviac zastupena
                    maxOrientations = max(orientationsInNeighbourhood)
                    maxOrientationsValue = orientationsInNeighbourhood.index(maxOrientations)
                    # Najdem pocet orientacii, ktore su rovnake alebo o -+1 vacsie/mensie
                    if maxOrientationsValue == 0:
                        numberOfNeighboursWithSimilarOrientations = maxOrientations + orientationsInNeighbourhood[11] + \
                                                                    orientationsInNeighbourhood[maxOrientationsValue + 1]
                    elif maxOrientationsValue == 11:
                        numberOfNeighboursWithSimilarOrientations = maxOrientations + orientationsInNeighbourhood[
                            maxOrientationsValue - 1] + \
                                                                    orientationsInNeighbourhood[0]
                    else:
                        numberOfNeighboursWithSimilarOrientations = maxOrientations + orientationsInNeighbourhood[
                            maxOrientationsValue - 1] + orientationsInNeighbourhood[maxOrientationsValue + 1]

                    # Ked je aspon 5 takych orientacii a orientacia v aktualnom bloku sa vyrazne lisi, tak zmenim
                    # orientaciu na tu, ktora je v susedstve najcastejsie
                    if numberOfNeighboursWithSimilarOrientations >= 5:
                        myOrientation = blocks[row][col]["orientation"]
                        difference = min(abs(myOrientation - maxOrientationsValue),
                                         abs((myOrientation + 12) - maxOrientationsValue),
                                         abs(myOrientation - (maxOrientationsValue + 12)))
                        if difference > 2:
                            blocks[row][col]["orientation"] = maxOrientationsValue

    return blocks


def splitIntoParts(value, numberOfParts):
    return np.linspace(0, value, numberOfParts + 1)[1:]


def findStartEndCols(row, currentRow, currentCol, blockWidth, angle):
    # Funkcia sluzi pre zrychlenie Radonovej transformacie - najde zaciatocny a koncovy stlpec, v ktorom sa
    # bude hladat"""

    angleInDegrees = math.degrees(angle)

    if row < currentRow:
        if angleInDegrees < 90.0:
            startCol = currentCol
            endCol = min(currentCol + 20 + 1, blockWidth)

        elif angleInDegrees == 90.0:
            startCol = currentCol
            endCol = currentCol + 1

        else:
            startCol = max(currentCol - 20, 0)
            endCol = currentCol + 1

    elif row == currentRow:
        startCol = max(currentCol - 20, 0)
        endCol = min(currentCol + 20 + 1, blockWidth)

    else:
        if angleInDegrees > 90.0:
            startCol = currentCol
            endCol = min(currentCol + 20 + 1, blockWidth)

        elif angleInDegrees == 90.0:
            startCol = currentCol
            endCol = currentCol + 1

        else:
            startCol = max(currentCol - 20, 0)
            endCol = currentCol + 1

    return startCol, endCol


def calculateSinOfAngles(angles):
    # Funckia vypocita sinusy danych uhlov

    sinOfAngles = []

    for angle in angles:
        sinOfAngles.append(math.sin(angle))

    return sinOfAngles


def calculateCosOfAngles(angles):
    # Funckia vypocita cosinusy danych uhlov

    cosOfAngles = []

    for angle in angles:
        cosOfAngles.append(math.cos(angle))

    return cosOfAngles


def getOrientation(rs):
    # Funckia ziska orientaciu bodu

    minValue = min(rs)
    orientation = rs.index(minValue)
    confidence = -minValue

    return orientation, confidence


def findRadonTransform(row, col, block, angles, pixelsForRadonTransform):
    # Implementacia algoritmu zalozenom na Radonovej transformacii

    rs = [0] * 12

    # Vytvori masku pixelov, ktore su vo vnutri kruhoveho regionu (True ked dnu, False ked von)
    X, Y = ogrid[:block.shape[0], :block.shape[1]]
    distance = sqrt((X - row) ** 2 + (Y - col) ** 2)
    circleRegionMask = distance <= block.shape[0] // 2

    # Vypocita priemer pixelov v kruhovom regione
    blockMean = mean(block[circleRegionMask])

    for angleNumber in range(len(angles)):
        total = 0
        # Ziska body pre ktore sa bude pocitat Radonova transfomracia
        pointsToSearch = pixelsForRadonTransform[row][col][angleNumber]

        # Postupne prechadza ziskane body a Radonova transformacia sa rovna intenzite pixelu v danom bode, od ktoreho
        # je odcitany priemer intenzit pixelov v kruhovom regione
        for point in pointsToSearch:
            x = point[0]
            y = point[1]
            pixelIntensity = block[x][y]
            total += (pixelIntensity - blockMean)

        rs[angleNumber] += total

    orientation, confidence = getOrientation(rs)

    return orientation


def inLineForRadonTransform(row, currentRow, col, currentCol, angleNumber, sinOfAngles, cosOfAngles):
    deltaFunction = (row - currentRow) * cosOfAngles[angleNumber] + (col - currentCol) * sinOfAngles[angleNumber]

    """ Ked je hodnota Delta funckie (dirac) 0, tak to znamena ze dany bod lezi na ciare v danom uhle """
    if -0.5 < deltaFunction < 0.5:
        return True

    return False


def findPixelsForRadonTransform(blockHeight, blockWidth, angles):
    sinOfAngles = calculateSinOfAngles(angles)
    cosOfAngles = calculateCosOfAngles(angles)
    circleRadius = blockHeight // 2
    pixelsForRadonTransform = []

    #print("Im here")

    """ Vytvorim si pole pre ukladanie bodov, v ktorych sa bude pocitat Radonova transformacia """
    for a in range(blockHeight):
        pixelsForRadonTransform.append([])
        for b in range(blockWidth):
            pixelsForRadonTransform[a].append([])
            for c in range(len(angles)):
                pixelsForRadonTransform[a][b].append([])

    """ Prehladavam kazdy pixel v bloku """
    for row in range(blockHeight):
        for col in range(blockWidth):
            """ Vytvori masku pixelov, ktore su vo vnutri kruhoveho regionu (True ked dnu, False ked von)"""
            X, Y = np.ogrid[:blockHeight, :blockWidth]
            distance = np.sqrt((X - row) ** 2 + (Y - col) ** 2)
            circleRegionMask = distance <= circleRadius

            angleNumber = 0

            """ Nemusim prehladavat vsetky riadky, ale iba tie, ktorych vzdialenost od aktualneho riadku je mensia 
                ako circleRadius """
            firstRow = max(0, row - circleRadius)
            lastRow = min(row + circleRadius + 1, blockHeight)

            """ Skusam vsetky uhly, v ramci coho prechadzam vsetky pixely v kruhovom regione a hladam body, v ktorych 
                sa bude pocitat Radonova transformacia """
            for angle in angles:
                pixels = []

                for rowCircle in range(firstRow, lastRow):

                    """ Nemusim prehladavat vsetky stlpce """
                    startCol, endCol = findStartEndCols(rowCircle, row, col, blockWidth, angle)

                    """ Skontrolujem, ci sa pixel nachadza v kruhovom regione a ak ano tak najdem body, ktore lezia
                        na pomyselnej ciare v danom uhle (v tych bodoch sa bude pocitat Radonova transformacia) """
                    for colCircle in range(startCol, endCol):
                        if circleRegionMask[row][col]:
                            if inLineForRadonTransform(rowCircle, row, colCircle, col, angleNumber, sinOfAngles,
                                                       cosOfAngles):
                                pixels.append((rowCircle, colCircle))

                """ Ulozim si dane body do pola pre zodpovedajuci riadok, stlpec a uhol """
                pixelsForRadonTransform[row][col][angleNumber] = pixels
                angleNumber += 1

    #print("Im finished")

    return pixelsForRadonTransform


def getBlockOrientation(pixelOrientations):
    # Funkcia urci na zaklade ziskanych orientacii v aktualnom bloku celkovu orientaciu bloku

    confidence = 0
    try:
        finalOrientation = Counter(pixelOrientations).most_common()[0][0]
        firstOrientation = Counter(pixelOrientations).most_common()[0][0]
        firstOrientationNumber = Counter(pixelOrientations).most_common()[0][1]
        secondOrientation = Counter(pixelOrientations).most_common()[1][0]
        secondOrientationNumber = Counter(pixelOrientations).most_common()[1][1]
        if abs(firstOrientation - secondOrientation) == 1 or abs(firstOrientation - secondOrientation) >= 10:
            if firstOrientationNumber + secondOrientationNumber >= 150:
                confidence = 100
            else:
                confidence = 75
        else:
            if firstOrientationNumber >= 2 * secondOrientationNumber and firstOrientationNumber >= 100:
                confidence = 100
            else:
                confidence = ((firstOrientationNumber / secondOrientationNumber) - 1) * 100
    except IndexError:
        finalOrientation = -1

    return finalOrientation, confidence


def orientationField(blockImage, blocks):
    # Funkcia najde pole orientacii

    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    angles = splitIntoParts(math.pi, 12)

    pixelsForRadonTransform = findPixelsForRadonTransform(blockHeight, blockWidth, angles)

    for blockRow in range(blockRows):
        for blockCol in range(blockCols):
            if blocks[blockRow][blockCol]["background"] == 0:
                pixelOrientations = []
                block = blockImage[blockRow][blockCol]
                blockMean = mean(block)
                i = 0
                # Spracovavane budu iba cierne pixely
                for row in range(blockHeight):
                    for col in range(blockWidth):
                        i += 1
                        value = block[row][col]
                        # Pre urychlenie algoritmu bude skumany iba kazdy treti pixel
                        if value < blockMean - 10 and i % 3 == 0:
                            r = findRadonTransform(row, col, block, angles, pixelsForRadonTransform)
                            pixelOrientations.append(r)

                blockOrientation, confidence = getBlockOrientation(pixelOrientations)

                blocks[blockRow][blockCol]["orientation"] = blockOrientation
                blocks[blockRow][blockCol]["orientationConfidence"] = confidence

    return blocks, angles
