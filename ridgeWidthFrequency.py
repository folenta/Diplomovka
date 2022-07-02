#! /usr/bin/env python

"""

Funckie pre ziskanie a ulozenie frekvencie a sirky papilarnych linii.

Autor: Bc. Jan Folenta
Email: xfolen00@stud.fit.vutbr.cz
Rok: 2022

"""

import cv2
import numpy as np

from orientation import calculateCosOfAngles, calculateSinOfAngles


def smoothenSineWave(sineWave, windowHeight):
    # Funkcia vyhladi x-signaturu

    sineWaveSmoothened = [0] * windowHeight
    sineWaveSmoothened[0] = sineWave[0]
    sineWaveSmoothened[-1] = sineWave[-1]

    # Vyhladenie prebieha na zaklade 3 susednych bodov
    for windowRow in range(1, len(sineWave) - 1):
        sineWaveSmoothened[windowRow] = int(
            (sineWave[windowRow - 1] + sineWave[windowRow] + sineWave[windowRow + 1]) / 3)

    return sineWaveSmoothened


def getSineWave(windowHeight, windowWidth, sineWaveOrientation, i, j, sinOfAngles, cosOfAngles, blurredBlock,
                blockHeight, blockWidth):
    # Funkcia ziska x-signaturu v aktualnom bloku

    sineWave = [0] * windowHeight
    for k in range(windowHeight):
        sineWave[k] = 0
        numberOfAdded = 0
        for d in range(windowWidth):
            u = i + (d - windowWidth / 2) * cosOfAngles[sineWaveOrientation] + (k - windowHeight / 2) * sinOfAngles[
                sineWaveOrientation]
            v = j + (d - windowWidth / 2) * sinOfAngles[sineWaveOrientation] + (windowHeight / 2 - k) * cosOfAngles[
                sineWaveOrientation]
            u = round(u)
            v = round(v)

            if 0 <= u < blockHeight and 0 <= v < blockWidth:
                sineWave[k] += blurredBlock[u][v]
                numberOfAdded += 1

        sineWave[k] = int(sineWave[k] / numberOfAdded)

    sineWaveSmoothened = smoothenSineWave(sineWave, windowHeight)

    return sineWaveSmoothened


def getCrossingBoundPoints(sineWave, windowHeight):
    # Funkcia najde body, v ktorych x-signatura prechadza z bodov nizsich ako hranica na body vyssie ako hranica
    # a naopak z bodov vyssich ako hranica na body nizsie ako hranica

    a = []
    b = []
    # Hranicou bude priemer bodov v x-signature
    sineWaveMean = np.mean(sineWave)

    if sineWave[0] > sineWaveMean:
        aboveBound = True
        belowBound = False
    else:
        aboveBound = False
        belowBound = True

    for k in range(1, windowHeight):
        if sineWave[k] > sineWaveMean and belowBound:
            a.append(k)
            aboveBound = True
            belowBound = False
        if sineWave[k] <= sineWaveMean and aboveBound:
            b.append(k)
            aboveBound = False
            belowBound = True

    return a, b


def getAverageDistance(crossingBound):
    # Funkcia najde priemernu vlnovu dlzku (vzdialenost medzi bodmi v poli 'a' alebo 'b') a overi, ci sa niektora
    # z vlnovych dlzok vyrazne nelisi od priemeru

    distance = 0
    valid = True
    crossingBoundTotal = len(crossingBound)

    for i in range(crossingBoundTotal - 1):
        distance += crossingBound[i + 1] - crossingBound[i]
    averageDistance = distance / (crossingBoundTotal - 1)

    for i in range(crossingBoundTotal - 1):
        distance = crossingBound[i + 1] - crossingBound[i]
        if abs(distance - averageDistance) > 3:
            valid = False
            break

    return averageDistance, valid


def getBlockFrequency(blocks, row, col, averageDistanceA, averageDistanceB, dpi):
    # Funkcia prevedie najdenu frekvenciu v aktualnom bloku na hustotu na 1 cm a ulozi vyslednu hustotu (frekvenciu)

    averageDistance = (averageDistanceA + averageDistanceB) / 2
    blockFrequency = (1 / averageDistance) * (dpi / 2.54)
    blocks[row][col]["frequency"] = round(blockFrequency, 4)

    return blocks


def getBlockWidth(blocks, row, col, a, b, dpi):
    # Funkcia najde sirku papilarnuch linii v aktualnom bloku, prevedie ju na sirku v mm a ulozi ju do struktury 'blocks'

    if a[0] > b[0]:
        b.pop(0)

    if len(a) > len(b):
        numberOfWidthCount = len(a) - 2
    else:
        numberOfWidthCount = len(a) - 1

    width = 0
    i = 0

    for crossingIndex in range(numberOfWidthCount):
        i += 1
        firstBottomUp = a[crossingIndex]
        secondBottomUp = a[crossingIndex + 1]
        firstUpBottom = b[crossingIndex]
        secondUpBottom = b[crossingIndex + 1]

        middleOfValley1 = (firstUpBottom + firstBottomUp) / 2
        middleOfValley2 = (secondUpBottom + secondBottomUp) / 2

        width += middleOfValley2 - middleOfValley1

    averageWidth = width / i
    width = (averageWidth * 25.4) / dpi
    blocks[row][col]["ridgeWidth"] = round(width, 4)

    return blocks


def getAverageWidthAndFrequency(blocks, blockRows, blockCols):
    # Funkcia najde priemernu sirku a frekvenciu papilarnych linii na odtlacku dlane

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

    averageFrequency = round(frequency / frequencyCount, 4)
    averageWidth = round(width / widthCount, 4)

    return averageFrequency, averageWidth


def ridgeWidthAndFrequency(blocks, blockImage, angles, directoryName, dpi):
    # Funkcia najde sirku a frekvenciu papilarnych linii

    sinOfAngles = calculateSinOfAngles(angles)
    cosOfAngles = calculateCosOfAngles(angles)
    rightAngleIndex = (len(angles) // 2) - 1

    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape

    # Okno v ktorom bude hladana x-signatura ma velkost 50 x 25 pixelov
    i = blockHeight // 2
    j = blockWidth // 2
    windowHeight = blockHeight
    windowWidth = blockWidth // 2

    for row in range(blockRows):
        for col in range(blockCols):
            if blocks[row][col]["orientationConfidence"] < 100:
                continue
            block = blockImage[row][col]
            ridgeOrientation = blocks[row][col]["orientation"]

            sineWaveOrientation = (ridgeOrientation + rightAngleIndex) % len(angles)
            blurredBlock = cv2.GaussianBlur(block, (5, 5), 0)

            sineWave = getSineWave(windowHeight, windowWidth, sineWaveOrientation, i, j, sinOfAngles, cosOfAngles,
                                   blurredBlock, blockHeight, blockWidth)

            #sineWaveMean = np.mean(sineWave)
            #bound = [sineWaveMean] * 50

            # Ked je pocet bodov 'a' alebo 'b' mensi ako 3, tak je frekvencia povazovana za nevalidnu
            a, b = getCrossingBoundPoints(sineWave, windowHeight)
            if len(a) < 3 or len(b) < 3:
                continue

            else:
                averageDistanceA, validA = getAverageDistance(a)
                averageDistanceB, validB = getAverageDistance(b)

                if validA and validB:
                    blocks = getBlockFrequency(blocks, row, col, averageDistanceA, averageDistanceB, dpi)
                    blocks = getBlockWidth(blocks, row, col, a, b, dpi)

    averageFrequency, averageWidth = getAverageWidthAndFrequency(blocks, blockRows, blockCols)

    return blocks, averageFrequency, averageWidth
