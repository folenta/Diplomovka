#! /usr/bin/env python

"""

Funkcie pre sledovanie hlavnych linii vratane urcenia ukoncenia hlavnych linii.

Autor: Bc. Jan Folenta
Email: xfolen00@stud.fit.vutbr.cz
Rok: 2022

"""

import cv2
import math
import numpy as np

from orientation import calculateCosOfAngles, calculateSinOfAngles


def comingFromWhere(x, y, nextX, nextY):
    # Funckia najde orientaciu predchadzajuceho segmentu hlavnej linie

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


def getNextPointMainLineA(comingFrom, previousOrientation, orientation, sinOfAngles, cosOfAngles, x, y, lengthOfLine):
    # Pravidla pre urcenie dalsieho bodu hlavnej linie A

    if lengthOfLine > 15:
        orientationDifference = min(abs(orientation - previousOrientation), abs((orientation + 12) - previousOrientation),
                                    abs(orientation - (previousOrientation + 12)))
        if orientationDifference > 2:
            orientation = previousOrientation
    else:
        orientationDifference = min(abs(orientation - previousOrientation),
                                    abs((orientation + 12) - previousOrientation),
                                    abs(orientation - (previousOrientation + 12)))

        if orientationDifference > 3 and previousOrientation < 10:
            orientation = previousOrientation

    if comingFrom <= 3:
        if lengthOfLine > 15 and previousOrientation - orientation > 2:
            orientation = previousOrientation

        nextX = int(x + (25 * sinOfAngles[orientation]))
        nextY = int(y - (25 * cosOfAngles[orientation]))

        comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if lengthOfLine > 15 and comingFrom == 0 and comingFromNew == 6 and 2 <= orientation <= 3:
            orientation = (orientation + 5) % 12

            nextX = int(x - (25 * sinOfAngles[orientation]))
            nextY = int(y + (25 * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 0 and (2 <= comingFromNew <= 5):
            if 6 <= previousOrientation <= 8 and 2 <= orientation <= 4:
                return nextX, nextY, comingFromNew, orientation

            nextX = int(x - (25 * sinOfAngles[orientation]))
            nextY = int(y + (25 * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 1 and (4 <= comingFromNew <= 6):
            nextX = int(x - (25 * sinOfAngles[orientation]))
            nextY = int(y + (25 * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 3 and (comingFromNew == 0 or comingFromNew >= 6):
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

        if lengthOfLine > 15 and comingFrom == 6 and comingFromNew == 6 and 2 <= orientation <= 4:
            orientation = (orientation + 5) % 12

            nextX = int(x + (25 * sinOfAngles[orientation]))
            nextY = int(y - (25 * cosOfAngles[orientation]))

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

    return nextX, nextY, comingFromNew, orientation


def getNextPointMainLineB(previousOrientation, orientation, sinOfAngles, cosOfAngles, x, y, alreadyWentLeft,
                          alreadyAddedPoints, smallerDistance):
    # Pravidla pre urcenie dalsieho bodu hlavnej linie B

    # Povolene orientacie 0-5 a 17-23 (ked som uz raz isiel dolava)
    #  - povolene ist dolava mam iba raz (a to iba v prvych 5 bodoch)
    #  - ked bude orientacia od 11 do 17 a pocet bodov je aspon 5 tak idem natvrdo doprava
    #  - ked je menej jak 5 bodov tak ked pojdem dolava tak jedine ked je uhol od 14 do 16 - inak idem priamo dole
    #  - ked je viac jak 5 bodov idem natvrdo vzdy doprava (ale obmedzim velkost zmeny)
    distance = 25
    if smallerDistance:
        distance = 10

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

    else:
        if alreadyAddedPoints > 30 and 8 > abs(previousOrientation - orientation) > 3:
            orientation = previousOrientation

        else:
            if 12 <= orientation <= 16:
                orientation = orientation - 12

    orientationDifference = min(abs(orientation - previousOrientation), abs((orientation + 24) - previousOrientation),
                                abs(orientation - (previousOrientation + 24)))

    if orientationDifference > 3 and alreadyAddedPoints > 5:
        orientation = previousOrientation

    if 8 <= orientation <= 14 and alreadyAddedPoints > 10:
        orientation = (orientation + 12) % 24

    if orientation > 11:
        nextX = int(x + (distance * sinOfAngles[orientation - 12]))
        nextY = int(y - (distance * cosOfAngles[orientation - 12]))
    else:
        nextX = int(x - (distance * sinOfAngles[orientation]))
        nextY = int(y + (distance * cosOfAngles[orientation]))

    return nextX, nextY, orientation, alreadyWentLeft


def getNextPointMainLineC(previousOrientation, orientation, sinOfAngles, cosOfAngles, x, y, comingFrom, smallerDistance):
    # Pravidla pre urcenie dalsieho bodu hlavnej linie C

    distance = 25
    if smallerDistance:
        distance = 10

    if comingFrom <= 3:
        nextX = int(x + (distance * sinOfAngles[orientation]))
        nextY = int(y - (distance * cosOfAngles[orientation]))

        comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 0 and (2 <= comingFromNew <= 5):
            if 5 < previousOrientation < 8 and 2 < orientation < 5:
                return nextX, nextY, comingFromNew, orientation

            nextX = int(x - (distance * sinOfAngles[orientation]))
            nextY = int(y + (distance * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 1 and (4 <= comingFromNew <= 6):
            nextX = int(x - (distance * sinOfAngles[orientation]))
            nextY = int(y + (distance * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if (comingFrom == 2 or comingFrom == 3) and (comingFromNew == 0 or comingFromNew >= 6):
            if 3 <= previousOrientation <= 4 and 5 <= orientation <= 7:
                pass
            else:
                nextX = int(x - (distance * sinOfAngles[orientation]))
                nextY = int(y + (distance * cosOfAngles[orientation]))

                comingFromNew = comingFromWhere(x, y, nextX, nextY)

    else:
        nextX = int(x - (distance * sinOfAngles[orientation]))
        nextY = int(y + (distance * cosOfAngles[orientation]))

        comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 6 and (0 <= comingFromNew <= 3):
            nextX = int(x + (distance * sinOfAngles[orientation]))
            nextY = int(y - (distance * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 6 and comingFromNew == 4 and previousOrientation < 2 and orientation > 8:
            nextX = int(x + (distance * sinOfAngles[orientation]))
            nextY = int(y - (distance * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 7 and (2 <= comingFromNew <= 4):
            nextX = int(x + (distance * sinOfAngles[orientation]))
            nextY = int(y - (distance * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if (comingFrom == 4 or comingFrom == 5) and (comingFromNew == 7 or comingFromNew <= 2):

            nextX = int(x + (distance * sinOfAngles[orientation]))
            nextY = int(y - (distance * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 4 and comingFromNew == 6:
            if 8 <= previousOrientation <= 10 and 0 <= orientation <= 2:
                nextX = int(x + (distance * sinOfAngles[orientation]))
                nextY = int(y - (distance * cosOfAngles[orientation]))

                comingFromNew = comingFromWhere(x, y, nextX, nextY)

    return nextX, nextY, comingFromNew, orientation


def getNextPointMainLineD(previousOrientation, orientation, sinOfAngles, cosOfAngles, x, y, comingFrom,
                          smallerDistance, lengthOfLine):
    # Pravidla pre urcenie dalsieho bodu hlavnej linie D

    orientationDifference = min(abs(orientation - previousOrientation), abs((orientation + 12) - previousOrientation),
                                abs(orientation - (previousOrientation + 12)))

    if orientationDifference > 3 and lengthOfLine > 10:
        if previousOrientation < orientation:
            if comingFrom == 2:
                orientation = (orientation + 4) % 12
            else:
                orientation = (orientation + 3) % 12

        else:
            if comingFrom == 4:
                orientation = (orientation + 4) % 12
            else:
                orientation = orientation - 3
                if orientation < 0:
                    orientation = orientation + 12

    distance = 25
    if smallerDistance:
        distance = 10

    if comingFrom == 2 and lengthOfLine > 10 and 0 <= previousOrientation <= 2 and 4 <= orientation <= 5 and \
            orientationDifference > 2:
        orientation = previousOrientation

    if comingFrom <= 3:
        nextX = int(x + (distance * sinOfAngles[orientation]))
        nextY = int(y - (distance * cosOfAngles[orientation]))

        comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 0 and (2 <= comingFromNew <= 5):
            nextX = int(x - (distance * sinOfAngles[orientation]))
            nextY = int(y + (distance * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 1 and (4 <= comingFromNew <= 6):
            nextX = int(x - (distance * sinOfAngles[orientation]))
            nextY = int(y + (distance * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if (comingFrom == 2 or comingFrom == 3) and (comingFromNew == 0 or comingFromNew >= 6):
            nextX = int(x - (distance * sinOfAngles[orientation]))
            nextY = int(y + (distance * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

    else:
        nextX = int(x - (distance * sinOfAngles[orientation]))
        nextY = int(y + (distance * cosOfAngles[orientation]))

        comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 6 and (0 <= comingFromNew <= 3):
            nextX = int(x + (distance * sinOfAngles[orientation]))
            nextY = int(y - (distance * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 6 and comingFromNew == 4 and previousOrientation <= 2 and 10 <= orientation <= 11:
            nextX = int(x + (distance * sinOfAngles[orientation]))
            nextY = int(y - (distance * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 7 and (2 <= comingFromNew <= 4):
            nextX = int(x + (distance * sinOfAngles[orientation]))
            nextY = int(y - (distance * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 4:
            pass

        if (comingFrom == 4 or comingFrom == 5) and (comingFromNew == 7 or comingFromNew <= 2):

            nextX = int(x + (distance * sinOfAngles[orientation]))
            nextY = int(y - (distance * cosOfAngles[orientation]))

            comingFromNew = comingFromWhere(x, y, nextX, nextY)

        if comingFrom == 4 and comingFromNew == 6:
            if 8 <= previousOrientation <= 11 and 0 <= orientation <= 2:
                nextX = int(x + (distance * sinOfAngles[orientation]))
                nextY = int(y - (distance * cosOfAngles[orientation]))

                comingFromNew = comingFromWhere(x, y, nextX, nextY)

    return nextX, nextY, comingFromNew, orientation


def getNextPointMainLineT(comingFrom, previousOrientation, orientation, sinOfAngles, cosOfAngles, x, y, lengthOfLine):
    # Pravidla pre urcenie dalsieho bodu hlavnej linie T

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


def findMainLineA(triradius, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks, contour):
    # Funkcia najde priebeh hlavnej linie A

    points = [(triradius[1], triradius[0])]

    actualBlockRow = triradius[0] // blockHeight
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

    # Vdialenost dalsieho bodu sa rovna polovici sirky bloku
    distance = blockWidth // 2
    currentX = int(pointX + (distance * sinOfAngles[orientation]))
    currentY = int(pointY - (distance * cosOfAngles[orientation]))

    points.append((currentY, currentX))

    comingFrom = comingFromWhere(pointX, pointY, currentX, currentY)
    i = 0
    while blocks[currentBlockRow][currentBlockCol]["background"] == 0:
        orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

        # Najdenie dalsieho bodu hlavnej linie
        currentX, currentY, comingFrom, orientation = getNextPointMainLineA(comingFrom, previousOrientation,
                                                                            orientation, sinOfAngles, cosOfAngles,
                                                                            currentX, currentY, len(points))
        currentBlockRow = currentX // blockHeight
        currentBlockCol = currentY // blockWidth
        previousOrientation = orientation
        points.append((currentY, currentX))
        i += 1

        # Pokracovanie v pripade, ze blok nema urcenu orientaciu ale nachadza sa vo vnutri odtlacku
        if blocks[currentBlockRow][currentBlockCol]["orientation"] == -1:
            if cv2.pointPolygonTest(contour, (currentBlockCol, currentBlockRow), False) > 0:
                blocks[currentBlockRow][currentBlockCol]["background"] = 0
                blocks[currentBlockRow][currentBlockCol]["orientation"] = previousOrientation

    return points


def findMainLineB(triradius, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks, contour, smallerDistance):
    # Funkcia najde priebeh hlavnej linie B

    points = [(triradius[1], triradius[0])]

    actualBlockRow = triradius[0] // blockHeight
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

    # Vdialenost dalsieho bodu sa rovna polovici sirky bloku
    distance = blockWidth // 2
    currentX = int(pointX + (distance * sinOfAngles[orientation]))
    currentY = int(pointY - (distance * cosOfAngles[orientation]))

    points.append((currentY, currentX))

    alreadyWentLeft = False
    i = 0
    while blocks[currentBlockRow][currentBlockCol]["background"] == 0:
        orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

        # Najdenie dalsieho bodu hlavnej linie
        currentX, currentY, orientation, alreadyWentLeft = getNextPointMainLineB(previousOrientation, orientation,
                                                                                 sinOfAngles, cosOfAngles, currentX,
                                                                                 currentY, alreadyWentLeft, len(points),
                                                                                 smallerDistance)
        currentBlockRow = currentX // blockHeight
        currentBlockCol = currentY // blockWidth
        previousOrientation = orientation
        points.append((currentY, currentX))
        i += 1

        # Pokracovanie v pripade, ze blok nema urcenu orientaciu ale nachadza sa vo vnutri odtlacku
        if blocks[currentBlockRow][currentBlockCol]["orientation"] == -1:
            if cv2.pointPolygonTest(contour, (currentBlockCol, currentBlockRow), False) > 0:
                blocks[currentBlockRow][currentBlockCol]["background"] = 0
                blocks[currentBlockRow][currentBlockCol]["orientation"] = previousOrientation

    return points


def findMainLineC(triradius, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks, contour, smallerDistance):
    # Funkcia najde priebeh hlavnej linie C

    points = [(triradius[1], triradius[0])]

    actualBlockRow = triradius[0] // blockHeight
    rowInBlock = triradius[0] % blockHeight
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

    # Vdialenost dalsieho bodu sa rovna polovici sirky bloku
    distance = blockWidth // 2
    currentX = int(pointX + (distance * sinOfAngles[orientation]))
    currentY = int(pointY - (distance * cosOfAngles[orientation]))

    if orientation > 8:
        currentX = int(pointX - (distance * sinOfAngles[orientation]))
        currentY = int(pointY + (distance * cosOfAngles[orientation]))

    points.append((currentY, currentX))

    comingFrom = comingFromWhere(pointX, pointY, currentX, currentY)
    i = 0
    while blocks[currentBlockRow][currentBlockCol]["background"] == 0:
        orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

        # Najdenie dalsieho bodu hlavnej linie
        currentX, currentY, comingFrom, orientation = getNextPointMainLineC(previousOrientation, orientation,
                                                                            sinOfAngles, cosOfAngles, currentX,
                                                                            currentY, comingFrom, smallerDistance)
        currentBlockRow = currentX // blockHeight
        currentBlockCol = currentY // blockWidth
        previousOrientation = orientation
        points.append((currentY, currentX))
        i += 1

        # Pokracovanie v pripade, ze blok nema urcenu orientaciu ale nachadza sa vo vnutri odtlacku
        if blocks[currentBlockRow][currentBlockCol]["orientation"] == -1:
            if cv2.pointPolygonTest(contour, (currentBlockCol, currentBlockRow), False) > 0:
                blocks[currentBlockRow][currentBlockCol]["background"] = 0
                blocks[currentBlockRow][currentBlockCol]["orientation"] = previousOrientation

    return points


def findMainLineD(triradius, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks, contour, smallerDistance):
    # Funkcia najde priebeh hlavnej linie D

    points = [(triradius[1], triradius[0])]

    actualBlockRow = (triradius[0] + 10) // blockHeight
    rowInBlock = (triradius[0] + 10) % blockHeight
    actualBlockCol = triradius[1] // blockWidth
    colInBlock = triradius[1] % blockWidth

    currentBlockRow = actualBlockRow
    currentBlockCol = actualBlockCol - 1
    previousOrientation = blocks[currentBlockRow][currentBlockCol]["orientation"]
    orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

    pointX = (currentBlockRow * blockHeight) + rowInBlock
    pointY = (currentBlockCol * blockWidth) + colInBlock
    points.append((pointY, pointX))

    # Vdialenost dalsieho bodu sa rovna polovici sirky bloku
    distance = blockWidth // 2
    currentX = int(pointX + (distance * sinOfAngles[orientation]))
    currentY = int(pointY - (distance * cosOfAngles[orientation]))

    points.append((currentY, currentX))

    comingFrom = 2
    i = 0
    while blocks[currentBlockRow][currentBlockCol]["background"] == 0:
        orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

        # Najdenie dalsieho bodu hlavnej linie
        currentX, currentY, comingFrom, orientation = getNextPointMainLineD(previousOrientation, orientation, sinOfAngles, cosOfAngles,
                                                            currentX, currentY, comingFrom, smallerDistance, len(points))
        currentBlockRow = currentX // blockHeight
        currentBlockCol = currentY // blockWidth
        previousOrientation = orientation
        points.append((currentY, currentX))
        i += 1

        # Pokracovanie v pripade, ze blok nema urcenu orientaciu ale nachadza sa vo vnutri odtlacku
        if blocks[currentBlockRow][currentBlockCol]["orientation"] == -1:
            if cv2.pointPolygonTest(contour, (currentBlockCol, currentBlockRow), False) > 0:
                blocks[currentBlockRow][currentBlockCol]["background"] = 0
                blocks[currentBlockRow][currentBlockCol]["orientation"] = previousOrientation

    return points


def findMainLineT(triradius, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks, contour):
    # Funkcia najde priebeh hlavnej linie T

    points = [(triradius[1], triradius[0])]

    actualBlockRow = triradius[0] // blockHeight
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

    # Vdialenost dalsieho bodu sa rovna polovici sirky bloku
    distance = blockWidth // 2

    currentX = int(pointX - (distance * sinOfAngles[orientation]))
    currentY = int(pointY + (distance * cosOfAngles[orientation]))

    points.append((currentY, currentX))

    comingFrom = 5
    i = 0
    while blocks[currentBlockRow][currentBlockCol]["background"] == 0:
        orientation = blocks[currentBlockRow][currentBlockCol]["orientation"]

        # Najdenie dalsieho bodu hlavnej linie
        currentX, currentY, comingFrom, orientation = getNextPointMainLineT(comingFrom, previousOrientation,
                                                                            orientation, sinOfAngles, cosOfAngles,
                                                                            currentX, currentY,  len(points))
        currentBlockRow = currentX // blockHeight
        currentBlockCol = currentY // blockWidth
        previousOrientation = orientation
        points.append((currentY, currentX))
        i += 1

        # Pokracovanie v pripade, ze blok nema urcenu orientaciu ale nachadza sa vo vnutri odtlacku
        if blocks[currentBlockRow][currentBlockCol]["orientation"] == -1:
            if cv2.pointPolygonTest(contour, (currentBlockCol, currentBlockRow), False) > 0:
                blocks[currentBlockRow][currentBlockCol]["background"] = 0
                blocks[currentBlockRow][currentBlockCol]["orientation"] = previousOrientation

    return points


def findMainLineEnding(point, blocks, blockHeight, blockWidth, edgePointsOfSegments):
    # Funkcia najde zakoncenie hlavnej linie

    currentBlockRow = point[1] // blockHeight
    currentBlockCol = point[0] // blockWidth

    if blocks[currentBlockRow][currentBlockCol]["palmprintSegment"] != 0:
        if blocks[currentBlockRow][currentBlockCol]["palmprintSegment"] == 55:
            return "5\'"
        return blocks[currentBlockRow][currentBlockCol]["palmprintSegment"]

    else:
        minDistance = 1000
        segment = 0
        for i in range(len(edgePointsOfSegments)):
            edgePointRow, edgePointCol = edgePointsOfSegments[i]
            distance = math.hypot(edgePointRow - currentBlockRow, edgePointCol - currentBlockCol)
            if distance < minDistance:
                minDistance = distance
                segment = i

        if segment == 4:
            segment = '5\''

        elif segment == 5:
            segment = '5\'\''

        elif segment % 2 == 0:
            segment += 1

        return segment


def measureDistanceFromTriradius(mainLine1, mainLine2, triradius1, triradius2):
    # Funkcia najde najmensiu vzdialenost medzi triradiami a hlavnymi liniami

    minDistanceFromTriradius1 = 1000000
    for point in mainLine2:
        distance = np.sqrt((point[1] - triradius1[0]) ** 2 + (point[0] - triradius1[1]) ** 2)
        if distance < minDistanceFromTriradius1:
            minDistanceFromTriradius1 = distance

    minDistanceFromTriradius2 = 1000000
    for point in mainLine1:
        distance = np.sqrt((point[1] - triradius2[0]) ** 2 + (point[0] - triradius2[1]) ** 2)
        if distance < minDistanceFromTriradius2:
            minDistanceFromTriradius2 = distance

    return minDistanceFromTriradius1, minDistanceFromTriradius2


def getModifiedAngles(angles, curve, positiveCurve):
    # Funkcia ziska zmenene sinusy a cosinusy uhlov na zaklade vstupneho zakrivenia

    sinOfAnglesNew = []
    cosOfAnglesNew = []
    for angle in angles:
        if positiveCurve:
            sinOfAnglesNew.append(math.sin(angle + math.radians(curve)))
            cosOfAnglesNew.append(math.cos(angle + math.radians(curve)))

        else:
            sinOfAnglesNew.append(math.sin(angle - math.radians(curve)))
            cosOfAnglesNew.append(math.cos(angle - math.radians(curve)))

    return sinOfAnglesNew, cosOfAnglesNew


def solveCrossingOfMainLinesBC(mainLines, triradiusB, triradiusC, mainLineB, mainLineC, mainLineEndingB,
                               mainLineEndingC, blocks, angles, contour, blockHeight, blockWidth, edgePointsOfSegments):
    # Funckia vyriesi problem, kedy sa krizuju hlavne linie B a C

    minDistanceFromTriradiusB, minDistanceFromTriradiusC = measureDistanceFromTriradius(mainLineB, mainLineC, triradiusB, triradiusC)

    curve = 0
    if minDistanceFromTriradiusC < minDistanceFromTriradiusB:
        while mainLineEndingB == 7:
            sinOfAnglesNew, cosOfAnglesNew = getModifiedAngles(angles, curve, positiveCurve=True)
            mainLineB = findMainLineB(triradiusB, blockHeight, blockWidth, sinOfAnglesNew, cosOfAnglesNew,
                                      blocks, contour, False)
            mainLineEndingB = findMainLineEnding(mainLineB[-1], blocks, blockHeight, blockWidth,
                                               edgePointsOfSegments)

    else:
        while mainLineEndingC == 11:
            curve += 5
            sinOfAnglesNew, cosOfAnglesNew = getModifiedAngles(angles, curve, positiveCurve=False)
            mainLineC = findMainLineC(triradiusC, blockHeight, blockWidth, sinOfAnglesNew, cosOfAnglesNew,
                                      blocks, contour, False)
            mainLineEndingC = findMainLineEnding(mainLineC[-1], blocks, blockHeight, blockWidth,
                                               edgePointsOfSegments)

    mainLines.pop(-1)
    mainLines.pop(-1)
    mainLines.append(mainLineB)
    mainLines.append(mainLineC)

    return mainLines, mainLineEndingB, mainLineEndingC


def solveMainLineCrossingCD(mainLines, triradiusC, triradiusD, blocks, angles, contour, blockHeight, blockWidth,
                            edgePointsOfSegments, sinOfAngles, cosOfAngles):
    # Funkcia vyriesi problem, kedy sa krizuju hlavne linie C a D

    mainLineC = findMainLineC(triradiusC, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks, contour, True)
    mainLineD = findMainLineD(triradiusD, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks, contour, True)
    mainLineEndingC = findMainLineEnding(mainLineC[-1], blocks, blockHeight, blockWidth, edgePointsOfSegments)
    mainLineEndingD = findMainLineEnding(mainLineD[-1], blocks, blockHeight, blockWidth, edgePointsOfSegments)
    mainLines.pop(-1)
    mainLines.append(mainLineC)

    if (mainLineEndingC == "5\'" or mainLineEndingC == "5\'\'") and mainLineEndingD == 9:
        minDistanceFromTriradiusC, minDistanceFromTriradiusD = measureDistanceFromTriradius(mainLineC, mainLineD,
                                                                                            triradiusC, triradiusD)

        curve = 0
        if minDistanceFromTriradiusC < minDistanceFromTriradiusD:
            while mainLineEndingD == 9:
                curve += 5
                sinOfAnglesNew, cosOfAnglesNew = getModifiedAngles(angles, curve, positiveCurve=False)
                mainLineD = findMainLineD(triradiusD, blockHeight, blockWidth, sinOfAnglesNew, cosOfAnglesNew,
                                          blocks, contour, False)
                mainLineEndingD = findMainLineEnding(mainLineD[-1], blocks, blockHeight, blockWidth,
                                                   edgePointsOfSegments)

        else:
            while mainLineEndingC == "5\'" or mainLineEndingC == "5\'\'":
                curve += 5
                sinOfAnglesNew, cosOfAnglesNew = getModifiedAngles(angles, curve, positiveCurve=True)
                mainLineC = findMainLineC(triradiusC, blockHeight, blockWidth, sinOfAnglesNew, cosOfAnglesNew,
                                          blocks, contour, False)
                mainLineEndingC = findMainLineEnding(mainLineC[-1], blocks, blockHeight, blockWidth,
                                                   edgePointsOfSegments)

            mainLines.pop(-1)
            mainLines.append(mainLineC)

    return mainLines, mainLineD, mainLineEndingC, mainLineEndingD


def solveMainLineCrossingBD(mainLines, triradiusB, triradiusD, blocks, angles, contour, blockHeight, blockWidth,
                            edgePointsOfSegments, sinOfAngles, cosOfAngles):
    # Funkcia vyriesi problem, kedy sa krizuju hlavne linie B a D

    mainLineB = findMainLineB(triradiusB, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks, contour,
                              True)
    mainLineEndingB = findMainLineEnding(mainLineB[-1], blocks, blockHeight, blockWidth, edgePointsOfSegments)
    mainLineD = findMainLineD(triradiusD, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks, contour,
                              True)
    mainLineEndingD = findMainLineEnding(mainLineD[-1], blocks, blockHeight, blockWidth, edgePointsOfSegments)
    if (mainLineEndingB == "5\'" or mainLineEndingB == "5\'\'") and mainLineEndingD == 11:
        minDistanceFromD = 10000000
        for point in mainLineB:
            distance = np.sqrt((point[0] - triradiusD[0]) ** 2 + (point[1] - triradiusD[1]) ** 2)
            if distance < minDistanceFromD:
                minDistanceFromD = distance
        minDistanceFromB = 10000000
        for point in mainLineD:
            distance = np.sqrt((point[0] - triradiusB[0]) ** 2 + (point[1] - triradiusB[1]) ** 2)
            if distance < minDistanceFromB:
                minDistanceFromB = distance

        curve = 0
        if minDistanceFromB < minDistanceFromD:
            while mainLineEndingD == 11:
                curve += 5
                sinOfAnglesNew = []
                cosOfAnglesNew = []
                for angle in angles:
                    sinOfAnglesNew.append(math.sin(angle - math.radians(curve)))
                    cosOfAnglesNew.append(math.cos(angle - math.radians(curve)))

                mainLineD = findMainLineD(triradiusD, blockHeight, blockWidth, sinOfAnglesNew, cosOfAnglesNew,
                                          blocks, contour, False)

                mainLineEndingD = findMainLineEnding(mainLineD[-1], blocks, blockHeight, blockWidth,
                                                     edgePointsOfSegments)

        else:
            while mainLineEndingB == "5\'" or mainLineEndingB == "5\'\'":
                curve += 5
                sinOfAnglesNew = []
                cosOfAnglesNew = []
                for angle in angles:
                    sinOfAnglesNew.append(math.sin(angle + math.radians(curve)))
                    cosOfAnglesNew.append(math.cos(angle + math.radians(curve)))

                mainLineB = findMainLineB(triradiusB, blockHeight, blockWidth, sinOfAnglesNew, cosOfAnglesNew,
                                          blocks, contour, False)

                mainLineEndingB = findMainLineEnding(mainLineB[-1], blocks, blockHeight, blockWidth,
                                                     edgePointsOfSegments)

            mainLines.pop(1)
            mainLines.insert(1, mainLineB)

    return mainLines, mainLineD, mainLineEndingB, mainLineEndingD


def findMainLines(blocks, blockImage, angles, triradius, edgePointsOfSegments, contour):
    # Funckia najde priebeh jednotlivych hlavnych linii a urci ich zakoncenie

    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    mainLines = []
    sinOfAngles = calculateSinOfAngles(angles)
    cosOfAngles = calculateCosOfAngles(angles)

    mainLineEndingA = 0
    triradiusA = triradius[0]
    if triradiusA != (0, 0):
        mainLineA = findMainLineA(triradiusA, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks, contour)
        mainLines.append(mainLineA)
        mainLineEndingA = findMainLineEnding(mainLineA[-1], blocks, blockHeight, blockWidth, edgePointsOfSegments)

    mainLineEndingB = 0
    triradiusB = triradius[1]
    if triradiusB != (0, 0):
        mainLineB = findMainLineB(triradiusB, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks, contour, False)
        mainLines.append(mainLineB)
        mainLineEndingB = findMainLineEnding(mainLineB[-1], blocks, blockHeight, blockWidth, edgePointsOfSegments)

    mainLineEndingC = 0
    triradiusC = triradius[2]
    if triradiusC != (0, 0):
        mainLineC = findMainLineC(triradiusC, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks, contour, False)
        mainLines.append(mainLineC)
        mainLineEndingC = findMainLineEnding(mainLineC[-1], blocks, blockHeight, blockWidth, edgePointsOfSegments)
        if mainLineEndingB == 7 and mainLineEndingC == 11:
            mainLines, mainLineEndingB, mainLineEndingC = solveCrossingOfMainLinesBC(mainLines, triradiusB, triradiusC, mainLineB, mainLineC, mainLineEndingB, mainLineEndingC, blocks, angles, contour, blockHeight, blockWidth, edgePointsOfSegments)

    mainLineEndingD = 0
    triradiusD = triradius[3]
    if triradiusD != (0, 0):
        mainLineD = findMainLineD(triradiusD, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks, contour, False)
        mainLineEndingD = findMainLineEnding(mainLineD[-1], blocks, blockHeight, blockWidth, edgePointsOfSegments)
        if (mainLineEndingC == "5\'" or mainLineEndingC == "5\'\'") and mainLineEndingD == 9:
            mainLines, mainLineD, mainLineEndingC, mainLineEndingD = solveMainLineCrossingCD(mainLines, triradiusC, triradiusD, blocks, angles, contour, blockHeight, blockWidth, edgePointsOfSegments, sinOfAngles, cosOfAngles)

        elif (mainLineEndingB == "5\'" or mainLineEndingB == "5\'\'") and mainLineEndingD == 11:
            mainLines, mainLineD, mainLineEndingB, mainLineEndingD = solveMainLineCrossingBD(mainLines, triradiusB, triradiusD, blocks, angles, contour, blockHeight, blockWidth, edgePointsOfSegments, sinOfAngles, cosOfAngles)

        mainLines.append(mainLineD)

    mainLineEndingT = 0
    triradiusT = triradius[4]
    if triradiusT != (0, 0):
        mainLineT = findMainLineT(triradiusT, blockHeight, blockWidth, sinOfAngles, cosOfAngles, blocks, contour)
        mainLines.append(mainLineT)
        mainLineEndingT = findMainLineEnding(mainLineT[-1], blocks, blockHeight, blockWidth, edgePointsOfSegments)

    mainLinesEndings = [mainLineEndingA, mainLineEndingB, mainLineEndingC, mainLineEndingD, mainLineEndingT]

    return mainLines, mainLinesEndings
