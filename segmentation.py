#! /usr/bin/env python

"""

Implementacia segmentacie odtlacku dlane. Vratane segmentacie subor obsahuje funkcie pre oddelenia odtlacku od pozadia,
oddelenie odtlacku od prstov, najdenie orientacie dlane a automaticke ocislovanie jednotlivych segmentov obrysu
odtlacku dlane.

Autor: Bc. Jan Folenta
Email: xfolen00@stud.fit.vutbr.cz
Rok: 2022

"""

import copy
import cv2
import numpy as np
from bresenham import bresenham


def findExtremePalmprintPoints(blocks, blockImage):
    # Funkcia najde a vrati cislo prveho a posledneho riadku a cislo prveho a posledneho stlpca segmentovaneho
    # odtlacku dlane.

    blockRows, blockCols, _, _ = blockImage.shape
    firstRowFound = False
    firstBlockRow = blockCols
    lastBlockRow = 0
    firstBlockCol = blockCols
    lastBlockCol = 0

    for row in range(blockRows):
        for col in range(blockCols):
            if blocks[row][col]["background"] != 1:
                if not firstRowFound:
                    firstBlockRow = row
                    firstRowFound = True
                if col < firstBlockCol:
                    firstBlockCol = col
                if col > lastBlockCol:
                    lastBlockCol = col
                lastBlockRow = row

    return firstBlockRow, lastBlockRow, firstBlockCol, lastBlockCol


def getWhitePixelRatio(block, totalPixels):
    # Funckia najde pomer bielych pixelov v danom bloku

    limit = 127

    x, y = (block > limit).nonzero()
    values = block[x, y]

    wpp = len(values) / totalPixels

    return wpp


def fillNeighborsWithIndex(foregroundRegions, currentBlock, index, blockImage, blocks):
    # Funkcia postupne vyplni vsetky susedne bloky, ktore patria do popredia odltacku danym indexom. Vznikne tak
    # spojity region patriaci do popredia odtlacku.
    blockRows, blockCols, _, _ = blockImage.shape
    toSearch = [currentBlock]

    # Semienkom je vstupny blok, ktory nasledne skuma svoje 4 susedne bloky. V pripade, ze susedny blok patri do
    # popredia odtlacku, je danemu bloku priradeny dany index a blok je vlozeny do pola toSearch.
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
    # Funckia najde a vrati najvacsi spojity region.

    maxSize = 0
    regionWithMaxSize = 0
    for indexRegion in range(1, index):
        regionSize = (foregroundRegions == indexRegion).sum()
        if regionSize > maxSize:
            maxSize = regionSize
            regionWithMaxSize = indexRegion

    return regionWithMaxSize


def putSmallerRegionsIntoBackground(blockRows, blockCols, foregroundRegions, palmprintRegion, blocks):
    # Vsetky bloky, ktore nepatria do najvacsieho spojiteho regionu budu priradene do pozadia odtlacku.

    for row in range(blockRows):
        for col in range(blockCols):
            if foregroundRegions[row][col] != 0 and foregroundRegions[row][col] != palmprintRegion:
                blocks[row][col]["background"] = 1

    return blocks


def checkFingerTip(fingerTips, point, lastPalmprintRow):
    # Funckia skontroluje potencialny koncek prsta.

    y, x = point
    alreadyFound = False
    minDistance = 7.5  # Minimalna vzdialenost medzi dvoma koncekmi prstov (hodnota bola ziskana experimentalne)

    # Potencialny koncek prsta sa nemoze nachadzat v spodnej casti odtlacku (jeho x-ova suradnica sa musi nachadzat
    # minimalne 10 blokov od spondneho riadku odtlacku).
    if x < lastPalmprintRow - 10:
        # Potencialny koncek prsta sa musi nachadzat v urcitej vzdialenosti od uz najdenych koncekov prstov
        if len(fingerTips) == 0:
            fingerTips.append((x, y))
        else:
            for fingerTip in fingerTips:
                xx = fingerTip[0]
                yy = fingerTip[1]
                distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

                if distance <= minDistance:
                    alreadyFound = True

            if not alreadyFound:
                fingerTips.append((x, y))

    return fingerTips


def checkPointBetweenFingers(pointsBetweenFingers, point, lastPalmprintRow):
    # Funkcia skontroluje potencialny bod v medziprstovom priestore

    y, x = point

    # Potencialny bod v medziprstovom priestore sa nemoze nachadzat v spodnej casti odtlacku (jeho x-ova suradnica
    # sa musi nachadzat minimalne 15 blokov od spondneho riadku odtlacku).
    if x < lastPalmprintRow - 15:
        pointsBetweenFingers.append((x, y))

    return pointsBetweenFingers


def findLastRowOfMask(palmprintMask):
    # Funkcia najde cislo prveho a posledneho riadku masky odtlacku dlane

    rows, cols = palmprintMask.shape
    lastRow = 0
    firstRow = 0
    found = False

    for row in range(rows):
        for col in range(cols):
            if palmprintMask[row][col] == 1:
                if not found:
                    firstRow = row
                    found = True
                lastRow = row
                break

    return firstRow, lastRow


def findBorderBetweenFingers(palmprintBorder, startingPoint, blocks, palmprintSegment):
    # Funckia najde zo startovancieho bodu nachadzajuceho sa v medziprstovom priestore cely medziprstovy priestor
    startingPointIndex = palmprintBorder.index(startingPoint)
    pointsBetweenFingers = []

    # Prehladavanie v lavom smere.
    currentPoint = startingPoint
    currentPointRow, currentPointCol = currentPoint
    pointsBetweenFingers.append(currentPoint)
    blocks[currentPointRow][currentPointCol]["palmprintSegment"] = palmprintSegment

    # Prehladava sa az dokym nezacne obrys odtlacku kolmo stupat.
    for distanceFromStartingPoint in range(1, len(palmprintBorder)):
        nextBorderPoint = palmprintBorder[startingPointIndex + distanceFromStartingPoint]
        nextBorderPointRow, nextBorderPointCol = nextBorderPoint
        if nextBorderPointCol >= currentPointCol:
            if nextBorderPointRow < currentPointRow:
                break
        pointsBetweenFingers.append(nextBorderPoint)
        blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = palmprintSegment
        currentPointRow, currentPointCol = nextBorderPoint

    # Prehladavanie v pravom smere.
    currentPoint = startingPoint
    currentPointRow, currentPointCol = currentPoint
    alreadyWentUp = False

    # Prehladava sa az dokym nezacne obrys odtlacku kolmo stupat.
    for distanceFromStartingPoint in range(1, len(palmprintBorder)):
        nextBorderPoint = palmprintBorder[startingPointIndex - distanceFromStartingPoint]
        nextBorderPointRow, nextBorderPointCol = nextBorderPoint
        if nextBorderPointCol <= currentPointCol:
            if nextBorderPointRow < currentPointRow:
                break
        # Priestor medzi prstami 4 a 5 (prstenik a malicek), teda segment 7 ma mierne rozdielny profil ako zvysne
        # priestory. Preto sa prehladava pokym obrys odtlacku nestupa 2-krat po sebe.
        if palmprintSegment == 7:
            if alreadyWentUp and nextBorderPointRow < currentPointRow:
                break
            if not alreadyWentUp and nextBorderPointRow < currentPointRow:
                alreadyWentUp = True
        pointsBetweenFingers.append(nextBorderPoint)
        blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = palmprintSegment
        currentPointRow, currentPointCol = nextBorderPoint

    # Najdene bloky su zoradene podla cisla stlpca
    pointsBetweenFingers.sort(key=lambda x: x[1])

    return blocks, pointsBetweenFingers


def cutFinger(startingPoint, endingPoint, palmprintBorder, palmprintMask, blockImage, blocks):
    # Funkcia vystrihne z odtlacku dlane prst a upravi obrys odtlacku.

    blockRows, blockCols, _, _ = blockImage.shape
    startingPointIndex = palmprintBorder.index(startingPoint)
    endingPointIndex = palmprintBorder.index(endingPoint)
    toSearch = [palmprintBorder[startingPointIndex + 10], palmprintBorder[endingPointIndex - 10]]

    # Najdenie spojnice medzi lavym a pravym blokom bazy prsta
    newFingerBorder = list(bresenham(startingPoint[0], startingPoint[1], endingPoint[0], endingPoint[1]))

    # Vymnazanie obrysu odtlacku medzi lavym a pravym blokom bazy prsta
    if startingPointIndex > endingPointIndex:
        del palmprintBorder[startingPointIndex:len(palmprintBorder)]
        del palmprintBorder[0:endingPointIndex + 1]
        startingPointIndex = 0
    else:
        del palmprintBorder[startingPointIndex:endingPointIndex + 1]

    # Vlozenie najdenej spojnice do obrysu odtlacku
    distanceFromStartingPoint = 0
    for borderPoint in newFingerBorder:
        palmprintMask[borderPoint[0]][borderPoint[1]] = 0
        palmprintBorder.insert(startingPointIndex + distanceFromStartingPoint, borderPoint)
        distanceFromStartingPoint += 1

    # Zmenou obrysu odltacku sa oddelil prst od odtlacku dlane. Bloky patriace vystrihnutemu prstu su nastavene ako
    # bloky patriace do pozadia odtlacku
    while toSearch:
        currentRow, currentCol = toSearch.pop(0)

        topRow = max(0, currentRow - 1)
        bottomRow = min(currentRow + 1, blockRows - 1)
        leftCol = max(0, currentCol - 1)
        rightCol = min(currentCol + 1, blockCols - 1)

        if palmprintMask[topRow][currentCol] == 1:
            toSearch.append((topRow, currentCol))
            blocks[topRow][currentCol]["background"] = 1
            palmprintMask[topRow][currentCol] = 0

        if palmprintMask[currentRow][rightCol] == 1:
            toSearch.append((currentRow, rightCol))
            blocks[currentRow][rightCol]["background"] = 1
            palmprintMask[currentRow][rightCol] = 0

        if palmprintMask[bottomRow][currentCol] == 1:
            toSearch.append((bottomRow, currentCol))
            blocks[bottomRow][currentCol]["background"] = 1
            palmprintMask[bottomRow][currentCol] = 0

        if palmprintMask[currentRow][leftCol] == 1:
            toSearch.append((currentRow, leftCol))
            blocks[currentRow][leftCol]["background"] = 1
            palmprintMask[currentRow][leftCol] = 0

    return blocks, palmprintBorder, palmprintMask


def findPointOnTheOtherSideOfFinger(startingPoint, palmprintBorder, blocks, reverse, palmprintSegment):
    # Funkcia najde zaciatok prsta na jeho opacnej strane (zaciatok prsta na jednej strane uz je najdeny a je
    # vstupnym argumentom funkcie).

    startingPointIndex = palmprintBorder.index(startingPoint)
    startingPointRow, startingPointCol = startingPoint

    minDistance = len(palmprintBorder)
    minDistanceIndex = 0

    # Postupne sa prechadza obrys odtlacku a v pripade, ze sa aktualny bod nachadza v podobnej vyske ako zaciatocny
    # bod (-+ 5 blokov), je odmerana vzdialenost medzi danym a zaciatocnym bodom. Bod, ktory je od zaciatocneho bodu
    # najblizsie je povazovany za zaciatok prsta na jeho druhej strane.
    for distanceFromStartingPoint in range(10, len(palmprintBorder)):
        if not reverse:
            nextPointIndex = startingPointIndex + distanceFromStartingPoint
            if nextPointIndex >= len(palmprintBorder):
                nextPointIndex = nextPointIndex - len(palmprintBorder)
            nextBorderPoint = palmprintBorder[nextPointIndex]
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
        borderIndex = startingPointIndex + minDistanceIndex
        if borderIndex >= len(palmprintBorder):
            borderIndex = borderIndex - len(palmprintBorder)
        endingPoint = palmprintBorder[borderIndex]
    else:
        endingPoint = palmprintBorder[startingPointIndex - minDistanceIndex]

    # Najdenemu bodu je priradene prislusne cislo segmentu odtlacku.
    blocks[endingPoint[0]][endingPoint[1]]["palmprintSegment"] = palmprintSegment

    return endingPoint, blocks


def findPointsBetweenThumb(startingPoint, palmprintBorder, blockImage, blocks, reverse, palmprintSegment,
                           thumbDetected):
    # Funkcia najde cast obrysu odtlacku, ktory sa nachadza medzi palcom a ukazovakom spolu s opacnym koncom palca.

    startingPointIndex = palmprintBorder.index(startingPoint)
    pointsBetweenFingers = []

    currentPoint = startingPoint
    currentPointRow, currentPointCol = currentPoint
    pointsBetweenFingers.append(currentPoint)

    # Ked bol detekovany palec, tak sa postupne prechazda obrys odtlacku, pricom zaciatocnym bodom je bod na lavej
    # strane ukazovaka. Prechadza sa dovtedy kym nezacne obrys odtlacku stupat. Dany bod je zaciatkom palca na
    # jednej strane.
    if thumbDetected:
        for distanceFromStartingPoint in range(1, len(palmprintBorder)):
            nextBorderPoint = palmprintBorder[startingPointIndex + distanceFromStartingPoint]
            nextBorderPointRow, nextBorderPointCol = nextBorderPoint
            if nextBorderPointRow < currentPointRow:
                break
            pointsBetweenFingers.append(nextBorderPoint)
            blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = palmprintSegment
            currentPointRow, currentPointCol = nextBorderPoint

        # Nasledujuca cast kodu najde zaciatocny bod palca na jeho druhej strane
        startingPoint = pointsBetweenFingers[-1]
        startingPointIndex = palmprintBorder.index(startingPoint)
        startingPointRow, startingPointCol = startingPoint

        minDistance = len(palmprintBorder)
        minDistanceIndex = 0

        for distanceFromStartingPoint in range(20, len(palmprintBorder)):
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

    # Ked palec detekovany nebol, tak sa postupne prechazda obrys odtlacku, pricom zaciatocnym bodom je bod na lavej
    # strane ukazovaka. Najlavejsi bod obrysu odltacku medzi zaciatocnym bodom a spodnym bodom odtlacku je urceny ako
    # zaciatocny bod palca.
    else:
        mostLeft = currentPointCol
        thumbPointDistance = 0
        lastRow = max(palmprintBorder, key=lambda t: t[0])[0]
        for distanceFromStartingPoint in range(1, len(palmprintBorder)):
            nextBorderPoint = palmprintBorder[startingPointIndex + distanceFromStartingPoint]
            nextBorderPointRow, nextBorderPointCol = nextBorderPoint
            if nextBorderPointRow == lastRow:
                break
            if nextBorderPointCol < mostLeft:
                mostLeft = nextBorderPointCol
                thumbPointDistance = distanceFromStartingPoint

        for distanceFromStartingPoint in range(1, thumbPointDistance):
            nextBorderPoint = palmprintBorder[startingPointIndex + distanceFromStartingPoint]
            nextBorderPointRow, nextBorderPointCol = nextBorderPoint
            pointsBetweenFingers.append(nextBorderPoint)
            blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = palmprintSegment

        # Druhym zaciatocnym bodom palca bude bod obrysu odtlacku, ktory sa nachadza vo vzdialenosti 2 bloky od prveho
        # zaciatocneho bodu.
        endingPoint = palmprintBorder[startingPointIndex + thumbPointDistance + 2]

    return endingPoint, pointsBetweenFingers, blockImage, blocks


def findPalmprintBorder(palmprintMask):
    # Funkcia najde obrys odtlacku.

    contours, _ = cv2.findContours(palmprintMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    palmprintBorder = []

    contour = max(contours, key=lambda t: len(t))

    for point in contour:
        y, x = point[0]
        palmprintBorder.append((x, y))

    return palmprintBorder, contour


def findMissingFingerTips(fingerTips, palmprintBorder, middlePalmprintRow):
    # Funkcia najde chybajuce konceky prstov

    increasing = False
    increasingLength = 0
    newFingerTips = []

    # Postupne je prehladavany obrys odtlacku, pricom sa zaznamenava ci obrys odtlacku stupa alebo klesa.
    # Ked obrys odtlacku stupa aspon 5 po sebe iducich blokov a nasledne zacne klesat, je dany bod uchovany
    # ako potencialny koncek prsta a je porovnany s uz najdenymi koncekmi prstov.
    for i in range(len(palmprintBorder) - 1):
        # Zaujima nas iba vrchna polovica odtlacku
        if palmprintBorder[i][0] > middlePalmprintRow:
            continue
        if palmprintBorder[i + 1][0] < palmprintBorder[i][0]:
            increasing = True
            increasingLength += 1
        if palmprintBorder[i + 1][0] > palmprintBorder[i][0] and increasing:
            increasing = False
            if increasingLength <= 5:
                increasingLength = 0
                continue
            increasingLength = 0
            currentRow, currentCol = palmprintBorder[i]
            found = False
            for tip in fingerTips:
                dist = abs(currentCol - tip[1])
                if dist <= 5:
                    found = True
            if not found:
                fingerTips.append((currentRow, currentCol))
                newFingerTips.append((currentRow, currentCol))

    return fingerTips, newFingerTips


def findMissingPointsBetweenFingers(newFingerTips, palmprintBorder, pointsBetweenFingers):
    # Funkcia najde chybajuce body medzi prstami

    # Postupne je prehladavany obrys odtlacku od novo najdeneho konceka prsta a v bod, v ktorom zacne obrys stupat
    # je urceny ako bod medzi prstami
    for fingerTip in newFingerTips:
        startingPointIndex = palmprintBorder.index(fingerTip)
        currentRow, currentCol = fingerTip

        # Prehladavanie lavej strany konceka
        for i in range(len(palmprintBorder)):
            nextRow, nextCol = palmprintBorder[startingPointIndex + i]
            if nextRow < currentRow:
                found = False
                for point in pointsBetweenFingers:
                    dist = abs(currentCol - point[1])
                    if dist <= 5:
                        found = True
                if not found:
                    pointsBetweenFingers.append((currentRow, currentCol))
                break
            currentRow = nextRow
            currentCol = nextCol

        startingPointIndex = palmprintBorder.index(fingerTip)
        currentRow, currentCol = fingerTip

        # Prehladavanie pravej strany konceka
        for i in range(len(palmprintBorder)):
            nextRow, nextCol = palmprintBorder[startingPointIndex - i]
            if nextRow < currentRow:
                found = False
                for point in pointsBetweenFingers:
                    dist = abs(currentCol - point[1])
                    if dist <= 5:
                        found = True
                if not found:
                    pointsBetweenFingers.append((currentRow, currentCol))
                break
            currentRow = nextRow
            currentCol = nextCol

    return pointsBetweenFingers


def findFingerPoints(palmprintMask, contour, palmprintBorder, blockImage):
    # Funkcia najde konceky prstov a body nachadzajuce sa medzi prstami (medzi dvoma susednymi prstami jeden bod).

    blockRows, blockCols, _, _ = blockImage.shape
    firstPalmprintRow, lastPalmprintRow = findLastRowOfMask(palmprintMask)
    middlePalmprintRow = (firstPalmprintRow + lastPalmprintRow) // 2

    fingerTips = []
    pointsBetweenFingers = []

    # Najdenie konvexneho obalu a konvexnych defektov odtlacku dlane
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)

    if defects is None:
        print("ERROR")

    # Postupne prehladavanie konvexnych defektov
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i][0]
        startX, startY = contour[s][0]
        endX, endY = contour[e][0]
        farX, farY = contour[f][0]
        a = np.sqrt((endX - startX) ** 2 + (endY - startY) ** 2)
        b = np.sqrt((farX - startX) ** 2 + (farY - startY) ** 2)
        c = np.sqrt((endX - farX) ** 2 + (endY - farY) ** 2)
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
        # Potencialne konceky prstov a bod medzi prstami su skontrolovane na zaklade uz najdenych bodov
        if angle <= np.pi / 2:
            fingerTips = checkFingerTip(fingerTips, (startX, startY), lastPalmprintRow)
            fingerTips = checkFingerTip(fingerTips, (endX, endY), lastPalmprintRow)
            pointsBetweenFingers = checkPointBetweenFingers(pointsBetweenFingers, (farX, farY), lastPalmprintRow)

    # V pripade, ze je najdenych menej ako 5 koncekov prstov, je potrebne najst zvysne.
    if len(fingerTips) < 5:
        # Ked sa nejaky koncek prsta nachadza v spodnej polovici odtlacku (co evokuje koncek palca), su v dalsej casti
        # hladane nedetekovane konceky prstov.
        allFingersFound = True
        for tip in fingerTips:
            if tip[0] > (firstPalmprintRow + middlePalmprintRow) // 2:
                allFingersFound = False

        if not allFingersFound:
            fingerTips, newFingerTips = findMissingFingerTips(fingerTips, palmprintBorder, middlePalmprintRow)

            # Dalsia cast kodu najde body medzi novo najdenymi koncekmi prstov a susednymi koncekmi prstov
            if len(newFingerTips) > 0:
                pointsBetweenFingers = findMissingPointsBetweenFingers(newFingerTips, palmprintBorder,
                                                                       pointsBetweenFingers)

    fingerTips.sort(key=lambda x: x[1])
    pointsBetweenFingers.sort(key=lambda x: x[1])

    return fingerTips, pointsBetweenFingers


def findPalmOrientation(fingerTips):
    # Funkcia najde orientaciu dlane (ci sa jedna o lavu alebo pravu dlan) a zisti, ci bol detekovany palec.

    palmOrientation = None
    thumbDetected = False

    # V pripade, ze je najdenych vsetkych 5 koncekov prstov, je orientacia dlane urcena podla najnizsie polozeneho
    # konceka prsta, teda palca.
    if len(fingerTips) == 5:
        thumbDetected = True
        maxValue = max(fingerTips, key=lambda t: t[0])
        maxValueIndex = fingerTips.index(maxValue)

        if maxValueIndex == 0:
            palmOrientation = "right"

        if maxValueIndex == 4:
            palmOrientation = "left"

    # V pripade, ze su najdene 4 konceky prstov (co znamena ze palec nebol detekovany), je orientacia dlane urcena
    # podla najnizsie polozeneho konceka prsta, teda malicka.
    if len(fingerTips) == 4:

        thumbDetected = False
        maxValue = max(fingerTips, key=lambda t: t[0])
        maxValueIndex = fingerTips.index(maxValue)

        if maxValueIndex == 0:
            palmOrientation = "left"

        if maxValueIndex == 3:
            palmOrientation = "right"

    return palmOrientation, thumbDetected


def flipPalmprint(image, blocks, blockImage):
    # Funkcia osovo otoci udaje o odtlacku dlane ziskane v predchadzajucich castiach

    imageHeight, imageWidth = image.shape
    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape

    flippedImage = cv2.flip(image, 1)
    flippedBlocks = copy.deepcopy(blocks)

    blockImage = flippedImage.reshape(imageHeight // blockHeight, blockHeight, imageWidth // blockWidth, blockWidth)
    blockImage = blockImage.swapaxes(1, 2)

    for blockRow in range(blockRows):
        for blockCol in range(blockCols):
            flippedBlocks[blockRow][blockCol]["background"] = blocks[blockRow][blockCols - blockCol - 1]["background"]
            flippedBlocks[blockRow][blockCol]["orientation"] = blocks[blockRow][blockCols - blockCol - 1]["orientation"]
            flippedBlocks[blockRow][blockCol]["orientationConfidence"] = blocks[blockRow][blockCols - blockCol - 1][
                "orientationConfidence"]
            flippedBlocks[blockRow][blockCol]["palmprintSegment"] = blocks[blockRow][blockCols - blockCol - 1][
                "palmprintSegment"]
            flippedBlocks[blockRow][blockCol]["triradiusRegion"] = blocks[blockRow][blockCols - blockCol - 1][
                "triradiusRegion"]

    palmprintMask = np.zeros((blockRows, blockCols), dtype=np.uint8)

    for row in range(blockRows):
        for col in range(blockCols):
            if flippedBlocks[row][col]["background"] == 0:
                palmprintMask[row][col] = 1

    palmprintBorder, contour = findPalmprintBorder(palmprintMask)
    fingerTips, pointsBetweenFingers = findFingerPoints(palmprintMask, contour, palmprintBorder, blockImage)

    return flippedImage, flippedBlocks, blockImage, fingerTips, pointsBetweenFingers, palmprintBorder, palmprintMask


def checkDetectedPalmprint(blocks, blockRows, blockCols, palmprintMask):
    # Funkcia sa vysporiada s pripadmi, kedy su bloky patriace do odtlack dlane detekovane aj na okraji vstupneho
    # obrazu odltacku. V takom pripade sa odtlacok bud napojil na text nad nim, alebo sa palec dotyka samostatnych
    # odtlackov prstov na lavej strane obrazu odltacku.

    found = False
    end = False
    firstRow = 0
    for row in range(blockRows // 5):
        for col in range(blockCols // 4, blockCols * 3 // 4):
            if row > 5 and not found:
                end = True
                break
            if palmprintMask[row][col]:
                if palmprintMask[row][col + 1] == 0 or palmprintMask[row][col + 2] == 0 or palmprintMask[row][col + 3] \
                        == 0:
                    found = True
                    firstRow = row
                else:
                    if found:
                        end = True
                break
        if end:
            break

    for row in range(firstRow + 1):
        for col in range(blockCols):
            palmprintMask[row][col] = 0
            blocks[row][col]["background"] = 1

    found = False
    firstCol = 0
    for col in range(blockCols // 2):
        if col > 5 and not found:
            break
        foregroundBlocks = 0
        for row in range(blockRows // 4, blockRows * 3 // 4):
            if palmprintMask[row][col]:
                foregroundBlocks += 1
                found = True
        if foregroundBlocks < 10:
            firstCol = col

    for col in range(firstCol):
        for row in range(blockRows):
            palmprintMask[row][col] = 0
            blocks[row][col]["background"] = 1

    return blocks, palmprintMask


def cutBottomOfPalmprint(blocks, blockRows, blockCols, palmprintMask):
    # Funkcia vyraze tenky spodok dlane. Sirka spodku dlane musi byt aspon 25 blokov.

    lastRow = blockRows
    for row in range(blockRows):
        length = 0
        firstFound = False
        for col in range(blockCols):
            if palmprintMask[blockRows - row - 1][col] == 0 and firstFound:
                break
            if palmprintMask[blockRows - row - 1][col] == 1:
                firstFound = True
                length += 1
        if firstFound and length >= 25:
            lastRow = blockRows - row - 1
            break

    for row in range(lastRow + 3, blockRows):
        for col in range(blockCols):
            palmprintMask[row][col] = 0
            blocks[row][col]["background"] = 1

    return blocks, palmprintMask


def removePossibleJoinedFingers(blocks, palmprintMask, palmprintBorder, contour):
    # Funckia ddstrani potencialny problem spojenych prstov

    firstPalmprintRow, lastPalmprintRow = findLastRowOfMask(palmprintMask)
    middlePalmprintRow = (firstPalmprintRow + lastPalmprintRow) // 2
    decreesing = False
    change = False
    for i in range(len(palmprintBorder) - 1):
        if palmprintBorder[i + 1][0] > palmprintBorder[i][0]:
            decreesing = True
        if palmprintBorder[i + 1][0] < palmprintBorder[i][0] and decreesing:
            decreesing = False
            currentRow, currentCol = palmprintBorder[i]
            if currentRow < middlePalmprintRow and (
                    palmprintMask[currentRow + 5][currentCol - 1] == 0 or palmprintMask[currentRow + 5][
                currentCol] == 0 or palmprintMask[currentRow + 5][currentCol + 1] == 0):
                change = True
                if palmprintMask[currentRow + 5][currentCol - 1] == 0:
                    for row in range(currentRow, currentRow + 5):
                        for col in range(currentCol - 1, currentCol + 1):
                            palmprintMask[row][col] = 0
                            blocks[row][col]["background"] = 1
                if palmprintMask[currentRow + 5][currentCol] == 0:
                    for row in range(currentRow, currentRow + 5):
                        for col in range(currentCol, currentCol + 1):
                            palmprintMask[row][col] = 0
                            blocks[row][col]["background"] = 1
                if palmprintMask[currentRow + 5][currentCol + 1] == 0:
                    for row in range(currentRow, currentRow + 5):
                        for col in range(currentCol, currentCol + 2):
                            palmprintMask[row][col] = 0
                            blocks[row][col]["background"] = 1

    if change:
        palmprintBorder, contour = findPalmprintBorder(palmprintMask)

    while change:
        change = False
        decreesing = False
        for i in range(len(palmprintBorder) - 1):
            if palmprintBorder[i + 1][0] > palmprintBorder[i][0]:
                decreesing = True
            if palmprintBorder[i + 1][0] < palmprintBorder[i][0] and decreesing:
                decreesing = False
                currentRow, currentCol = palmprintBorder[i]
                if currentRow < middlePalmprintRow and (
                        palmprintMask[currentRow + 5][currentCol - 1] == 0 or palmprintMask[currentRow + 5][
                    currentCol] == 0 or palmprintMask[currentRow + 5][currentCol + 1] == 0):
                    change = True
                    if palmprintMask[currentRow + 5][currentCol - 1] == 0:
                        for row in range(currentRow, currentRow + 5):
                            for col in range(currentCol - 1, currentCol + 1):
                                palmprintMask[row][col] = 0
                                blocks[row][col]["background"] = 1
                    if palmprintMask[currentRow + 5][currentCol] == 0:
                        for row in range(currentRow, currentRow + 5):
                            for col in range(currentCol, currentCol + 1):
                                palmprintMask[row][col] = 0
                                blocks[row][col]["background"] = 1
                    if palmprintMask[currentRow + 5][currentCol + 1] == 0:
                        for row in range(currentRow, currentRow + 5):
                            for col in range(currentCol, currentCol + 2):
                                palmprintMask[row][col] = 0
                                blocks[row][col]["background"] = 1

        if change:
            palmprintBorder, contour = findPalmprintBorder(palmprintMask)

    return blocks, palmprintMask, palmprintBorder, contour


def detectPalmprint(blockImage, blocks, image):
    # Funkcia detekuje samotny odtlacok dlane (spolu s prstami)

    blockRows, blockCols, _, _ = blockImage.shape
    foregroundRegions = np.zeros((blockRows, blockCols))
    index = 1

    # Postupne prechadza vsetky bloky a ked najde blok patriaci do popredia tak ho oznaci indexom a vyplni rovnakym
    # indexom vsetky susedne bloky v popredi - takym sposobom vznikne niekolko regionov
    for row in range(blockRows):
        for col in range(blockCols):
            if blocks[row][col]["background"] == 0:
                if foregroundRegions[row][col] == 0:
                    foregroundRegions[row][col] = index
                    foregroundRegions = fillNeighborsWithIndex(foregroundRegions, (row, col), index, blockImage, blocks)
                    index += 1

    # Ked su aspon 2 najdene regiony (tym nultym regionom je pozadie), tak najde najvacsi a zvysne bloky da do
    # pozadia
    if index > 2:
        palmprintRegion = findRegionWithMaxSize(foregroundRegions, index)
        blocks = putSmallerRegionsIntoBackground(blockRows, blockCols, foregroundRegions, palmprintRegion, blocks)

    palmprintMask = np.zeros((blockRows, blockCols), dtype=np.uint8)

    # Najde sa maska odtlacku dlane
    for row in range(blockRows):
        for col in range(blockCols):
            if blocks[row][col]["background"] == 0:
                palmprintMask[row][col] = 1

    blocks, palmprintMask = checkDetectedPalmprint(blocks, blockRows, blockCols, palmprintMask)
    blocks, palmprintMask = cutBottomOfPalmprint(blocks, blockRows, blockCols, palmprintMask)

    # Najde sa obrys odtlacku
    palmprintBorder, contour = findPalmprintBorder(palmprintMask)

    blocks, palmprintMask, palmprintBorder, contour = removePossibleJoinedFingers(blocks, palmprintMask,
                                                                                  palmprintBorder, contour)

    # Najdu sa konceky prstov a body medzi prstami
    fingerTips, pointsBetweenFingers = findFingerPoints(palmprintMask, contour, palmprintBorder, blockImage)

    # Najde sa orientacia dlane
    handOrientation, thumbDetected = findPalmOrientation(fingerTips)

    # V pripade lavej dlane sa odtlacok osovo otoci
    if handOrientation == "left":
        image, blocks, blockImage, fingerTips, pointsBetweenFingers, palmprintBorder, palmprintMask = flipPalmprint(
            image, blocks, blockImage)

    return image, blocks, blockImage, fingerTips, pointsBetweenFingers, palmprintBorder, palmprintMask, thumbDetected, contour


def connectFingersWithPalmprint(blockImage, blocks):
    # Funkcia sa pokusi spojit clanky prstov s odtlackom """

    blockRows, blockCols, _, _ = blockImage.shape

    for row in range(0, blockRows):
        for col in range(0, blockCols):
            if blocks[row][col]["background"]:
                topRow = max(0, row - 1)
                bottomRow = min(row + 1, blockRows - 1)
                belowBottom = min(row + 2, blockRows - 1)
                belowBelowBottom = min(row + 3, blockRows - 1)

                if blocks[topRow][col]["background"] == 0 and (
                        blocks[bottomRow][col]["background"] == 0 or blocks[belowBottom][col]["background"] == 0 or
                        blocks[belowBelowBottom][col]["background"] == 0):
                    blocks[row][col]["background"] = 0

    return blocks


def fillSegment1(blocks, palmprintBorder, bottomPointOfThumb):
    # Funkcia vyplni segment obrysu odtlacku pod palcom hodnotou 1

    lastRow = max(palmprintBorder, key=lambda t: t[0])[0]
    startingPointIndex = palmprintBorder.index(bottomPointOfThumb)
    currentPointRow, currentPointCol = bottomPointOfThumb

    for distanceFromStartingPoint in range(1, len(palmprintBorder)):
        nextBorderPoint = palmprintBorder[startingPointIndex + distanceFromStartingPoint]
        nextBorderPointRow, nextBorderPointCol = nextBorderPoint
        if currentPointRow == nextBorderPointRow and nextBorderPointRow > lastRow - 3:
            break
        blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = 1
        currentPointRow, currentPointCol = nextBorderPoint

    bottomPointOfSegment1 = currentPointRow, currentPointCol

    return blocks, bottomPointOfSegment1


def fillUlnarSide(blocks, palmprintBorder, rightPointOfFinger5):
    # Funkcia vyplni ulnarny okraj obrysu odtlacku hodnotami 3, 4 a 5

    lastRow = max(palmprintBorder, key=lambda t: t[0])[0]
    startingPointIndex = palmprintBorder.index(rightPointOfFinger5)
    currentPointRow, currentPointCol = rightPointOfFinger5

    for distanceFromStartingPoint in range(1, len(palmprintBorder)):
        nextBorderPoint = palmprintBorder[startingPointIndex - distanceFromStartingPoint]
        nextBorderPointRow, nextBorderPointCol = nextBorderPoint
        if currentPointRow == nextBorderPointRow and nextBorderPointRow > lastRow - 3:
            break
        blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = 5
        currentPointRow, currentPointCol = nextBorderPoint

    bottomPointOfSegment3 = currentPointRow, currentPointCol

    firstUlnarRow = rightPointOfFinger5[0]
    lastUlnarRow = currentPointRow
    middleUlnarRow = (firstUlnarRow + lastUlnarRow) // 2
    middleUlnarPoint = (0, 0)

    for distanceFromStartingPoint in range(1, len(palmprintBorder)):
        nextBorderPoint = palmprintBorder[startingPointIndex - distanceFromStartingPoint]
        nextBorderPointRow, nextBorderPointCol = nextBorderPoint
        if nextBorderPointRow == lastUlnarRow:
            blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = 3
            break
        elif nextBorderPointRow == middleUlnarRow + 1:
            blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = 4
            middleUlnarPoint = (nextBorderPointRow, nextBorderPointCol)
        elif nextBorderPointRow > middleUlnarRow + 1:
            blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = 3

    return blocks, middleUlnarPoint, bottomPointOfSegment3


def splitPalmprintBorderIntoSegments(blocks, blockImage, palmprintBorder, palmprintMask, pointsBetweenFingers,
                                     thumbDetected):
    # Funckia rozdeli obrys odtlacku na jednotlive segmenty a kazdemu segmentu priradi prislusnu hodnotu

    if len(pointsBetweenFingers) == 4:
        pointsBetweenFingers.pop(0)

    # Najdenie hranice medzi prstom 2 a 3 (ukazovak a prostrednik), najdenie bodu na druhej strane
    # ukazovaka a odstrihnutie ukazovaka
    blocks, between23 = findBorderBetweenFingers(palmprintBorder, pointsBetweenFingers[0], blocks, 11)
    leftPointOfFinger2, blocks = findPointOnTheOtherSideOfFinger(between23[0], palmprintBorder, blocks, False,
                                                                 13)
    blocks, palmprintBorder, palmprintMask = cutFinger(between23[0], leftPointOfFinger2, palmprintBorder,
                                                       palmprintMask, blockImage, blocks)

    # Najdenie hranice medzi prstom 3 a 4 (prostrednik a prstenik), a odstrihnutie prostrednika
    blocks, between34 = findBorderBetweenFingers(palmprintBorder, pointsBetweenFingers[1], blocks, 9)
    blocks, palmprintBorder, palmprintMask = cutFinger(between34[0], between23[-1], palmprintBorder,
                                                       palmprintMask, blockImage, blocks)

    # Najdenie hranice medzi prstom 4 a 5 (prstenik a malicek), a odstrihnutie prstenika
    blocks, between45 = findBorderBetweenFingers(palmprintBorder, pointsBetweenFingers[2], blocks, 7)
    blocks, palmprintBorder, palmprintMask = cutFinger(between45[0], between34[-1], palmprintBorder,
                                                       palmprintMask, blockImage, blocks)

    # Najdenie druhej strany malicka a odstrihnutie malicka
    rightPointOfFinger5, blocks = findPointOnTheOtherSideOfFinger(between45[-1], palmprintBorder,
                                                                  blocks, True, 5)
    blocks, palmprintBorder, palmprintMask = cutFinger(rightPointOfFinger5, between45[-1], palmprintBorder,
                                                       palmprintMask, blockImage, blocks)

    # Najdenie oboch stran palca a jeho odstrihnutie
    bottomPointOfThumb, between12, blockImage, blocks = findPointsBetweenThumb(leftPointOfFinger2,
                                                                               palmprintBorder,
                                                                               blockImage, blocks, False, 13,
                                                                               thumbDetected)
    if thumbDetected:
        blocks, palmprintBorder, palmprintMask = cutFinger(between12[-1], bottomPointOfThumb, palmprintBorder,
                                                           palmprintMask, blockImage, blocks)

    blocks, bottomPointOfSegment1 = fillSegment1(blocks, palmprintBorder, bottomPointOfThumb)
    blocks, middleUlnarPoint, bottomPointOfSegment3 = fillUlnarSide(blocks, palmprintBorder, rightPointOfFinger5)

    for borderPoint in palmprintBorder:
        blocks[borderPoint[0]][borderPoint[1]]["background"] = 2

    leftOfFinger5 = max(between45, key=lambda t: t[1])
    rightOfFinger4 = min(between45, key=lambda t: t[1])
    leftOfFinger4 = max(between34, key=lambda t: t[1])
    rightOfFinger3 = min(between34, key=lambda t: t[1])
    leftOfFinger3 = max(between23, key=lambda t: t[1])
    rightOfFinger2 = min(between23, key=lambda t: t[1])

    edgePointsOfSegments = [bottomPointOfThumb, bottomPointOfSegment1, bottomPointOfSegment3, bottomPointOfSegment3,
                            middleUlnarPoint, rightPointOfFinger5, leftOfFinger5, rightOfFinger4, leftOfFinger4,
                            rightOfFinger3, leftOfFinger3, rightOfFinger2, leftPointOfFinger2]

    betweenFingersAreas = [leftPointOfFinger2, between23, between34, between45, rightPointOfFinger5]
    radialSide = [leftPointOfFinger2, bottomPointOfSegment1]
    ulnarSide = [rightPointOfFinger5, bottomPointOfSegment3]

    return blockImage, blocks, blockImage, betweenFingersAreas, ulnarSide, radialSide, palmprintBorder, \
           edgePointsOfSegments


def segmentation(blockImage, blocks, image):
    # Funkcia oddeli odtlacok dlane od pozadia a prstov

    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    totalPixels = blockHeight * blockWidth
    success = True

    for row in range(blockRows):
        for col in range(blockCols):
            block = blockImage[row][col]
            wpp = getWhitePixelRatio(block, totalPixels)
            var = np.var(block)
            v = wpp - 0.001 * var  # Koeficienty boli ziskane experimentalne

            if v > 0.95:
                blocks[row][col]["background"] = 1

    blocks = connectFingersWithPalmprint(blockImage, blocks)

    image, blocks, blockImage, fingerTips, pointsBetweenFingers, palmprintBorder, palmprintMask, thumbDetected, contour \
        = detectPalmprint(blockImage, blocks, image)

    if len(pointsBetweenFingers) != 4 and len(pointsBetweenFingers) != 3:
        success = False
        return success, None, None, None, None, None, None, None, None, None

    blockImage, blocks, blockImage, betweenFingersAreas, ulnarSide, radialSide, palmprintBorder, \
    edgePointsOfSegments = splitPalmprintBorderIntoSegments(blocks, blockImage, palmprintBorder, palmprintMask,
                                                            pointsBetweenFingers, thumbDetected)

    contours, _ = cv2.findContours(palmprintMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        contour = max(contours, key=lambda t: len(t))
    else:
        success = False

    return success, image, blockImage, blocks, betweenFingersAreas, ulnarSide, radialSide, palmprintBorder, edgePointsOfSegments, contour
