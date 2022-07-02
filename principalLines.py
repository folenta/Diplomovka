#! /usr/bin/env python

"""

Implementacia detekcie flekcnych ryh.

Autor: Bc. Jan Folenta
Email: xfolen00@stud.fit.vutbr.cz
Rok: 2022

"""

import numpy as np
from skimage.morphology import flood
from imageProcessing import mergeBlocksToImage


def getColCandidateForProjection(ulnarSide, radialSide, blocks, left):
    # Funkcia najde stlpce, v ktorych moze prebehnut smerova projekcia.

    colCandidates: list[int] = []
    topRowLeft, topColLeft = radialSide[0]
    bottomRowLeft, bottomColLeft = radialSide[1]

    topRowRight, topColRight = ulnarSide[0]
    bottomRowRight, bottomColRight = ulnarSide[1]

    middleCol = (topColLeft + topColRight) // 2

    # V pripade projekcie na lavej strane odtlacku pripadaju do uvahy stlpce od prveho bodu radialnej strany odtlacku
    # az po horizontalny stred odtlacku. Aby mohol byt stlpec kandidatom na smerovu projekciu, musia bloky pozdlz
    # vysky okna neprerusene patrit do odtlacku dlane.
    if left:
        middleRowLeft = (topRowLeft + bottomRowLeft) // 2
        middleSearchRow = (middleRowLeft + topRowLeft) // 2
        quarterDistanceFromMiddleSearchRow = (middleSearchRow - topRowLeft) // 2

        firstSearchRow = (middleSearchRow - quarterDistanceFromMiddleSearchRow)
        lastSearchRow = (middleSearchRow + quarterDistanceFromMiddleSearchRow)

        firstCol = 0
        for col in range(topColLeft + 1, middleCol):
            valid = True
            firstCol = col
            for row in range(firstSearchRow, lastSearchRow):
                if blocks[row][col]["background"] != 0:
                    valid = False
                    break
            if valid:
                break

        for col in range(firstCol, middleCol):
            colCandidates.append(col)

        return colCandidates

    # V pripade projekcie na pravej strane odtlacku pripadaju do uvahy stlpce od horizontalneho stredu odtlacku az
    # po prvy bod ulnarneho okraja dlane. Aby mohol byt stlpec kandidatom na smerovu projekciu, musia bloky pozdlz
    # vysky okna neprerusene patrit do odtlacku dlane.
    else:
        middleRowRight = (topRowRight + bottomRowRight) // 2
        middleSearchRow = (middleRowRight + topRowRight) // 2
        quarterDistanceFromMiddleSearchRow = (middleSearchRow - topRowRight) // 2

        firstSearchRow = (middleSearchRow - quarterDistanceFromMiddleSearchRow)
        lastSearchRow = (middleSearchRow + quarterDistanceFromMiddleSearchRow)

        lastCol = 0
        i = 0
        for col in range(middleCol, topColRight):
            valid = True
            lastCol = topColRight - i
            i += 1
            for row in range(firstSearchRow, lastSearchRow):
                if blocks[row][topColRight - i]["background"] != 0:
                    valid = False
            if valid:
                break

        for col in range(middleCol, lastCol):
            colCandidates.append(col)

        colCandidates.reverse()

        return colCandidates


def projectionSmoothing(sumIntensityInRow):
    # Funkcia vyhladi smerovu projekciu. Pri vyhladeni sa pozera na 5 susednych pixelov.

    step = 5
    smoothenedSumIntensityInRow = [0] * len(sumIntensityInRow)

    for row in range(len(sumIntensityInRow)):
        rowsInSmoothing = 0
        for rowIndex in range(max(0, row - step), min(len(sumIntensityInRow), row + step)):
            smoothenedSumIntensityInRow[row] += sumIntensityInRow[rowIndex]
            rowsInSmoothing += 1

        smoothenedSumIntensityInRow[row] = smoothenedSumIntensityInRow[row] // rowsInSmoothing

    return smoothenedSumIntensityInRow


def orientedProjection(image, blockImage, colCandidate, topPoint, bottomPoint):
    # Funkcia najde bod na flekcnej ryhe pomocou algoritmu smerovej projekcie.

    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape

    firstBlockRow = topPoint[0]
    lastBlockRow = bottomPoint[0]

    # Urcenie okna pre smerovu projekciu
    middleBlockRow = (firstBlockRow + lastBlockRow) // 2

    middleSearchRow = (middleBlockRow + firstBlockRow) // 2
    quarterDistanceFromMiddleSearchRow = (middleSearchRow - firstBlockRow) // 2

    firstSearchRow = (middleSearchRow - quarterDistanceFromMiddleSearchRow) * blockHeight
    lastSearchRow = (middleSearchRow + quarterDistanceFromMiddleSearchRow) * blockHeight

    firstSearchCol = colCandidate * blockWidth
    lastSearchCol = firstSearchCol + blockWidth

    sumIntensityInRow = [0] * (lastSearchRow - firstSearchRow)
    rowIndex = 0

    # Vypocet sumy intenzit pixelov pozdlz kazdeho bodu stlpca v okne smerovej projekcie
    for row in range(firstSearchRow, lastSearchRow):
        for col in range(firstSearchCol, lastSearchCol):
            sumIntensityInRow[rowIndex] += image[row, col]
        rowIndex += 1

    # Vzhladenie projekcie
    sumIntensityInRow = projectionSmoothing(sumIntensityInRow)

    # Maximalna hodnota znamena bod na flekcnej ryhe - jeho index sluzi potom pre najdenie suradnic
    index = np.argmax(sumIntensityInRow)

    # Najdenie suradnic bodu leziaceho na flekcnej ryhe
    pointRowOnPrincipleLine = firstSearchRow + index
    pointColOnPrincipleLine = (firstSearchCol + lastSearchCol) // 2

    return pointRowOnPrincipleLine, pointColOnPrincipleLine


def orientedProjectionLeft(image, blockImage, colCandidatesForProjection, radialSide):
    # Funkcia vrati riadok, v ktorom radialna longitudinalna ryha vyustuje z dlane

    _, _, blockHeight, blockWidth = blockImage.shape

    principalPointSeeds = []
    for ind in range(0, min(len(colCandidatesForProjection), 11)):
        colCandidate = colCandidatesForProjection[ind]

        # Vykonanie smerovej projekcie v danom stlpci
        pointOnPrincipleLine = orientedProjection(image, blockImage, colCandidate, radialSide[0], radialSide[1])
        principalPointSeeds.append(pointOnPrincipleLine)

    # Najdenie riadku, v ktorom radialna longitudinalna ryha vyustuje z dlane
    pointRow = 0
    found = False
    for point in principalPointSeeds:
        seeds = []
        inOneLine = 0
        pointRow = point[0]
        for i in range(len(principalPointSeeds)):
            if pointRow - 100 <= principalPointSeeds[i][0] <= pointRow + 200:
                inOneLine += 1
                seeds.append(principalPointSeeds[i])
        if inOneLine >= 5:
            found = True
            break

    principalLinePoints = []

    # Ked taky riadok nie je najdeny, bude riadkom, v ktorom radialna longitudinalna ryha vyustuje z dlane stred
    # medzi vrchnym bodom radialneho okraja dlane a stredom radialneho okraja dlane
    if found:
        principalLineRow = pointRow // blockHeight
        # principalLinePoints = fillPrincipleLine(seeds, blockImage, blocks, [])

    else:
        firstBlockRow = radialSide[0][0]
        lastBlockRow = radialSide[1][0]

        # Vypocet stredu medzi vrchnym bodom radialneho okraja dlane a stredom radialneho okraja dlane
        middleBlockRow = (firstBlockRow + lastBlockRow) // 2

        principalLineRow = (middleBlockRow + firstBlockRow) // 2

    return principalLineRow, principalLinePoints


def orientedProjectionRight(image, blockImage, colCandidatesForProjection, ulnarSide, principalLinePoints):
    # Funkcia vrati riadok, v ktorom distalna transverzalna ryha vyustuje z dlane

    _, _, blockHeight, blockWidth = blockImage.shape

    principalPointSeeds = []
    for ind in range(1, min(len(colCandidatesForProjection), 11)):
        colCandidate = colCandidatesForProjection[ind]

        # Vykonanie smerovej projekcie v danom stlpci
        pointOnPrincipleLine = orientedProjection(image, blockImage, colCandidate, ulnarSide[0], ulnarSide[1])
        principalPointSeeds.append(pointOnPrincipleLine)

    # Najdenie riadku, v ktorom distalna transverzalna ryha vyustuje z dlane
    pointRow = 0
    found = False
    for point in principalPointSeeds:
        seeds = []
        inOneLine = 0
        pointRow = point[0]
        for i in range(len(principalPointSeeds)):
            if pointRow - 200 <= principalPointSeeds[i][0] <= pointRow + 100:
                inOneLine += 1
                seeds.append(principalPointSeeds[i])
        if inOneLine >= 5:
            found = True
            break

    # Ked taky riadok nie je najdeny, bude riadkom, v ktorom distalna transverzalna ryha vyustuje z dlane stred
    # medzi vrchnym bodom ulnarneho okraja dlane a stredom ulnarneho okraja dlane
    if found:
        principalLineRow = (pointRow // blockHeight) + 1
        #principalLinePoints = fillPrincipleLine(seeds, blockImage, blocks, principalLinePoints)

    else:
        firstBlockRow = ulnarSide[0][0]
        lastBlockRow = ulnarSide[1][0]

        # Vypocet stredu medzi vrchnym bodom ulnarneho okraja dlane a stredom ulnarneho okraja dlane
        middleBlockRow = (firstBlockRow + lastBlockRow) // 2
        principalLineRow = (middleBlockRow + firstBlockRow) // 2

    return principalLineRow


def fillPrincipleLine(seed, blockImage, blocks, principalLinePoints):
    # Funkcia vyplni na zaklade vstupnych bodov flekcnu ryhu

    for row in range(blockImage.shape[0]):
        for col in range(blockImage.shape[1]):
            if blocks[row][col]["background"] == 2:
                blockImage[row][col] = 0

    image = mergeBlocksToImage(blockImage)
    whitePixelsMask = flood(image, seed[0], tolerance=30)

    toSearch = seed
    alreadySearched = []

    i = 0
    while toSearch:
        i += 1
        if i == 100000:
            break
        seedRow, seedCol = toSearch.pop(0)

        if not whitePixelsMask[seedRow, seedCol]:
            alreadySearched.append((seedRow, seedCol))
            continue

        if (seedRow, seedCol) in alreadySearched:
            continue

        regionSum = sum(sum(whitePixelsMask[seedRow - 2:seedRow + 3, seedCol - 2:seedCol + 3]))
        if regionSum > 23:
            #principalLinePoints[seedRow - 3:seedRow + 4, seedCol - 3:seedCol + 4] = True
            toSearch.append((seedRow - 3, seedCol))
            toSearch.append((seedRow - 3, seedCol + 3))
            toSearch.append((seedRow, seedCol + 3))
            toSearch.append((seedRow + 3, seedCol + 3))
            toSearch.append((seedRow + 3, seedCol))
            toSearch.append((seedRow + 3, seedCol - 3))
            toSearch.append((seedRow, seedCol - 3))
            toSearch.append((seedRow - 3, seedCol - 3))

        alreadySearched.append((seedRow, seedCol))
        principalLinePoints.append((seedRow, seedCol))

    return principalLinePoints


def principleLines(image, blocks, blockImage, ulnarSide, radialSide, palmprintBorder):
    # Funkcia detekuje flekcne ryhy.

    # Zacina sa detekciou radialnej longitudinalnej ryhy na lavej strane odtlacku
    left = True

    # Najdenie stlpcov, v ktorych moze prebehnut smerova projekcia
    colCandidatesForProjection = getColCandidateForProjection(ulnarSide, radialSide, blocks, left)

    # Smerova projekcia pre najdenie bodov patriacich do radialnej longitudinalnej ryhy
    principleLineRow, principalLinePoints = orientedProjectionLeft(image, blockImage, colCandidatesForProjection,
                                                                   radialSide)

    # Rozdelenie segmentu 13 na segmenty 13' a 13''
    startingPointIndex = palmprintBorder.index(radialSide[0])
    for distanceFromStartingPoint in range(1, len(palmprintBorder)):
        nextBorderPoint = palmprintBorder[startingPointIndex + distanceFromStartingPoint]
        nextBorderPointRow, nextBorderPointCol = nextBorderPoint
        if blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] != 13:
            break
        if nextBorderPointRow <= principleLineRow:
            blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = "13\'"
        if nextBorderPointRow > principleLineRow:
            blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = "13\'\'"

    # Pokracuje sa detekciou distalnej transverzalnej ryhy
    left = False

    # Najdenie stlpcov, v ktorych moze prebehnut smerova projekcia
    colCandidatesForProjection = getColCandidateForProjection(ulnarSide, radialSide, blocks, left)

    # Smerova projekcia pre najdenie bodov patriacich do distalnej transverzalnej ryhy
    principleLineRow = orientedProjectionRight(image, blockImage, colCandidatesForProjection, ulnarSide,
                                               principalLinePoints)

    # Rozdelenie segmentu 5 an segmenty 5' a 5''
    startingPointIndex = palmprintBorder.index(ulnarSide[0])
    for distanceFromStartingPoint in range(1, len(palmprintBorder)):
        nextBorderPoint = palmprintBorder[startingPointIndex - distanceFromStartingPoint]
        nextBorderPointRow, nextBorderPointCol = nextBorderPoint
        if blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] == 4:
            break
        if nextBorderPointRow < principleLineRow:
            blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = "5\'\'"
        if nextBorderPointRow >= principleLineRow:
            blocks[nextBorderPointRow][nextBorderPointCol]["palmprintSegment"] = "5\'"

    return blocks
