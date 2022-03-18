import cv2
import numpy as np
from skimage.morphology import flood


def savePalmprint(image):
    cv2.imwrite('newPalmprint.bmp', image)


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