import numpy as np
import cv2
import math
from collections import Counter
from timeit import default_timer as timer


def mergeBlocksToImage(blockImage):
    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    imageHeight = blockRows * blockHeight
    imageWidth = blockCols * blockWidth

    """ Spojenie blokov do celkoveho obrazu"""
    image = blockImage.reshape(imageHeight // blockHeight, imageWidth // blockWidth, blockHeight, blockWidth)
    image = image.swapaxes(1, 2).reshape(imageHeight, imageWidth)

    return image


def showPalmprint(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def savePalmprint(image, directoryName):
    cv2.imwrite(f'{directoryName}/orientation.bmp', image)


def saveOrientationsImage(image, blockImage, blocks, angles, directoryName):
    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape

    """for row in range(blockRows):
        for col in range(blockCols):
            if blocks[row][col]["orientationConfidence"] != 100:
                blockImage[row][col] = 0

    image = mergeBlocksToImage(blockImage)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if i % 50 == 0 or j % 50 == 0:
                image[i][j] = 255
            if i % 498 == 0 or i % 499 == 0 or i % 500 == 0 or i % 501 == 0 or i % 502 == 0 or j % 498 == 0 or j % 499 == 0 or j % 500 == 0 or j % 501 == 0 or j % 502 == 0:
                image[i][j] = 255"""

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

    savePalmprint(orientationImage, directoryName)


def splitIntoParts(value, numberOfParts):
    return np.linspace(0, value, numberOfParts + 1)[1:]


def findStartEndCols(row, currentRow, currentCol, blockWidth, angle):
    """ Sluzi pre zrychlenie Radonovej transformacie - najde zaciatocny a koncovy stlpec, v ktorom sa bude hladat"""
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
    sinOfAngles = []

    for angle in angles:
        sinOfAngles.append(math.sin(angle))

    return sinOfAngles


def calculateCosOfAngles(angles):
    cosOfAngles = []

    for angle in angles:
        cosOfAngles.append(math.cos(angle))

    return cosOfAngles


def getOrientation(rs):
    minValue = min(rs)
    orientation = rs.index(minValue)
    confidence = -minValue

    return orientation, confidence


def findRadonTransform(row, col, block, angles, pixelsForRadonTransform):
    rs = [0] * 12

    """ Vytvori masku pixelov, ktore su vo vnutri kruhoveho regionu (True ked dnu, False ked von)"""
    X, Y = np.ogrid[:block.shape[0], :block.shape[1]]
    distance = np.sqrt((X - row) ** 2 + (Y - col) ** 2)
    circleRegionMask = distance <= block.shape[0] // 2

    """ Vypocita priemer pixelov v kruhovom regione """
    mean = np.mean(block[circleRegionMask])

    """ Pre dany bod prechadza vsetky uhly """
    for angleNumber in range(len(angles)):
        total = 0
        """ Ziska body pre ktore sa bude pocitat Radonova transfomracia """
        pointsToSearch = pixelsForRadonTransform[row][col][angleNumber]

        """ Postupne prechadza ziskane body a Radonova transformacia sa rovna intenzite pixelu v danom bode, od ktoreho
            je odcitany priemer intenzit pixelov v kruhovom regione """
        for point in pointsToSearch:
            x = point[0]
            y = point[1]
            pixelIntensity = block[x][y]
            total += (pixelIntensity - mean)

        rs[angleNumber] += total

    """ Ziska sa orientacia a dovera v danu orientaciu """
    orientation, confidence = getOrientation(rs)
    # print(f"Celkovo: {-confidence} - {orientation}")

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

    print("Im here")

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

    print("Im finished")

    return pixelsForRadonTransform


def orientationField(blockImage, blocks, image, directoryName):
    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    angles = splitIntoParts(math.pi, 12)

    start1 = timer()
    pixelsForRadonTransform = findPixelsForRadonTransform(blockHeight, blockWidth, angles)
    end1 = timer()
    print(f"Najdenie pixelov pre RadonTransform: {end1 - start1}")

    blocksAmount = 50000
    out = False

    for blockRow in range(blockRows):
        for blockCol in range(blockCols):
            if blocksAmount == 0:
                out = True
                break

            if blocks[blockRow][blockCol]["background"] == 0:
                pixelOrientations = []
                block = blockImage[blockRow][blockCol]
                mean = np.mean(block)
                print(f"Mean: {mean}")

                #start2 = timer()
                i = 0
                """ Spracovane budu iba cierne pixely """
                for row in range(blockHeight):
                    for col in range(blockWidth):
                        i += 1
                        value = block[row][col]
                        if value < mean - 10 and i % 3 == 0:

                            r = findRadonTransform(row, col, block, angles, pixelsForRadonTransform)
                            pixelOrientations.append(r)
                #end2 = timer()
                #print(f"Radon Transform All: {end2 - start2}")

                print(Counter(pixelOrientations))
                #start3 = timer()
                confidence = 0
                try:
                    finalOrientation = Counter(pixelOrientations).most_common()[0][0]
                    firstOrientation = Counter(pixelOrientations).most_common()[0][0]
                    firstOrientationNumber = Counter(pixelOrientations).most_common()[0][1]
                    secondOrientation = Counter(pixelOrientations).most_common()[1][0]
                    secondOrientationNumber = Counter(pixelOrientations).most_common()[1][1]
                    if abs(firstOrientation - secondOrientation) == 1:
                        #finalOrientation = firstOrientation
                        if firstOrientationNumber + secondOrientationNumber >= 150:
                            confidence = 100
                        else:
                            confidence = 50
                    else:
                        if firstOrientationNumber >= 2 * secondOrientationNumber and firstOrientationNumber >= 100:
                            confidence = 100
                            #finalOrientation = firstOrientation
                        else:
                            confidence = ((firstOrientationNumber / secondOrientationNumber) - 1) * 100
                except IndexError:
                    finalOrientation = -1

                #end3 = timer()
                #print(f"Counter: {end3 - start3}")

                #showPalmprint(block)

                blocks[blockRow][blockCol]["orientation"] = finalOrientation
                blocks[blockRow][blockCol]["orientationConfidence"] = confidence
                print(f"{blockRow}:{blockCol} --> {finalOrientation}")
                blocksAmount -= 1
        if out:
            break

    print("end")

    saveOrientationsImage(image, blockImage, blocks, angles, directoryName)

    return blocks, angles
