#! /usr/bin/env python

"""

Funkcie pre manipulaciu s obrazom - nacitanie obrazu, rozelenie obrazu na bloky, spojenie blokov do obrazu.

Autor: Bc. Jan Folenta
Email: xfolen00@stud.fit.vutbr.cz
Rok: 2022

"""

import cv2


def loadPalmprint(fileName):
    # Nacitanie vstupneho suboru

    image = cv2.imread(fileName, 0)
    return image


def initializeBlocks(blockRows, blockColumns):
    # Funkcia vytvori strukturu 'blocks', ktora uchovava informacie o jednotlivych blokoch

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
    # Funkcia rozdeli vstupny obraz na bloky o danej velkosti

    imageHeight, imageWidth = image.shape

    # blockImage[pocet blokov (riadky), pocet blokov(stlpce), vyska bloku, sirka bloku]
    blockImage = image.reshape(imageHeight // blockHeight, blockHeight, imageWidth // blockWidth, blockWidth)
    blockImage = blockImage.swapaxes(1, 2)

    blockRows, blockCols, _, _ = blockImage.shape

    blocks = initializeBlocks(blockRows, blockCols)

    return blockImage, blocks


def mergeBlocksToImage(blockImage):
    # Funkcia spoji obraz rozdeleny na bloky do vystupneho obrazu

    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    imageHeight = blockRows * blockHeight
    imageWidth = blockCols * blockWidth

    image = blockImage.reshape(imageHeight // blockHeight, imageWidth // blockWidth, blockHeight, blockWidth)
    image = image.swapaxes(1, 2).reshape(imageHeight, imageWidth)

    return image
