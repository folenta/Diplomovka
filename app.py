#! /usr/bin/env python

"""

Implementacia zakladnej logiky aplikacie pre ziskavanie podrobnych informacii z odltackov dlane vratane grafickeho
uzivatelskeho rozhrania.

Autor: Bc. Jan Folenta
Email: xfolen00@stud.fit.vutbr.cz
Rok: 2022

"""

import math
import tkinter
import os
import threading
from xml.dom import minidom
import xml.etree.ElementTree as ET

import cv2
import copy
import tkinter.filedialog
from tkinter import *
from PIL import Image, ImageTk

from imageProcessing import mergeBlocksToImage, splitImageToBlocks
from orientation import orientationField, orientationSmoothing
from segmentation import segmentation, findExtremePalmprintPoints
from principalLines import principleLines
from triradiusDetection import findTriradius
from ridgeWidthFrequency import ridgeWidthAndFrequency
from mainLineTracking import findMainLines


def evaluateMainLineEndings(mainLineEndings, palmprintName, textColors):
    # Funkcia vyhodnoti spravnost najdenych ukonceni hlavnych linii

    directory = 'anotatedData'
    foundData = False

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            tmpCreated = False
            leftPalmName = ""
            rightPalmName = ""
            try:
                anotatedTree = ET.parse(f)
                anotatedRoot = anotatedTree.getroot()
            except:
                tmpFile = open(f)
                lines = tmpFile.readlines()
                tmpFile.close()
                lines.pop(-2)
                updatedFile = "".join(lines)

                save_path_file = "tmp.xml"

                with open(save_path_file, "w") as f:
                    f.write(updatedFile)

                tmpCreated = True

                anotatedTree = ET.parse('tmp.xml')
                anotatedRoot = anotatedTree.getroot()

            i = 0
            images = anotatedRoot.find('images')
            for image in images.iter('image'):

                if i == 10:
                    leftPalmName = image.get('src').split('.')[0]
                if i == 11:
                    rightPalmName = image.get('src').split('.')[0]

                i += 1

            anotatedPalmprintName = leftPalmName
            if anotatedPalmprintName == palmprintName:
                foundData = True
                palmprintData = anotatedRoot.findall('dermatoglyph')[10]
                i = 0
                mainLinesAnotated = []
                for mainLine in palmprintData.iter('setting'):
                    if i < 5:
                        mainLinesAnotated.append(mainLine.get('second'))
                    i += 1

            anotatedPalmprintName = rightPalmName
            if anotatedPalmprintName == palmprintName:
                foundData = True
                palmprintData = anotatedRoot.findall('dermatoglyph')[11]
                i = 0
                mainLinesAnotated = []
                for mainLine in palmprintData.iter('setting'):
                    if i < 5:
                        mainLinesAnotated.append(mainLine.get('second'))
                    i += 1

            if foundData:
                mainLinesAnotated[0], mainLinesAnotated[1], mainLinesAnotated[2], mainLinesAnotated[3] = \
                    mainLinesAnotated[3],  mainLinesAnotated[2], mainLinesAnotated[1], mainLinesAnotated[0]

                for i in range(len(mainLineEndings)):
                    if str(mainLinesAnotated[i]) == '4' and (str(mainLineEndings[i]) == '3' or str(mainLineEndings[i]) == '5\''):
                        textColors[i] = "green"

                    elif str(mainLinesAnotated[i]) == str(mainLineEndings[i]):
                        textColors[i] = "green"

                    else:
                        textColors[i] = "red"

            if tmpCreated:
                os.remove('tmp.xml')

            if foundData:
                break

    return textColors


def findMainLineIndex(mainLinesEndings):
    # Funckia urci zo ziskanych linii index hlavnych linii

    mainLineA = mainLinesEndings[0]
    mainLineD = mainLinesEndings[3]

    if mainLineA == '5\'':
        mainLineA = 5
    elif mainLineA == '5\'\'':
        mainLineA = 6
    elif mainLineA == '13\'' or mainLineA == '13\'\'':
        mainLineA = 8
    elif mainLineA >= 7:
        mainLineA = mainLineA - 5

    if mainLineD == '5\'':
        mainLineD = 5
    elif mainLineD == '5\'\'':
        mainLineD = 6
    elif mainLineD == '13\'' or mainLineD == '13\'\'':
        mainLineD = 8
    elif mainLineD >= 7:
        mainLineD = mainLineD - 5

    mainLineIndex = mainLineA + mainLineD

    return mainLineIndex


def showSegmentedPalmprint(blockImage, blocks):
    # Funckia vytvori segmentovany odtlacok dlane

    blockRows, blockCols, _, _ = blockImage.shape

    for row in range(blockRows):
        for col in range(blockCols):
            if blocks[row][col]["background"] == 2:
                blockImage[row][col] = 0

    firstRow, lastRow, firstCol, lastCol = findExtremePalmprintPoints(blocks, blockImage)

    segmentedImage = mergeBlocksToImage(blockImage[firstRow - 1:lastRow + 1, firstCol - 1:lastCol + 1])
    imageHeight, imageWidth = segmentedImage.shape[:2]
    segmentedImage = cv2.resize(segmentedImage, (int((700/imageHeight)*imageWidth), 700), interpolation=cv2.INTER_AREA)

    return segmentedImage


def showOrientationPalmprint(image, blockImage, blocks, angles):
    # Funkcia vytvori odtlacok dlane s najdenou orientaciou papilarnych linii

    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape

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

                orientationImage = cv2.line(orientationImage, (x2, y), (x, y2), (255, 0, 0), 2)

    firstRow, lastRow, firstCol, lastCol = findExtremePalmprintPoints(blocks, blockImage)

    orientationImage = orientationImage[(firstRow*blockHeight)-blockHeight:(lastRow*blockHeight)+blockHeight,
                                        (firstCol*blockWidth)-blockWidth:(lastCol*blockWidth)+blockWidth]
    imageHeight, imageWidth = orientationImage.shape[:2]
    orientationImage = cv2.resize(orientationImage, (int((700 / imageHeight) * imageWidth), 700),
                                interpolation=cv2.INTER_AREA)

    return orientationImage


def showTriradiusPalmprint(triradius, blockImage):
    # Funkcia vyznaci v odtlacku detekovane triradia

    global blocks
    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape

    image = mergeBlocksToImage(blockImage)
    triradiusImage = copy.deepcopy(image)

    triradiusA = triradius[0]
    triradiusB = triradius[1]
    triradiusC = triradius[2]
    triradiusD = triradius[3]
    triradiusT = triradius[4]

    triradiusImage = cv2.circle(triradiusImage, (triradiusA[1], triradiusA[0]), 10, (0, 0, 255), 3)
    triradiusImage = cv2.circle(triradiusImage, (triradiusB[1], triradiusB[0]), 10, (0, 0, 255), 3)
    triradiusImage = cv2.circle(triradiusImage, (triradiusC[1], triradiusC[0]), 10, (0, 0, 255), 3)
    triradiusImage = cv2.circle(triradiusImage, (triradiusD[1], triradiusD[0]), 10, (0, 0, 255), 3)
    triradiusImage = cv2.circle(triradiusImage, (triradiusT[1], triradiusT[0]), 10, (0, 0, 255), 3)

    firstRow, lastRow, firstCol, lastCol = findExtremePalmprintPoints(blocks, blockImage)

    triradiusImage = triradiusImage[(firstRow * blockHeight) - blockHeight:(lastRow * blockHeight) + blockHeight,
                       (firstCol * blockWidth) - blockWidth:(lastCol * blockWidth) + blockWidth]
    imageHeight, imageWidth = triradiusImage.shape[:2]
    triradiusImage = cv2.resize(triradiusImage, (int((700 / imageHeight) * imageWidth), 700),
                                  interpolation=cv2.INTER_AREA)

    return triradiusImage


def showMainLinePalmprint(triradius, mainLines, blockImage):
    # Funkcia zobrazi v odtlacku priebeh hlavnych linii

    global blocks
    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape

    image = mergeBlocksToImage(blockImage)
    mainLineImage = copy.deepcopy(image)

    triradiusA = triradius[0]
    triradiusB = triradius[1]
    triradiusC = triradius[2]
    triradiusD = triradius[3]
    triradiusT = triradius[4]

    mainLineImage = cv2.circle(mainLineImage, (triradiusA[1], triradiusA[0]), 10, (0, 0, 255), 3)
    mainLineImage = cv2.circle(mainLineImage, (triradiusB[1], triradiusB[0]), 10, (0, 0, 255), 3)
    mainLineImage = cv2.circle(mainLineImage, (triradiusC[1], triradiusC[0]), 10, (0, 0, 255), 3)
    mainLineImage = cv2.circle(mainLineImage, (triradiusD[1], triradiusD[0]), 10, (0, 0, 255), 3)
    mainLineImage = cv2.circle(mainLineImage, (triradiusT[1], triradiusT[0]), 10, (0, 0, 255), 3)

    for mainLine in mainLines:
        for mainLinePoint in range(len(mainLine) - 1):
            mainLineImage = cv2.line(mainLineImage, mainLine[mainLinePoint], mainLine[mainLinePoint + 1], (0, 0, 255), 3)

    firstRow, lastRow, firstCol, lastCol = findExtremePalmprintPoints(blocks, blockImage)

    mainLineImage = mainLineImage[(firstRow * blockHeight) - blockHeight:(lastRow * blockHeight) + blockHeight,
                       (firstCol * blockWidth) - blockWidth:(lastCol * blockWidth) + blockWidth]
    imageHeight, imageWidth = mainLineImage.shape[:2]
    mainLineImage = cv2.resize(mainLineImage, (int((700 / imageHeight) * imageWidth), 700),
                                  interpolation=cv2.INTER_AREA)

    return mainLineImage


def showWidthFrequencyPalmprint(blocks, blockImage):
    # Funkcia zobrazi sirku a frekvenciu papilarnych linii v odtlacku dlane

    blockRows, blockCols, blockHeight, blockWidth = blockImage.shape
    minWidth = 100
    maxWidth = -1

    for row in range(blockRows):
        for col in range(blockCols):
            if blocks[row][col]["ridgeWidth"] != 0:
                if blocks[row][col]["ridgeWidth"] < minWidth:
                    minWidth = blocks[row][col]["ridgeWidth"]
                if blocks[row][col]["ridgeWidth"] > maxWidth:
                    maxWidth = blocks[row][col]["ridgeWidth"]

    widthRange = maxWidth - minWidth

    widthAndFrequencyBlocks = copy.deepcopy(blockImage)

    for row in range(blockRows):
        for col in range(blockCols):
            if blocks[row][col]["ridgeWidth"] != 0:
                diffFromMin = blocks[row][col]["ridgeWidth"] - minWidth
                widthAndFrequencyBlocks[row][col] = int(75 + (diffFromMin/widthRange * 100))

    widthImage = mergeBlocksToImage(widthAndFrequencyBlocks)

    firstRow, lastRow, firstCol, lastCol = findExtremePalmprintPoints(blocks, blockImage)

    widthImage = widthImage[(firstRow*blockHeight)-blockHeight:(lastRow*blockHeight)+blockHeight,
                                        (firstCol*blockWidth)-blockWidth:(lastCol*blockWidth)+blockWidth]
    imageHeight, imageWidth = widthImage.shape[:2]
    widthImage = cv2.resize(widthImage, (int((700 / imageHeight) * imageWidth), 700),
                                interpolation=cv2.INTER_AREA)

    return widthImage
    #savePalmprint(orientationImage, directoryName)


def showOriginal():
    # Funkcia zobrazi originalny obraz odtlacku

    global palmprint, originalPalmprint, canvas

    palmprint = originalPalmprint
    canvas.create_image(700, 0, anchor=NE, image=palmprint)


def showSegmented():
    # Funkcia zobrazi originalny obraz odtlacku

    global palmprint, segmentedPalmprint, canvas

    palmprint = ImageTk.PhotoImage(Image.fromarray(segmentedPalmprint))
    canvas.create_image(700, 0, anchor=NE, image=palmprint)


def showOrientations():
    # Funkcia zobrazi odtlacok s vyznacenymi orientaciami papilarnych linii

    global palmprint, orientationPalmprint, canvas

    palmprint = ImageTk.PhotoImage(Image.fromarray(orientationPalmprint))
    canvas.create_image(700, 0, anchor=NE, image=palmprint)


def showTriradius():
    # Funkcia zobrazi odtlacok s vyznacenymi triradiami

    global palmprint, triradiusPalmprint, canvas

    palmprint = ImageTk.PhotoImage(Image.fromarray(triradiusPalmprint))
    canvas.create_image(700, 0, anchor=NE, image=palmprint)


def showMainLines():
    # Funkcia zobrazi odtlacok s vyznacenym priebehom hlavnych linii

    global palmprint, mainLinesPalmprint, canvas

    palmprint = ImageTk.PhotoImage(Image.fromarray(mainLinesPalmprint))
    canvas.create_image(700, 0, anchor=NE, image=palmprint)


def showWidthAndFrequency():
    # Funkcia zobrazi odtlacok s vyznacenou sirkou a frekvenciou papilarnych linii

    global palmprint, ridgeWidthFrequencyPalmprint, canvas

    palmprint = ImageTk.PhotoImage(Image.fromarray(ridgeWidthFrequencyPalmprint))
    canvas.create_image(700, 0, anchor=NE, image=palmprint)


def runSegmentation(grayscalePalmprint, baseName, blockImage, dpi):
    # Funckia spusti segmentaciu odtlacku dlane

    global segmentedButton, segmentedPalmprint, blocks, animation

    loadingAnimation(count, "segmentation")

    success, grayscalePalmprint, blockImage, blocks, betweenFingersAreas, ulnarSide, radialSide, \
    palmprintBorder, edgePointsOfSegments, contour = segmentation(blockImage, blocks, grayscalePalmprint)

    if success:
        grayscalePalmprint = mergeBlocksToImage(blockImage)

        blocks = principleLines(grayscalePalmprint, blocks, blockImage, ulnarSide, radialSide, palmprintBorder)

        segmentedPalmprint = showSegmentedPalmprint(blockImage, blocks)
        segmentedButton["state"] = "normal"
        root.after_cancel(animation)
        segmentedButtonState.configure(image=successImage)
        segmentedButtonState.image = successImage

        threading.Thread(target=runOrientations, args=(blockImage, grayscalePalmprint, baseName, betweenFingersAreas,
                                                       edgePointsOfSegments, contour, dpi, )).start()
    else:
        root.after_cancel(animation)
        segmentedButtonState.configure(image=errorImage)
        segmentedButtonState.image = errorImage
        orientationButtonState.configure(image=errorImage)
        orientationButtonState.image = errorImage
        ridgeWidthFrequencyButtonState.configure(image=errorImage)
        ridgeWidthFrequencyButtonState.image = errorImage
        triradiusButtonState.configure(image=errorImage)
        triradiusButtonState.image = errorImage
        mainLinesButtonState.configure(image=errorImage)
        mainLinesButtonState.image = errorImage
        root.after_cancel(animation)


def runOrientations(blockImage, grayscalePalmprint, baseName, betweenFingersAreas, edgePointsOfSegments, contour, dpi):
    # Funkcia spusti urcovanie lokalnej orientacie papilarnych linii

    global orientationButton, orientationPalmprint, blocks

    root.after_cancel(animation)
    segmentedButtonState.configure(image=successImage)
    segmentedButtonState.image = successImage

    loadingAnimation(count, "orientations")

    blocks, angles = orientationField(blockImage, blocks)
    orientationPalmprint = showOrientationPalmprint(grayscalePalmprint, blockImage, blocks, angles)
    orientationButton["state"] = "normal"
    root.after_cancel(animation)
    orientationButtonState.configure(image=successImage)
    orientationButtonState.image = successImage

    threading.Thread(target=runRidgeWidthFrequency, args=(blockImage, baseName, angles, grayscalePalmprint,
                                                         betweenFingersAreas, edgePointsOfSegments, contour, dpi, )).start()


def runRidgeWidthFrequency(blockImage, baseName, angles, grayscalePalmprint, betweenFingersAreas, edgePointsOfSegments,
                           contour, dpi):
    # Funkcia spusti urcovanie sirky a frekvencie papilarnych linii

    global ridgeWidthFrequencyButton, ridgeWidthFrequencyPalmprint, blocks

    root.after_cancel(animation)
    orientationButtonState.configure(image=successImage)
    orientationButtonState.image = successImage

    loadingAnimation(count, "widthFrequency")

    blocks, frequency, width = ridgeWidthAndFrequency(blocks, blockImage, angles, baseName, dpi)
    ridgeWidthFrequencyPalmprint = showWidthFrequencyPalmprint(blocks, blockImage)
    ridgeWidthFrequencyButton["state"] = "normal"
    root.after_cancel(animation)
    ridgeWidthFrequencyButtonState.configure(image=successImage)
    ridgeWidthFrequencyButtonState.image = successImage

    widthText.configure(text=f"Ridge width: {width}")
    frequencyText.configure(text=f"Ridge frequency: {frequency}")

    threading.Thread(target=runTriradiusDetection, args=(blockImage, baseName, angles, grayscalePalmprint,
                                                         betweenFingersAreas, edgePointsOfSegments, contour,)).start()


def runTriradiusDetection(blockImage, baseName, angles, grayscalePalmprint, betweenFingersAreas, edgePointsOfSegments,
                          contour):
    # Funkcia spusti detekciu triradii

    global triradiusButton, triradiusPalmprint, orientationPalmprint, blocks, triradius

    root.after_cancel(animation)
    ridgeWidthFrequencyButtonState.configure(image=successImage)
    ridgeWidthFrequencyButtonState.image = successImage

    loadingAnimation(count, "triradius")

    blocks, triradius = findTriradius(blocks, blockImage, angles, betweenFingersAreas)
    blocks = orientationSmoothing(blocks, blockImage, angles)
    triradiusPalmprint = showTriradiusPalmprint(triradius, blockImage)
    orientationPalmprint = showOrientationPalmprint(grayscalePalmprint, blockImage, blocks, angles)
    triradiusButton["state"] = "normal"
    root.after_cancel(animation)
    triradiusButtonState.configure(image=successImage)
    triradiusButtonState.image = successImage

    threading.Thread(target=runMainLinesTracking, args=(blockImage, baseName, angles,  edgePointsOfSegments, contour,
                                                        triradius, )).start()


def runMainLinesTracking(blockImage, baseName, angles, edgePointsOfSegments, contour, triradius):
    # Funkcia spusti sledovanie priebehu hlavnych linii

    global mainLinesButton, exportButton, mainLinesPalmprint, blocks, mainLinesEndings

    root.after_cancel(animation)
    triradiusButtonState.configure(image=successImage)
    triradiusButtonState.image = successImage

    loadingAnimation(count, "mainLines")

    mainLines, mainLinesEndings = findMainLines(blocks, blockImage, angles, triradius, edgePointsOfSegments, contour)
    mainLinesPalmprint = showMainLinePalmprint(triradius, mainLines, blockImage)
    mainLinesButton["state"] = "normal"
    exportButton["state"] = "normal"
    root.after_cancel(animation)
    mainLinesButtonState.configure(image=successImage)
    mainLinesButtonState.image = successImage

    textColors = ["black"] * 5

    textColors = evaluateMainLineEndings(mainLinesEndings, baseName, textColors)

    mainLineAText.configure(text=f"Main line A: {mainLinesEndings[0]}", fg=textColors[0])
    mainLineBText.configure(text=f"Main line B: {mainLinesEndings[1]}", fg=textColors[1])
    mainLineCText.configure(text=f"Main line C: {mainLinesEndings[2]}", fg=textColors[2])
    mainLineDText.configure(text=f"Main line D: {mainLinesEndings[3]}", fg=textColors[3])
    mainLineTText.configure(text=f"Main line T: {mainLinesEndings[4]}", fg=textColors[4])

    mainLineIndex = findMainLineIndex(mainLinesEndings)

    mliColor = "black"
    if textColors[0] == "green" and textColors[3] == "green":
        mliColor = "green"
    elif textColors[0] == "red" or textColors[3] == "red":
        mliColor = "red"

    mainLineIndexText.configure(text=f"Main line index: {mainLineIndex}", fg=mliColor)

    root.after_cancel(animation)
    mainLinesButtonState.configure(image=successImage)
    mainLinesButtonState.image = successImage


def export():
    # Funkcia poskytne export ziskanych dat

    global blocks, triradius, mainLinesEndings, fileName

    doc = minidom.Document()

    baseName = os.path.basename(fileName)
    baseName = baseName.split('\\')[-1]

    palmprint = doc.createElement('palmprint')
    doc.appendChild(palmprint)
    palmprint.setAttribute('src', baseName)
    palmprint.setAttribute('width', str(imageWidth))
    palmprint.setAttribute('height', str(imageHeight))
    palmprint.setAttribute('xdpi', str(xdpi))
    palmprint.setAttribute('ydpi', str(ydpi))
    palmprint.setAttribute('blockWidth', str(blockWidth))
    palmprint.setAttribute('blockHeight', str(blockHeight))

    frequencyTotal = 0
    frequencyCount = 0
    widthTotal = 0
    widthCount = 0
    for row in blocks:
        for col in blocks[row]:
            if blocks[row][col]["frequency"] != 0:
                frequencyTotal += blocks[row][col]["frequency"]
                frequencyCount += 1
            if blocks[row][col]["ridgeWidth"] != 0:
                widthTotal += blocks[row][col]["ridgeWidth"]
                widthCount += 1

            background = blocks[row][col]["background"]
            if background == 2:
                background = 1

            block = doc.createElement('block')
            block.setAttribute('row', str(row))
            block.setAttribute('col', str(col))
            block.setAttribute('background', str(background))

            if background == 0:
                orientation = doc.createElement('orientation')
                orientation.appendChild(doc.createTextNode(str(15 * blocks[row][col]["orientation"] + 15)))
                block.appendChild(orientation)

                ridgeWidth = doc.createElement('ridgeWidth')
                ridgeWidth.appendChild(doc.createTextNode(str(blocks[row][col]["ridgeWidth"])))
                block.appendChild(ridgeWidth)

                ridgeFrequency = doc.createElement('ridgeFrequency')
                ridgeFrequency.appendChild(doc.createTextNode(str(blocks[row][col]["frequency"])))
                block.appendChild(ridgeFrequency)

            palmprint.appendChild(block)

    palmprint.appendChild(doc.createTextNode(' '))

    width = doc.createElement('averageWidth')
    width.appendChild(doc.createTextNode(str(widthTotal / widthCount)))
    palmprint.appendChild(width)

    frequency = doc.createElement('averageFrequency')
    frequency.appendChild(doc.createTextNode(str(frequencyTotal / frequencyCount)))
    palmprint.appendChild(frequency)

    palmprint.appendChild(doc.createTextNode(' '))

    triradiusA = doc.createElement('triradiusA')
    triradiusA.setAttribute('x', str(triradius[0][0]))
    triradiusA.setAttribute('y', str(triradius[0][1]))
    palmprint.appendChild(triradiusA)

    triradiusB = doc.createElement('triradiusB')
    triradiusB.setAttribute('x', str(triradius[1][0]))
    triradiusB.setAttribute('y', str(triradius[1][1]))
    palmprint.appendChild(triradiusB)

    triradiusC= doc.createElement('triradiusC')
    triradiusC.setAttribute('x', str(triradius[2][0]))
    triradiusC.setAttribute('y', str(triradius[2][1]))
    palmprint.appendChild(triradiusC)

    triradiusD = doc.createElement('triradiusD')
    triradiusD.setAttribute('x', str(triradius[3][0]))
    triradiusD.setAttribute('y', str(triradius[3][1]))
    palmprint.appendChild(triradiusD)

    triradiusT = doc.createElement('triradiusT')
    triradiusT.setAttribute('x', str(triradius[4][0]))
    triradiusT.setAttribute('y', str(triradius[4][1]))
    palmprint.appendChild(triradiusT)

    palmprint.appendChild(doc.createTextNode(' '))

    mainLineA = doc.createElement('mainLineA')
    mainLineA.appendChild(doc.createTextNode(str(mainLinesEndings[0])))
    palmprint.appendChild(mainLineA)

    mainLineB = doc.createElement('mainLineB')
    mainLineB.appendChild(doc.createTextNode(str(mainLinesEndings[1])))
    palmprint.appendChild(mainLineB)

    mainLineC = doc.createElement('mainLineC')
    mainLineC.appendChild(doc.createTextNode(str(mainLinesEndings[2])))
    palmprint.appendChild(mainLineC)

    mainLineD = doc.createElement('mainLineD')
    mainLineD.appendChild(doc.createTextNode(str(mainLinesEndings[3])))
    palmprint.appendChild(mainLineD)

    mainLineT = doc.createElement('mainLineT')
    mainLineT.appendChild(doc.createTextNode(str(mainLinesEndings[4])))
    palmprint.appendChild(mainLineT)

    palmprint.appendChild(doc.createTextNode(' '))

    mainLineIndex = findMainLineIndex(mainLinesEndings)

    mli = doc.createElement('mainLineIndex')
    mli.appendChild(doc.createTextNode(str(mainLineIndex)))
    palmprint.appendChild(mli)

    xml_str = doc.toprettyxml(indent='   ')

    baseName = os.path.basename(fileName)
    baseName = baseName.split('\\')[-1]
    baseName = baseName.split('.')[0]

    savePathFile = baseName + "_out.xml"
    savePathFile = os.path.join("out", savePathFile)

    with open(savePathFile, "w") as f:
        f.write(xml_str)


def getImage():
    # Funkcia nacita vstupny odtlacok dlane

    global fileName, orientationButton, canvas, palmprint, originalPalmprint, blocks, xdpi, ydpi, blockHeight, \
        blockWidth, imageHeight, imageWidth

    blockHeight = 50
    blockWidth = 50

    fileName = tkinter.filedialog.askopenfilename(parent=root, title="Choose a file",
                                                  filetypes=[("Image Files", "*.tif *.png *.jpg *.jpeg *.bmp")])

    baseName = os.path.basename(fileName)
    baseName = baseName.split('\\')[-1]
    baseName = baseName.split('.')[0]

    imageDPI = Image.open(fileName)
    xdpi, ydpi = imageDPI.info['dpi']
    xdpi = round(xdpi)
    ydpi = round(ydpi)

    image = cv2.cvtColor(cv2.imread(fileName), cv2.COLOR_BGR2RGB)

    imageHeight = image.shape[0]
    imageWidth = image.shape[1]

    if image.shape[0] % blockHeight != 0:
        image = image[:-(image.shape[0] % blockHeight)]
    if image.shape[1] % blockWidth != 0:
        image = image[:, :-(image.shape[1] % blockWidth)]

    palmprint = ImageTk.PhotoImage(Image.fromarray(cv2.resize(image, (500, 700), interpolation=cv2.INTER_AREA)))
    canvas.create_image(700, 0, anchor=NE, image=palmprint)

    text.delete("1.0", tkinter.END)
    text.insert(INSERT, fileName)

    originalButton["state"] = "normal"
    originalButtonState.configure(image=successImage)
    originalButtonState.image = successImage

    segmentedButton["state"] = "disabled"
    segmentedButtonState.configure(image="")
    segmentedButtonState.image = ""
    orientationButton["state"] = "disabled"
    orientationButtonState.configure(image="")
    orientationButtonState.image = ""
    ridgeWidthFrequencyButton["state"] = "disabled"
    ridgeWidthFrequencyButtonState.configure(image="")
    ridgeWidthFrequencyButtonState.image = ""
    triradiusButton["state"] = "disabled"
    triradiusButtonState.configure(image="")
    triradiusButtonState.image = ""
    mainLinesButton["state"] = "disabled"
    mainLinesButtonState.configure(image="")
    mainLinesButtonState.image = ""
    exportButton["state"] = "disabled"
    widthText.configure(text="Ridge width:")
    frequencyText.configure(text="Ridge frequency:")
    mainLineAText.configure(text="Main line A:", fg="black")
    mainLineBText.configure(text="Main line B:", fg="black")
    mainLineCText.configure(text="Main line C:", fg="black")
    mainLineDText.configure(text="Main line D:", fg="black")
    mainLineTText.configure(text="Main line T:", fg="black")
    mainLineIndexText.configure(text="Main line index:", fg="black")

    originalPalmprint = ImageTk.PhotoImage(Image.fromarray(cv2.resize(image, (500, 700), interpolation=cv2.INTER_AREA)))

    grayscalePalmprint = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blockImage, blocks = splitImageToBlocks(grayscalePalmprint, blockHeight, blockWidth)

    threading.Thread(target=runSegmentation, args=(grayscalePalmprint, baseName, blockImage, xdpi, )).start()


def loadingAnimation(count, currentPart):
    # Implementacie animacie loading

    global animation
    loadingImage = loading[count]
    if currentPart == "segmentation":
        segmentedButtonState.configure(image=loadingImage)

    if currentPart == "orientations":
        orientationButtonState.configure(image=loadingImage)

    if currentPart == "widthFrequency":
        ridgeWidthFrequencyButtonState.configure(image=loadingImage)

    if currentPart == "triradius":
        triradiusButtonState.configure(image=loadingImage)

    if currentPart == "mainLines":
        mainLinesButtonState.configure(image=loadingImage)

    count += 1
    if count == gifFrames:
        count = 0

    animation = root.after(10, lambda: loadingAnimation(count, currentPart))


fileName = ""
animation = ""
palmprint, blocks, triradius, mainLineEndings = None, None, None, None
mainLinesEndings = []
originalPalmprint, segmentedPalmprint, orientationPalmprint, ridgeWidthFrequencyPalmprint, triradiusPalmprint, \
mainLinesPalmprint = None, None, None, None, None, None
count = 0
xdpi, ydpi = 0, 0
imageHeight, imageWidth, blockHeight, blockWidth = 0, 0, 0, 0

root = Tk()
root.title('PalmprintInformationExtractor')
p1 = PhotoImage(file='images/logo.png')

root.iconphoto(False, p1)

loadingGif = Image.open('images/loadingGif.gif')
gifFrames = loadingGif.n_frames

loading = [PhotoImage(file='images/loadingGif.gif', format=f'gif -index {i}') for i in range(gifFrames)]

successImage = ImageTk.PhotoImage(Image.open('images/correct.png').resize((30, 30)))
errorImage = ImageTk.PhotoImage(Image.open('images/remove.png').resize((30, 30)))

text = Text(root, height=1, width=75, font=('Helvetica', '9'), pady=5, padx=5)
text.grid(row=0, column=0, columnspan=8, padx=(20, 20), pady=(10,0))

canvas = Canvas(root, width=700, height=700, bg="white")
canvas.grid(row=1, column=0, rowspan=500, columnspan=10, padx=10, pady=10)

browseButton = Button(root, text="Browse", bg="black", fg="white", command=lambda: getImage(), width=10, relief=FLAT, font=('Helvetica 12 bold'))
browseButton.grid(row=0, column=8, pady=(10, 0))

originalButton = Button(root, text="Original", bg="black", fg="white", state=DISABLED, command=lambda: showOriginal(), width=15, height=2, relief=FLAT, font=('Helvetica 12 bold'))
originalButton.grid(row=55, column=11, padx=10)

originalButtonState = Label(root, image="")
originalButtonState.grid(row=55, column=12, padx=(0, 20))

segmentedButton = Button(root, text="Segmented", bg="black", fg="white", state=DISABLED, command=lambda: showSegmented(), width=15, height=2, relief=FLAT, font=('Helvetica 12 bold'))
segmentedButton.grid(row=60, column=11, padx=10)

segmentedButtonState = Label(root, image="")
segmentedButtonState.grid(row=60, column=12, padx=(0, 20))

orientationButton = Button(root, text="Orientations", bg="black", fg="white", state=DISABLED, command=lambda: showOrientations(), width=15, height=2, relief=FLAT, font=('Helvetica 12 bold'))
orientationButton.grid(row=65, column=11, padx=10)

orientationButtonState = Label(root, image="")
orientationButtonState.grid(row=65, column=12, padx=(0, 20))

ridgeWidthFrequencyButton = Button(root, text="Width/frequency", bg="black", fg="white", state=DISABLED, command=lambda: showWidthAndFrequency(), width=15, height=2, relief=FLAT, font=('Helvetica 12 bold'))
ridgeWidthFrequencyButton.grid(row=70, column=11, padx=10)

ridgeWidthFrequencyButtonState = Label(root, image="")
ridgeWidthFrequencyButtonState.grid(row=70, column=12, padx=(0, 20))

triradiusButton = Button(root, text="Triradius", bg="black", fg="white", state=DISABLED, command=lambda: showTriradius(), width=15, height=2, relief=FLAT, font=('Helvetica 12 bold'))
triradiusButton.grid(row=75, column=11, padx=10)

triradiusButtonState = Label(root, image="")
triradiusButtonState.grid(row=75, column=12, padx=(0, 20))

mainLinesButton = Button(root, text="Main Lines", bg="black", fg="white", state=DISABLED, command=lambda: showMainLines(), width=15, height=2, relief=FLAT, font=('Helvetica 12 bold'))
mainLinesButton.grid(row=80, column=11, padx=10)

mainLinesButtonState = Label(root, image="")
mainLinesButtonState.grid(row=80, column=12, padx=(0, 20))

exportButton = Button(root, text="Export", bg="black", fg="white", state=DISABLED, command=lambda: export(), width=15, height=2, relief=FLAT, font=('Helvetica 12 bold'))
exportButton.grid(row=450, column=11, padx=10)

widthText = Label(root, text="Ridge width:", anchor="w", font=('Helvetica 12 bold'), width=20)
widthText.grid(row=150, column=11, columnspan=2, padx=(10, 10))

frequencyText = Label(root, text="Ridge frequency:", font=('Helvetica 12 bold'), anchor='w', width=20)
frequencyText.grid(row=151, column=11, columnspan=2, padx=(10, 10))

mainLineAText = Label(root, text="Main line A:", font=('Helvetica 12 bold'), anchor='w', width=20)
mainLineAText.grid(row=200, column=11, columnspan=2, padx=(10, 10))

mainLineBText = Label(root, text="Main line B:", font=('Helvetica 12 bold'), anchor='w', width=20)
mainLineBText.grid(row=201, column=11, columnspan=2, padx=(10, 10))

mainLineCText = Label(root, text="Main line C:", font=('Helvetica 12 bold'), anchor="w", width=20)
mainLineCText.grid(row=202, column=11, columnspan=2, padx=(10, 10))

mainLineDText = Label(root, text="Main line D:", font=('Helvetica 12 bold'), anchor='w', width=20)
mainLineDText.grid(row=203, column=11, columnspan=2, padx=(10, 10))

mainLineTText = Label(root, text="Main line T:", font=('Helvetica 12 bold'), anchor='w', width=20)
mainLineTText.grid(row=204, column=11, columnspan=2, padx=(10, 10))

mainLineIndexText = Label(root, text="Main line index:", font=('Helvetica 12 bold'), anchor='w', width=20)
mainLineIndexText.grid(row=250, column=11, columnspan=2, padx=(10, 10))

root.mainloop()
