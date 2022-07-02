#! /usr/bin/env python

"""

Skript sluziaci pre vyhodnotenie aplikacie.

Autor: Bc. Jan Folenta
Email: xfolen00@stud.fit.vutbr.cz
Rok: 2022

"""

import os.path
from math import sqrt
import xml.etree.ElementTree as ET
from numpy import sum, min, max


def checkDistance(foundPoint, anotatedPoint, correct):
    # Funkcia skontroluje vzdialenost medzi najdenym a anotovanym triradiom

    distanceOK = False
    distance = sqrt((anotatedPoint[0] - foundPoint[0]) ** 2 + (anotatedPoint[1] - foundPoint[1]) ** 2)
    if distance < 150:
        correct += 1
        distanceOK = True

    return correct, distanceOK


def checkMainLine(mainLineAnoted, mainLineFound, mainLineCorrect):
    # Funkcia overi spravnost urcenia zakoncenia hlavnej linie

    ok = False

    if str(mainLineAnoted) == '4' and (str(mainLineFound) == '3' or str(mainLineFound) == '5\''):
        mainLineCorrect += 1
        ok = True

    elif str(mainLineAnoted) == str(mainLineFound):
        mainLineCorrect += 1
        ok = True

    return mainLineCorrect, ok


def checkMainLineIndex(mainLineAanoted, mainLineDanoted, mainLineAfound, mainLineDfound, correct):
    # Funkcia overi spravnost urcenia indexu hlavnych linii

    okA = False
    okD = False

    if str(mainLineAanoted) == '4' and (str(mainLineAfound) == '3' or str(mainLineAfound) == '5\''):
        okA = True

    elif str(mainLineAanoted) == str(mainLineAfound):
        okA = True

    if str(mainLineDanoted) == '4' and (str(mainLineDfound) == '3' or str(mainLineDfound) == '5\''):
        okD = True

    elif str(mainLineDanoted) == str(mainLineDfound):
        okD = True

    if okA and okD:
        correct += 1

    return correct


def getAnotatedData(palmData, leftPalm):
    # Funkcia ziska anotovane data

    palmTriradius = []
    cropInfo = palmData.find('crop')
    cropped = False

    minX = 10000000
    maxX = 0
    minY = 10000000
    for point in cropInfo:
        cropped = True
        x = float(point.get('x'))
        y = float(point.get('y'))

        if x > maxX:
            maxX = x
        if x < minX:
            minX = x
        if y < minY:
            minY = y

    triradius = palmData.find('triradium')
    for triradiusPoint in triradius:
        triradiusPointX = triradiusPoint.get('x')
        triradiusPointY = triradiusPoint.get('y')
        if triradiusPointX != '0' and triradiusPointY != '0':
            if leftPalm:
                if cropped:
                    triradiusPointX = int(5100 - (maxX - float(triradiusPointX)))
                    triradiusPointY = int(minY + float(triradiusPointY))
                else:
                    triradiusPointX = int(float(triradiusPointX))
                    triradiusPointY = int(float(triradiusPointY))
            else:
                if cropped:
                    triradiusPointX = int(minX + float(triradiusPointX))
                    triradiusPointY = int(minY + float(triradiusPointY))
                else:
                    triradiusPointX = int(float(triradiusPointX))
                    triradiusPointY = int(float(triradiusPointY))
            palmTriradius.append((triradiusPointY, triradiusPointX))

    i = 0
    mainLinesAnotated = []
    for mainLine in palmData.iter('setting'):
        if i < 5:
            mainLinesAnotated.append(mainLine.get('second'))
        i += 1

    i = 0
    for mainLine in mainLinesAnotated:
        if mainLine == "0":
            if i != 4:
                j = 3 - i
                palmTriradius.insert(j, (0, 0))
            else:
                palmTriradius.insert(i, (0, 0))

        i += 1

    palmTriradius = palmTriradius[0:5]

    return palmTriradius, mainLinesAnotated


def main():

    triradiustotal = [0, 0]

    triradiusAcorrect = [0, 0]
    triradiusBcorrect = [0, 0]
    triradiusCcorrect = [0, 0]
    triradiusDcorrect = [0, 0]
    triradiusTcorrect = [0, 0]

    mainLineAcorrect = [0, 0]
    mainLineBcorrect = [0, 0]
    mainLineCcorrect = [0, 0]
    mainLineDcorrect = [0, 0]
    mainLineTcorrect = [0, 0]

    mainLineAtotal = [0, 0]
    mainLineBtotal = [0, 0]
    mainLineCtotal = [0, 0]
    mainLineDtotal = [0, 0]
    mainLineTtotal = [0, 0]

    mliCorrect = [0, 0]
    mliTotal = [0, 0]

    frequenciesMen = []
    frequenciesWomen = []
    widthsMen = []
    widthsWomen = []

    directory = 'anotatedData'
    for filename in os.listdir(directory):
        tmpCreated = False
        f = os.path.join(directory, filename)
        if os.path.isfile(f):

            try:
                anotatedTree = ET.parse(f)
                anotatedRoot = anotatedTree.getroot()
            except:
                anotatedFile = open(f)
                lines = anotatedFile.readlines()
                anotatedFile.close()
                lines.pop(-2)
                updatedFile = "".join(lines)

                save_path_file = "tmp.xml"

                with open(save_path_file, "w") as f:
                    f.write(updatedFile)

                tmpCreated = True

                anotatedTree = ET.parse('tmp.xml')
                anotatedRoot = anotatedTree.getroot()

            palmNames = []
            i = 0
            images = anotatedRoot.find('images')
            for image in images.iter('image'):

                if i == 10:
                    leftPalmName = image.get('src').split('.')[0]
                    palmNames.append(leftPalmName)
                if i == 11:
                    rightPalmName = image.get('src').split('.')[0]
                    palmNames.append(rightPalmName)

                i += 1

            leftPalm = True

            for palmName in palmNames:
                if palmName == "":
                    leftPalm = False
                    continue
                if palmName == palmNames[0]:
                    palmData = anotatedRoot.findall('dermatoglyph')[10]
                else:
                    palmData = anotatedRoot.findall('dermatoglyph')[11]

                if palmName[0:4] == "NORM":
                    sex = 0
                else:
                    sex = 1

                palmTriradius, mainLinesAnotated = getAnotatedData(palmData, leftPalm)

                if os.path.isfile(f"outEvaluation/{palmName}_out.xml"):

                    triradiustotal[sex] += 1

                    tree = ET.parse(f'outEvaluation/{palmName}_out.xml')
                    root = tree.getroot()

                    if sex == 0:
                        frequenciesMen.append(float(root.find('averageFrequency').text))
                        widthsMen.append(float(root.find('averageWidth').text))
                    else:
                        frequenciesWomen.append(float(root.find('averageFrequency').text))
                        widthsWomen.append(float(root.find('averageWidth').text))

                    triradiusAx = int(root.find('triradiusA').get('x'))
                    triradiusAy = int(root.find('triradiusA').get('y'))

                    triradiusBx = int(root.find('triradiusB').get('x'))
                    triradiusBy = int(root.find('triradiusB').get('y'))

                    triradiusCx = int(root.find('triradiusC').get('x'))
                    triradiusCy = int(root.find('triradiusC').get('y'))

                    triradiusDx = int(root.find('triradiusD').get('x'))
                    triradiusDy = int(root.find('triradiusD').get('y'))

                    triradiusTx = int(root.find('triradiusT').get('x'))
                    triradiusTy = int(root.find('triradiusT').get('y'))

                    triradiusAcorrect[sex], triradiusAok = checkDistance((triradiusAx, triradiusAy), palmTriradius[0],
                                                                    triradiusAcorrect[sex])
                    triradiusBcorrect[sex], triradiusBok = checkDistance((triradiusBx, triradiusBy), palmTriradius[1],
                                                                    triradiusBcorrect[sex])
                    triradiusCcorrect[sex], triradiusCok = checkDistance((triradiusCx, triradiusCy), palmTriradius[2],
                                                                    triradiusCcorrect[sex])
                    triradiusDcorrect[sex], triradiusDok = checkDistance((triradiusDx, triradiusDy), palmTriradius[3],
                                                                    triradiusDcorrect[sex])
                    triradiusTcorrect[sex], triradiusTok = checkDistance((triradiusTx, triradiusTy), palmTriradius[4],
                                                                    triradiusTcorrect[sex])

                    mainLineA = root.find('mainLineA').text
                    mainLineB = root.find('mainLineB').text
                    mainLineC = root.find('mainLineC').text
                    mainLineD = root.find('mainLineD').text
                    mainLineT = root.find('mainLineT').text

                    mainLineAok = False
                    mainLineDok = False

                    if triradiusAok:
                        mainLineAtotal[sex] += 1
                        mainLineAcorrect[sex], mainLineAok = checkMainLine(mainLinesAnotated[3], mainLineA,
                                                                           mainLineAcorrect[sex])

                    if triradiusBok:
                        mainLineBtotal[sex] += 1
                        mainLineBcorrect[sex], mainLineBok = checkMainLine(mainLinesAnotated[2], mainLineB,
                                                                           mainLineBcorrect[sex])

                    if triradiusCok:
                        mainLineCtotal[sex] += 1
                        mainLineCcorrect[sex], mainLineCok = checkMainLine(mainLinesAnotated[1], mainLineC,
                                                                           mainLineCcorrect[sex])

                    if triradiusDok:
                        #print(f"{palmName} -- OK")
                        mainLineDtotal[sex] += 1
                        mainLineDcorrect[sex], mainLineDok = checkMainLine(mainLinesAnotated[0], mainLineD,
                                                                           mainLineDcorrect[sex])

                    if triradiusTok:
                        mainLineTtotal[sex] += 1
                        mainLineTcorrect[sex], mainLineTok = checkMainLine(mainLinesAnotated[4], mainLineT,
                                                                           mainLineTcorrect[sex])

                    mliCorrect[sex] = checkMainLineIndex(mainLinesAnotated[3], mainLinesAnotated[0], mainLineA,
                                                         mainLineD, mliCorrect[sex])
                    mliTotal[sex] += 1

                leftPalm = False

            if tmpCreated:
                os.remove('tmp.xml')

    print(f"FREQUENCY [min -- max -> average]")
    print(f"Men => {min(frequenciesMen)} -- {max(frequenciesMen)} -> {sum(frequenciesMen) / len(frequenciesMen)}")
    print(f"Women => {min(frequenciesWomen)} -- {max(frequenciesWomen)} -> {sum(frequenciesWomen) / len(frequenciesWomen)}")

    print(f"\nWIDTH [min -- max -> average]")
    print(f"Men => {min(widthsMen)} -- {max(widthsMen)} -> {sum(widthsMen) / len(widthsMen)}")
    print(f"Women => {min(widthsWomen)} -- {max(widthsWomen)} -> {sum(widthsWomen) / len(widthsWomen)}")

    print("\nTRIRADIUS DETECTION [a, b, c, d, t  - total]")
    print("==========================")

    if triradiustotal[0] != 0:
        triradiusTotalMen = ((triradiusTcorrect[0] + triradiusDcorrect[0] + triradiusCcorrect[0] + triradiusBcorrect[0] +
                              triradiusAcorrect[0]) / (5 * triradiustotal[0])) * 100
        triradiusTotalWomen = ((triradiusTcorrect[1] + triradiusDcorrect[1] + triradiusCcorrect[1] + triradiusBcorrect[1] +
                               triradiusAcorrect[1]) / (5 * triradiustotal[1])) * 100

        triradiusTotalAll = ((sum(triradiusTcorrect) + sum(triradiusDcorrect) + sum(triradiusCcorrect) + sum(triradiusBcorrect) +
                               sum(triradiusAcorrect)) / (5 * sum(triradiustotal))) * 100

        print(f"Men => {(triradiusAcorrect[0] / triradiustotal[0]) * 100}, {(triradiusBcorrect[0] / triradiustotal[0]) * 100}, "
              f"{(triradiusCcorrect[0] / triradiustotal[0]) * 100}, {(triradiusDcorrect[0] / triradiustotal[0]) * 100}, "
              f"{(triradiusTcorrect[0] / triradiustotal[0]) * 100} - {triradiusTotalMen}")

        print(
            f"Women => {(triradiusAcorrect[1] / triradiustotal[1]) * 100}, {(triradiusBcorrect[1] / triradiustotal[1]) * 100}, "
            f"{(triradiusCcorrect[1] / triradiustotal[1]) * 100}, {(triradiusDcorrect[1] / triradiustotal[1]) * 100}, "
            f"{(triradiusTcorrect[1] / triradiustotal[1]) * 100} - {triradiusTotalWomen}")

        print(
            f"Total => {(sum(triradiusAcorrect) / sum(triradiustotal)) * 100}, {(sum(triradiusBcorrect) / sum(triradiustotal)) * 100}, "
            f"{(sum(triradiusCcorrect) / sum(triradiustotal)) * 100}, {(sum(triradiusDcorrect) / sum(triradiustotal)) * 100}, "
            f"{(sum(triradiusTcorrect) / sum(triradiustotal)) * 100} - {triradiusTotalAll}")

    print("\nMAIN LINE TRACKING [A, B, C, D, T  - total]")
    print("==========================")

    mainLinesTotalMen = ((mainLineAcorrect[0] + mainLineBcorrect[0] + mainLineCcorrect[0] + mainLineDcorrect[0] +
                              mainLineTcorrect[0]) / (mainLineAtotal[0] + mainLineBtotal[0] + mainLineCtotal[0] + mainLineDtotal[0] + mainLineTtotal[0])) * 100

    mainLinesTotalWomen = ((mainLineAcorrect[1] + mainLineBcorrect[1] + mainLineCcorrect[1] + mainLineDcorrect[1] +
                              mainLineTcorrect[1]) / (mainLineAtotal[1] + mainLineBtotal[1] + mainLineCtotal[1] + mainLineDtotal[1] + mainLineTtotal[1])) * 100

    mainLinesTotalAll = ((sum(mainLineAcorrect) + sum(mainLineBcorrect) + sum(mainLineCcorrect) + sum(mainLineDcorrect) +
                              sum(mainLineTcorrect)) / (sum(mainLineAtotal) + sum(mainLineBtotal) + sum(mainLineCtotal) + sum(mainLineDtotal) + sum(mainLineTtotal))) * 100

    print(f"Men => {(mainLineAcorrect[0] / mainLineAtotal[0]) * 100}, {(mainLineBcorrect[0] / mainLineBtotal[0]) * 100}, "
          f"{(mainLineCcorrect[0] / mainLineCtotal[0]) * 100}, {(mainLineDcorrect[0] / mainLineDtotal[0]) * 100}, "
          f"{(mainLineTcorrect[0] / mainLineTtotal[0]) * 100} - {mainLinesTotalMen}")

    print(f"Women => {(mainLineAcorrect[1] / mainLineAtotal[1]) * 100}, {(mainLineBcorrect[1] / mainLineBtotal[1]) * 100}, "
          f"{(mainLineCcorrect[1] / mainLineCtotal[1]) * 100}, {(mainLineDcorrect[1] / mainLineDtotal[1]) * 100}, "
          f"{(mainLineTcorrect[1] / mainLineTtotal[1]) * 100} - {mainLinesTotalWomen}")

    print(f"Total => {(sum(mainLineAcorrect) / sum(mainLineAtotal)) * 100}, {(sum(mainLineBcorrect) / sum(mainLineBtotal)) * 100}, "
          f"{(sum(mainLineCcorrect) / sum(mainLineCtotal)) * 100}, {(sum(mainLineDcorrect) / sum(mainLineDtotal)) * 100}, "
          f"{(sum(mainLineTcorrect) / sum(mainLineTtotal)) * 100} - {mainLinesTotalAll}")

    print("\nMAIN LINE INDEX")
    print("==========================")

    print(f"Men: {(mliCorrect[0] / mliTotal[0]) * 100}")
    print(f"Women: {(mliCorrect[1] / mliTotal[1]) * 100}")
    print(f"Total: {(sum(mliCorrect) / sum(mliTotal)) * 100}")


if __name__ == '__main__':
    main()
