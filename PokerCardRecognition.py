import cv2
import numpy as np
import json

kernel = np.ones((5, 5))
image1 = cv2.imread("resources/KH.png")
image2 = cv2.imread("resources/KS.png")

images = [image1, image2]

train_ranks = []
train_suits = []

class Card:
    def __init__(self):
        self.rank = ""
        self.suit = ""

class Rank:
    def __init__(self):
        self.value = ""
        self.img = []

class Suit:
    def __init__(self):
        self.value = ""
        self.img = []

def fillTrainArrays():
    for Card in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]:
        img = cv2.imread("train/" + Card + "_TRAIN.png", cv2.IMREAD_GRAYSCALE)

        rank = Rank()
        rank.value = Card
        rank.img = img

        train_ranks.append(rank)

    for SuitCard in ["Diamonds", "Hearts", "Clubs", "Spades"]:
        img = cv2.imread("train/" + SuitCard + ".png", cv2.IMREAD_GRAYSCALE)

        suit = Suit()
        suit.value = SuitCard
        suit.img = img
        train_suits.append(suit)

def processImage(img):
    img = img[54:600, 25:110]
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, imgThres = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY_INV)

    return imgThres

def getContour(img):
    cnts, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    if len(cnts) != 0:
        x1, y1, w1, h1 = cv2.boundingRect(cnts[0])
        rect = img[y1:y1 + h1, x1:x1 + w1]
        sized = cv2.resize(rect, (70, 125), 0, 0)
        img = sized

    return img

def matchCardsRanks(rank):
    height_img1 = rank.img.shape[0]
    width_img1 = rank.img.shape[1]
    bestMatchRank = 10000
    bestCardRank = ""

    for Rank in train_ranks:
        height_img2 = Rank.img.shape[0]
        width_img2 = Rank.img.shape[1]

        if height_img1 > height_img2:
            rank.img = rank.img[:height_img2, :]
        else:
            Rank.img = Rank.img[:height_img1, :]

        if width_img1 > width_img2:
            rank.img = rank.img[:, :width_img2]
        else:
            Rank.img = Rank.img[:, :width_img1]

        diffImg = cv2.absdiff(rank.img, Rank.img)
        rank_diff = int(np.sum(diffImg) / 255)

        if rank_diff < bestMatchRank:
            bestMatchRank = rank_diff
            bestCardRank = Rank.value

    return bestCardRank

def matchCardsSuits(suit):
    height_img1 = suit.img.shape[0]
    width_img1 = suit.img.shape[1]
    bestMatchSuit = 10000
    bestCardSuit = ""

    for Suit in train_suits:
        height_img2 = Suit.img.shape[0]
        width_img2 = Suit.img.shape[1]

        if height_img1 > height_img2:
            suit.img = suit.img[:height_img2, :]
        else:
            Suit.img = Suit.img[:height_img1, :]

        if width_img1 > width_img2:
            suit.img = suit.img[:, :width_img2]
        else:
            Suit.img = Suit.img[:, :width_img1]

        diffImg = cv2.absdiff(suit.img, Suit.img)
        rank_diff = int(np.sum(diffImg) / 255)

        if rank_diff < bestMatchSuit:
            bestMatchSuit = rank_diff
            bestCardSuit = Suit.value

    return bestCardSuit

def run():
    fillTrainArrays()
    hand = []

    for image in images:
        card = Card()

        rank = Rank()
        suit = Suit()
        rank.img = image

        imgProcessed = processImage(rank.img)
        rank.img = imgProcessed[:120]
        suit.img = imgProcessed[120:230]

        rank.img = getContour(rank.img)
        suit.img = getContour(suit.img)

        bestRank = matchCardsRanks(rank)
        bestSuit = matchCardsSuits(suit)

        card.rank = bestRank
        card.suit = bestSuit

        hand.append(card.__dict__)

    if hand[0]['rank'] == hand[1]['rank']:
        print("Você já tem um par, mizera!!!")
    else:
        print("Bixo, tu vai perder!!!")

run()

def buildTrain():
    for Card in ["C", "D", "H", "S"]:
        filename = "J" + Card
        img = cv2.imread("resources/" + filename + ".png")
        processedImg = processImage(img)
        processedImg = processedImg[120:230]
        processedImg = getContour(processedImg)
        cv2.imwrite("F:/train_images/" + filename + ".png", processedImg)