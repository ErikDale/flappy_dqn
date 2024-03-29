from itertools import cycle
import random
import sys
import pygame
from pygame.locals import *
from pre_processing import pre_process_cnn_input, pre_process_dnn_input

'''
Heavily inspired by the code found at https://github.com/sourabhv/FlapPyBird
@date 27.02.2023 
'''

FPS = 30
SCREENWIDTH = 288
SCREENHEIGHT = 512
PIPEGAPSIZE = 200  # gap between upper and lower part of pipe, decides the difficult of the game
BASEY = SCREENHEIGHT * 0.79
# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

# How to get pygame screen as pixels: pygame.surfarray.array3d(self.screen)


# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
)

try:
    xrange
except NameError:
    xrange = range


class flappyGame:
    def __init__(self, cnn_model=True):
        self.cnn_model = cnn_model
        self.screen = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
        self.fpsClock = pygame.time.Clock()

        self.score = self.playerIndex = self.loopIter = 0

        self.dt = self.fpsClock.tick(FPS) / 1000
        self.pipeVelX = -128 * self.dt

        # player velocity, max velocity, downward acceleration, acceleration on flap
        self.playerVelY = -9  # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY = 10  # max vel along Y, max descend speed
        self.playerMinVelY = -8  # min vel along Y, max ascend speed
        self.playerAccY = 1  # players downward acceleration
        self.playerRot = 45  # player's rotation
        self.playerVelRot = 3  # angular speed
        self.playerRotThr = 20  # rotation threshold
        self.playerFlapAcc = -9  # players speed on flapping
        self.playerFlapped = False  # True when player flaps

        self.movementInfo = {}
        self.playerIndexGen = None
        self.playerx, self.playery = None, None

        self.basex = None
        self.baseShift = None

        self.newPipe1 = None
        self.newPipe2 = None

        # list of upper pipes
        self.upperPipes = []

        # list of lowerpipe
        self.lowerPipes = []

    def main(self):
        self.score = 0
        pygame.init()
        pygame.display.set_caption('Flappy Bird')

        # If it is a cnn model, use black background
        if self.cnn_model:
            # list of backgrounds
            BACKGROUNDS_LIST = (
                'assets/sprites/background-day_black.png',
                'assets/sprites/background-night_black.png',
            )
        # If not, use normal background
        else:
            # list of backgrounds
            BACKGROUNDS_LIST = (
                'assets/sprites/background-day.png',
                'assets/sprites/background-night.png',
            )

        # numbers sprites for score display
        IMAGES['numbers'] = (
            pygame.image.load('assets/sprites/0.png').convert_alpha(),
            pygame.image.load('assets/sprites/1.png').convert_alpha(),
            pygame.image.load('assets/sprites/2.png').convert_alpha(),
            pygame.image.load('assets/sprites/3.png').convert_alpha(),
            pygame.image.load('assets/sprites/4.png').convert_alpha(),
            pygame.image.load('assets/sprites/5.png').convert_alpha(),
            pygame.image.load('assets/sprites/6.png').convert_alpha(),
            pygame.image.load('assets/sprites/7.png').convert_alpha(),
            pygame.image.load('assets/sprites/8.png').convert_alpha(),
            pygame.image.load('assets/sprites/9.png').convert_alpha()
        )

        # game over sprite
        IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
        # message sprite for welcome screen
        IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
        # base (ground) sprite
        IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

        # sounds
        if 'win' in sys.platform:
            soundExt = '.wav'
        else:
            soundExt = '.ogg'

        SOUNDS['die'] = pygame.mixer.Sound('assets/audio/die' + soundExt)
        SOUNDS['hit'] = pygame.mixer.Sound('assets/audio/hit' + soundExt)
        SOUNDS['point'] = pygame.mixer.Sound('assets/audio/point' + soundExt)
        SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
        SOUNDS['wing'] = pygame.mixer.Sound('assets/audio/wing' + soundExt)

        # select random background sprites
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.flip(
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), False, True),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hitmask for pipes
        HITMASKS['pipe'] = (
            self.getHitmask(IMAGES['pipe'][0]),
            self.getHitmask(IMAGES['pipe'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            self.getHitmask(IMAGES['player'][0]),
            self.getHitmask(IMAGES['player'][1]),
            self.getHitmask(IMAGES['player'][2]),
        )

        # self.movementInfo = self.showWelcomeAnimation()

        # Initial position
        playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)
        # player shm for up-down motion on welcome screen
        playerShmVals = {'val': 0, 'dir': 1}
        basex = 0

        playerIndexGen = cycle([0, 1, 2, 1])

        self.movementInfo = {
            'playery': playery + playerShmVals['val'],
            'basex': basex,
            'playerIndexGen': playerIndexGen,
        }

        self.playerIndexGen = self.movementInfo['playerIndexGen']
        self.playerx, self.playery = int(SCREENWIDTH * 0.2), self.movementInfo['playery']

        self.basex = self.movementInfo['basex']
        self.baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

        # get 2 new pipes to add to upperPipes lowerPipes list
        self.newPipe1 = self.getRandomPipe()
        self.newPipe2 = self.getRandomPipe()

        # list of upper pipes
        self.upperPipes = [
            {'x': SCREENWIDTH + 200, 'y': self.newPipe1[0]['y']},
            {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': self.newPipe2[0]['y']},
        ]

        # list of lowerpipe
        self.lowerPipes = [
            {'x': SCREENWIDTH + 200, 'y': self.newPipe1[1]['y']},
            {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': self.newPipe2[1]['y']},
        ]

        # draw sprites
        self.screen.blit(IMAGES['background'], (0, 0))
        self.screen.blit(IMAGES['player'][self.playerIndex],
                         (self.playerx, self.playery + playerShmVals['val']))
        self.screen.blit(IMAGES['base'], (basex, BASEY))

        pygame.display.update()

        if self.cnn_model:
            return pre_process_cnn_input(pygame.surfarray.array3d(self.screen))
        else:
            return pre_process_dnn_input({
                'y': self.playery,
                'groundCrash': False,
                'basex': self.basex,
                'upperPipes': self.upperPipes,
                'lowerPipes': self.lowerPipes,
                'score': self.score,
                'playerVelY': self.playerVelY,
                'playerRot': self.playerRot,
                'state': pygame.surfarray.array3d(self.screen),
                'reward': 1,
                'done': False
            })


    def showWelcomeAnimation(self):
        """Shows welcome screen animation of flappy bird"""
        # index of player to blit on screen
        playerIndex = 0
        playerIndexGen = cycle([0, 1, 2, 1])
        # iterator used to change playerIndex after every 5th iteration
        loopIter = 0

        playerx = int(SCREENWIDTH * 0.2)
        playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)

        messagex = int((SCREENWIDTH - IMAGES['message'].get_width()) / 2)
        messagey = int(SCREENHEIGHT * 0.12)

        basex = 0
        # amount by which base can maximum shift to left
        baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

        # player shm for up-down motion on welcome screen
        playerShmVals = {'val': 0, 'dir': 1}

        while True:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                    # make first flap sound and return values for mainGame
                    SOUNDS['wing'].play()
                    return {
                        'playery': playery + playerShmVals['val'],
                        'basex': basex,
                        'playerIndexGen': playerIndexGen,
                    }

            # adjust playery, playerIndex, basex
            if (loopIter + 1) % 5 == 0:
                playerIndex = next(playerIndexGen)
            loopIter = (loopIter + 1) % 30
            basex = -((-basex + 4) % baseShift)
            self.playerShm(playerShmVals)

            # draw sprites
            self.screen.blit(IMAGES['background'], (0, 0))
            self.screen.blit(IMAGES['player'][playerIndex],
                             (playerx, playery + playerShmVals['val']))
            self.screen.blit(IMAGES['message'], (messagex, messagey))
            self.screen.blit(IMAGES['base'], (basex, BASEY))

            pygame.display.update()
            self.fpsClock.tick(FPS)

    def take_step(self, action):
        reward = 0.1
        # Player presses flap button (space or up key)
        if action == 1:
            if self.playery > -2 * IMAGES['player'][0].get_height():
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True
                SOUNDS['wing'].play()
                reward = 0.1

        # check for crash here
        crashTest = self.checkCrash({'x': self.playerx, 'y': self.playery, 'index': self.playerIndex},
                                    self.upperPipes, self.lowerPipes)

        # If crash, break out of game loop
        if crashTest[0]:
            reward = -1
            return {
                'y': self.playery,
                'groundCrash': crashTest[1],
                'basex': self.basex,
                'upperPipes': self.upperPipes,
                'lowerPipes': self.lowerPipes,
                'score': self.score,
                'playerVelY': self.playerVelY,
                'playerRot': self.playerRot,
                'state': pygame.surfarray.array3d(self.screen),
                'reward': reward,
                'done': True
            }

        # check for score
        playerMidPos = self.playerx + IMAGES['player'][0].get_width() / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                reward = 1
                self.score += 1
                SOUNDS['point'].play()

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(self.playerIndexGen)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # rotate the player
        if self.playerRot > -90:
            self.playerRot -= self.playerVelRot

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False

            # more rotation to cover the threshold (calculated in visible rotation)
            self.playerRot = 45

        playerHeight = IMAGES['player'][self.playerIndex].get_height()
        self.playery += min(self.playerVelY, BASEY - self.playery - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 3 > len(self.upperPipes) > 0 and 0 < self.upperPipes[0]['x'] < 5:
            newPipe = self.getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if len(self.upperPipes) > 0 and self.upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # draw sprites
        self.screen.blit(IMAGES['background'], (0, 0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            self.screen.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            self.screen.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        self.screen.blit(IMAGES['base'], (self.basex, BASEY))

        # print score so player overlaps the score
        if not self.cnn_model:
            self.showScore(self.score)

        # Player rotation has a threshold
        visibleRot = self.playerRotThr
        if self.playerRot <= self.playerRotThr:
            visibleRot = self.playerRot

        playerSurface = pygame.transform.rotate(IMAGES['player'][self.playerIndex], visibleRot)
        self.screen.blit(playerSurface, (self.playerx, self.playery))

        pygame.display.update()
        self.fpsClock.tick(FPS)

        return {
                'y': self.playery,
                'groundCrash': crashTest[1],
                'basex': self.basex,
                'upperPipes': self.upperPipes,
                'lowerPipes': self.lowerPipes,
                'score': self.score,
                'playerVelY': self.playerVelY,
                'playerRot': self.playerRot,
                'state': pygame.surfarray.array3d(self.screen),
                'reward': reward,
                'done': False
            }


    def showGameOverScreen(self, crashInfo):
        """crashes the player down and shows gameover image"""
        score = crashInfo['score']
        playerx = SCREENWIDTH * 0.2
        playery = crashInfo['y']
        playerHeight = IMAGES['player'][0].get_height()
        playerVelY = crashInfo['playerVelY']
        playerAccY = 2
        playerRot = crashInfo['playerRot']
        playerVelRot = 7

        basex = crashInfo['basex']

        upperPipes, lowerPipes = crashInfo['upperPipes'], crashInfo['lowerPipes']

        # play hit and die sounds
        SOUNDS['hit'].play()
        if not crashInfo['groundCrash']:
            SOUNDS['die'].play()

        while True:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                    if playery + playerHeight >= BASEY - 1:
                        return

            # player y shift
            if playery + playerHeight < BASEY - 1:
                playery += min(playerVelY, BASEY - playery - playerHeight)

            # player velocity change
            if playerVelY < 15:
                playerVelY += playerAccY

            # rotate only when it's a pipe crash
            if not crashInfo['groundCrash']:
                if playerRot > -90:
                    playerRot -= playerVelRot

            # draw sprites
            self.screen.blit(IMAGES['background'], (0, 0))

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                self.screen.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
                self.screen.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

            self.screen.blit(IMAGES['base'], (basex, BASEY))
            self.showScore(score)

            playerSurface = pygame.transform.rotate(IMAGES['player'][1], playerRot)
            self.screen.blit(playerSurface, (playerx, playery))
            self.screen.blit(IMAGES['gameover'], (50, 180))

            self.fpsClock.tick(FPS)
            pygame.display.update()

    def playerShm(self, playerShm):
        """oscillates the value of playerShm['val'] between 8 and -8"""
        if abs(playerShm['val']) == 8:
            playerShm['dir'] *= -1

        if playerShm['dir'] == 1:
            playerShm['val'] += 1
        else:
            playerShm['val'] -= 1

    def getRandomPipe(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
        gapY += int(BASEY * 0.2)
        pipeHeight = IMAGES['pipe'][0].get_height()
        pipeX = SCREENWIDTH + 10

        return [
            {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
            {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # lower pipe
        ]

    def showScore(self, score):
        """displays score in center of screen"""
        scoreDigits = [int(x) for x in list(str(score))]
        totalWidth = 0  # total width of all numbers to be printed

        for digit in scoreDigits:
            totalWidth += IMAGES['numbers'][digit].get_width()

        Xoffset = (SCREENWIDTH - totalWidth) / 2

        for digit in scoreDigits:
            self.screen.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
            Xoffset += IMAGES['numbers'][digit].get_width()

    def checkCrash(self, player, upperPipes, lowerPipes):
        """returns True if player collides with base or pipes."""
        pi = player['index']
        player['w'] = IMAGES['player'][0].get_width()
        player['h'] = IMAGES['player'][0].get_height()

        # if player crashes into ground
        if player['y'] + player['h'] >= BASEY - 1:
            return [True, True]
        # if player crashes into top of screen
        elif player['y'] + player['h'] < 22:
            return [True, True]
        else:

            playerRect = pygame.Rect(player['x'], player['y'],
                                     player['w'], player['h'])
            pipeW = IMAGES['pipe'][0].get_width()
            pipeH = IMAGES['pipe'][0].get_height()

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

                # player and upper/lower pipe hitmasks
                pHitMask = HITMASKS['player'][pi]
                uHitmask = HITMASKS['pipe'][0]
                lHitmask = HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = self.pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = self.pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

                if uCollide or lCollide:
                    return [True, False]

        return [False, False]

    def pixelCollision(self, rect1, rect2, hitmask1, hitmask2):
        """Checks if two objects collide and not just their rects"""
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in xrange(rect.width):
            for y in xrange(rect.height):
                if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                    return True
        return False

    def getHitmask(self, image):
        """returns a hitmask using an image's alpha."""
        mask = []
        for x in xrange(image.get_width()):
            mask.append([])
            for y in xrange(image.get_height()):
                mask[x].append(bool(image.get_at((x, y))[3]))
        return mask

