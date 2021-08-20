import sys
import pygame
from pygame.math import Vector2

#  window size
pygame.init()
size = width,height = 450,300
screen = pygame.display.set_mode(size)

# initial positions of each arm
ini_position=[(100, 100), (200, 100), (300, 100), (400, 100)];
arms = list(map(Vector2, ini_position));

# target position
# target = Vector2(220, 120)
# target = Vector2(350, 150)
target = Vector2(50, 50)
# target_speed = Vector2(0, 0)
target_speed = Vector2(3, 3)

rel_arms = [] # list for all arms, stores the movement between two arm
angles = []   # ik solution, list for all adjust angle
for i in range(1, len(arms)):
    rel_arms.append(arms[i] - arms[i-1])
    angles.append(0)

def solve_ik(i, end_joint, target):

    if i < len(arms) - 2:
        end_joint = solve_ik(i+1, end_joint, target)
    current_joint = arms[i]

    angle = (end_joint-current_joint).angle_to(target-current_joint)

    angles[i] = min( max(-2, angle), 2 ) + angles[i]  # move angle

    angles[i] = min( max( -360, (angles[i])%360 ), 360 )

    return current_joint + (end_joint-current_joint).rotate(angle)

def draw():

    black = 0,0,0
    white = 255,255,255
    blue = 0,0,255

    screen.fill(white)
    for i in range(1, len(arms)):
        prev = arms[i-1]
        next = arms[i]
        pygame.draw.aaline(screen, black, prev, next)
    for point in arms:
        pygame.draw.circle(screen, blue, (int(point[0]), int(point[1])), 10)
    pygame.draw.circle(screen, black, (int(target[0]), int(target[1])), 11)
    pygame.display.flip()


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    solve_ik(0, arms[-1], target)
    angle = 0
    for i in range(1, len(arms)):
        angle += angles[i-1]
        arms[i] = arms[i-1] + rel_arms[i-1].rotate(angle)

    # target change
    target += target_speed
    # Rebound when hitting the edge
    if target.x <= 0 or target.x >= width:
        target_speed.x = -target_speed.x
    if target.y <= 0 or target.y >= height:
        target_speed.y = -target_speed.y

    draw()
    pygame.time.wait(int(1000/30))