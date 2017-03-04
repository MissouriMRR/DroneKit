import pygame, sys, math, random, time


RUN_SPEED = 5 # times.  Note: limited by hardware

PI = 3.14159265359
MAX_NOISE = 20
ROOMBA_FRAME_DELAY = 200 # milliseconds, 1/5 second
GROUND_ROBOT_NOISE_INTERVAL = 5 # seconds
GROUND_ROBOT_REVERSE_INTERVAL = 20 # seconds
GROUND_ROBOT_SPEED = 0.333 # meters per second
METERS_PER_WINDOW = 24
PIXELS_PER_METER = 20
ROOMBA_TURN_SPEED = 0.25 # rotations per second
GROUND_ROBOT_COUNT = 10
GROUND_ROBOT_START_RADIUS = 1 # meter
ROOMBA_RADIUS = 0.2 # meters
OBSTACLE_ROBOT_COUNT = 4
OBSTACLE_ROBOT_START_RADIUS = 5 # meters
OBSTACLE_ROBOT_NOISE = 2 # maximum degrees per roomba frame.  In the videos the obstacle robots get way off their circle, so this simulation has noise in the obstacle robots.
DRONE_FRAME_RATE = 60 # frames per second
DRONE_PIXEL_SPEED = 2 # the drone's agility, treated as a safety factor for slowing down to hit targets in Sim9AI()
DRONE_ACCEL_NUMBER = 0.005 # I have no idea what unit this is.  This is from the JavaScript simulation.
DRONE_COUNT = 1
DRONE_RADIUS = 0.18 # meters
ROOMBA_HEIGHT = 0.1 # meters
ROOMBA_TAPPABLE_HEIGHT = 0.12 # meters from floor
OBSTACLE_RADIUS = 0.05 # meters
OK_TO_LEAVE_THRESHOLD = 16 # of 24 meters

def rotate2d(pos,rad): x,y=pos; s,c = math.sin(rad),math.cos(rad); return x*c-y*s,y*c+x*s;
def distance2d(pos1,pos2): return math.sqrt( pow(pos1[0] - pos2[0], 2) + pow(pos1[1] - pos2[1], 2) )
def inBounds(pos): TWENTY_FOURTHS = METERS_PER_WINDOW * PIXELS_PER_METER / 24; return (pos[0] >= 2 * TWENTY_FOURTHS and pos[0] <= 22 * TWENTY_FOURTHS and pos[1] >= 2 * TWENTY_FOURTHS and pos[1] <= 22 * TWENTY_FOURTHS)

class Cam:
  def __init__(self,pos=(0,0,0),rot=(0,0)):
    self.pos = list(pos)
    self.rot = list(rot)

  """
  def events(self,event):
    if event.type == pygame.MOUSEMOTION:
      x,y = event.rel
      x/=240; y/=240
      self.rot[0]+=y; self.rot[1]+=x
  """

  def update(self,dt,key):
    s = dt*10   

    """
    for i in range(len(key)):
      if key[i]:
        print(i)
    """

    if key[113]: self.pos[1]-=s # q
    if key[101]: self.pos[1]+=s # e
    
    if key[119]: self.pos[2]+=s # w
    if key[115]: self.pos[2]-=s # s
    if key[97]: self.pos[0]-=s # a
    if key[100]: self.pos[0]+=s # d


class roomba:
  def __init__(self,pos=[PIXELS_PER_METER*METERS_PER_WINDOW/2,PIXELS_PER_METER*METERS_PER_WINDOW/2],rotDeg=(0)):
    self.pos = [PIXELS_PER_METER*METERS_PER_WINDOW/2,PIXELS_PER_METER*METERS_PER_WINDOW/2]
    self.rotDeg = 0 # degrees

  def collision(self, otherRoomba ):
    collision = False
    if inBounds( self.pos ) and inBounds( otherRoomba.pos ):
      distance = distance2d( self.pos, otherRoomba.pos )
      collision = distance < 2 * ROOMBA_RADIUS * PIXELS_PER_METER
      if collision:
        if distance > distance2d( (self.pos[0] + math.cos(self.rotDeg * PI / 180), self.pos[1] + math.sin(self.rotDeg * PI / 180)), otherRoomba.pos ):
          self.bump()
        if distance > distance2d( (otherRoomba.pos[0] + math.cos(otherRoomba.rotDeg * PI / 180), otherRoomba.pos[1] + math.sin(otherRoomba.rotDeg * PI / 180)), self.pos ):
          otherRoomba.bump()
    return collision




class groundRobot(roomba):
  def __init__(self,noiseCountdown=(GROUND_ROBOT_NOISE_INTERVAL * 1000 / ROOMBA_FRAME_DELAY),noiseSelection=(0),turningCountdown=(0),reverseCountdown=(100),stillIn=(True)):
    roomba.__init__(self)
    self.noiseCountdown = GROUND_ROBOT_NOISE_INTERVAL * 1000 / ROOMBA_FRAME_DELAY
    self.turningCountdown = turningCountdown
    self.reverseCountdown = 100
    self.stillIn = stillIn

  def bump(self):
    self.turningCountdown = 0.5 * 1000/ROOMBA_FRAME_DELAY/ROOMBA_TURN_SPEED

  def tap(self):
    self.turningCountdown = 0.125 * 1000/ROOMBA_FRAME_DELAY/ROOMBA_TURN_SPEED

  def iterate(self, times = 1 ):
    if not( self.stillIn and inBounds( self.pos ) ):
      self.stillIn = False
    else:
      for time in range( times ):
        stop = False
        if( self.noiseCountdown >= 5 ):
          self.noiseCountdown -= 1
        if self.reverseCountdown > 0.5 * 1000/ROOMBA_FRAME_DELAY/ROOMBA_TURN_SPEED:
          self.reverseCountdown -= 1
        if self.noiseCountdown == 5:
          self.noiseSelection = (random.random() - 0.5) * MAX_NOISE / (1000 / ROOMBA_FRAME_DELAY)
        if self.turningCountdown > 0:
          stop = True
          self.rotDeg += ROOMBA_TURN_SPEED * ROOMBA_FRAME_DELAY * 360 / 1000
          self.turningCountdown -= 1
        elif self.reverseCountdown <= 0.5 * 1000/ROOMBA_FRAME_DELAY/ROOMBA_TURN_SPEED:
          stop = True
          self.rotDeg += ROOMBA_TURN_SPEED * ROOMBA_FRAME_DELAY * 360 / 1000
          self.reverseCountdown -= 1
          if self.reverseCountdown <= 0:
            self.reverseCountdown = GROUND_ROBOT_REVERSE_INTERVAL * 1000 / ROOMBA_FRAME_DELAY # 1000 milliseconds in one second
        elif self.noiseCountdown < 5:
          self.noiseCountdown -= 1
          self.rotDeg += self.noiseSelection
          if self.noiseCountdown <= 0:
            self.noiseCountdown = GROUND_ROBOT_NOISE_INTERVAL * 1000 / ROOMBA_FRAME_DELAY # 1000 milliseconds in one second
        
        if not stop:
          coordChange = rotate2d((GROUND_ROBOT_SPEED*PIXELS_PER_METER*ROOMBA_FRAME_DELAY/1000,0),self.rotDeg*PI/180)
          for k in range(2):
            self.pos[k] += coordChange[k]






class obstacleRobot(roomba):
  def __init__(self,height=( (random.random() + 1) * PIXELS_PER_METER ),go=True):
    roomba.__init__(self)
    self.height = height
    self.go = True
  
  def bump(self):
    self.go = False

  def iterate(self, times = 1 ):
    for time in range( times ):
      if self.go:
        self.rotDeg -= 360 * GROUND_ROBOT_SPEED * ROOMBA_FRAME_DELAY / 1000 / (2 * PI * 5)
        coordChange = rotate2d((GROUND_ROBOT_SPEED*PIXELS_PER_METER*ROOMBA_FRAME_DELAY/1000,0),self.rotDeg*PI/180)
        for k in range(2):
          self.pos[k] += coordChange[k]
        self.rotDeg += (random.random() - 0.5) * OBSTACLE_ROBOT_NOISE
      else:
        self.go = True

  """
  def collision(self, otherRoomba ):
    distance = distance2d( self.pos, otherRoomba.pos )
    collision = distance < 2 * ROOMBA_RADIUS * PIXELS_PER_METER
    if collision:
      if self.reverseCountdown > 0.5 * 1000/ROOMBA_FRAME_DELAY/ROOMBA_TURN_SPEED and distance > distance2d( (self.pos[0] + math.cos(self.rotDeg * PI / 180), self.pos[1] + math.sin(self.rotDeg * PI / 180)), otherRoomba.pos ):
        self.reverseCountdown = 0.5 * 1000/ROOMBA_FRAME_DELAY/ROOMBA_TURN_SPEED
      if otherRoomba.reverseCountdown > 0.5 * 1000/ROOMBA_FRAME_DELAY/ROOMBA_TURN_SPEED and distance > distance2d( (otherRoomba.pos[0] + math.cos(otherRoomba.rotDeg * PI / 180), otherRoomba.pos[1] + math.sin(otherRoomba.rotDeg * PI / 180)), self.pos ):
        otherRoomba.reverseCountdown = 0.5 * 1000/ROOMBA_FRAME_DELAY/ROOMBA_TURN_SPEED
    return collision
  """



# for AI from JavaScript simulation
class drone:
  def __init__(self,X=(METERS_PER_WINDOW*PIXELS_PER_METER * 22/24),Y=(METERS_PER_WINDOW*PIXELS_PER_METER / 2),Z=(3 * PIXELS_PER_METER),XSpeed=(0),YSpeed=(0),TargetRobot=(0),OkToLeave=(True),NextRotate=(20 * DRONE_FRAME_RATE)):
    self.X=X
    self.Y=Y
    self.Z=Z
    self.XSpeed=XSpeed
    self.YSpeed=YSpeed
    self.TargetRobot=TargetRobot
    self.OkToLeave=OkToLeave
    self.NextRotate=NextRotate

  def Sim9AI( self, groundRobotsPos, groundRobotsRotDeg, groundRobotsStillIn, obstacleRobotsPos, times = 1 ):
  
    #print( groundRobotsPos, groundRobotsRotDeg, groundRobotsStillIn, obstacleRobotsPos )
  
    for k in range( times ):
    
      self.iterate()
      
      TargetRot = 270
    
      PrevTargetRobot = self.TargetRobot
      #alert(DataArray[CurrentFrame].Drones[index].OkToLeave + "\n" + DataArray[CurrentFrame].Drones[index].TargetRobot);
      if(self.OkToLeave == True and self.NextRotate / DRONE_FRAME_RATE > 12 or groundRobotsStillIn[self.TargetRobot] == False):
        TempX = 0
        TempY = 0
        self.TargetRobot = -1
        for index2 in range( GROUND_ROBOT_COUNT ):
          Rot = ( groundRobotsRotDeg[index2] % 360 + 360) if ( groundRobotsRotDeg[index2] % 360 < 0) else ( groundRobotsRotDeg[index2] % 360)
          if groundRobotsStillIn[index2] == True and (Rot < TargetRot - 70 / 2 or Rot > TargetRot + 70 / 2):
            SafeTarget = True;
            for index3 in range( OBSTACLE_ROBOT_COUNT ):
              A = groundRobotsPos[index2][0] - obstacleRobotsPos[index3][0]
              B = groundRobotsPos[index2][1] - obstacleRobotsPos[index3][1]
              if(math.sqrt(math.pow(A, 2) + math.pow(B, 2)) / PIXELS_PER_METER < (ROOMBA_RADIUS + ROOMBA_RADIUS) / PIXELS_PER_METER + 1):
                SafeTarget = False
            if SafeTarget:
              # get the number of robots still in
              numStillIn = 0
              for bot in groundRobotsStillIn:
                if bot:
                  numStillIn += 1
              if( TempX == 0 or groundRobotsPos[index2][1] < TempY ):
                self.TargetRobot = index2
                TempX = groundRobotsPos[index2][0]
                TempY = groundRobotsPos[index2][1]
    
      if self.TargetRobot >= 0:
        TargetX = groundRobotsPos[self.TargetRobot][0]
        TargetY = groundRobotsPos[self.TargetRobot][1]
      else:
        TargetX = METERS_PER_WINDOW*PIXELS_PER_METER/2
        TargetY = METERS_PER_WINDOW*PIXELS_PER_METER/2
    
      XDiff = self.X - TargetX
      YDiff = self.Y - TargetY
      RotationToGroundRobot = math.atan2(XDiff, YDiff) * 180 / PI # now, atan2 accepts (y,x), but in sim9 this is how it is (it's the same way in JavaScript)
      PerpLine = RotationToGroundRobot + 90
        
      XSpeed = self.XSpeed
      YSpeed = self.YSpeed
      XPos = self.X
      YPos = self.Y
      while XSpeed != 0 or YSpeed != 0:
        if(XSpeed > DRONE_ACCEL_NUMBER):
          XSpeed -= DRONE_ACCEL_NUMBER
        elif(XSpeed < -DRONE_ACCEL_NUMBER):
          XSpeed += DRONE_ACCEL_NUMBER
        else:
          XSpeed = 0
          
        if(YSpeed > DRONE_ACCEL_NUMBER):
          YSpeed = YSpeed - DRONE_ACCEL_NUMBER
        elif(YSpeed < -DRONE_ACCEL_NUMBER):
          YSpeed = YSpeed + DRONE_ACCEL_NUMBER
        else:
          YSpeed = 0
          
        XPos += XSpeed
        YPos += YSpeed
      
      if((XPos < TargetX and self.X < TargetX) or (XPos < TargetX and self.X > TargetX)):
        self.XSpeed += DRONE_PIXEL_SPEED * DRONE_ACCEL_NUMBER
      else:
        self.XSpeed += DRONE_PIXEL_SPEED * -DRONE_ACCEL_NUMBER
                          
      if((YPos < TargetY and self.Y < TargetY) or (YPos < TargetY and self.Y > TargetY)):
        self.YSpeed = self.YSpeed + DRONE_PIXEL_SPEED * DRONE_ACCEL_NUMBER
      else:
        self.YSpeed = self.YSpeed + DRONE_PIXEL_SPEED * -DRONE_ACCEL_NUMBER

      TargetRobotRot = groundRobotsRotDeg[self.TargetRobot]
      TargetRobotRot = ( TargetRobotRot % 360 + 360) if ( TargetRobotRot % 360 < 0) else ( TargetRobotRot % 360)
      if ( ( TargetRobotRot > TargetRot + 70 / 2 or TargetRobotRot < TargetRot - 70 / 2 ) and math.pow( ROOMBA_RADIUS * PIXELS_PER_METER, 2) > math.pow( self.X - groundRobotsPos[self.TargetRobot][0], 2) + math.pow( self.Y - groundRobotsPos[self.TargetRobot][1], 2) ):
        self.Z = ROOMBA_HEIGHT
        if groundRobotsPos[self.TargetRobot][1] > METERS_PER_WINDOW * PIXELS_PER_METER * OK_TO_LEAVE_THRESHOLD / 24:
          self.OkToLeave = False
        else:
          self.OkToLeave = True
      else:
        self.Z = 3 * PIXELS_PER_METER
        


  def iterate( self, times = 1 ):
    
    self.X += self.XSpeed
    self.Y += self.YSpeed





pygame.init()
w,h = PIXELS_PER_METER*METERS_PER_WINDOW, PIXELS_PER_METER*METERS_PER_WINDOW; cx,cy = w//2, h//2
screen = pygame.display.set_mode((w,h))
clock = pygame.time.Clock()


groundRobots = []
for k in range( GROUND_ROBOT_COUNT ):
  groundRobots.append( groundRobot() )
  posChange = rotate2d( ( GROUND_ROBOT_START_RADIUS * PIXELS_PER_METER, 0), 2 * PI * k / GROUND_ROBOT_COUNT )
  groundRobots[k].pos[0] += posChange[0]; groundRobots[k].pos[1] += posChange[1];
  groundRobots[k].rotDeg += 360 * k / GROUND_ROBOT_COUNT


obstacleRobots = []
for k in range( OBSTACLE_ROBOT_COUNT ):
  obstacleRobots.append( obstacleRobot() )
  posChange = rotate2d( ( OBSTACLE_ROBOT_START_RADIUS * PIXELS_PER_METER, 0), 2 * PI * k / OBSTACLE_ROBOT_COUNT )
  obstacleRobots[k].pos[0] += posChange[0]; obstacleRobots[k].pos[1] += posChange[1];
  obstacleRobots[k].rotDeg += 360 * k / OBSTACLE_ROBOT_COUNT
  obstacleRobots[k].rotDeg -= 90

drones = []
for k in range( DRONE_COUNT ):
  drones.append( drone() )



#pygame.event.get(); pygame.mouse.get_rel()
#pygame.mouse.set_visible(0); pygame.event.set_grab(1)

seconds = float(0)

while seconds < 600:
  seconds += float(ROOMBA_FRAME_DELAY) / float(1000)
  
  for event in pygame.event.get():
    if event.type == pygame.QUIT: pygame.quit(); sys.exit()
    if event.type == pygame.KEYDOWN:
      if event.key == pygame.K_ESCAPE: pygame.quit(); sys.exit()
    



  for k in range(GROUND_ROBOT_COUNT):
    for j in range( GROUND_ROBOT_COUNT ):
      if j != k:
        groundRobots[k].collision(groundRobots[j])
    for j in range( OBSTACLE_ROBOT_COUNT ):
      groundRobots[k].collision(obstacleRobots[j])
    groundRobots[k].iterate()

  for k in range(OBSTACLE_ROBOT_COUNT):
    for j in range( OBSTACLE_ROBOT_COUNT ):
      obstacleRobots[k].collision(obstacleRobots[j])
    obstacleRobots[k].iterate()


  for k in range( DRONE_COUNT ):
    groundBotsPos, groundBotsStillIn, groundBotsRotDeg, obstacleBotsPos = [], [], [], []
    for groundBot in groundRobots:
      groundBotsPos.append( groundBot.pos ); groundBotsStillIn.append( groundBot.stillIn ); groundBotsRotDeg.append( groundBot.rotDeg )
    for obstacleBot in obstacleRobots: obstacleBotsPos.append( obstacleBot.pos )

    drones[k].Sim9AI( groundBotsPos, groundBotsRotDeg, groundBotsStillIn, obstacleBotsPos, int( DRONE_FRAME_RATE * ROOMBA_FRAME_DELAY / 1000 + 0.5 ) )

    for j in range( GROUND_ROBOT_COUNT ):
      if math.pow( ROOMBA_RADIUS * PIXELS_PER_METER, 2) > math.pow( drones[k].X - groundRobots[j].pos[0], 2) + math.pow( drones[k].Y - groundRobots[j].pos[1], 2) and drones[k].Z < ROOMBA_TAPPABLE_HEIGHT:
        groundRobots[j].tap()
    for j in range( OBSTACLE_ROBOT_COUNT ):
      if math.pow( (OBSTACLE_RADIUS + DRONE_RADIUS) * PIXELS_PER_METER, 2) > math.pow( drones[k].X - obstacleRobots[j].pos[0], 2) + math.pow( drones[k].Y - obstacleRobots[j].pos[1], 2) and drones[k].Z < obstacleRobots[j].height:
        seconds = 999999999






  screen.fill((0,0,0))

  for k in range(20):
    pygame.draw.line(screen,(255,255,255),(60+(20*k),40),(60+(20*k),440),1)
    pygame.draw.line(screen,(255,255,255),(40,60+(20*k)),(440,60+(20*k)),1)
  pygame.draw.line(screen,(255,255,255),(40,40),(40,440),1)
  pygame.draw.line(screen,(255,0,0),(40,440),(440,440),1)
  pygame.draw.line(screen,(255,255,255),(440,440),(440,40),1)
  pygame.draw.line(screen,(0,255,0),(440,40),(40,40),1)
  for k in range(GROUND_ROBOT_COUNT):
    pygame.draw.circle(screen, (255,255,255), (int(groundRobots[k].pos[0] + 0.5),int(groundRobots[k].pos[1] + 0.5)), int(ROOMBA_RADIUS * PIXELS_PER_METER + 0.5), 0)
  for k in range(OBSTACLE_ROBOT_COUNT):
    pygame.draw.circle(screen, (255,255,0), (int(obstacleRobots[k].pos[0] + 0.5),int(obstacleRobots[k].pos[1] + 0.5)), int(ROOMBA_RADIUS * PIXELS_PER_METER + 0.5), 0)
  for k in range( DRONE_COUNT ):
    pygame.draw.circle(screen, (255,0,255), (int(drones[k].X + 0.5),int(drones[k].Y + 0.5)), int( DRONE_RADIUS * PIXELS_PER_METER + 0.5 ), 0)






  myfont = pygame.font.SysFont("monospace", 15)
  label = myfont.render( str(seconds), 1, (255,255,0))
  screen.blit(label, (METERS_PER_WINDOW*PIXELS_PER_METER - 50, METERS_PER_WINDOW*PIXELS_PER_METER - 20))

  time.sleep( (ROOMBA_FRAME_DELAY / float(1000)) / float(RUN_SPEED) )

  
  pygame.display.flip()

  key = pygame.key.get_pressed()


while True:
  time.sleep(0.1)

  for event in pygame.event.get():
    if event.type == pygame.QUIT: pygame.quit(); sys.exit()
    if event.type == pygame.KEYDOWN:
      if event.key == pygame.K_ESCAPE: pygame.quit(); sys.exit()

