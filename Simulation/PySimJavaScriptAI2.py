# Python simulation of International Aerial Robotics Competition, Mission 7
# by Tanner Winkelman

# HOW TO RUN:
#   On a computer in the Computer Science buiding, double click on this file.
#   A window of the Python GUI opens.
#   Click Run>Run Module.
#   It takes 5 seconds to load.

# You can also launch this program from a command line
#   with the command "python [filename]"


import pygame, sys, math, random, time


# AI weights, not on same scale
#these weights are applied after robots that are too close to obstacle robots have been eliminated and robots that are already pointing in a good direction have been eliminated
OB_WEIGHT = 50
OB_PROJECT_ROOMBA_FRAMES = 60 # 60f / 5fps * 0.33m/s == 4 meters
CLOSENESS_WEIGHT = 100000
CLOSENESS_WIDTH = 10 # pixels
GREENLINE_WIEGHT = 0.1
GOOD_DIRECTION_WIDTH = 100 # degrees; the range is half of this in each direction from the direction to the target point in the top of the window

RUN_SPEED = 20 # times.  Note: limited by hardware
FRAMES_TO_SKIP = 0 # the number of roomba frames that pass for every time the simulation is drawn

TRACKER_OKAY_ROT_WIDTH = 90 # degrees, the angle width of acceptable direction for a roomba the drone is tracking.  This value is divided by two when it is used.
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
OBSTACLE_ROBOT_BUMP_FACTOR = 0.1 # to prevent obstacle robots from getting stuck
DRONE_FRAME_RATE = 60 # frames per second
DRONE_PIXEL_SPEED = 2 # the drone's agility, treated as a safety factor for slowing down to hit targets in Sim9AI()
DRONE_ACCEL_NUMBER = 0.005 # I have no idea what unit this is.  This is from the JavaScript simulation.
DRONE_COUNT = 1
DRONE_RADIUS = 0.18 # meters
ROOMBA_HEIGHT = 0.1 # meters
ROOMBA_TAPPABLE_HEIGHT = 0.12 # meters from floor
OBSTACLE_RADIUS = 0.05 # meters
OK_TO_LEAVE_THRESHOLD = 16 # of 24 meters
DIRECTION_LINE_LENGTH = 15 # pixels
PI = 3.14159265359 # don't change this value


def rotate2d(pos,rad): x,y=pos; s,c = math.sin(rad),math.cos(rad); return x*c-y*s,y*c+x*s;
def distance2d(pos1,pos2): return math.sqrt( pow(pos1[0] - pos2[0], 2) + pow(pos1[1] - pos2[1], 2) )
def inBounds(pos): TWENTY_FOURTHS = METERS_PER_WINDOW * PIXELS_PER_METER / 24; return (pos[0] >= 2 * TWENTY_FOURTHS and pos[0] <= 22 * TWENTY_FOURTHS and pos[1] >= 2 * TWENTY_FOURTHS and pos[1] <= 22 * TWENTY_FOURTHS)

"""
  Desc: inPolygon() returns whether or not the point is in the polygon
  formed by the points in the arrayOfVertices.
  inPolygon() is not consistent if the point is on an edge or vertex.
  Pre: arrayOfVertices is a list of points, where each point is a
  list of two floats.  point is a list of two floats.
  Post: a bool is returned
"""
def inPolygon(point,arrayOfVertices):
  numVertices = len( arrayOfVertices )
  index = 0
  counter = 0
  while index < numVertices:
    vert1 = arrayOfVertices[index]
    vert2 = arrayOfVertices[(index + 1) % numVertices]
    if( (vert1[1] > point[1]) != (vert2[1] > point[1]) ):
      if( vert1[1] != vert2[1] ):
        if( point[0] - vert1[0] < (point[1] - vert1[1]) * (vert2[0] - vert1[0]) / (vert2[1] - vert1[1]) ):
          counter += 1
    index += 1
  return counter % 2 == 1


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
    coordChange = rotate2d((GROUND_ROBOT_SPEED*PIXELS_PER_METER*ROOMBA_FRAME_DELAY/1000,0),self.rotDeg*PI/180)
    self.pos[0] -= coordChange[0] * OBSTACLE_ROBOT_BUMP_FACTOR; self.pos[1] -= coordChange[1] * OBSTACLE_ROBOT_BUMP_FACTOR

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
  
  
  def pos(self):
    returnable = []
    returnable.append( self.X )
    returnable.append( self.Y )
    return returnable

  #self -> called object
  #groundRobotsPos    -> array of active ground robot's positions (x and y)
  #groundRobotsRotDeg -> array of directions of ground robots
  #obstacleRobotsPos  -> array of active obstacle robot's positions (x and y)
  #time               -> number of times to run (defaults to 1)
  def Sim9AI( self, groundRobotsPos, groundRobotsRotDeg, obstacleRobotsPos, times = 1 ):
  
    #print( groundRobotsPos, groundRobotsRotDeg, groundRobotsStillIn, obstacleRobotsPos )
  
    for k in range( times ):

      try:
        
        self.iterate()
        if self.NextRotate <= 0:
          self.NextRotate = (20 * DRONE_FRAME_RATE)
        self.NextRotate -= (DRONE_FRAME_RATE/12)
        TargetRot = 270

        if self.TargetRobot >= len( groundRobotsPos ):
          self.OkToLeave = True
          self.TargetRobot = -1
      
        PrevTargetRobot = self.TargetRobot
        #alert(DataArray[CurrentFrame].Drones[index].OkToLeave + "\n" + DataArray[CurrentFrame].Drones[index].TargetRobot);
        if(self.OkToLeave == True and self.NextRotate / DRONE_FRAME_RATE > 12 ):
          TempX = 0
          TempY = 0
          self.TargetRobot = -1
          TempScore = 0
          scores = []
          scoreBefore = 0 # for printing purposes
          for k in range( len( groundRobotsPos ) ):
            scores.append(0)
          for index2 in range( len( groundRobotsPos ) ):
            Rot = ( groundRobotsRotDeg[index2] % 360 + 360) if ( groundRobotsRotDeg[index2] % 360 < 0) else ( groundRobotsRotDeg[index2] % 360)
            TargetRot = 180 + (180 / PI) * math.atan2( groundRobotsPos[index2][1] - (METERS_PER_WINDOW * PIXELS_PER_METER * 2 / 24), groundRobotsPos[index2][0] - (METERS_PER_WINDOW * PIXELS_PER_METER / 2) );
            #print( index2, groundRobotsPos[index2][0], groundRobotsPos[index2][1], TargetRot )
            
            if (Rot < TargetRot - GOOD_DIRECTION_WIDTH / 2 or Rot > TargetRot + GOOD_DIRECTION_WIDTH / 2):
              # priority #1, is the target roomba safely far from
              SafeTarget = True;
              for index3 in range( OBSTACLE_ROBOT_COUNT ):
                A = groundRobotsPos[index2][0] - obstacleRobotsPos[index3][0]
                B = groundRobotsPos[index2][1] - obstacleRobotsPos[index3][1]
                if(math.sqrt(math.pow(A, 2) + math.pow(B, 2)) / PIXELS_PER_METER < (ROOMBA_RADIUS + ROOMBA_RADIUS) / PIXELS_PER_METER + 1):
                  SafeTarget = False
              if not SafeTarget:
                scores[index2] = 0
              else:
                #priority #2, is the target roomba pointing out of bounds
                projectedPos = []
                projectedPos.append( groundRobotsPos[index2][0] + math.cos(groundRobotsRotDeg[index2] * PI / 180) * OB_PROJECT_ROOMBA_FRAMES * GROUND_ROBOT_SPEED*PIXELS_PER_METER*ROOMBA_FRAME_DELAY/1000 )
                projectedPos.append( groundRobotsPos[index2][1] + math.sin(groundRobotsRotDeg[index2] * PI / 180) * OB_PROJECT_ROOMBA_FRAMES * GROUND_ROBOT_SPEED*PIXELS_PER_METER*ROOMBA_FRAME_DELAY/1000 )
                #scoreBefore = scores[index2]
                if( not inBounds( projectedPos ) ):
                  scores[index2] += OB_WEIGHT
                #sys.stdout.write( "OB:" + str( scores[index2] - scoreBefore ) )
                #sys.stdout.flush()
                #priority #3, how close is the roomba to the green line
                #scoreBefore = scores[index2]
                scores[index2] += GREENLINE_WIEGHT * (METERS_PER_WINDOW*PIXELS_PER_METER - groundRobotsPos[index2][1])
                #sys.stdout.write( "GreenLine:" + str( scores[index2] - scoreBefore ) )
                #sys.stdout.flush()
                #priority #4, how close is the roomba to us (bell curve)
                #scoreBefore = scores[index2]
                scores[index2] += CLOSENESS_WEIGHT / (CLOSENESS_WIDTH + math.pow( distance2d( groundRobotsPos[index2], self.pos() ), 2 ) )
                #sys.stdout.write( "Closeness:" + str( scores[index2] - scoreBefore ) )
                #sys.stdout.flush()
                if( TempX == 0 or scores[index2] > TempScore ):
                  self.TargetRobot = index2
                  TempX = groundRobotsPos[index2][0]
                  TempY = groundRobotsPos[index2][1]
                  TempScore = scores[index2]
        
        TargetRot = 270
        
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


        if self.TargetRobot >= 0:
          TargetRobotRot = groundRobotsRotDeg[self.TargetRobot]
          TargetRobotRot = ( TargetRobotRot % 360 + 360) if ( TargetRobotRot % 360 < 0) else ( TargetRobotRot % 360)
          TargetRot = 180 + (180 / PI) * math.atan2( groundRobotsPos[self.TargetRobot][1] - (METERS_PER_WINDOW * PIXELS_PER_METER * 0), groundRobotsPos[self.TargetRobot][0] - (METERS_PER_WINDOW * PIXELS_PER_METER / 2) )
          if ( ( TargetRobotRot > TargetRot + GOOD_DIRECTION_WIDTH / 2 or TargetRobotRot < TargetRot - GOOD_DIRECTION_WIDTH / 2 ) and math.pow( ROOMBA_RADIUS * PIXELS_PER_METER, 2) > math.pow( self.X - groundRobotsPos[self.TargetRobot][0], 2) + math.pow( self.Y - groundRobotsPos[self.TargetRobot][1], 2) ):
            self.Z = ROOMBA_HEIGHT
            
            if groundRobotsPos[self.TargetRobot][1] > METERS_PER_WINDOW * PIXELS_PER_METER * OK_TO_LEAVE_THRESHOLD / 24:
              self.OkToLeave = False
            else:
              self.OkToLeave = True
            
          else:
            self.Z = 3 * PIXELS_PER_METER
      except Exception as e:
        print( "Error: TargetRobot:" , self.TargetRobot, "Message:", str(e) )


  #def idealTarget( self, robotPos, , nextRotation, isSafe = True):
    """
    d = distance2d( (self.XPos, self.YPos), robotPos)
    e1 = robotPos[1] - METERS_PER_WINDOW * PIXELS_PER_METER * 2/24
    e2 = abs(robotPos[0] - METERS_PER_WINDOW * PIXELS_PER_METER * 22/24)
    e3 = abs(robotPos[1] - METERS_PER_WINDOW * PIXELS_PER_METER * 22/24)
    e4 = robotPos[0] - METERS_PER_WINDOW * PIXELS_PER_METER * 2/24
    if GROUND_ROBOT_SPEED * nextRotation =
    """
    #Danger of Exit (lump w/ 3)
    #isSafe (use other function as parameter) (iterate through distance function w/ obst)
    #Distance to Goal (lump w/ 1)
    #Distance from Drone (complete, need to implement)

  def Tracker( self, groundRobotsPos, groundRobotsRotDeg, obstacleRobotsPos, times = 1 ):
    
    for k in range( times ):
    
      self.iterate()
    
      closestDist = METERS_PER_WINDOW * PIXELS_PER_METER
      closestDistIndex = -1
      dist = 0
      for index in range(len(groundRobotsPos)):
        dist = distance2d( (self.X, self.Y), groundRobotsPos[index] )
        if dist < closestDist:
          SafeTarget = True
          for obstaclePos in obstacleRobotsPos:
            if SafeTarget and distance2d( obstaclePos, groundRobotsPos[index] ) / PIXELS_PER_METER < (ROOMBA_RADIUS + ROOMBA_RADIUS) / PIXELS_PER_METER + 1:
              SafeTarget = False
          if SafeTarget:
            closestDistIndex = index
            closestDist = dist
      self.TargetRobot = closestDistIndex
      self.OkToLeave = False
      
      
      TargetX = 0
      TargetY = 0
      
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
      
      XSpeed = 0
      YSpeed = 0
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
      
      if(XPos < TargetX):
        self.XSpeed += DRONE_PIXEL_SPEED * DRONE_ACCEL_NUMBER
      else:
        self.XSpeed += DRONE_PIXEL_SPEED * -DRONE_ACCEL_NUMBER
    
      if(YPos < TargetY):
        self.YSpeed = self.YSpeed + DRONE_PIXEL_SPEED * DRONE_ACCEL_NUMBER
      else:
        self.YSpeed = self.YSpeed + DRONE_PIXEL_SPEED * -DRONE_ACCEL_NUMBER
  
      if self.TargetRobot >= 0:
        TargetRobotRot = groundRobotsRotDeg[self.TargetRobot]
        TargetRobotRot = ( TargetRobotRot % 360 + 360) if ( TargetRobotRot % 360 < 0) else ( TargetRobotRot % 360)
        TargetRot = 180 + (180 / PI) * math.atan2( groundRobotsPos[self.TargetRobot][1] - (METERS_PER_WINDOW * PIXELS_PER_METER * 0), groundRobotsPos[self.TargetRobot][0] - (METERS_PER_WINDOW * PIXELS_PER_METER / 2) )
        if ( ( TargetRobotRot > TargetRot + TRACKER_OKAY_ROT_WIDTH / 2 or TargetRobotRot < TargetRot - TRACKER_OKAY_ROT_WIDTH / 2 ) and math.pow( ROOMBA_RADIUS * PIXELS_PER_METER, 2) > math.pow( self.X - groundRobotsPos[self.TargetRobot][0], 2) + math.pow( self.Y - groundRobotsPos[self.TargetRobot][1], 2) ):
          self.Z = ROOMBA_HEIGHT
      
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

frameSkip = 0

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
    groundBotsPos, groundBotsRotDeg, obstacleBotsPos = [], [], []
    for groundBot in groundRobots:
      if groundBot.stillIn:
        groundBotsPos.append( groundBot.pos ); groundBotsRotDeg.append( groundBot.rotDeg )
    for obstacleBot in obstacleRobots: obstacleBotsPos.append( obstacleBot.pos )

    
    drones[k].Sim9AI( groundBotsPos, groundBotsRotDeg, obstacleBotsPos, int( DRONE_FRAME_RATE * ROOMBA_FRAME_DELAY / 1000 + 0.5 ) )

    for j in range( GROUND_ROBOT_COUNT ):
      if math.pow( ROOMBA_RADIUS * PIXELS_PER_METER, 2) > math.pow( drones[k].X - groundRobots[j].pos[0], 2) + math.pow( drones[k].Y - groundRobots[j].pos[1], 2) and drones[k].Z < ROOMBA_TAPPABLE_HEIGHT:
        groundRobots[j].tap()
    for j in range( OBSTACLE_ROBOT_COUNT ):
      if math.pow( (OBSTACLE_RADIUS + DRONE_RADIUS) * PIXELS_PER_METER, 2) > math.pow( drones[k].X - obstacleRobots[j].pos[0], 2) + math.pow( drones[k].Y - obstacleRobots[j].pos[1], 2) and drones[k].Z < obstacleRobots[j].height:
        seconds = 999999999





  if frameSkip <= 0:
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
      directionLineEndPoint = []
      directionLineEndPoint += rotate2d((DIRECTION_LINE_LENGTH,0),groundRobots[k].rotDeg * PI / 180)
      directionLineEndPoint[0] += groundRobots[k].pos[0]
      directionLineEndPoint[1] += groundRobots[k].pos[1]
      pygame.draw.line(screen,(255,255,255),(int(groundRobots[k].pos[0] + 0.5),int(groundRobots[k].pos[1] + 0.5)),(int(directionLineEndPoint[0] + 0.5),int(directionLineEndPoint[1] + 0.5)),1)
    for k in range(OBSTACLE_ROBOT_COUNT):
      pygame.draw.circle(screen, (255,255,0), (int(obstacleRobots[k].pos[0] + 0.5),int(obstacleRobots[k].pos[1] + 0.5)), int(ROOMBA_RADIUS * PIXELS_PER_METER + 0.5), 0)
    for k in range( DRONE_COUNT ):
      pygame.draw.circle(screen, (255,0,255), (int(drones[k].X + 0.5),int(drones[k].Y + 0.5)), int( DRONE_RADIUS * PIXELS_PER_METER + 0.5 ), 0)
      pygame.draw.circle(screen, (240,230,180), (int(drones[k].X + 0.5),int(drones[k].Y + 0.5 - drones[k].Z)), int( DRONE_RADIUS * PIXELS_PER_METER + 0.5 ), 0)






    myfont = pygame.font.SysFont("monospace", 15)
    label = myfont.render( str(seconds), 1, (255,255,0))
    screen.blit(label, (METERS_PER_WINDOW*PIXELS_PER_METER - 50, METERS_PER_WINDOW*PIXELS_PER_METER - 20))

    pygame.display.flip()

    frameSkip = FRAMES_TO_SKIP
  else:
    frameSkip = frameSkip - 1

  time.sleep( (ROOMBA_FRAME_DELAY / float(1000)) / float(RUN_SPEED) )

  


  key = pygame.key.get_pressed()


while True:
  time.sleep(0.1)

  for event in pygame.event.get():
    if event.type == pygame.QUIT: pygame.quit(); sys.exit()
    if event.type == pygame.KEYDOWN:
      if event.key == pygame.K_ESCAPE: pygame.quit(); sys.exit()

