# Python simulation of International Aerial Robotics Competition, Mission 7
# by Tanner Winkelman

# HOW TO RUN:
#   On a computer in the Computer Science buiding, double click on this file.
#   A window of the Python GUI opens.
#   Click Run>Run Module.
#   It takes 5 seconds to load.

# You can also launch this program from a command line
#   with the command "python [filename]"



"""
Sim9AI is derived from the JavaScript Simulation "Sim9.html".
Tracker is designed to be more realistic, guiding ground robots across the green line one at a time.
The AIs are passed an array of ground robot and an array of obstacle robots.
The AIs use member variables on the drone object for internal storage and for output.
The AI output is in the form of chaning the XSpeed, YSpeed and ZSpeed member variables
(which should have been named XVelocity, YVelocity, and ZVelocity).
"""


import pygame, sys, math, random, time

RUN_SPEED = 20 # times.  Note: limited by hardware
FRAMES_TO_SKIP = 0 # the number of roomba frames that pass for every time the simulation is drawn

# choose which AI runs, there are 2 of them
AI = "Sim9AI"
# AI = "Tracker"

# Sim9AI weights, not on same scale
#these weights are applied after robots that are too close to obstacle robots have been eliminated and robots that are already pointing in a good direction have been eliminated
OB_WEIGHT = 50
OB_PROJECT_ROOMBA_FRAMES = 60 # 60f / 5fps * 0.33m/s == 4 meters
CLOSENESS_WEIGHT = 100000
CLOSENESS_WIDTH = 10 # pixels
GREENLINE_WIEGHT = 0.1
GOOD_DIRECTION_WIDTH = 100 # degrees; the range is half of this in each direction from the direction to the target point in the top of the window
NO_TARGET_POINT = 12 # seconds; after this many seconds sim9AI no will longer leave its roomba

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
OBSTACLE_CAMERA_ALERT_THRESHOLD = 1 # meter
DRONE_FRAME_RATE = 60 # frames per second
DRONE_PIXEL_SPEED = 2 # the drone's agility, treated as a safety factor for slowing down to hit targets in Sim9AI()
DRONE_ACCEL_NUMBER = 0.005 # I have no idea what unit this is.  This is from the JavaScript simulation.
DRONE_OBSTACLE_DODGE_ACCEL = 0.001 # pixels per drone frame per drone frame
DRONE_VERTICAL_ACCEL = 0.015 # pixels per drone frame per drone frame, it think; this is the drone's acceleration both up and down
TAP_TARGET_Z = -5 # pixels
HOVER_TARGET_Z = 0.25 * PIXELS_PER_METER # pixels
BOUNCE_SPEED_RETENTION = 0.5 # the fraction of the velocity the drone hit the ground with that is put into the upward velocity the drone bounces up with
DRONE_CAMERA_WIDTH_DEG = 40 # degrees
DRONE_CAMERA_CIRCLE_DETECTION_RADIUS = 1 # meter
DRONE_COUNT = 1
DRONE_RADIUS = 0.18 # meters
ROOMBA_HEIGHT = 0.1 # meters
ROOMBA_TAPPABLE_HEIGHT = 0.12 # meters from floor
OBSTACLE_RADIUS = 0.05 # meters
OK_TO_LEAVE_THRESHOLD = 16 # of 24 meters
DIRECTION_LINE_LENGTH = 15 # pixels
PI = 3.14159265359 # don't change this value

# Description:  rotate2d() returns the pos (x,y) rotated about (0,0) by rad radians.
def rotate2d(pos,rad): x,y=pos; s,c = math.sin(rad),math.cos(rad); return x*c-y*s,y*c+x*s;

# Description:  distance2d() returns the distance between points pos1 and pos2
def distance2d(pos1,pos2): return math.sqrt( pow(pos1[0] - pos2[0], 2) + pow(pos1[1] - pos2[1], 2) )

# Description:  inBounds returns whether or not the pos it is passed is in the 20x20 grid.
def inBounds(pos): TWENTY_FOURTHS = METERS_PER_WINDOW * PIXELS_PER_METER / 24; return (pos[0] >= 2 * TWENTY_FOURTHS and pos[0] <= 22 * TWENTY_FOURTHS and pos[1] >= 2 * TWENTY_FOURTHS and pos[1] <= 22 * TWENTY_FOURTHS)

# Description: mod() mods floating point numbers.
def mod( n, d ):
  r = n
  if r > d:
    while r >= d:
      r -= d
  if r < 0:
    while r < 0:
      r += d
  return r


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



class roomba:
  def __init__(self,pos=[PIXELS_PER_METER*METERS_PER_WINDOW/2,PIXELS_PER_METER*METERS_PER_WINDOW/2],rotDeg=(0)):
    self.pos = [PIXELS_PER_METER*METERS_PER_WINDOW/2,PIXELS_PER_METER*METERS_PER_WINDOW/2]
    self.rotDeg = 0 # degrees

  def collision(self, otherRoomba ):
    collision = False
    if inBounds( self.pos ) and inBounds( otherRoomba.pos ):
      distance = distance2d( self.pos, otherRoomba.pos )
      collision = distance < 2 * ROOMBA_RADIUS * PIXELS_PER_METER
      #print(type(self),type(otherRoomba), collision)
      if collision:
        angle = math.atan2( otherRoomba.pos[1] - self.pos[1], otherRoomba.pos[0] - self.pos[0] )
        #if distance > distance2d( ( self.pos[0] + selfCoordChange[0], self.pos[1] + selfCoordChange[1] ), otherRoomba.pos ):
        selfCoordChange = rotate2d((GROUND_ROBOT_SPEED*PIXELS_PER_METER*ROOMBA_FRAME_DELAY/1000,0),self.rotDeg*PI/180)
        if distance > distance2d( (self.pos[0] + selfCoordChange[0], self.pos[1] + selfCoordChange[1]), otherRoomba.pos ):
          self.bump()
        otherRoombaCoordChange = rotate2d((GROUND_ROBOT_SPEED*PIXELS_PER_METER*ROOMBA_FRAME_DELAY/1000,0),otherRoomba.rotDeg*PI/180)
        if distance > distance2d( (otherRoomba.pos[0] + otherRoombaCoordChange[0], otherRoomba.pos[1] + otherRoombaCoordChange[1]), self.pos ):
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
    if( self.turningCountdown <= 0 ):
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
    self.height = height # meters
    self.go = True
  
  def bump(self):
    self.go = False
    coordChange = rotate2d((GROUND_ROBOT_SPEED*PIXELS_PER_METER*ROOMBA_FRAME_DELAY/1000,0),self.rotDeg*PI/180)
    # get bumped backward, to prevent roomba jams
    self.pos[0] -= coordChange[0] * OBSTACLE_ROBOT_BUMP_FACTOR
    self.pos[1] -= coordChange[1] * OBSTACLE_ROBOT_BUMP_FACTOR

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
  def __init__(self,X=(METERS_PER_WINDOW*PIXELS_PER_METER * 22/24),Y=(METERS_PER_WINDOW*PIXELS_PER_METER / 2),Z=(3 * PIXELS_PER_METER),XSpeed=(0),YSpeed=(0),ZSpeed=(0),rotDeg=(0),targetZ=(3*PIXELS_PER_METER),TargetRobot=(0),OkToLeave=(True),NextRotate=(20 * DRONE_FRAME_RATE),TargetRobotLastRoombaFrameRotDeg=(-1),TargetRobotRotSampleCounter=(-1),TargetRobotTurning=(False)):
    self.X=X
    self.Y=Y
    self.Z=Z
    self.XSpeed=XSpeed
    self.YSpeed=YSpeed
    self.ZSpeed=ZSpeed
    self.rotDeg=rotDeg
    self.targetZ=targetZ
    self.TargetRobot=TargetRobot
    self.OkToLeave=OkToLeave
    self.NextRotate=NextRotate
    self.TargetRobotLastRoombaFrameRotDeg=TargetRobotLastRoombaFrameRotDeg
    self.TargetRobotRotSampleCounter=TargetRobotRotSampleCounter
    self.TargetRobotTurning=TargetRobotTurning
  
  
  def pos(self):
    returnable = []
    returnable.append( self.X )
    returnable.append( self.Y )
    return returnable

  #self -> called object
  #groundRobotsPos    -> array of active ground robot's positions (x and y)
  #groundRobotsRotDeg -> array of directions of ground robots (degrees)
  #obstacleRobotsPos  -> array of active obstacle robot's positions (x and y)
  #time               -> number of times to run (defaults to 1)
  # Output is in the form of changing the x, y, and z speed member variables.  The other member variables are used by Sim9AI also.
  def Sim9AI( self, groundRobotsPos, groundRobotsRotDeg, obstacleRobotsPos, obstacleRobotsHeight = [], times = 1 ):
  
    #print( groundRobotsPos, groundRobotsRotDeg, groundRobotsStillIn, obstacleRobotsPos )
  
    for k in range( times ):

      try:
        
        self.iterate()
        if self.NextRotate <= 0:
          # 20 seconds, + 0.05 for those annoying roombas that lose synchrony
          self.NextRotate = (20.05 * DRONE_FRAME_RATE)
        self.NextRotate -= 1
        TargetRot = 270

        if self.TargetRobot >= len( groundRobotsPos ):
          self.OkToLeave = True
          self.TargetRobot = -1
          self.TargetRobotLastRoombaFrameRotDeg = -1
      
      
        obstacleDodge = -1
        for obsIndex in range(len(obstacleRobotsPos)):
          if obstacleRobotsHeight[obsIndex] >= self.Z and distance2d( (self.X,self.Y), obstacleRobotsPos[obsIndex] ) < PIXELS_PER_METER * OBSTACLE_CAMERA_ALERT_THRESHOLD:
            obstacleDodge = obsIndex
      
        if( obstacleDodge >= 0 ):
          self.OkToLeave = True
          self.TargetRobot = -1
          dxSign = -1 if self.X - obstacleRobotsPos[obstacleDodge][0] < 0 else 1
          dySign = -1 if self.Y - obstacleRobotsPos[obstacleDodge][1] < 0 else 1
          self.YSpeed = self.YSpeed + ( dySign * math.sqrt(DRONE_OBSTACLE_DODGE_ACCEL/(math.pow((self.X - obstacleRobotsPos[obstacleDodge][0])/(self.Y - obstacleRobotsPos[obstacleDodge][1]), 2) + 1)))
          self.XSpeed = self.XSpeed + ( dxSign * math.sqrt(DRONE_OBSTACLE_DODGE_ACCEL/(math.pow((self.Y - obstacleRobotsPos[obstacleDodge][1])/(self.X - obstacleRobotsPos[obstacleDodge][0]), 2) + 1)))
        
        
        PrevTargetRobot = self.TargetRobot
        #alert(DataArray[CurrentFrame].Drones[index].OkToLeave + "\n" + DataArray[CurrentFrame].Drones[index].TargetRobot);
        if(self.OkToLeave == True and self.NextRotate / DRONE_FRAME_RATE > NO_TARGET_POINT ):
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
              # priority #1, is the target roomba safely far from obstacle robots
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
    
          if( self.TargetRobot != PrevTargetRobot ):
            #sys.stdout.write("target robot:" + str(self.TargetRobot) + ",PrevTargetRobot:" + str(PrevTargetRobot) + " ")
            #sys.stdout.flush()
            self.TargetRobotLastRoombaFrameRotDeg = -1
            self.TargetRobotRotSampleCounter = -1
            self.TargetRobotTurning = False
        else:
          self.targetZ = 3 * PIXELS_PER_METER
      
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
          if ( TargetRobotRot > TargetRot + (GOOD_DIRECTION_WIDTH / 2) or TargetRobotRot < TargetRot - (GOOD_DIRECTION_WIDTH / 2) ):
            if math.pow( ROOMBA_RADIUS * PIXELS_PER_METER, 2) > math.pow( self.X - groundRobotsPos[self.TargetRobot][0], 2) + math.pow( self.Y - groundRobotsPos[self.TargetRobot][1], 2):
              if self.TargetRobotTurning:
                self.targetZ = HOVER_TARGET_Z
              else:
                self.targetZ = TAP_TARGET_Z
              
              if groundRobotsPos[self.TargetRobot][1] > METERS_PER_WINDOW * PIXELS_PER_METER * OK_TO_LEAVE_THRESHOLD / 24:
                self.OkToLeave = False
              else:
                self.OkToLeave = True
            else:
              self.targetZ = PIXELS_PER_METER * ( 3 - (3 / ( 1 + ( distance2d( ( groundRobotsPos[self.TargetRobot][0], groundRobotsPos[self.TargetRobot][1] ), ( self.X, self.Y ) ) / PIXELS_PER_METER ) ) ) )
          else:
            self.targetZ = HOVER_TARGET_Z
        else: # targetRobot < 0
          self.targetZ = 3 * PIXELS_PER_METER
        if ( self.Z < (self.targetZ) - (self.ZSpeed/DRONE_VERTICAL_ACCEL) ):
          self.ZSpeed += DRONE_VERTICAL_ACCEL
        if ( self.Z > (self.targetZ) - (self.ZSpeed/DRONE_VERTICAL_ACCEL) ):
          self.ZSpeed -= DRONE_VERTICAL_ACCEL
        
        if self.TargetRobot >= 0 and self.TargetRobot < len(groundRobotsPos):
          if self.TargetRobotRotSampleCounter < 0:
            if self.TargetRobotLastRoombaFrameRotDeg >= 0:
              roombaTurningThreshold = (360 * ROOMBA_TURN_SPEED * ROOMBA_FRAME_DELAY / 1000) / 2 # degrees per roomba frame
              self.TargetRobotTurning = roombaTurningThreshold < abs( mod(self.TargetRobotLastRoombaFrameRotDeg, 360) - mod(groundRobotsRotDeg[self.TargetRobot], 360) ) or roombaTurningThreshold < abs( mod(self.TargetRobotLastRoombaFrameRotDeg + 180, 360) - mod(groundRobotsRotDeg[self.TargetRobot] + 180, 360) )
            self.TargetRobotLastRoombaFrameRotDeg = mod( groundRobotsRotDeg[self.TargetRobot], 360 )
            self.TargetRobotRotSampleCounter = DRONE_FRAME_RATE * ROOMBA_FRAME_DELAY / 1000
          else:
            self.TargetRobotRotSampleCounter -= 1
        else:
          self.TargetRobotLastRoombaFrameRotDeg = -1
          self.TargetRobotRotSampleCounter = -1
          self.TargetRobotTurning = False
        #sys.stdout.write(str(self.TargetRobotTurning) + " " + str(self.TargetRobot) + "; ")
        #sys.stdout.flush()
          
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

  def Tracker( self, groundRobotsPos, groundRobotsRotDeg, obstacleRobotsPos, obstacleRobotsHeight = [], times = 1 ):
    
    for k in range( times ):
      
      self.iterate()
      
      obstacleDodge = -1
      for obsIndex in range(len(obstacleRobotsPos)):
        if obstacleRobotsHeight[obsIndex] >= self.Z and distance2d( (self.X,self.Y), obstacleRobotsPos[obsIndex] ) < PIXELS_PER_METER * OBSTACLE_CAMERA_ALERT_THRESHOLD:
          obstacleDodge = obsIndex
      
      if( obstacleDodge >= 0 ):
        self.OkToLeave = True
        self.TargetRobot = -1
        dxSign = -1 if self.X - obstacleRobotsPos[obstacleDodge][0] < 0 else 1
        dySign = -1 if self.Y - obstacleRobotsPos[obstacleDodge][1] < 0 else 1
        self.YSpeed = self.YSpeed + ( dySign * math.sqrt(DRONE_OBSTACLE_DODGE_ACCEL/(math.pow((self.X - obstacleRobotsPos[obstacleDodge][0])/(self.Y - obstacleRobotsPos[obstacleDodge][1]), 2) + 1)))
        self.XSpeed = self.XSpeed + ( dxSign * math.sqrt(DRONE_OBSTACLE_DODGE_ACCEL/(math.pow((self.Y - obstacleRobotsPos[obstacleDodge][1])/(self.X - obstacleRobotsPos[obstacleDodge][0]), 2) + 1)))
    
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
        if math.pow( ROOMBA_RADIUS * PIXELS_PER_METER, 2) > math.pow( self.X - groundRobotsPos[self.TargetRobot][0], 2) + math.pow( self.Y - groundRobotsPos[self.TargetRobot][1], 2):
          if TargetRobotRot > TargetRot + TRACKER_OKAY_ROT_WIDTH / 2 or TargetRobotRot < TargetRot - TRACKER_OKAY_ROT_WIDTH / 2:
            if self.TargetRobotTurning:
              self.targetZ = HOVER_TARGET_Z
            else:
              self.targetZ = TAP_TARGET_Z
          else:
            self.targetZ = HOVER_TARGET_Z
        else:
          self.targetZ = 3 * PIXELS_PER_METER
      else:
        self.targetZ = 3 * PIXELS_PER_METER
      if ( self.Z < (self.targetZ) - (self.ZSpeed/DRONE_VERTICAL_ACCEL) ):
        self.ZSpeed += DRONE_VERTICAL_ACCEL
      if ( self.Z > (self.targetZ) - (self.ZSpeed/DRONE_VERTICAL_ACCEL) ):
        self.ZSpeed -= DRONE_VERTICAL_ACCEL
      
      if self.TargetRobot >= 0 and self.TargetRobot < len(groundRobotsPos):
        if self.TargetRobotRotSampleCounter < 0:
          if self.TargetRobotLastRoombaFrameRotDeg >= 0:
            roombaTurningThreshold = (360 * ROOMBA_TURN_SPEED * ROOMBA_FRAME_DELAY / 1000) / 2 # degrees per roomba frame
            self.TargetRobotTurning = roombaTurningThreshold < abs( mod(self.TargetRobotLastRoombaFrameRotDeg, 360) - mod(groundRobotsRotDeg[self.TargetRobot], 360) ) or roombaTurningThreshold < abs( mod(self.TargetRobotLastRoombaFrameRotDeg + 180, 360) - mod(groundRobotsRotDeg[self.TargetRobot] + 180, 360) )
          self.TargetRobotLastRoombaFrameRotDeg = mod( groundRobotsRotDeg[self.TargetRobot], 360 )
          self.TargetRobotRotSampleCounter = DRONE_FRAME_RATE * ROOMBA_FRAME_DELAY / 1000
        else:
          self.TargetRobotRotSampleCounter -= 1
      else:
        self.TargetRobotLastRoombaFrameRotDeg = -1
        self.TargetRobotRotSampleCounter = -1
        self.TargetRobotTurning = False
        

  def iterate( self, times = 1 ):
    
    self.X += self.XSpeed
    self.Y += self.YSpeed
    self.Z += self.ZSpeed

    if self.Z < 0:
      self.ZSpeed = abs(self.ZSpeed) * BOUNCE_SPEED_RETENTION
      self.Z = 0
    #if self.TargetRobot >= 0 and self.TargetRobot < len( groundBotsPos ):
    #  if mod(math.atan2( self.Y - groundBotsPos[self.TargetRobot][1], self.X - groundBotsPos[self.TargetRobot][0] ) * 180 / PI - self.rotDeg + 180, 360 ) > 180:
    #    self.rotDeg -= 1
    #  else:
    #    self.rotDeg -= 1







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

obstacleCollision = False

MYFONT = pygame.font.SysFont("monospace", 15)

while seconds < 600 and not obstacleCollision:
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
      if j != k:
        obstacleRobots[k].collision(obstacleRobots[j])
    obstacleRobots[k].iterate()



  groundBotsPos, groundBotsRotDeg, obstacleBotsPos, obstacleBotsHeight = [], [], [], []
  groundBotCounter = 0
  for groundBot in groundRobots:
    if groundBot.stillIn:
      #  if DRONE_CAMERA_WIDTH_DEG > abs( math.atan2( groundBot.pos[1] - drones[k].Y, groundBot.pos[0] - drones[k].X ) * 180 / PI - drones[k].rotDeg ) or distance2d((drones[k].X,drones[k].Y),groundBot.pos) < DRONE_CAMERA_CIRCLE_DETECTION_RADIUS * PIXELS_PER_METER:
      groundBotsPos.append( groundBot.pos ); groundBotsRotDeg.append( groundBot.rotDeg )
  for obstacleBot in obstacleRobots:
    obstacleBotsPos.append( obstacleBot.pos )
    obstacleBotsHeight.append( obstacleBot.height )



  for k in range( DRONE_COUNT ):
    
    # THE CALL TO THE AI
    if( AI == "Sim9AI" ):
      drones[k].Sim9AI( groundBotsPos, groundBotsRotDeg, obstacleBotsPos, obstacleBotsHeight, int( DRONE_FRAME_RATE * ROOMBA_FRAME_DELAY / 1000 + 0.5 ) )
    elif( AI == "Tracker" ):
      drones[k].Tracker( groundBotsPos, groundBotsRotDeg, obstacleBotsPos, obstacleBotsHeight, int( DRONE_FRAME_RATE * ROOMBA_FRAME_DELAY / 1000 + 0.5 ) )

    for j in range( GROUND_ROBOT_COUNT ):
      if math.pow( ROOMBA_RADIUS * PIXELS_PER_METER, 2) > math.pow( drones[k].X - groundRobots[j].pos[0], 2) + math.pow( drones[k].Y - groundRobots[j].pos[1], 2) and drones[k].Z < ROOMBA_TAPPABLE_HEIGHT:
        groundRobots[j].tap()
    for j in range( OBSTACLE_ROBOT_COUNT ):
      if math.pow( (OBSTACLE_RADIUS + DRONE_RADIUS) * PIXELS_PER_METER, 2) > math.pow( drones[k].X - obstacleRobots[j].pos[0], 2) + math.pow( drones[k].Y - obstacleRobots[j].pos[1], 2) and drones[k].Z < obstacleRobots[j].height:
        obstacleCollision = True





  if frameSkip <= 0:
    screen.fill((0,0,0))


    pPm = PIXELS_PER_METER # pixels per meter
    for k in range(20):
      pygame.draw.line(screen,(255,255,255),(3*pPm+(pPm*k),2*pPm),(3*pPm+(pPm*k),22*pPm),1)
      pygame.draw.line(screen,(255,255,255),(2*pPm,3*pPm+(pPm*k)),(22*pPm,3*pPm+(pPm*k)),1)
    pygame.draw.line(screen,(255,255,255),(2*pPm,2*pPm),(2*pPm,22*pPm),1)
    pygame.draw.line(screen,(255,0,0),(2*pPm,22*pPm),(22*pPm,22*pPm),1)
    pygame.draw.line(screen,(255,255,255),(22*pPm,22*pPm),(22*pPm,2*pPm),1)
    pygame.draw.line(screen,(0,255,0),(22*pPm,2*pPm),(2*pPm,2*pPm),1)
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
      label = MYFONT.render( str(drones[k].TargetRobot), 1, (255,255,0))
      screen.blit(label, (METERS_PER_WINDOW*PIXELS_PER_METER - 150 - (20 * k), METERS_PER_WINDOW*PIXELS_PER_METER - 20))
      #pygame.draw.line(screen, (240,230,180), (int(drones[k].X),int(drones[k].Y)), (int(drones[k].X + (1234 * math.cos((drones[k].rotDeg + DRONE_CAMERA_WIDTH_DEG) * PI / 180))),int(drones[k].Y + (1234 * math.sin((drones[k].rotDeg + DRONE_CAMERA_WIDTH_DEG) * PI / 180)))), 1)
      #pygame.draw.line(screen, (240,230,180), (int(drones[k].X),int(drones[k].Y)), (int(drones[k].X + (1234 * math.cos((drones[k].rotDeg - DRONE_CAMERA_WIDTH_DEG) * PI / 180))),int(drones[k].Y + (1234 * math.sin((drones[k].rotDeg - DRONE_CAMERA_WIDTH_DEG) * PI / 180)))), 1)
    label = MYFONT.render( "TargetRobot:", 1, (255,255,0))
    screen.blit(label, (METERS_PER_WINDOW*PIXELS_PER_METER - 200 - (20 * DRONE_COUNT), METERS_PER_WINDOW*PIXELS_PER_METER - 20))

    for grpi in range(len(groundBotsPos)):
      label = MYFONT.render( str(grpi), 1, (255,255,0))
      screen.blit(label, (groundBotsPos[grpi][0], groundBotsPos[grpi][1] - 20))
      groundBotCounter += 1




    if not obstacleCollision:
      label = MYFONT.render( "Seconds:", 1, (255,255,0))
      screen.blit(label, (METERS_PER_WINDOW*PIXELS_PER_METER - 90, METERS_PER_WINDOW*PIXELS_PER_METER - 20))
      label = MYFONT.render( str(seconds), 1, (255,255,0))
      screen.blit(label, (METERS_PER_WINDOW*PIXELS_PER_METER - 40, METERS_PER_WINDOW*PIXELS_PER_METER - 20))
    else: # obstacle collision
      label = MYFONT.render( "Obstacle Collision", 1, (255,255,0))
      screen.blit(label, (METERS_PER_WINDOW*PIXELS_PER_METER - 100, METERS_PER_WINDOW*PIXELS_PER_METER - 20))
    
    

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

