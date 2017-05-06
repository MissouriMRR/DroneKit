/*
 * @author: jstuder
 * 
 */
//from arduino #include <Servo.h>

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <termios.h>

// Open the Maestro's virtual COM port.
//const char * device = "\\\\.\\USBSER000";  // Windows, "\\\\.\\COM6" also works
const char * device = "/dev/ttyACM0";  // Linux
//const char * device = "/dev/cu.usbmodem00034567";  // Mac OS X
int serialControl = open(device, O_RDWR | O_NOCTTY);

//the servo to counter the pitch of the drone
int pitch = 0;
int roll = 1;
const int PARALLEL = 50; //45 allows for 30 degrees of + and - with ease
const int PERPENDICULAR = PARALLEL + 90; 

//A constant to control the speed of angle change, lower is faster
const int SPEED = 10;

// Gets the position of a Maestro channel.
// See the "Serial Servo Commands" section of the user's guide.
int maestroGetPosition(int serialControl, unsigned char channel)
{
  unsigned char command[] = {0x90, channel};
  if(write(serialControl, command, sizeof(command)) == -1)
  {
    perror("error writing");
    return -1;
  }
 
  unsigned char response[2];
  if(read(serialControl,response,2) != 2)
  {
    perror("error reading");
    return -1;
  }
 
  return response[0] + 256*response[1];
}

// Sets the target of a Maestro channel.
// See the "Serial Servo Commands" section of the user's guide.
// The units of 'target' are quarter-microseconds.
int maestroSetTarget(int serialControl, unsigned char channel, unsigned short target)
{
  unsigned char command[] = {0x84, channel, target & 0x7F, target >> 7 & 0x7F};
  if (write(serialControl, command, sizeof(command)) == -1)
  {
    perror("error writing");
    return -1;
  }
  return 0;
}

/*
 * @parameters
 *  int angle -> the angle you want to go to
 *  int speed -> the sleep between a move of 1 degree (lower is faster)
 */
void goToAngle(int pitchAngle, int rollAngle, int speed);

/*
 * @parameters
 * bool isPLL -> true if gimbal needs to be parallel to the ground, false if it
 *               needs to be perpendicular
 * int xAdj -> pass a positive or negative value to adjust the left-right axis of the gimbal
 *             (modifies servo2)
 * int yAdj -> pass a positive or negative value to adjust the forward-backward axis of the gimbal
 *             (modifies pitch)
 */
void mntn(bool isPLL, int rAdj, int pAdj);

void setup() 
{
  if (serialControl == -1)
  {
    perror(device);
    return 1;
  }
 
  #ifdef _WIN32
    _setmode(serialControl, _O_BINARY);
  #else
    struct termios options;
    tcgetattr(serialControl, &options);
    options.c_iflag &= ~(INLCR | IGNCR | ICRNL | IXON | IXOFF);
    options.c_oflag &= ~(ONLCR | OCRNL);
    options.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
    tcsetattr(serialControl, TCSANOW, &options);
  #endif  

  goToAngle(50, 40, SPEED); //sets the gimbal to parallel as default

}

bool sig_handler(int signo)
{
  if (signo == SIGUSR1)
  {
    return true;
  }
}

void main() 
{
  //used to track parallel or perpendicular
  static bool isParallel = true;
  
  if (Serial.available())
  {
    //allows yChange to default to the previous value. If input is invalid the gimbal doesn't move.
    int pChange = maestroGetPosition(serialControl, pitch);
    int rChange = maestroGetPosition(serialControl, roll);
    
    //if SIGUSR1 is sent, allows the gimbal to switch orientation
    if (sig_handler(SIGUSR1)) 
    {
      isParallel = !isParallel;
      p = 0;
      r = 0;
    }
    else
      mntn(isParallel, rChange, pChange);
  }
}

void mntn(bool isPLL, int rAdj, int pAdj)
{
  //sets the new angle equal to the change from the old angle
  //runs if the gimbal needs to be parallel to the ground
  rAdj += 40;
  if (isPLL)
  {
    pAdj += PARALLEL;
    goToAngle(pAdj, rAdj, SPEED);
  }
  else if (!isPLL)
  {
    pAdj += PERPENDICULAR;
    goToAngle(pAdj, rAdj, SPEED);
  }
}


void goToAngle(int pitchAngle, int rollAngle, int speed)
{
  int currentPAngle = maestroGetPosition(serialControl, pitch);
  int currentRAngle = maestroGetPosition(serialControl, roll);
  if(pitchAngle > currentPAngle)
  {
    for (int i = currentPAngle; i < pitchAngle; i++)
    {
      maestroSetTarget(serialControl, pitch, i);
      sleep(speed);
    }
  }
  else
  {
    for (int i = currentPAngle; i > pitchAngle; i--)
    {
      maestroSetTarget(serialControl, pitch, i);
      sleep(speed);
    }
  }

  if(rollAngle > currentRAngle)
  {
    for (int i = currentRAngle; i < rollAngle; i++)
    {
      maestroSetTarget(serialControl, roll, i);
      sleep(speed);
    }
  }
  else
  {
    for (int i = currentRAngle; i > rollAngle; i--)
    {
      maestroSetTarget(serialControl, roll, i);
      sleep(speed);
    }
  }
}

