/*
   @author: jstuder

*/
#include <Servo.h>
#include <SoftwareSerial.h>

const byte rxPin = 3;
const byte txPin = 4;

SoftwareSerial upBoard (rxPin, txPin);  //Creating a software serial for the Arduino Nano

//the servo to counter the pitch of the drone
Servo pitch, roll;
const int PARALLEL = 50; //45 allows for 30 degrees of + and - with ease
const int PERPENDICULAR = PARALLEL + 90;

//A constant to control the speed of angle change, lower is faster
const int SPEED = 10;

/*
   @parameters
    byte angle -> the angle you want to go to
    int speed -> the delay between a move of 1 degree (lower is faster)
*/
void goToAngle(byte pitchAngle, byte rollAngle, int speed);

/*
   @parameters
   bool isPLL -> true if gimbal needs to be parallel to the ground, false if it
                 needs to be perpendicular
   int xAdj -> pass a positive or negative value to adjust the left-right axis of the gimbal
               (modifies servo2)
   int yAdj -> pass a positive or negative value to adjust the forward-backward axis of the gimbal
               (modifies pitch)
*/
void mntn(bool isPLL, byte rAdj, byte pAdj);

void setup() {

  //assigns analog pin 0 to output
  pinMode(A0, OUTPUT);
  pinMode(A2, OUTPUT);
  pitch.attach(A0); //attaches pitch to analog 0
  roll.attach(A2);

  delay(1000); //gives a little leeway before start up. Not essential
  goToAngle(50, 40, SPEED); //sets the gimbal to parallel as default

  //enables serial
  //SoftwareSerial upBoard (rxPin, txPin);
  upBoard.begin(115200);
  Serial.begin(115200);
  Serial.println("Ready");
   
  upBoard.println("hello??");
  
}

void loop()
{
 
  //used to track parallel or perpendicular
  static boolean isParallel = true;
  if (upBoard.available())
  {
    //reads as a string to allow for 'change' to be a condition
    String in;
    int i = 0;
    in = upBoard.readString();
    Serial.println(in);
    char inC[in.length()];
    in.toCharArray(inC, in.length());

    //allows yChange to default to the previous value. If input is invalid the gimbal doesn't move.
    byte pChange = pitch.read();
    byte rChange = roll.read();
    //if 'change' is sent, allows the gimbal to switch orientation
    if (inC[0] == 's')
    {
      isParallel = !isParallel;
      Serial.println("CHANGE"); //debugging purposes
      pChange = 0; //allows the change to be precise. on change, you should be hovering
      rChange = 0;
    }
    else
    {
      String rString = "";
      String pString = "";
      int count = 0;
      do
      {
        rString += inC[count];
        count++;
      } while (inC[count] != ' ');

      do
      {
        pString += inC[count];
        count++;
      } while (inC[count] != ' ');

      rChange = rString.toInt();
      pChange = pString.toInt();

    }
    Serial.print(rChange); //debugging purposes
    Serial.print(" ");
    Serial.println(pChange);
    //calls the "maintain" function
    mntn(isParallel, rChange, pChange);
  }
 
}

void mntn(bool isPLL, byte rAdj, byte pAdj)
{
  //sets the new angle equal to the change from the old angle
  //runs if the gimbal needs to be parallel to the ground
  rAdj += 40;
  if (isPLL)
  {
    Serial.println("PLL"); //debugging purposes
    pAdj += PARALLEL;
    goToAngle(pAdj, rAdj, SPEED);
  }
  else if (!isPLL)
  {
    Serial.println("PERP"); //debugging purposes
    pAdj += PERPENDICULAR;
    goToAngle(pAdj, rAdj, SPEED);
  }
}


void goToAngle(byte pitchAngle, byte rollAngle, int speed)
{
  byte currentPAngle = pitch.read();
  byte currentRAngle = roll.read();
  if (pitchAngle > currentPAngle)
  {
    for (int i = currentPAngle; i < pitchAngle; i++)
    {
      pitch.write(i);
      //delay(speed);
    }
  }
  else
  {
    for (int i = currentPAngle; i > pitchAngle; i--)
    {
      pitch.write(i);
      //delay(speed);
    }
  }

  if (rollAngle > currentRAngle)
  {
    for (int i = currentRAngle; i < rollAngle; i++)
    {
      roll.write(i);
      //delay(speed);
    }
  }
  else
  {
    for (int i = currentRAngle; i > rollAngle; i--)
    {
      roll.write(i);
      //delay(speed);
    }
  }
}

