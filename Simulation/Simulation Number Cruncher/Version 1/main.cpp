/*
  Programmer: Lucas Winkelman & Tanner Winkelman
  File: main.cpp
  Purpose: To simulate matches with a varying drone heuristic
*/

#include <windows.h>
#include <gdiplus.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
using namespace Gdiplus;
using namespace std;

// Frames per second
const float FRAME_RATE = 120;

// The width and height in pixels of the arena (the drawing space for the field, but not the field)
const float ARENA_WIDTH = 650;
const float ARENA_HEIGHT = 650;

// A gdiplus color object for the outline of the arena
const Color ARENA_EDGE_COLOR = Color(255, 200, 200, 200);

// The space in the window to the left and top of the arena
const int ARENA_LEFT_MARGIN = 10;
const int ARENA_TOP_MARGIN = 10;

// The number of ground robots, obstacle robots, and drones
const int GROUND_ROBOT_NUMBER = 10;
const int OBSTACLE_ROBOT_NUMBER =  4;
const int DRONE_NUMBER = 1;

// The length in minutes of a match
const int MATCH_LENGTH = 10;

const int SECONDS_IN_MINUTE = 60;

// Number of margins around the field
const int FIELD_MARGIN_COUNT = 2;

// The size in meters of each margin
const int FIELD_MARGIN_SIZE = 2;

// The size of the field in meters
const int FIELD_METER_WIDTH = 20;

const int CENTI_IN_METER = 100;

// The size of the field's tape in centimeters
const float TAPE_CM_WIDTH = 7;

const float TO_RADIANS = 0.0174533;
const float TO_DEGREES = 57.29580;
const float DEGREES_IN_CIRCLE = 360;
const float PI = 3.14159265359;

const float METER_PIXEL_DIST = ARENA_HEIGHT / (FIELD_METER_WIDTH + (FIELD_MARGIN_SIZE * FIELD_MARGIN_COUNT));

// Height and width in pixels of the field
const float FIELD_HEIGHT = (METER_PIXEL_DIST * FIELD_METER_WIDTH);
const float FIELD_WIDTH = FIELD_HEIGHT;

// X and Y pixel position of the field in the arena
const float FIELD_X = (ARENA_WIDTH - FIELD_WIDTH) / 2;
const float FIELD_Y = (ARENA_HEIGHT - FIELD_HEIGHT) / 2;

// The field's tape's width in pixels
const float TAPE_WIDTH = TAPE_CM_WIDTH / CENTI_IN_METER * METER_PIXEL_DIST;

// The obstacle robot's difference in rotation from its rotation from with the center of the field
const float OBSTACLE_ROBOT_TURN_AMOUNT = 90;

// The radius of an obstacle robot's path around the center
const float OBSTACLE_ROBOT_PATH_RADIUS = 5;

// The drone's diameter in meters
const float DRONE_DIA = 0.3;

// The drone's argb color
const Color DRONE_COLOR = Color(255, 255, 255, 0);

// The field's tape's argb color
const Color TAPE_COLOR = Color(255, 0, 0, 0);

// The ground robots' color
const Color GROUND_ROBOT_COLOR = Color(255, 255, 255, 255);

// The ground robots' diameter in meters
const float GROUND_ROBOT_DIA = 0.353;

// The obstacle robots' diameter in meters
const float OBSTACLE_ROBOT_DIA = 0.353;

// The ground robots' color
const Color OBSTACLE_ROBOT_COLOR = Color(255, 200, 200, 200);

// The arena's background color
const Color ARENA_BACKGROUND_COLOR = Color(255, 150, 150, 150);

// The color red ground robots' direction pointers
const Color RED_ROBOT_COLOR = Color(255, 255, 0, 0);

// The color blue ground robots' direction pointers
const Color BLUE_ROBOT_COLOR = Color(255, 0, 0, 255);

// The direction pointer's length
const float DIRECTION_POINTER_LENGTH = 1;

// The ground robots' speed in m/s
const float GROUND_ROBOT_SPEED = 0.33;

// The obstacle robots' speed in m/s
const float OBSTACLE_ROBOT_SPEED = 0.33;

// The time it takes an obstacle robot to do a full circle in seconds
const float OBSTACLE_ROBOT_FULL_CIRLE_TIME = 2 * OBSTACLE_ROBOT_PATH_RADIUS * PI / OBSTACLE_ROBOT_SPEED;

// The obstacle robots' speed in pixels
const float OBSTACLE_ROBOT_PIXEL_SPEED = OBSTACLE_ROBOT_SPEED * FIELD_HEIGHT / FIELD_METER_WIDTH / FRAME_RATE;

// The amount of time of a match in seconds
const float TIME_LIMIT = SECONDS_IN_MINUTE * MATCH_LENGTH;

// The number of frames in one match
const int FRAME_LIMIT = FRAME_RATE * TIME_LIMIT;

// The ground robots' speed in pixels
const float GROUND_ROBOT_PIXEL_SPEED = GROUND_ROBOT_SPEED * FIELD_HEIGHT / FIELD_METER_WIDTH / FRAME_RATE;

// The numbers for the various turning modes
const int FORWARD_MODE = 0;
const int REVERSE_MODE = 1;
const int NOISE_MODE = 2;
const int COLLISION_MODE = 3;
const int TAP_MODE = 4;

// The rotation in degrees of reverse and noise modes
const float REVERSE_DURATION = 2.150;
const float NOISE_MAX_ROTATION = 20;
const float NOISE_MIN_ROTATION = 0;

// The interval time between each reverse and noise action
const float REVERSE_INTERVAL = 20;
const float NOISE_INTERVAL = 5;

// The degrees turned when the robot reverses
const float REVERSE_DEGREES = 180;

// The degrees turned when the robot collides
const float COLLISION_DEGREES = 180;

// The speed the robot turns at in degrees per second
const float GROUND_ROBOT_TURN_SPEED = REVERSE_DEGREES / REVERSE_DURATION;

// IDs for the two ground robot colors
const int RED_ID = 0;
const int BLUE_ID = 1;



struct Ground_Robot_Data
{
    float X;
    float Y;
    float Rotation;
    int Color;
    int Frames_To_Turn;
    int Turn_Direction_Mult;
    bool Still_In;
    bool Scored;
    char Mode;
    int Last_Frame_Reversed;
    int Last_Frame_Noise;
    int Time_Left_To_Rotate;
};

struct Obstacle_Robot_Data
{
    float X;
    float Y;
    float Rotation;
};

struct Drone_Data
{
    float X;
    float Y;
    float X_Speed;
    float Y_Speed;
    int Target_Robot;
    bool Ok_To_Leave;
};

struct Game_Frame
{
    vector<Ground_Robot_Data> Ground_Robots;
    vector<Obstacle_Robot_Data> Obstacle_Robots;
    vector<Drone_Data> Drones;
};


// Desc: Initializes the first frame of a match
// Pre: Relies on global variables, especially Array_Of_Frames
// Post: The first frame is set up and we are ready to run the match (Array_Of_Frames[0] is set up)
void Init();

// Desc: The function that get called when the window gets a Windows Message.
//   This keeps the window responsive
// Pre: None (that we care about, the OS takes care of calling it the right way)
// Post: An action may be taken inside the function that may effect global variables or perform
//   an action on the window
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

// Desc: This function draws each from on the screen
// Pre: hWnd must be a handle to our main window, the global variable Current_Frame is the frame drawn
// Post: The frame is drawn to the screen
void DrawScreen(HWND hWnd);

// Desc: This function makes a DIB section that can be attached to a Device context
// Pre: hdc must be a Handle to the Device Context we want this bitmap to be compatible with
// Post: returns a Handle to a BITMAP that is compatible with the Device Context given and with the dimensions w and h
HBITMAP CreateDIBSectionFunction(HDC hdc, int W,  int H);

// Desc: This function runs a frame of the game and changes variables
// Pre: Array_Of_Frames and Current_Frame must exist
// Post: the values in Array_Of_Frames are changed
void Engine();


void Reload_Match();



void Resize_Arrays();



// We are pretty much forced to use global variables with the way a windows c++ program is set up :(
vector<Game_Frame> Array_Of_Frames(FRAME_LIMIT + 1);
int Current_Frame = 0;

INT WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, PSTR, INT iCmdShow)
{
    HWND                hWnd;
    MSG                 msg;
    WNDCLASS            wndClass;
    GdiplusStartupInput gdiplusStartupInput;
    ULONG_PTR           gdiplusToken;
    HDC  WindowHDC;
    RECT Window_Size_Rect;
    float Milli_From_Last_Frame = 0;

    srand(time(NULL));

    // Initialize GDI+.
    GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

    wndClass.style          = CS_HREDRAW | CS_VREDRAW;
    wndClass.lpfnWndProc    = WndProc;
    wndClass.cbClsExtra     = 0;
    wndClass.cbWndExtra     = 0;
    wndClass.hInstance      = hInstance;
    wndClass.hIcon          = LoadIcon(NULL, IDI_APPLICATION);
    wndClass.hCursor        = LoadCursor(NULL, IDC_ARROW);
    wndClass.hbrBackground  = (HBRUSH)GetStockObject(WHITE_BRUSH);
    wndClass.lpszMenuName   = NULL;
    wndClass.lpszClassName  = TEXT("GettingStarted");

    RegisterClass(&wndClass);

    hWnd = CreateWindow(
        TEXT("GettingStarted"),   // window class name
        TEXT("Getting Started"),  // window caption
        WS_OVERLAPPEDWINDOW,      // window style
        CW_USEDEFAULT,            // initial x position
        CW_USEDEFAULT,            // initial y position
        CW_USEDEFAULT,            // initial x size
        CW_USEDEFAULT,            // initial y size
        NULL,                     // parent window handle
        NULL,                     // window menu handle
        hInstance,                // program instance handle
        NULL);                    // creation parameters

    ShowWindow(hWnd, iCmdShow);

    WindowHDC = GetDC(hWnd);

    cout << "before init" << endl;
    Resize_Arrays();
    Init();
    cout << "made it 1";

    // The message loop
    while(TRUE)
    {

        while(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        if(msg.message == WM_QUIT)
            break;

        while(Current_Frame < FRAME_LIMIT)
        {
            Current_Frame++;
            Engine();
            //DrawScreen(hWnd);
        }
        //cout << "just finished" << endl;
        Init();
        Current_Frame = 0;
        //cout << "finished" << endl;
	}

   GdiplusShutdown(gdiplusToken);
   return msg.wParam;
}  // WinMain


// This function gets called in response to messages to the window
// It gets called after the DispatchMessage function in the message loop
LRESULT CALLBACK WndProc(HWND hWnd, UINT message,
   WPARAM wParam, LPARAM lParam)
{
    HDC          WindowHDC;
    PAINTSTRUCT  ps;

    switch(message)
    {
        // If the window needs to be repainted
        case WM_PAINT:
        {
            WindowHDC = BeginPaint(hWnd, &ps);
            DrawScreen(hWnd);
            EndPaint(hWnd, &ps);
            break;
        }

        // If the window's close (X) button has been clicked
        case WM_DESTROY:
        {
            PostQuitMessage(0);
            break;
        }

        // Otherwise do the default
        default:
        {
            return DefWindowProc(hWnd, message, wParam, lParam);
        }
    }
    return 0;
}



void DrawScreen(HWND hWnd)
{
    RECT Window_Size_Rect;
    HDC Window_HDC;

    // Get the window's width and height
    GetWindowRect(hWnd, &Window_Size_Rect);
    int Window_Width = Window_Size_Rect.right;
    int Window_Height = Window_Size_Rect.bottom;

    // Get a Handle to the Device Context of hWnd
    Window_HDC = GetDC(hWnd);

    //***** GDI and GDI+ stuff to get ready for drawing *****//
    HDC Back_Buffer_HDC = CreateCompatibleDC(Window_HDC);
    HBITMAP Back_Buffer_HBITMAP = CreateDIBSectionFunction(Back_Buffer_HDC, Window_Width, Window_Height);
    HBITMAP Original_Back_Buffer_HBITMAP = (HBITMAP)SelectObject(Back_Buffer_HDC, Back_Buffer_HBITMAP);

    // This is the object that "draws" stuff for us
    Graphics Back_Buffer_Drawer(Back_Buffer_HDC);


    //****** Now ready to draw *****//
    // Clear the back buffer to white
    Back_Buffer_Drawer.Clear(Color(255, 255, 255, 255));

    //***** Create the "arena" and its parts *****//
    HDC Arena_HDC = CreateCompatibleDC(Back_Buffer_HDC);
    HBITMAP Arena_HBITMAP = CreateDIBSectionFunction(Arena_HDC, ARENA_WIDTH, ARENA_HEIGHT);
    HBITMAP Original_Arena_HBITMAP = (HBITMAP)SelectObject(Arena_HDC, Arena_HBITMAP);
    Graphics Arena_Drawer(Arena_HDC);

    // Turn on anti-aliasing for Arena_Drawer
    Arena_Drawer.SetSmoothingMode(SmoothingModeAntiAlias);

    // Clear to white
    Arena_Drawer.Clear(ARENA_BACKGROUND_COLOR);

    // Draw an outline around the arena
    Pen Arena_Edge_Pen(ARENA_EDGE_COLOR);
    Arena_Drawer.DrawRectangle(&Arena_Edge_Pen, 0, 0, static_cast<int>(ARENA_WIDTH - 1), static_cast<int>(ARENA_HEIGHT - 1));

    // Draw the horizontal pieces tape
    Pen Tape_Pen(TAPE_COLOR, TAPE_WIDTH);
    for(int index = 0; index < FIELD_METER_WIDTH + 1; index++)
    {
        Arena_Drawer.DrawLine(&Tape_Pen, FIELD_X, FIELD_Y + (METER_PIXEL_DIST * index), FIELD_X + FIELD_WIDTH, FIELD_Y + (METER_PIXEL_DIST * index));
    }

    // Draw the vertical pieces of tape
    for(int index = 0; index < FIELD_METER_WIDTH + 1; index++)
    {
        Arena_Drawer.DrawLine(&Tape_Pen, FIELD_X + (METER_PIXEL_DIST * index), FIELD_Y, FIELD_X + (METER_PIXEL_DIST * index), FIELD_Y + FIELD_HEIGHT);
    }

    // Draw the ground robots
    SolidBrush Ground_Robot_Brush(GROUND_ROBOT_COLOR);
    Pen Red_Robot_Pen(RED_ROBOT_COLOR);
    Pen Blue_Robot_Pen(BLUE_ROBOT_COLOR);
    for(int index = 0; index < GROUND_ROBOT_NUMBER; index++)
    {
        float Temp_X = Array_Of_Frames[Current_Frame].Ground_Robots[index].X;
        float Temp_Y = Array_Of_Frames[Current_Frame].Ground_Robots[index].Y;
        Arena_Drawer.FillEllipse(&Ground_Robot_Brush, Array_Of_Frames[Current_Frame].Ground_Robots[index].X - ((GROUND_ROBOT_DIA * METER_PIXEL_DIST) / 2), Array_Of_Frames[Current_Frame].Ground_Robots[index].Y - ((GROUND_ROBOT_DIA * METER_PIXEL_DIST) / 2),
            GROUND_ROBOT_DIA * METER_PIXEL_DIST, GROUND_ROBOT_DIA * METER_PIXEL_DIST);

        // Draw direction pointer on robot
        if(Array_Of_Frames[Current_Frame].Ground_Robots[index].Color == RED_ID)
        {
            Arena_Drawer.DrawLine(&Red_Robot_Pen, Temp_X, Temp_Y,
                Temp_X + sin(Array_Of_Frames[Current_Frame].Ground_Robots[index].Rotation * TO_RADIANS) * DIRECTION_POINTER_LENGTH * METER_PIXEL_DIST,
                Temp_Y + cos(Array_Of_Frames[Current_Frame].Ground_Robots[index].Rotation * TO_RADIANS) * DIRECTION_POINTER_LENGTH * METER_PIXEL_DIST);
        }
        else
        {
            Arena_Drawer.DrawLine(&Blue_Robot_Pen, Temp_X, Temp_Y,
                Temp_X + sin(Array_Of_Frames[Current_Frame].Ground_Robots[index].Rotation * TO_RADIANS) * DIRECTION_POINTER_LENGTH * METER_PIXEL_DIST,
                Temp_Y + cos(Array_Of_Frames[Current_Frame].Ground_Robots[index].Rotation * TO_RADIANS) * DIRECTION_POINTER_LENGTH * METER_PIXEL_DIST);
        }
    }

    // Draw obstacle robots
    SolidBrush Obstacle_Robot_Brush(OBSTACLE_ROBOT_COLOR);
    Pen Obstacle_Robot_Pen(OBSTACLE_ROBOT_COLOR);
    for(int index = 0; index < OBSTACLE_ROBOT_NUMBER; index++)
    {
        float Temp_X = Array_Of_Frames[Current_Frame].Obstacle_Robots[index].X;
        float Temp_Y = Array_Of_Frames[Current_Frame].Obstacle_Robots[index].Y;
        Arena_Drawer.FillEllipse(&Obstacle_Robot_Brush, Array_Of_Frames[Current_Frame].Obstacle_Robots[index].X - ((OBSTACLE_ROBOT_DIA * METER_PIXEL_DIST) / 2), Array_Of_Frames[Current_Frame].Obstacle_Robots[index].Y - ((OBSTACLE_ROBOT_DIA * METER_PIXEL_DIST) / 2),
            OBSTACLE_ROBOT_DIA * METER_PIXEL_DIST, OBSTACLE_ROBOT_DIA * METER_PIXEL_DIST);
        Arena_Drawer.DrawLine(&Obstacle_Robot_Pen, Temp_X, Temp_Y,
            Temp_X + sin(Array_Of_Frames[Current_Frame].Obstacle_Robots[index].Rotation * TO_RADIANS) * DIRECTION_POINTER_LENGTH * METER_PIXEL_DIST,
            Temp_Y + cos(Array_Of_Frames[Current_Frame].Obstacle_Robots[index].Rotation * TO_RADIANS) * DIRECTION_POINTER_LENGTH * METER_PIXEL_DIST);
    }


    // Draw the drone
    SolidBrush Drone_Brush(DRONE_COLOR);
    Arena_Drawer.FillEllipse(&Drone_Brush, Array_Of_Frames[Current_Frame].Drones[0].X - ((DRONE_DIA * METER_PIXEL_DIST) / 2), Array_Of_Frames[Current_Frame].Drones[0].Y - ((DRONE_DIA * METER_PIXEL_DIST) / 2),
        DRONE_DIA * METER_PIXEL_DIST, DRONE_DIA * METER_PIXEL_DIST);

    // Draw arena onto the back buffer
    BitBlt(Back_Buffer_HDC, ARENA_LEFT_MARGIN, ARENA_TOP_MARGIN, Window_Width, Window_Height, Arena_HDC, 0, 0, SRCCOPY);
    //cout << "fire2" << endl;
    DeleteDC(Arena_HDC);
    DeleteObject(Arena_HBITMAP);


	LinearGradientBrush LGB(Point(0,0), Point(0,500), Color(255,0,0,255), Color(255,0,0,200));
	//int ok = Back_Buffer_Drawer.FillRectangle(&LGB, 0, 0, 400, 500);

	// Copy the back buffer to the window
	BitBlt(Window_HDC, 0, 0, Window_Width, Window_Height, Back_Buffer_HDC, 0, 0, SRCCOPY);

	// This is supposed to be done before at the end of drawing
	SelectObject(Back_Buffer_HDC, Original_Back_Buffer_HBITMAP);
	DeleteDC(Back_Buffer_HDC);
	DeleteObject(Back_Buffer_HBITMAP);
}



void Engine()
{
    //*** Copy data from last frame ***//
    for(int index = 0; index < DRONE_NUMBER; index++)
    {

        Array_Of_Frames[Current_Frame].Drones[index].X_Speed = Array_Of_Frames[Current_Frame - 1].Drones[index].X_Speed;
        Array_Of_Frames[Current_Frame].Drones[index].Y_Speed = Array_Of_Frames[Current_Frame - 1].Drones[index].Y_Speed;
        Array_Of_Frames[Current_Frame].Drones[index].X = Array_Of_Frames[Current_Frame - 1].Drones[index].X;
        Array_Of_Frames[Current_Frame].Drones[index].Y = Array_Of_Frames[Current_Frame - 1].Drones[index].Y;
        Array_Of_Frames[Current_Frame].Drones[index].Target_Robot = Array_Of_Frames[Current_Frame - 1].Drones[index].Target_Robot;
        Array_Of_Frames[Current_Frame].Drones[index].Ok_To_Leave = Array_Of_Frames[Current_Frame - 1].Drones[index].Ok_To_Leave;

    }


    for(int index = 0; index < GROUND_ROBOT_NUMBER; index++)
    {

        Array_Of_Frames[Current_Frame].Ground_Robots[index].Rotation = Array_Of_Frames[Current_Frame - 1].Ground_Robots[index].Rotation;
        Array_Of_Frames[Current_Frame].Ground_Robots[index].X = Array_Of_Frames[Current_Frame - 1].Ground_Robots[index].X;
        Array_Of_Frames[Current_Frame].Ground_Robots[index].Y = Array_Of_Frames[Current_Frame - 1].Ground_Robots[index].Y;
        Array_Of_Frames[Current_Frame].Ground_Robots[index].Color = Array_Of_Frames[Current_Frame - 1].Ground_Robots[index].Color;
        Array_Of_Frames[Current_Frame].Ground_Robots[index].Frames_To_Turn = Array_Of_Frames[Current_Frame - 1].Ground_Robots[index].Frames_To_Turn;
        Array_Of_Frames[Current_Frame].Ground_Robots[index].Turn_Direction_Mult = Array_Of_Frames[Current_Frame - 1].Ground_Robots[index].Turn_Direction_Mult;
        Array_Of_Frames[Current_Frame].Ground_Robots[index].Still_In = Array_Of_Frames[Current_Frame - 1].Ground_Robots[index].Still_In;
        Array_Of_Frames[Current_Frame].Ground_Robots[index].Scored = Array_Of_Frames[Current_Frame - 1].Ground_Robots[index].Scored;
        Array_Of_Frames[Current_Frame].Ground_Robots[index].Last_Frame_Reversed = Array_Of_Frames[Current_Frame - 1].Ground_Robots[index].Last_Frame_Reversed;
        Array_Of_Frames[Current_Frame].Ground_Robots[index].Last_Frame_Noise = Array_Of_Frames[Current_Frame - 1].Ground_Robots[index].Last_Frame_Noise;
        Array_Of_Frames[Current_Frame].Ground_Robots[index].Time_Left_To_Rotate = Array_Of_Frames[Current_Frame - 1].Ground_Robots[index].Time_Left_To_Rotate;
        Array_Of_Frames[Current_Frame].Ground_Robots[index].Mode = Array_Of_Frames[Current_Frame - 1].Ground_Robots[index].Mode;

    }


    for(int index = 0; index < OBSTACLE_ROBOT_NUMBER; index++)
    {
        Array_Of_Frames[Current_Frame].Obstacle_Robots[index].Rotation = Array_Of_Frames[Current_Frame - 1].Obstacle_Robots[index].Rotation;
        Array_Of_Frames[Current_Frame].Obstacle_Robots[index].X = Array_Of_Frames[Current_Frame - 1].Obstacle_Robots[index].X;
        Array_Of_Frames[Current_Frame].Obstacle_Robots[index].Y = Array_Of_Frames[Current_Frame - 1].Obstacle_Robots[index].Y;
    }

    //*** Now for changing data ***//

    // Move the ground robots
    for(int index = 0; index < GROUND_ROBOT_NUMBER; index++)
    {
        Ground_Robot_Data * Current_G_Robot;
        float Prev_X = 0;
        float Prev_Y = 0;

        Current_G_Robot = &Array_Of_Frames[Current_Frame].Ground_Robots[index];
        Prev_X = Current_G_Robot->X;
        Prev_Y = Current_G_Robot->Y;

        if(Current_G_Robot->Mode == FORWARD_MODE)
        {
            // If its time to reverse
            if(Current_G_Robot->Last_Frame_Reversed < Current_Frame - (FRAME_RATE * REVERSE_INTERVAL))
            {
                Current_G_Robot->Mode = REVERSE_MODE;
                Current_G_Robot->Last_Frame_Reversed = Current_Frame;
                Current_G_Robot->Frames_To_Turn = (REVERSE_DEGREES / GROUND_ROBOT_TURN_SPEED) * FRAME_RATE;
            }
            else if(Current_G_Robot->Last_Frame_Noise < Current_Frame - (FRAME_RATE * NOISE_INTERVAL))
            {
                Current_G_Robot->Mode =  NOISE_MODE;
                Current_G_Robot->Last_Frame_Noise = Current_Frame;
                Current_G_Robot->Frames_To_Turn = ((rand() % (static_cast<int>(NOISE_MAX_ROTATION - NOISE_MIN_ROTATION) + 1)) / GROUND_ROBOT_TURN_SPEED) * FRAME_RATE;
            }

            // Assign new X and Y
            Current_G_Robot->X = Current_G_Robot->X + sin(Current_G_Robot->Rotation * TO_RADIANS) * GROUND_ROBOT_PIXEL_SPEED;
            Current_G_Robot->Y = Current_G_Robot->Y + cos(Current_G_Robot->Rotation * TO_RADIANS) * GROUND_ROBOT_PIXEL_SPEED;

            for(int index2 = 0; index2 < OBSTACLE_ROBOT_NUMBER; index2++)
            {
                float A;
                float B;
                Obstacle_Robot_Data * Current_O_Robot;

                Current_O_Robot = &Array_Of_Frames[Current_Frame].Obstacle_Robots[index2];
                A = Current_G_Robot->X - Current_O_Robot->X;
                B = Current_G_Robot->Y - Current_O_Robot->Y;
                if(sqrt(A * A + B * B) < (GROUND_ROBOT_DIA / 2 + OBSTACLE_ROBOT_DIA / 2) * METER_PIXEL_DIST)
                {
                    Current_G_Robot->X = Prev_X;
                    Current_G_Robot->Y = Prev_Y;
                    Current_G_Robot->Frames_To_Turn = (COLLISION_DEGREES / GROUND_ROBOT_TURN_SPEED) * FRAME_RATE;
                    Current_G_Robot->Mode = COLLISION_MODE;
                }
            }

            for(int index2 = 0; index2 < GROUND_ROBOT_NUMBER; index2++)
            {
                float A;
                float B;
                Ground_Robot_Data * Current_G_Robot2;

                if(index2 != index)
                {
                    Current_G_Robot2 = &Array_Of_Frames[Current_Frame].Ground_Robots[index2];
                    A = Current_G_Robot->X - Current_G_Robot2->X;
                    B = Current_G_Robot->Y - Current_G_Robot2->Y;
                    if(sqrt(A * A + B * B) < (GROUND_ROBOT_DIA / 2 + OBSTACLE_ROBOT_DIA / 2) * METER_PIXEL_DIST)
                    {
                        Current_G_Robot->X = Prev_X;
                        Current_G_Robot->Y = Prev_Y;
                        Current_G_Robot->Frames_To_Turn = (COLLISION_DEGREES / GROUND_ROBOT_TURN_SPEED) * FRAME_RATE;
                        Current_G_Robot->Mode = COLLISION_MODE;
                    }
                }
            }
        }

        else if(Current_G_Robot->Mode == REVERSE_MODE)
        {
            // Turn
            Current_G_Robot->Rotation = Current_G_Robot->Rotation - (GROUND_ROBOT_TURN_SPEED / FRAME_RATE);

            // Decrement the frames left in the turn
            Current_G_Robot->Frames_To_Turn = Current_G_Robot->Frames_To_Turn - 1;

            // If the turn is done
            if(Current_G_Robot->Frames_To_Turn <= 0)
            {
                // Start driving forward again
                Current_G_Robot->Mode = FORWARD_MODE;
            }
        }
        else if(Current_G_Robot->Mode == NOISE_MODE)
        {
            // Turn
            Current_G_Robot->Rotation = Current_G_Robot->Rotation - (GROUND_ROBOT_TURN_SPEED / FRAME_RATE);

            // Decrement the frames left in the turn
            Current_G_Robot->Frames_To_Turn = Current_G_Robot->Frames_To_Turn - 1;

            // If the turn is done
            if(Current_G_Robot->Frames_To_Turn <= 0)
            {
                // Start driving forward again
                Current_G_Robot->Mode = FORWARD_MODE;
            }
        }
        else if(Current_G_Robot->Mode == COLLISION_MODE)
        {
            // Turn
            Current_G_Robot->Rotation = Current_G_Robot->Rotation - (GROUND_ROBOT_TURN_SPEED / FRAME_RATE);

            // Decrement the frames left in the turn
            Current_G_Robot->Frames_To_Turn = Current_G_Robot->Frames_To_Turn - 1;

            // If the turn is done
            if(Current_G_Robot->Frames_To_Turn <= 0)
            {
                // Start driving forward again
                Current_G_Robot->Mode = FORWARD_MODE;
            }
        }
    }

    // Move the obstacle robots
    for(int index = 0; index < OBSTACLE_ROBOT_NUMBER; index++)
    {
        Obstacle_Robot_Data * Current_O_Robot;
        float Prev_X = 0;
        float Prev_Y = 0;
        int Prev_Rotation = 0;

        Current_O_Robot = &Array_Of_Frames[Current_Frame].Obstacle_Robots[index];

        Prev_X = Current_O_Robot->X;
        Prev_Y = Current_O_Robot->Y;
        Prev_Rotation = Current_O_Robot->Rotation;

        float Rotation_Change = DEGREES_IN_CIRCLE / FRAME_RATE / OBSTACLE_ROBOT_FULL_CIRLE_TIME;
        Current_O_Robot->Rotation = Current_O_Robot->Rotation - Rotation_Change;
        Current_O_Robot->X = Current_O_Robot->X + sin(Current_O_Robot->Rotation * TO_RADIANS) * OBSTACLE_ROBOT_PIXEL_SPEED;
        Current_O_Robot->Y = Current_O_Robot->Y + cos(Current_O_Robot->Rotation * TO_RADIANS) * OBSTACLE_ROBOT_PIXEL_SPEED;

        for(int index2 = 0; index2 < GROUND_ROBOT_NUMBER; index2++)
        {
            float A;
            float B;
            Ground_Robot_Data * Current_G_Robot;

            Current_G_Robot = &Array_Of_Frames[Current_Frame].Ground_Robots[index2];
            A = Current_O_Robot->X - Current_G_Robot->X;
            B = Current_O_Robot->Y - Current_G_Robot->Y;
            if(sqrt(A * A + B * B) < (GROUND_ROBOT_DIA / 2 + OBSTACLE_ROBOT_DIA / 2) * METER_PIXEL_DIST)
            {
                Current_O_Robot->X = Prev_X;
                Current_O_Robot->Y = Prev_Y;
                Current_O_Robot->Rotation = Prev_Rotation;
            }
        }
    }
}



void Resize_Arrays()
{
    for(int index = 0; index < FRAME_LIMIT + 1; index++)
    {
        Array_Of_Frames[index].Ground_Robots.resize(GROUND_ROBOT_NUMBER);
        Array_Of_Frames[index].Obstacle_Robots.resize(OBSTACLE_ROBOT_NUMBER);
        Array_Of_Frames[index].Drones.resize(DRONE_NUMBER);
    }
    return;
}



void Init()
{
    for(int index = 0; index < GROUND_ROBOT_NUMBER; index++)
    {
        float Temp_Rotation = index * DEGREES_IN_CIRCLE / GROUND_ROBOT_NUMBER;
        Array_Of_Frames[0].Ground_Robots[index].X = ARENA_WIDTH / 2 + sin(Temp_Rotation * TO_RADIANS) *  METER_PIXEL_DIST;
        Array_Of_Frames[0].Ground_Robots[index].Y = ARENA_HEIGHT / 2 + cos(Temp_Rotation * TO_RADIANS) *  METER_PIXEL_DIST;
        Array_Of_Frames[0].Ground_Robots[index].Rotation = Temp_Rotation;
        Array_Of_Frames[0].Ground_Robots[index].Color = (index % 2 == 0) ? RED_ID : BLUE_ID;
        Array_Of_Frames[0].Ground_Robots[index].Frames_To_Turn = 0;
        Array_Of_Frames[0].Ground_Robots[index].Turn_Direction_Mult = 0;
        Array_Of_Frames[0].Ground_Robots[index].Still_In = true;
        Array_Of_Frames[0].Ground_Robots[index].Scored = false;
        Array_Of_Frames[0].Ground_Robots[index].Mode = FORWARD_MODE;
        Array_Of_Frames[0].Ground_Robots[index].Last_Frame_Reversed = 0;
        Array_Of_Frames[0].Ground_Robots[index].Last_Frame_Noise = 0;
        Array_Of_Frames[0].Ground_Robots[index].Time_Left_To_Rotate = 0;
    }


    for(int index = 0; index < OBSTACLE_ROBOT_NUMBER; index++)
    {
        float Temp_Rotation = index * DEGREES_IN_CIRCLE / OBSTACLE_ROBOT_NUMBER;
        float Obstacle_Robot_Direction = Temp_Rotation - 90;
        Array_Of_Frames[0].Obstacle_Robots[index].X = FIELD_X + FIELD_WIDTH / 2 + sin(Temp_Rotation * TO_RADIANS) * METER_PIXEL_DIST * OBSTACLE_ROBOT_PATH_RADIUS;
        Array_Of_Frames[0].Obstacle_Robots[index].Y = FIELD_Y + FIELD_HEIGHT / 2 + cos(Temp_Rotation * TO_RADIANS) * METER_PIXEL_DIST * OBSTACLE_ROBOT_PATH_RADIUS;
        Array_Of_Frames[0].Obstacle_Robots[index].Rotation = Obstacle_Robot_Direction;
    }


    for(int index = 0; index < DRONE_NUMBER; index++)
    {
        float Temp_Rotation = index * DEGREES_IN_CIRCLE / GROUND_ROBOT_NUMBER;
        Array_Of_Frames[0].Drones[index].X = FIELD_X + FIELD_WIDTH + (DRONE_DIA * METER_PIXEL_DIST);
        Array_Of_Frames[0].Drones[index].Y = FIELD_Y + FIELD_HEIGHT / 2;
        Array_Of_Frames[0].Drones[index].X_Speed = 0;
        Array_Of_Frames[0].Drones[index].Y_Speed = 0;
        Array_Of_Frames[0].Drones[index].Target_Robot = 0;
        Array_Of_Frames[0].Drones[index].Ok_To_Leave = true;
    }
    cout << "after init";
    return;
}



void Reload_Match()
{
    cout << endl << "there" << endl;
    Array_Of_Frames.clear();
    cout << "after 1" << endl;
    Array_Of_Frames.resize(FRAME_LIMIT + 1);
    cout << "end" << endl;
}



HBITMAP CreateDIBSectionFunction(HDC hdc, int W,  int H)
{
	HBITMAP hbm;
	BITMAPINFO bmi;
  			BITMAPINFOHEADER bmih = {0};
			bmih.biSize = sizeof(BITMAPINFOHEADER);
			bmih.biWidth = W;
			bmih.biHeight = -H;
			bmih.biPlanes = 1;
			bmih.biBitCount = 32;
			bmih.biCompression = BI_RGB;
			bmih.biXPelsPerMeter = 0;
			bmih.biYPelsPerMeter = 0;
			bmih.biClrUsed = 0;
			bmih.biClrImportant = 0;
			bmih.biSizeImage = W * H * 4;
			bmi.bmiHeader = bmih;
  			hbm = CreateDIBSection(((hdc==NULL) ? GetDC(NULL) : hdc), &bmi, DIB_RGB_COLORS, NULL, NULL, 0);
  			return hbm;
}
