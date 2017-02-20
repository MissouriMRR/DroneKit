/*
  Programmer: Lucas Winkelman & Tanner Winkelman
  File: main.cpp
  Purpose: To simulate matches with a varying drone heuristic
  Warning!  This program uses global variables.
*/

#include <windows.h>
#include <gdiplus.h>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <future>
#include <random>
#include <chrono>
#include <string>
using namespace Gdiplus;
using namespace std;

// The starting score of a match
const float MATCH_START_SCORE = 12000;

// The points to add to a match per robot scored
const float ROBOT_SCORED_POINTS = 2000;

// The points to add to a match per robot not scored
const float ROBOT_NOT_SCORED_POINTS = -1000;

// The points to add to a match per minute used in the match
const float MINUTE_USED_POINTS = -100;

// Frames per second when replaying
const float DISPLAY_FRAME_RATE = 30;

// Frames per second when computing matches
const float COMP_FRAME_RATE = 30;

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
const Color WHITE_TAPE_COLOR = Color(255, 255, 255, 255);
const Color GREEN_TAPE_COLOR = Color(255, 0, 255, 0);
const Color RED_TAPE_COLOR = Color(255, 255, 0, 0);

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
const float OBSTACLE_ROBOT_PIXEL_SPEED = OBSTACLE_ROBOT_SPEED * FIELD_HEIGHT / FIELD_METER_WIDTH / COMP_FRAME_RATE;

// The amount of time of a match in seconds
const float TIME_LIMIT = SECONDS_IN_MINUTE * MATCH_LENGTH;

// The number of frames in one match
const int FRAME_LIMIT = COMP_FRAME_RATE * TIME_LIMIT;

// The ground robots' speed in pixels
const float GROUND_ROBOT_PIXEL_SPEED = GROUND_ROBOT_SPEED * FIELD_HEIGHT / FIELD_METER_WIDTH / COMP_FRAME_RATE;

// The numbers for the various turning modes
const int FORWARD_MODE = 0;
const int REVERSE_MODE = 1;
const int NOISE_MODE = 2;
const int COLLISION_MODE = 3;
const int TAP_MODE = 4;
const int TOP_TOUCH_MODE = 5;

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

// The degrees turned when a ground robot gets touched on top
const float TOP_TOUCH_DEGREES = 45;

// The speed the robot turns at in degrees per second
const float GROUND_ROBOT_TURN_SPEED = REVERSE_DEGREES / REVERSE_DURATION;

// IDs for the two ground robot colors
const int RED_ID = 0;
const int BLUE_ID = 1;

// The drone's acceleration
const float DRONE_ACCEL = 1 / COMP_FRAME_RATE;

// The min and max of the range of multipliers that get multiplied with the distance
//   of a roomba from the green line
const float GREEN_LINE_MULT_MIN = -5;
const float GREEN_LINE_MULT_MAX = 5;

// The min and max of the range of multipliers that get multiplied with the closest distance to an edge
//   except for the green one
const float DIST_FROM_EDGE_MIN = -5;
const float DIST_FROM_EDGE_MAX = 5;

// The min and max of the distance from drone multiplier
const float DIST_FROM_DRONE_MIN = -5;
const float DIST_FROM_DRONE_MAX = 5;

// The min and max of the mult of the importance of helping a ground robot that is going to going to almost go off the map
const float HEAD_OFF_EDGE_MIN = -5;
const float HEAD_OFF_EDGE_MAX = 5;

// The min and max of the mult for the ground robot facing a certain direction when giving a score
const float FIND_DIRECTION_MAX = -5;
const float FIND_DIRECTION_MIN = 5;

// The min and max of target rotation
const float ROT_MIN = 0;
const float ROT_MAX = 360;

// The min and max of target rotation
const float AFTER_HALF_ROT_MIN = 0;
const float AFTER_HALF_ROT_MAX = 720;

// The min and max of a ground robot's rotation when its on the left side
const float LEFT_ROT_MIN = 0;
const float LEFT_ROT_MAX = 720;

// The min and max of target rotation
const float HALF_MIN = 0;
const float HALF_MAX = 20;

// The min and max of target rotation
const bool CONTINUE_AFTER_HALF_OFF = false;
const bool CONTINUE_AFTER_HALF_ON = true;

// The number of meters ground robot must be from any edge at the next reverse time to be considered safe
const float HEAD_OFF_EDGE_SAFETY_DIST = 0.5;

// The maximum distance the drone can from right above the ground robot to top touch it (in meters)
const float TOP_TOUCH_MAX_OFFSET = 0.1;

// The number of data sets
const int DATA_SET_COUNT = 30;

// The maximum speed difference between the drone and the ground robot it wants to touch
const float TOP_TOUCH_MAX_SPEED_DIFF = 0.2 / COMP_FRAME_RATE;

// The minimum safe distance from an obstacle robot
const float DRONE_FROM_OBS_MIN_SAFE_DIST = 1;


const float EVO_MATCH_MULT = 0;
const float EVO_MATCH_ADDER = 4;


const int MATCHES_MAX = 2000;


const float ROT_MIN_START = 220;
const float ROT_MAX_START = 330;

const float AFTER_HALF_ROT_MIN_START = 400;
const float AFTER_HALF_ROT_MAX_START = 510;

const float MUT_ON_VARIABLE_MIN = 0;
const float MUT_ON_VARIABLE_MAX = 100;

const float RANDOM_MUT_MAX = 1;
const float INCREMENT_MUT_MAX = 5;

const float ON_RIGHT_ROT_MIN_START = 120;
const float ON_RIGHT_ROT_MAX_START = 220;

const float ON_LEFT_ROT_MIN_START = 310;
const float ON_LEFT_ROT_MAX_START = 400;

const float RIGHT_SIDE_ROT_DIST_START = 2;
const float LEFT_SIDE_ROT_DIST_START = 2;

const float LEFT_SIDE_MIN = 0;
const float LEFT_SIDE_MAX = 10;

const float RIGHT_SIDE_MIN = 0;
const float RIGHT_SIDE_MAX = 10;

const bool SWITCH_BEFORE_DONE_START = false;

const int FRAME_SKIP = 3;

const int MAX_MUT_TRIES = 10;

const float RAND_MUT_CHANCE = 10;
const float INCREMENT_MUT_CHANCE = 50;

const int AFTER_HALF_TOP_SKIP_MIN = 0;
const int AFTER_HALF_TOP_SKIP_MAX = 10;


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
    int Prev_Target_Robot;
};

struct Game_Frame
{
    vector<Ground_Robot_Data> Ground_Robots;
    vector<Obstacle_Robot_Data> Obstacle_Robots;
    vector<Drone_Data> Drones;
    bool Target_In_Position;
};

struct Result
{
    float Ground_Robots_Scored;
    float Time_Taken;
    float Score;
};

struct Data_Set
{
    float Green_Line_Mult;
    float Dist_From_Edge_Mult;
    float Dist_From_Drone_Mult;
    float Head_Off_Edge_Mult;
    float Find_Direction_Mult;
    float Rot_Min;
    float Rot_Max;
    bool Cont_After_Half;
    float After_Half_Rot_Min;
    float After_Half_Rot_Max;
    float Half;
    float On_Right_Rot_Min;
    float On_Right_Rot_Max;
    float On_Left_Rot_Min;
    float On_Left_Rot_Max;
    bool Switch_Before_Done;
    vector<Result> Results;
    float Ave_Robots;
    float Ave_Time_Taken;
    float Ave_Score;
    int After_Half_Top_Skip;
};

struct Match
{
    vector<Game_Frame> Array_Of_Frames;
    int Current_Frame;
    int Frame_Ended;
    bool Has_Ended;
};

struct Pos
{
    float X;
    float Y;
};

// Desc: Initializes the first frame of a match
// Pre: Relies on global variables, especially Array_Of_Frames
// Post: The first frame is set up and we are ready to run the match (Array_Of_Frames[0] is set up)
void Init(int In_Match);

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
void Engine(int In_Match);

// Desc: This function initializes the multipliers in the data sets
// Pre: None
// Post: The multipliers in Data_Sets are initialized
void Init_Mults(int index, int Matches);

// Desc: Scores match at Current_Frame in Array_Of_Frames
// Pre: None
// Post: Fills in Data_Sets[Current_Data_Set].Results[Current_Match_In_Set] with the scoring data from the match
void Score_Match(int In_Match);

// Desc: Calculate the average of a data set's matches's scores
// Pre: None
// Post: Calculate the averages for Data_Set[Current_Data_Set] and sets its vcounter * 2ariables
void Calc_Set_Ave(int Num_Of_Matches);

// Desc: Print out a set on the screen
// Pre: index must be positive
// Post: Data_Sets[index] will be outputted
void Output_Set(int index);

// Desc: Data_Sets by each items score by bubble sort
// Pre: None
// Post: Data_Sets will be sorted
void Sort_Sets_By_Score();



void Make_New_Sets(int Matches_Num);



void Gen_Random_Set(int index);



void Run_Match(int In_Match);



bool Is_Ground_Robot_In_Rotation_Range(int G_R_Index, int In_Match);



// We are pretty much forced to use global variables with the way a windows c++ program is set up :(
vector<Data_Set> Data_Sets(DATA_SET_COUNT);
vector<Match> Matches;
vector<thread *> Threads;
atomic<int> Current_Data_Set(0);
atomic<int> Current_Match_In_Set(0);
int Evo_Iteration = 0;
int G_Robot_W_High_Score = 0;
int Matches_Per_Set = 4;
int Cores = 4;
bool Draw_On_Screen = false;
atomic<int> Threads_Done(0);

vector<Pos> Points(10);

HWND global_hwnd;

// http://stackoverflow.com/questions/22105867/seeding-default-random-engine
unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
default_random_engine generator(seed1);

float firex = 0;
float firey = 0;

HWND hWnd;

INT WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, PSTR, INT iCmdShow)
{
    MSG                 msg;
    WNDCLASS            wndClass;
    GdiplusStartupInput gdiplusStartupInput;
    ULONG_PTR           gdiplusToken;
    int Generations_Per_Pause = 0;

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

    global_hwnd = hWnd;

    //ShowWindow(hWnd, SW_MAXIMIZE);
    ShowWindow(hWnd, iCmdShow);


    do
    {
        cout << "How many cores in machine?" << endl;
        cin >> Cores;
    }while(Cores <= 0);
    Threads.resize(Cores);

    do
    {
        cout << "How many matches per set?" << endl;
        cin >> Matches_Per_Set;
    }while(Matches_Per_Set < 0);
    Matches.resize(Matches_Per_Set);

    do
    {
        cout << "How many generations per pause?" << endl;
        cin >> Generations_Per_Pause;
    }while(Generations_Per_Pause <= 0);

    do
    {
        cout << "Draw on screen? (0 or 1)" << endl;
        cin >> Draw_On_Screen;
    }while(Draw_On_Screen < 0);

    for(int counter1 = 0; counter1 < DATA_SET_COUNT; counter1++)
    {
        Init_Mults(counter1, Matches_Per_Set);
    }

    for(int counter = 0; counter < Cores; counter++)
    {
        Init(counter);
    }

    // The message loop
    while(TRUE)
    {
        cout << "Number of Matches: " << Matches_Per_Set << endl;
        cout << "Generation: " << Evo_Iteration << endl;
        while(Current_Data_Set < DATA_SET_COUNT)
        {
            while(Current_Match_In_Set < Matches_Per_Set)
            {
                while(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
                {
                    TranslateMessage(&msg);
                    DispatchMessage(&msg);
                }
                if(msg.message == WM_QUIT)
                    break;

                Threads_Done = 0;
                for(int counter = 0; counter < Cores; counter++)
                {
                    Threads[counter] = new thread(Run_Match, counter);
                }
                int adder = 0;
                while(Threads_Done < Cores)
                {
                    while(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
                    {
                        TranslateMessage(&msg);
                        DispatchMessage(&msg);
                    }
                    if(msg.message == WM_QUIT)
                        break;
                }
                 for(int counter = 0; counter < Cores; counter++)
                {
                    Threads[counter]->join();
                    delete Threads[counter];
                }
                for(int counter = 0; counter < Cores; counter++)
                {
                    if(Current_Match_In_Set < Matches_Per_Set)
                    {
                        Score_Match(counter);
                        Current_Match_In_Set++;
                    }
                }

                for(int counter = 0; counter < Cores; counter++)
                {
                    Matches[counter].Current_Frame = 0;
                    Init(counter);
                }
            }
            // End of data set
            Current_Match_In_Set = 0;
            Calc_Set_Ave(Matches_Per_Set);
            Output_Set(Current_Data_Set);

            Current_Data_Set++;
        }

        Current_Data_Set = 0;
        Sort_Sets_By_Score();
        Make_New_Sets(Matches_Per_Set);
        Evo_Iteration++;
        if(Evo_Iteration % Generations_Per_Pause == 0)
        {
            do
            {
                cout << "How many cores in machine?" << endl;
                cin >> Cores;
            }while(Cores <= 0);
            Threads.resize(Cores);

            do
            {
                cout << "How many matches per set?" << endl;
                cin >> Matches_Per_Set;
            }while(Matches_Per_Set < 0);
            Matches.resize(Matches_Per_Set);
            for(int counter = 0; counter < DATA_SET_COUNT; counter++)
            {
                Data_Sets[counter].Results.resize(Matches_Per_Set);
            }

            do
            {
                cout << "How many generations per pause?" << endl;
                cin >> Generations_Per_Pause;
            }while(Generations_Per_Pause <= 0);

            do
            {
                cout << "Draw on screen? (0 or 1)" << endl;
                cin >> Draw_On_Screen;
            }while(Draw_On_Screen < 0);
        }
        /*
        if(Winner_Shown == false)
        {
            int Best_Set = 0;
            float High_Score = 0;
            for(int counter = 0; counter < DATA_SET_COUNT; counter++)
            {
                if(Data_Sets[counter].Ave_Score > High_Score)
                {
                    High_Score = Data_Sets[counter].Ave_Score;
                    Best_Set = counter;
                }
            }

            Winner_Shown = true;
            Output_Set(Best_Set);
        }
        */
    }

   GdiplusShutdown(gdiplusToken);
   return msg.wParam;
}  // WinMain


// This function gets called in response to messages to the window
// It gets called after the DispatchMessage function in the message loop
LRESULT CALLBACK WndProc(HWND hWnd, UINT message,
   WPARAM wParam, LPARAM lParam)
{
    PAINTSTRUCT  ps;

    switch(message)
    {
        // If the window needs to be repainted
        case WM_PAINT:
        {
            BeginPaint(hWnd, &ps);
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
    Pen White_Tape_Pen(WHITE_TAPE_COLOR, TAPE_WIDTH);
    for(int index = 1; index < FIELD_METER_WIDTH; index++)
    {
        Arena_Drawer.DrawLine(&White_Tape_Pen, FIELD_X, FIELD_Y + (METER_PIXEL_DIST * index), FIELD_X + FIELD_WIDTH, FIELD_Y + (METER_PIXEL_DIST * index));
    }

    // Draw the vertical pieces of tape
    for(int index = 0; index < FIELD_METER_WIDTH + 1; index++)
    {
        Arena_Drawer.DrawLine(&White_Tape_Pen, FIELD_X + (METER_PIXEL_DIST * index), FIELD_Y, FIELD_X + (METER_PIXEL_DIST * index), FIELD_Y + FIELD_HEIGHT);
    }

    // Create red and green pens
    Pen Red_Tape_Pen(RED_TAPE_COLOR, TAPE_WIDTH);
    Pen Green_Tape_Pen(GREEN_TAPE_COLOR, TAPE_WIDTH);

    // Draw a red and a green line
    Arena_Drawer.DrawLine(&Red_Tape_Pen, FIELD_X, FIELD_Y + FIELD_HEIGHT, FIELD_X + FIELD_WIDTH, FIELD_Y + FIELD_HEIGHT);
    Arena_Drawer.DrawLine(&Green_Tape_Pen, FIELD_X, FIELD_Y, FIELD_X + FIELD_WIDTH, FIELD_Y);

    // Draw the ground robots
    SolidBrush Ground_Robot_Brush(GROUND_ROBOT_COLOR);
    Pen Red_Robot_Pen(RED_ROBOT_COLOR);
    Pen Blue_Robot_Pen(BLUE_ROBOT_COLOR);
    for(int index = 0; index < GROUND_ROBOT_NUMBER; index++)
    {
        float Temp_X = Matches[0].Array_Of_Frames[Matches[0].Current_Frame].Ground_Robots[index].X;
        float Temp_Y = Matches[0].Array_Of_Frames[Matches[0].Current_Frame].Ground_Robots[index].Y;
        Arena_Drawer.FillEllipse(&Ground_Robot_Brush, Matches[0].Array_Of_Frames[Matches[0].Current_Frame].Ground_Robots[index].X - ((GROUND_ROBOT_DIA * METER_PIXEL_DIST) / 2),
            Matches[0].Array_Of_Frames[Matches[0].Current_Frame].Ground_Robots[index].Y - ((GROUND_ROBOT_DIA * METER_PIXEL_DIST) / 2),
            GROUND_ROBOT_DIA * METER_PIXEL_DIST, GROUND_ROBOT_DIA * METER_PIXEL_DIST);

        Arena_Drawer.FillEllipse(&Ground_Robot_Brush, Points[index].X - ((GROUND_ROBOT_DIA * METER_PIXEL_DIST) / 2),
            Points[index].Y - ((GROUND_ROBOT_DIA * METER_PIXEL_DIST) / 2),
            GROUND_ROBOT_DIA * METER_PIXEL_DIST, GROUND_ROBOT_DIA * METER_PIXEL_DIST);

        // Draw direction pointer on robot
        if(Matches[0].Array_Of_Frames[Matches[0].Current_Frame].Ground_Robots[index].Color == RED_ID)
        {
            Arena_Drawer.DrawLine(&Red_Robot_Pen, Temp_X, Temp_Y,
                Temp_X + cos(Matches[0].Array_Of_Frames[Matches[0].Current_Frame].Ground_Robots[index].Rotation * TO_RADIANS) * DIRECTION_POINTER_LENGTH * METER_PIXEL_DIST,
                Temp_Y + sin(Matches[0].Array_Of_Frames[Matches[0].Current_Frame].Ground_Robots[index].Rotation * TO_RADIANS) * DIRECTION_POINTER_LENGTH * METER_PIXEL_DIST);
        }
        else
        {
            Arena_Drawer.DrawLine(&Blue_Robot_Pen, Temp_X, Temp_Y,
                Temp_X + cos(Matches[0].Array_Of_Frames[Matches[0].Current_Frame].Ground_Robots[index].Rotation * TO_RADIANS) * DIRECTION_POINTER_LENGTH * METER_PIXEL_DIST,
                Temp_Y + sin(Matches[0].Array_Of_Frames[Matches[0].Current_Frame].Ground_Robots[index].Rotation * TO_RADIANS) * DIRECTION_POINTER_LENGTH * METER_PIXEL_DIST);
        }
    }

    // Draw obstacle robots
    SolidBrush Obstacle_Robot_Brush(OBSTACLE_ROBOT_COLOR);
    Pen Obstacle_Robot_Pen(OBSTACLE_ROBOT_COLOR);
    for(int index = 0; index < OBSTACLE_ROBOT_NUMBER; index++)
    {
        float Temp_X = Matches[0].Array_Of_Frames[Matches[0].Current_Frame].Obstacle_Robots[index].X;
        float Temp_Y = Matches[0].Array_Of_Frames[Matches[0].Current_Frame].Obstacle_Robots[index].Y;
        Arena_Drawer.FillEllipse(&Obstacle_Robot_Brush, Matches[0].Array_Of_Frames[Matches[0].Current_Frame].Obstacle_Robots[index].X - ((OBSTACLE_ROBOT_DIA * METER_PIXEL_DIST) / 2),
            Matches[0].Array_Of_Frames[Matches[0].Current_Frame].Obstacle_Robots[index].Y - ((OBSTACLE_ROBOT_DIA * METER_PIXEL_DIST) / 2),
            OBSTACLE_ROBOT_DIA * METER_PIXEL_DIST, OBSTACLE_ROBOT_DIA * METER_PIXEL_DIST);
        Arena_Drawer.DrawLine(&Obstacle_Robot_Pen, Temp_X, Temp_Y,
            Temp_X + cos(Matches[0].Array_Of_Frames[Matches[0].Current_Frame].Obstacle_Robots[index].Rotation * TO_RADIANS) * DIRECTION_POINTER_LENGTH * METER_PIXEL_DIST,
            Temp_Y + sin(Matches[0].Array_Of_Frames[Matches[0].Current_Frame].Obstacle_Robots[index].Rotation * TO_RADIANS) * DIRECTION_POINTER_LENGTH * METER_PIXEL_DIST);
    }


    // Draw the drone
    SolidBrush Drone_Brush(DRONE_COLOR);
    Arena_Drawer.FillEllipse(&Drone_Brush, Matches[0].Array_Of_Frames[Matches[0].Current_Frame].Drones[0].X - ((DRONE_DIA * METER_PIXEL_DIST) / 2),
        Matches[0].Array_Of_Frames[Matches[0].Current_Frame].Drones[0].Y - ((DRONE_DIA * METER_PIXEL_DIST) / 2),
        DRONE_DIA * METER_PIXEL_DIST, DRONE_DIA * METER_PIXEL_DIST);
    Arena_Drawer.FillEllipse(&Drone_Brush, firex - ((DRONE_DIA * METER_PIXEL_DIST) / 2), firey - ((DRONE_DIA * METER_PIXEL_DIST) / 2),
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
	SelectObject(Arena_HDC, Original_Arena_HBITMAP);
	SelectObject(Back_Buffer_HDC, Original_Back_Buffer_HBITMAP);
	DeleteDC(Back_Buffer_HDC);
	DeleteObject(Back_Buffer_HBITMAP);
}



void Engine(int In_Match)
{
    Drone_Data * Current_Drone;

    //*** Copy data from last frame ***//
    for(int index = 0; index < DRONE_NUMBER; index++)
    {

        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Drones[index].X_Speed = Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame - 1].Drones[index].X_Speed;
        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Drones[index].Y_Speed = Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame - 1].Drones[index].Y_Speed;
        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Drones[index].X = Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame - 1].Drones[index].X;
        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Drones[index].Y = Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame - 1].Drones[index].Y;
        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Drones[index].Target_Robot = Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame - 1].Drones[index].Target_Robot;
        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Drones[index].Ok_To_Leave = Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame - 1].Drones[index].Ok_To_Leave;
        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Drones[index].Prev_Target_Robot = Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame - 1].Drones[index].Prev_Target_Robot;

    }



    for(int index = 0; index < GROUND_ROBOT_NUMBER; index++)
    {

        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[index].Rotation = Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame - 1].Ground_Robots[index].Rotation;
        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[index].X = Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame - 1].Ground_Robots[index].X;
        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[index].Y = Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame - 1].Ground_Robots[index].Y;
        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[index].Color = Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame - 1].Ground_Robots[index].Color;
        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[index].Frames_To_Turn = Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame - 1].Ground_Robots[index].Frames_To_Turn;
        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[index].Turn_Direction_Mult = Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame - 1].Ground_Robots[index].Turn_Direction_Mult;
        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[index].Still_In = Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame - 1].Ground_Robots[index].Still_In;
        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[index].Scored = Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame - 1].Ground_Robots[index].Scored;
        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[index].Last_Frame_Reversed = Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame - 1].Ground_Robots[index].Last_Frame_Reversed;
        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[index].Last_Frame_Noise = Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame - 1].Ground_Robots[index].Last_Frame_Noise;
        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[index].Time_Left_To_Rotate = Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame - 1].Ground_Robots[index].Time_Left_To_Rotate;
        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[index].Mode = Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame - 1].Ground_Robots[index].Mode;

    }


    for(int index = 0; index < OBSTACLE_ROBOT_NUMBER; index++)
    {
        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Obstacle_Robots[index].Rotation = Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame - 1].Obstacle_Robots[index].Rotation;
        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Obstacle_Robots[index].X = Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame - 1].Obstacle_Robots[index].X;
        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Obstacle_Robots[index].Y = Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame - 1].Obstacle_Robots[index].Y;
    }

    //*** Now for changing data ***//

    for(int index1 = 0; index1 < DRONE_NUMBER; index1++)
    {
        Ground_Robot_Data * Target_G_Robot;

        Current_Drone = &Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Drones[index1];

        float Robot_High_Score = -99999;
        int Robot_With_High_Score = Current_Drone->Prev_Target_Robot;

        bool Compute = false;
        for(int index = 0; index < GROUND_ROBOT_NUMBER; index++)
        {
            Ground_Robot_Data * Current_G_Robot;
            Current_G_Robot = &Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[index];
            if(Current_G_Robot->Mode == FORWARD_MODE)
            {
                // Determine if the ground robot is going to get too near to the edge
                float Seconds_To_Next_Reverse = ((Current_G_Robot->Last_Frame_Reversed + (REVERSE_INTERVAL * COMP_FRAME_RATE)
                    - Matches[In_Match].Current_Frame) / COMP_FRAME_RATE);

                float Temp_X = Current_G_Robot->X + cos(TO_RADIANS * Current_G_Robot->Rotation) * GROUND_ROBOT_SPEED * METER_PIXEL_DIST *
                    Seconds_To_Next_Reverse;
                float Temp_Y = Current_G_Robot->Y + sin(TO_RADIANS * Current_G_Robot->Rotation) * GROUND_ROBOT_SPEED * METER_PIXEL_DIST *
                    Seconds_To_Next_Reverse;
                if(Temp_Y > FIELD_Y + FIELD_HEIGHT - (HEAD_OFF_EDGE_SAFETY_DIST * METER_PIXEL_DIST) ||
                    Temp_X < FIELD_X + (HEAD_OFF_EDGE_SAFETY_DIST * METER_PIXEL_DIST) ||
                    Temp_X > FIELD_X + FIELD_WIDTH - (HEAD_OFF_EDGE_SAFETY_DIST * METER_PIXEL_DIST))
                {
                    Compute = true;
                }
            }
        }
        bool oho = false;

        //cout << "made it" << endl;
       // if(Draw_On_Screen)
        //cout << Is_Ground_Robot_In_Rotation_Range(Current_Drone->Prev_Target_Robot, In_Match) << endl;
        if(Is_Ground_Robot_In_Rotation_Range(Current_Drone->Prev_Target_Robot, In_Match) == true || Data_Sets[Current_Data_Set].Switch_Before_Done == true || Matches[In_Match].Current_Frame <= 3 ||
            Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[Current_Drone->Prev_Target_Robot].Still_In == false || Compute == true)
        {
            for(int index = 0; index < GROUND_ROBOT_NUMBER; index++)
            {
                Ground_Robot_Data * Current_G_Robot;
                Current_G_Robot = &Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[index];
                if(Current_G_Robot->Still_In == true)
                {
                    // Variable to hold this robot's score
                    float Temp_Score = 0;

                    if(Is_Ground_Robot_In_Rotation_Range(Current_Drone->Prev_Target_Robot, In_Match) == true || Data_Sets[Current_Data_Set].Switch_Before_Done == true || Matches[In_Match].Current_Frame <= 3 ||
                        Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[Current_Drone->Prev_Target_Robot].Still_In == false)
                    {
                        // Add (dist from green line * multiplier)
                        Temp_Score = Temp_Score + (abs(Current_G_Robot->Y - FIELD_Y) / METER_PIXEL_DIST * Data_Sets[Current_Data_Set].Green_Line_Mult);

                        // Determine dist in meters from closest edge (other than green) the multiply then add
                        float Dist_To_Closest_Edge = 0;
                        float From_Left_Dist = abs(Current_G_Robot->X - FIELD_X);
                        float From_Right_Dist = abs((FIELD_X + FIELD_WIDTH) - Current_G_Robot->X);
                        float From_Bottom_Dist = abs((FIELD_Y + FIELD_HEIGHT) - Current_G_Robot->Y);
                        if(From_Left_Dist > From_Right_Dist)
                        {
                            if(From_Right_Dist > From_Bottom_Dist)
                            {
                                Dist_To_Closest_Edge = From_Bottom_Dist;
                            }
                            else
                            {
                                Dist_To_Closest_Edge = From_Right_Dist;
                            }
                        }
                        else
                        {
                            if(From_Left_Dist > From_Bottom_Dist)
                            {
                                Dist_To_Closest_Edge = From_Bottom_Dist;
                            }
                            else
                            {
                                Dist_To_Closest_Edge = From_Left_Dist;
                            }
                        }
                        Temp_Score = Temp_Score + (Dist_To_Closest_Edge * Data_Sets[Current_Data_Set].Dist_From_Edge_Mult);


                        // Determine dist from drone, multiply and add to Temp_Score
                        float A = Current_Drone->X - Current_G_Robot->X;
                        float B = Current_Drone->Y - Current_G_Robot->Y;
                        float Dist = sqrt(A * A + B * B) / METER_PIXEL_DIST;
                        Temp_Score = Temp_Score + (Dist * Data_Sets[Current_Data_Set].Dist_From_Drone_Mult);


                        // Calculate what rotation we want the ground robot
                        float Current_Robot_Rotation = static_cast<int>(Current_G_Robot->Rotation) % 360;
                        if(Current_G_Robot->Mode == REVERSE_MODE)
                            Current_Robot_Rotation = Current_G_Robot->Frames_To_Turn * (GROUND_ROBOT_TURN_SPEED / COMP_FRAME_RATE);
                        if(Current_Robot_Rotation < 0)
                            Current_Robot_Rotation += 360;



                        // See if the ground robot's rotation is in the range we want
                        if((Matches[In_Match].Current_Frame - Current_G_Robot->Last_Frame_Reversed) / COMP_FRAME_RATE < Data_Sets[Current_Data_Set].Half)
                        {
                            if((Current_Robot_Rotation > Data_Sets[Current_Data_Set].Rot_Min && Current_Robot_Rotation < Data_Sets[Current_Data_Set].Rot_Max))
                            {
                                Temp_Score = Temp_Score + Data_Sets[Current_Data_Set].Find_Direction_Mult;
                            }
                        }
                        else
                        {
                          //  if(Draw_On_Screen == true)
                            //cout << "fire " << (Matches[In_Match].Current_Frame - Current_G_Robot->Last_Frame_Reversed) / COMP_FRAME_RATE << endl;
                            int Number_From_Top = 0;
                            for(int counter = 0; counter < GROUND_ROBOT_NUMBER; counter++)
                            {
                                if(counter != index && Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[counter].Y < Current_G_Robot->Y
                                    && Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[counter].Still_In == true)
                                {
                                    Number_From_Top = Number_From_Top + 1;
                                }
                            }
                           // if(Draw_On_Screen == true)
                            //cout << Number_From_Top << " = Number_From_Top " << Data_Sets[Current_Data_Set].After_Half_Top_Skip << " " << index << endl;
                            oho = true;
                            if(Number_From_Top < Data_Sets[Current_Data_Set].After_Half_Top_Skip)
                            {
                                Temp_Score = Temp_Score + Data_Sets[Current_Data_Set].Find_Direction_Mult;
                            }
                            else if((Current_Robot_Rotation > Data_Sets[Current_Data_Set].After_Half_Rot_Min && Current_Robot_Rotation < Data_Sets[Current_Data_Set].After_Half_Rot_Max) ||
                                (Current_Robot_Rotation > Data_Sets[Current_Data_Set].After_Half_Rot_Min - 360 && Current_Robot_Rotation < Data_Sets[Current_Data_Set].After_Half_Rot_Max - 360))
                            {
                                //if(Draw_On_Screen)
                               // cout << Number_From_Top << " >= "  << Data_Sets[Current_Data_Set].After_Half_Top_Skip << " " << Current_Robot_Rotation << " "
                                //    << (Current_Robot_Rotation > Data_Sets[Current_Data_Set].After_Half_Rot_Min && Current_Robot_Rotation < Data_Sets[Current_Data_Set].After_Half_Rot_Max)
                                //    << " " << endl;
                                Temp_Score = Temp_Score + Data_Sets[Current_Data_Set].Find_Direction_Mult;
                            }
                        }

                        if(Current_G_Robot->Mode == FORWARD_MODE)
                        {
                            // Determine if the ground robot is going to get too near to the edge
                            float Seconds_To_Next_Reverse = ((Current_G_Robot->Last_Frame_Reversed + (REVERSE_INTERVAL * COMP_FRAME_RATE)
                                - Matches[In_Match].Current_Frame) / COMP_FRAME_RATE);

                            float Temp_X = Current_G_Robot->X + cos(TO_RADIANS * Current_G_Robot->Rotation) * GROUND_ROBOT_SPEED * METER_PIXEL_DIST *
                                Seconds_To_Next_Reverse;
                            float Temp_Y = Current_G_Robot->Y + sin(TO_RADIANS * Current_G_Robot->Rotation) * GROUND_ROBOT_SPEED * METER_PIXEL_DIST *
                                Seconds_To_Next_Reverse;
                            if(Temp_Y > FIELD_Y + FIELD_HEIGHT - (HEAD_OFF_EDGE_SAFETY_DIST * METER_PIXEL_DIST) ||
                                Temp_X < FIELD_X + (HEAD_OFF_EDGE_SAFETY_DIST * METER_PIXEL_DIST) ||
                                Temp_X > FIELD_X + FIELD_WIDTH - (HEAD_OFF_EDGE_SAFETY_DIST * METER_PIXEL_DIST))
                            {
                                Points[index].X = Temp_X;
                                Points[index].Y = Temp_Y;
                                Temp_Score = Temp_Score + Data_Sets[Current_Data_Set].Head_Off_Edge_Mult;
                                if(Draw_On_Screen)
                                    cout << "oho" << endl;
                                //cout << "fire 1 " << Data_Sets[Current_Data_Set].Head_Off_Edge_Mult << " " << Temp_Score << endl;
                            }
                        }

                        //cout << Temp_Score << endl;
                        // If the ground robot has a new high score
                        if(Temp_Score > Robot_High_Score && ((Matches[In_Match].Current_Frame - Current_G_Robot->Last_Frame_Reversed) / COMP_FRAME_RATE < Data_Sets[Current_Data_Set].Half || Data_Sets[Current_Data_Set].Cont_After_Half))
                        {
                            Robot_High_Score = Temp_Score;
                            Robot_With_High_Score = index;
                        }
                       // if(oho == true)
                            //if(Draw_On_Screen)
                            //cout << Temp_Score << " = TempScore" << endl;
                    }
                }

                //cout << Robot_With_High_Score << " " << Robot_High_Score << " " << Is_Ground_Robot_In_Rotation_Range(Current_Drone->Prev_Target_Robot, In_Match) << endl;
            }
        }
        G_Robot_W_High_Score = Robot_With_High_Score;
        Current_Drone->Prev_Target_Robot = Robot_With_High_Score;
        Target_G_Robot = &Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[Robot_With_High_Score];
        //cout << Target_G_Robot->X << " " << Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[3].X << endl;
        //cout << Current_Drone->Prev_Target_Robot << endl;

     //   if(oho == true)
        //    if(Draw_On_Screen)
           // cout << Robot_With_High_Score << " was the winner " << endl;

        // Check to see what obstacle robot is closest to the drone
        float Closest_Obs_Robot_Dist = 9999999;
        float Closest_Obs_Robot = 0;
        for(int counter = 0; counter < OBSTACLE_ROBOT_NUMBER; counter++)
        {
            float A = Current_Drone->X - Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Obstacle_Robots[counter].X;
            float B = Current_Drone->Y - Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Obstacle_Robots[counter].Y;
            float Temp_Dist = sqrt(A * A + B * B);

            if(Temp_Dist < Closest_Obs_Robot_Dist)
            {
                Closest_Obs_Robot_Dist = Temp_Dist;
                Closest_Obs_Robot = counter;
            }
        }

        // If closest obstacle robot is too close, move away
        /*float min_dist = (DRONE_DIA * METER_PIXEL_DIST / 2 + OBSTACLE_ROBOT_DIA * METER_PIXEL_DIST / 2) + DRONE_FROM_OBS_MIN_SAFE_DIST * METER_PIXEL_DIST;
        if(Closest_Obs_Robot_Dist < min_dist)
        {
            float Rotation_To_Closest_Obs = TO_DEGREES * atan2(Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Obstacle_Robots[Closest_Obs_Robot].Y - Current_Drone->Y,
                Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Obstacle_Robots[Closest_Obs_Robot].X - Current_Drone->X) - 90;
            Current_Drone->X_Speed = Current_Drone->X_Speed + cos((Rotation_To_Closest_Obs + 180) * TO_RADIANS) * DRONE_ACCEL;
            Current_Drone->Y_Speed = Current_Drone->Y_Speed + sin((Rotation_To_Closest_Obs + 180) * TO_RADIANS) * DRONE_ACCEL;
        }

        // Else move toward target ground robot
        else
        {*/

            float Temp_XSpeed = Current_Drone->X_Speed;
            float Temp_YSpeed = Current_Drone->Y_Speed;
            float Temp_X = Current_Drone->X + Current_Drone->X_Speed;
            float Temp_Y = Current_Drone->Y + Current_Drone->Y_Speed;
            while(Temp_XSpeed != 0 || Temp_YSpeed != 0)
            {
                Temp_X = Temp_X + Temp_XSpeed;
                Temp_Y = Temp_Y + Temp_YSpeed;

                if(Temp_XSpeed > DRONE_ACCEL)
                {
                  Temp_XSpeed = Temp_XSpeed - DRONE_ACCEL;
                }
                else if(Temp_XSpeed < -DRONE_ACCEL)
                {
                  Temp_XSpeed = Temp_XSpeed + DRONE_ACCEL;
                }
                else
                {
                  Temp_XSpeed = 0;
                }

                if(Temp_YSpeed > DRONE_ACCEL)
                {
                  Temp_YSpeed = Temp_YSpeed - DRONE_ACCEL;
                }
                else if(Temp_YSpeed < -DRONE_ACCEL)
                {
                  Temp_YSpeed = Temp_YSpeed + DRONE_ACCEL;
                }
                else
                {
                  Temp_YSpeed = 0;
                }
            }

            //cout << Temp_XSpeed << " " << Current_Drone->X_Speed << endl;
            if(Temp_X > Target_G_Robot->X)
            {
                Current_Drone->X_Speed = Current_Drone->X_Speed + -DRONE_ACCEL;
            }
            else
            {
                Current_Drone->X_Speed = Current_Drone->X_Speed + DRONE_ACCEL;
            }

            if(Temp_Y > Target_G_Robot->Y)
            {
                Current_Drone->Y_Speed = Current_Drone->Y_Speed + -DRONE_ACCEL;
            }
            else
            {
                Current_Drone->Y_Speed = Current_Drone->Y_Speed + DRONE_ACCEL;
            }
        //}
        //cout << Target_G_Robot->X << endl;

        Current_Drone->X = Current_Drone->X + Current_Drone->X_Speed;
        Current_Drone->Y = Current_Drone->Y + Current_Drone->Y_Speed;

        float Rotation = static_cast<int>(Target_G_Robot->Rotation) % 360;
                if(Rotation < 0)
                    Rotation += 360;
        //cout << ((Matches[In_Match].Current_Frame - Target_G_Robot->Last_Frame_Reversed) / COMP_FRAME_RATE < Data_Sets[Current_Data_Set].Half) << " || " << Data_Sets[Current_Data_Set].Cont_After_Half << endl;

        float A = Current_Drone->X - Target_G_Robot->X;
        float B = Current_Drone->Y - Target_G_Robot->Y;
        if(sqrt(A * A + B * B) / METER_PIXEL_DIST < TOP_TOUCH_MAX_OFFSET)
        {
            // Determine if drone is moving slow enough compared to ground robot to turn it
            float Drone_Movement_X = Current_Drone->X_Speed;
            float Drone_Movement_Y = Current_Drone->Y_Speed;
            float G_Robot_Movement_X = cos(Target_G_Robot->Rotation * TO_RADIANS) * GROUND_ROBOT_PIXEL_SPEED;
            float G_Robot_Movement_Y = sin(Target_G_Robot->Rotation * TO_RADIANS) * GROUND_ROBOT_PIXEL_SPEED;
            float A = Drone_Movement_X - G_Robot_Movement_X;
            float B = Drone_Movement_Y - G_Robot_Movement_Y;

            //cout << (Matches[In_Match].Current_Frame - Target_G_Robot->Last_Frame_Reversed) / COMP_FRAME_RATE << " < " << Data_Sets[Current_Data_Set].Half << endl;
            if(sqrt(A * A + B * B) / METER_PIXEL_DIST < TOP_TOUCH_MAX_SPEED_DIFF)
            {
                bool Going_Off_Edge = false;
                if(Target_G_Robot->Mode == FORWARD_MODE)
                {
                    // Determine if the ground robot is going to get too near to the edge
                    float Seconds_To_Next_Reverse = ((Target_G_Robot->Last_Frame_Reversed + (REVERSE_INTERVAL * COMP_FRAME_RATE)
                        - Matches[In_Match].Current_Frame) / COMP_FRAME_RATE);

                    float Temp_X = Target_G_Robot->X + cos(TO_RADIANS * Target_G_Robot->Rotation) * GROUND_ROBOT_SPEED * METER_PIXEL_DIST *
                        Seconds_To_Next_Reverse;
                    float Temp_Y = Target_G_Robot->Y + sin(TO_RADIANS * Target_G_Robot->Rotation) * GROUND_ROBOT_SPEED * METER_PIXEL_DIST *
                        Seconds_To_Next_Reverse;
                    if(Temp_Y > FIELD_Y + FIELD_HEIGHT - (HEAD_OFF_EDGE_SAFETY_DIST * METER_PIXEL_DIST) ||
                        Temp_X < FIELD_X + (HEAD_OFF_EDGE_SAFETY_DIST * METER_PIXEL_DIST) ||
                        Temp_X > FIELD_X + FIELD_WIDTH - (HEAD_OFF_EDGE_SAFETY_DIST * METER_PIXEL_DIST))
                    {
                        Going_Off_Edge = true;
                    }
                }
                if(Going_Off_Edge == true)
                {
                    if(Temp_X < FIELD_X + (HEAD_OFF_EDGE_SAFETY_DIST * METER_PIXEL_DIST))
                    {
                        float Seconds_To_Next_Reverse = ((Target_G_Robot->Last_Frame_Reversed + (REVERSE_INTERVAL * COMP_FRAME_RATE)
                            - Matches[In_Match].Current_Frame) / COMP_FRAME_RATE);
                        int Dist_To_Edge = GROUND_ROBOT_SPEED * METER_PIXEL_DIST * Seconds_To_Next_Reverse;
                        if(Temp_X - Dist_To_Edge < FIELD_X + (HEAD_OFF_EDGE_SAFETY_DIST * METER_PIXEL_DIST))
                        {
                            if((Rotation < Data_Sets[Current_Data_Set].On_Left_Rot_Min || Rotation > Data_Sets[Current_Data_Set].On_Left_Rot_Max) &&
                                (Rotation < Data_Sets[Current_Data_Set].On_Left_Rot_Min - 360 || Rotation > Data_Sets[Current_Data_Set].On_Left_Rot_Max - 360))
                            {
                                Target_G_Robot->Mode = TOP_TOUCH_MODE;
                                Target_G_Robot->Frames_To_Turn = (TOP_TOUCH_DEGREES / GROUND_ROBOT_TURN_SPEED) * COMP_FRAME_RATE;
                            }
                        }
                        else
                        {
                            if(Rotation < Data_Sets[Current_Data_Set].On_Right_Rot_Min || Rotation > Data_Sets[Current_Data_Set].On_Right_Rot_Max)
                            {
                                Target_G_Robot->Mode = TOP_TOUCH_MODE;
                                Target_G_Robot->Frames_To_Turn = (TOP_TOUCH_DEGREES / GROUND_ROBOT_TURN_SPEED) * COMP_FRAME_RATE;
                            }
                        }
                    }
                    if(Temp_X > FIELD_X + FIELD_WIDTH - (HEAD_OFF_EDGE_SAFETY_DIST * METER_PIXEL_DIST))
                    {
                        float Seconds_To_Next_Reverse = ((Target_G_Robot->Last_Frame_Reversed + (REVERSE_INTERVAL * COMP_FRAME_RATE)
                            - Matches[In_Match].Current_Frame) / COMP_FRAME_RATE);
                        int Dist_To_Edge = GROUND_ROBOT_SPEED * METER_PIXEL_DIST * Seconds_To_Next_Reverse;
                        if(Temp_X - Dist_To_Edge > FIELD_X + FIELD_WIDTH - (HEAD_OFF_EDGE_SAFETY_DIST * METER_PIXEL_DIST))
                        {
                            if((Rotation < Data_Sets[Current_Data_Set].On_Right_Rot_Min || Rotation > Data_Sets[Current_Data_Set].On_Right_Rot_Max))
                            {
                                Target_G_Robot->Mode = TOP_TOUCH_MODE;
                                Target_G_Robot->Frames_To_Turn = (TOP_TOUCH_DEGREES / GROUND_ROBOT_TURN_SPEED) * COMP_FRAME_RATE;
                            }
                        }
                        else
                        {
                            if((Rotation < Data_Sets[Current_Data_Set].On_Left_Rot_Min || Rotation > Data_Sets[Current_Data_Set].On_Left_Rot_Max) &&
                                (Rotation < Data_Sets[Current_Data_Set].On_Left_Rot_Min - 360 || Rotation > Data_Sets[Current_Data_Set].On_Left_Rot_Max - 360))
                            {
                                Target_G_Robot->Mode = TOP_TOUCH_MODE;
                                Target_G_Robot->Frames_To_Turn = (TOP_TOUCH_DEGREES / GROUND_ROBOT_TURN_SPEED) * COMP_FRAME_RATE;
                            }
                        }
                    }
                }
                else
                {
                    if(((Matches[In_Match].Current_Frame - Target_G_Robot->Last_Frame_Reversed) / COMP_FRAME_RATE < Data_Sets[Current_Data_Set].Half))
                    {
                        if((Rotation < Data_Sets[Current_Data_Set].Rot_Min || Rotation > Data_Sets[Current_Data_Set].Rot_Max))
                        {
                            //if(Draw_On_Screen)
                            //cout << Rotation << " < " << Data_Sets[Current_Data_Set].Rot_Min << endl;
                            Target_G_Robot->Mode = TOP_TOUCH_MODE;
                            Target_G_Robot->Frames_To_Turn = (TOP_TOUCH_DEGREES / GROUND_ROBOT_TURN_SPEED) * COMP_FRAME_RATE;
                        }
                    }


                    // If after the half way point to the 180 degree turn
                    if(Data_Sets[Current_Data_Set].Cont_After_Half == true &&
                        (Matches[In_Match].Current_Frame - Target_G_Robot->Last_Frame_Reversed) / COMP_FRAME_RATE > Data_Sets[Current_Data_Set].Half)
                    {
                         int Number_From_Top = 0;
                         // Determine if we should turn this ground robot down
                        for(int counter = 0; counter < GROUND_ROBOT_NUMBER; counter++)
                        {
                            if(counter != Robot_With_High_Score && Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[counter].Y < Target_G_Robot->Y
                                && Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[counter].Still_In == true)
                            {
                                Number_From_Top = Number_From_Top + 1;
                            }
                        }
                        if(Number_From_Top >= Data_Sets[Current_Data_Set].After_Half_Top_Skip)
                        {
                            if((Rotation < Data_Sets[Current_Data_Set].After_Half_Rot_Min || Rotation > Data_Sets[Current_Data_Set].After_Half_Rot_Max) &&
                                (Rotation < Data_Sets[Current_Data_Set].After_Half_Rot_Min - 360 || Rotation > Data_Sets[Current_Data_Set].After_Half_Rot_Max - 360))
                            {
                                //if(Draw_On_Screen)
                                //cout << "oho" << endl;
                                Target_G_Robot->Mode = TOP_TOUCH_MODE;
                                Target_G_Robot->Frames_To_Turn = (TOP_TOUCH_DEGREES / GROUND_ROBOT_TURN_SPEED) * COMP_FRAME_RATE;
                            }
                        }
                    }
                }
            }
        }
    }


    // Move the ground robots
    for(int index = 0; index < GROUND_ROBOT_NUMBER; index++)
    {
        Ground_Robot_Data * Current_G_Robot;
        float Prev_X = 0;
        float Prev_Y = 0;


        Current_G_Robot = &Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[index];
        if(Current_G_Robot->Still_In == true)
        {
            Prev_X = Current_G_Robot->X;
            Prev_Y = Current_G_Robot->Y;


            if(Current_G_Robot->Mode == FORWARD_MODE)
            {
                // If its time to reverse
                if(Current_G_Robot->Last_Frame_Reversed < Matches[In_Match].Current_Frame - (COMP_FRAME_RATE * REVERSE_INTERVAL))
                {
                    Current_G_Robot->Mode = REVERSE_MODE;
                    Current_G_Robot->Last_Frame_Reversed = Matches[In_Match].Current_Frame;
                    Current_G_Robot->Frames_To_Turn = (REVERSE_DEGREES / GROUND_ROBOT_TURN_SPEED) * COMP_FRAME_RATE;
                }
                else if(Current_G_Robot->Last_Frame_Noise < Matches[In_Match].Current_Frame - (COMP_FRAME_RATE * NOISE_INTERVAL))
                {
                    Current_G_Robot->Mode =  NOISE_MODE;
                    Current_G_Robot->Last_Frame_Noise = Matches[In_Match].Current_Frame;
                    uniform_int_distribution<int> distribution(0, (static_cast<int>(NOISE_MAX_ROTATION - NOISE_MIN_ROTATION) + 1));
                    Current_G_Robot->Frames_To_Turn = (distribution(generator) / GROUND_ROBOT_TURN_SPEED) * COMP_FRAME_RATE;
                }

                // Assign new X and Y to ground robot
                Current_G_Robot->X = Current_G_Robot->X + cos(Current_G_Robot->Rotation * TO_RADIANS) * GROUND_ROBOT_PIXEL_SPEED;
                Current_G_Robot->Y = Current_G_Robot->Y + sin(Current_G_Robot->Rotation * TO_RADIANS) * GROUND_ROBOT_PIXEL_SPEED;

                for(int index2 = 0; index2 < OBSTACLE_ROBOT_NUMBER; index2++)
                {
                    float A;
                    float B;
                    Obstacle_Robot_Data * Current_O_Robot;

                    Current_O_Robot = &Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Obstacle_Robots[index2];
                    A = Current_G_Robot->X - Current_O_Robot->X;
                    B = Current_G_Robot->Y - Current_O_Robot->Y;
                    if(sqrt(A * A + B * B) < (GROUND_ROBOT_DIA / 2 + OBSTACLE_ROBOT_DIA / 2) * METER_PIXEL_DIST)
                    {
                        Current_G_Robot->X = Prev_X;
                        Current_G_Robot->Y = Prev_Y;
                        Current_G_Robot->Frames_To_Turn = (COLLISION_DEGREES / GROUND_ROBOT_TURN_SPEED) * COMP_FRAME_RATE;
                        Current_G_Robot->Mode = COLLISION_MODE;
                    }
                }

                for(int index2 = 0; index2 < GROUND_ROBOT_NUMBER; index2++)
                {
                    float A;
                    float B;
                    Ground_Robot_Data * Current_G_Robot2;
                    Current_G_Robot2 = &Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[index2];
                    if(Current_G_Robot2->Still_In == true)
                    {
                        if(index2 != index)
                        {
                            A = Current_G_Robot->X - Current_G_Robot2->X;
                            B = Current_G_Robot->Y - Current_G_Robot2->Y;
                            if(sqrt(A * A + B * B) < (GROUND_ROBOT_DIA / 2 + OBSTACLE_ROBOT_DIA / 2) * METER_PIXEL_DIST)
                            {
                                Current_G_Robot->X = Prev_X;
                                Current_G_Robot->Y = Prev_Y;
                                Current_G_Robot->Frames_To_Turn = (COLLISION_DEGREES / GROUND_ROBOT_TURN_SPEED) * COMP_FRAME_RATE;
                                Current_G_Robot->Mode = COLLISION_MODE;
                            }
                        }
                    }
                }
            }

        else if(Current_G_Robot->Mode == REVERSE_MODE)
        {
            // Turn
            Current_G_Robot->Rotation = Current_G_Robot->Rotation + (GROUND_ROBOT_TURN_SPEED / COMP_FRAME_RATE);

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
            Current_G_Robot->Rotation = Current_G_Robot->Rotation + (GROUND_ROBOT_TURN_SPEED / COMP_FRAME_RATE);

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
            Current_G_Robot->Rotation = Current_G_Robot->Rotation + (GROUND_ROBOT_TURN_SPEED / COMP_FRAME_RATE);

            // Decrement the frames left in the turn
            Current_G_Robot->Frames_To_Turn = Current_G_Robot->Frames_To_Turn - 1;

            // If the turn is done
            if(Current_G_Robot->Frames_To_Turn <= 0)
            {
                // Start driving forward again
                Current_G_Robot->Mode = FORWARD_MODE;
            }
        }
        else if(Current_G_Robot->Mode == TOP_TOUCH_MODE)
        {
            // Turn
            Current_G_Robot->Rotation = Current_G_Robot->Rotation + (GROUND_ROBOT_TURN_SPEED / COMP_FRAME_RATE);

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
    }
    // Move the obstacle robots
    for(int index = 0; index < OBSTACLE_ROBOT_NUMBER; index++)
    {
        Obstacle_Robot_Data * Current_O_Robot;
        float Prev_X = 0;
        float Prev_Y = 0;
        int Prev_Rotation = 0;

        Current_O_Robot = &Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Obstacle_Robots[index];

        Prev_X = Current_O_Robot->X;
        Prev_Y = Current_O_Robot->Y;
        Prev_Rotation = Current_O_Robot->Rotation;

        float Rotation_Change = DEGREES_IN_CIRCLE / COMP_FRAME_RATE / OBSTACLE_ROBOT_FULL_CIRLE_TIME;
        Current_O_Robot->Rotation = Current_O_Robot->Rotation - Rotation_Change;
        Current_O_Robot->X = Current_O_Robot->X + cos(Current_O_Robot->Rotation * TO_RADIANS) * OBSTACLE_ROBOT_PIXEL_SPEED;
        Current_O_Robot->Y = Current_O_Robot->Y + sin(Current_O_Robot->Rotation * TO_RADIANS) * OBSTACLE_ROBOT_PIXEL_SPEED;

        for(int index2 = 0; index2 < GROUND_ROBOT_NUMBER; index2++)
        {
            float A;
            float B;
            Ground_Robot_Data * Current_G_Robot;

            Current_G_Robot = &Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[index2];
            if(Current_G_Robot->Still_In == true)
            {
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

    for(int index = 0; index < GROUND_ROBOT_NUMBER; index++)
    {
        Ground_Robot_Data * Current_G_Robot;
        Current_G_Robot = &Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[index];
        if(Current_G_Robot->X < FIELD_X || Current_G_Robot->X > FIELD_X + FIELD_WIDTH || Current_G_Robot->Y > FIELD_Y + FIELD_HEIGHT)
        {
            Current_G_Robot->Still_In = false;
        }
        else if(Current_G_Robot->Y < FIELD_Y)
        {
            Current_G_Robot->Still_In = false;
            Current_G_Robot->Scored = true;
        }
    }

    bool done = true;
    for(int index = 0; index < GROUND_ROBOT_NUMBER; index++)
    {
        Ground_Robot_Data * Current_G_Robot;
        Current_G_Robot = &Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[index];
        if(Current_G_Robot->Still_In == true)
        {
            done = false;
        }
    }
    if(done == true && Matches[In_Match].Has_Ended == false)
    {
        Matches[In_Match].Frame_Ended = Matches[In_Match].Current_Frame;
        Matches[In_Match].Has_Ended = true;
    }
}



void Init(int In_Match)
{

    Matches[In_Match].Array_Of_Frames.resize(FRAME_LIMIT + 1);
    for(int index = 0; index < FRAME_LIMIT + 1; index++)
    {
        Matches[In_Match].Array_Of_Frames[index].Ground_Robots.resize(GROUND_ROBOT_NUMBER);
        Matches[In_Match].Array_Of_Frames[index].Obstacle_Robots.resize(OBSTACLE_ROBOT_NUMBER);
        Matches[In_Match].Array_Of_Frames[index].Drones.resize(DRONE_NUMBER);
        Matches[In_Match].Array_Of_Frames[index].Target_In_Position = false;
    }
    for(int index = 0; index < GROUND_ROBOT_NUMBER; index++)
    {
        float Temp_Rotation = index * DEGREES_IN_CIRCLE / GROUND_ROBOT_NUMBER;
        Matches[In_Match].Array_Of_Frames[0].Ground_Robots[index].X = ARENA_WIDTH / 2 + cos(Temp_Rotation * TO_RADIANS) *  METER_PIXEL_DIST;
        Matches[In_Match].Array_Of_Frames[0].Ground_Robots[index].Y = ARENA_HEIGHT / 2 + sin(Temp_Rotation * TO_RADIANS) *  METER_PIXEL_DIST;
        Matches[In_Match].Array_Of_Frames[0].Ground_Robots[index].Rotation = Temp_Rotation;
        Matches[In_Match].Array_Of_Frames[0].Ground_Robots[index].Color = (index % 2 == 0) ? RED_ID : BLUE_ID;
        Matches[In_Match].Array_Of_Frames[0].Ground_Robots[index].Frames_To_Turn = 0;
        Matches[In_Match].Array_Of_Frames[0].Ground_Robots[index].Turn_Direction_Mult = 0;
        Matches[In_Match].Array_Of_Frames[0].Ground_Robots[index].Still_In = true;
        Matches[In_Match].Array_Of_Frames[0].Ground_Robots[index].Scored = false;
        Matches[In_Match].Array_Of_Frames[0].Ground_Robots[index].Mode = FORWARD_MODE;
        Matches[In_Match].Array_Of_Frames[0].Ground_Robots[index].Last_Frame_Reversed = 0;
        Matches[In_Match].Array_Of_Frames[0].Ground_Robots[index].Last_Frame_Noise = 0;
        Matches[In_Match].Array_Of_Frames[0].Ground_Robots[index].Time_Left_To_Rotate = 0;
    }


    for(int index = 0; index < OBSTACLE_ROBOT_NUMBER; index++)
    {
        float Temp_Rotation = index * DEGREES_IN_CIRCLE / OBSTACLE_ROBOT_NUMBER;
        float Obstacle_Robot_Direction = Temp_Rotation - 90;
        Matches[In_Match].Array_Of_Frames[0].Obstacle_Robots[index].X = FIELD_X + FIELD_WIDTH / 2 + cos(Temp_Rotation * TO_RADIANS) * METER_PIXEL_DIST * OBSTACLE_ROBOT_PATH_RADIUS;
        Matches[In_Match].Array_Of_Frames[0].Obstacle_Robots[index].Y = FIELD_Y + FIELD_HEIGHT / 2 + sin(Temp_Rotation * TO_RADIANS) * METER_PIXEL_DIST * OBSTACLE_ROBOT_PATH_RADIUS;
        Matches[In_Match].Array_Of_Frames[0].Obstacle_Robots[index].Rotation = Obstacle_Robot_Direction;
    }


    for(int index = 0; index < DRONE_NUMBER; index++)
    {
        Matches[In_Match].Array_Of_Frames[0].Drones[index].X = FIELD_X + FIELD_WIDTH + (DRONE_DIA * METER_PIXEL_DIST);
        Matches[In_Match].Array_Of_Frames[0].Drones[index].Y = FIELD_Y + FIELD_HEIGHT / 2;
        Matches[In_Match].Array_Of_Frames[0].Drones[index].X_Speed = 0;
        Matches[In_Match].Array_Of_Frames[0].Drones[index].Y_Speed = 0;
        Matches[In_Match].Array_Of_Frames[0].Drones[index].Target_Robot = 0;
        Matches[In_Match].Array_Of_Frames[0].Drones[index].Ok_To_Leave = true;
        Matches[In_Match].Array_Of_Frames[0].Drones[index].Prev_Target_Robot = 0;
    }
    Matches[In_Match].Frame_Ended = FRAME_LIMIT;
    Matches[In_Match].Has_Ended = false;
    Matches[In_Match].Current_Frame = 0;

    /*int Ground_Robots_Scored = 0;
    for(int counter = 0; counter < GROUND_ROBOT_NUMBER; counter++)
    {
        if(Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[counter].Scored == true)
        {
            Ground_Robots_Scored++;
        }
    }
    cout  << Ground_Robots_Scored << "fire ";*/
    return;
}



void Reload_Match()
{
}



void Init_Mults(int index, int Matches)
{
    Gen_Random_Set(index);
    Data_Sets[index].Results.resize(Matches);
    return;
}



void Score_Match(int In_Match)
{
    int Ground_Robots_Scored = 0;

    // Count ground robots scored
    for(int counter = 0; counter < GROUND_ROBOT_NUMBER; counter++)
    {
        if(Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[counter].Scored == true)
        {
            Ground_Robots_Scored++;
        }
    }
    Data_Sets[Current_Data_Set].Results[Current_Match_In_Set].Ground_Robots_Scored = Ground_Robots_Scored;
    //cout << Ground_Robots_Scored << "ok ";
    Data_Sets[Current_Data_Set].Results[Current_Match_In_Set].Time_Taken = Matches[In_Match].Frame_Ended;
    // Start the score with a starting score
    float Temp_Score = MATCH_START_SCORE;

    // Add points per robot scored
    Temp_Score = Temp_Score + (Ground_Robots_Scored * ROBOT_SCORED_POINTS);

    // Subtract points per robot not scored
    Temp_Score = Temp_Score + (GROUND_ROBOT_NUMBER - Ground_Robots_Scored) * ROBOT_NOT_SCORED_POINTS;

    // Subtract points per minute used in the match
    Temp_Score = Temp_Score + static_cast<int>(Matches[In_Match].Frame_Ended / COMP_FRAME_RATE / SECONDS_IN_MINUTE) * MINUTE_USED_POINTS;

    // Set the result's score to Temp_Score
    Data_Sets[Current_Data_Set].Results[Current_Match_In_Set].Score = Temp_Score;
}



void Calc_Set_Ave(int Num_Of_Matches)
{
    float Robots_Scored_Sum = 0;
    float Time_Used_Sum = 0;
    float Score_Sum = 0;
    for(int counter = 0; counter < Num_Of_Matches; counter++)
    {
        Robots_Scored_Sum = Robots_Scored_Sum + Data_Sets[Current_Data_Set].Results[counter].Ground_Robots_Scored;
        Time_Used_Sum = Time_Used_Sum + Data_Sets[Current_Data_Set].Results[counter].Time_Taken;
        Score_Sum = Score_Sum + Data_Sets[Current_Data_Set].Results[counter].Score;
    }
    Data_Sets[Current_Data_Set].Ave_Robots = Robots_Scored_Sum / Num_Of_Matches;
    Data_Sets[Current_Data_Set].Ave_Time_Taken = Time_Used_Sum / Num_Of_Matches;
    Data_Sets[Current_Data_Set].Ave_Score = Score_Sum / Num_Of_Matches;
}



void Output_Set(int index)
{
    cout << "Set " << index << endl;
    cout << "Data_Sets[index].Green_Line_Mult = " << Data_Sets[index].Green_Line_Mult << ";" << endl;
    cout << "Data_Sets[index].Dist_From_Edge_Mult = " << Data_Sets[index].Dist_From_Edge_Mult << ";" << endl;
    cout << "Data_Sets[index].Dist_From_Drone_Mult = " << Data_Sets[index].Dist_From_Drone_Mult << ";" << endl;
    cout << "Data_Sets[index].Head_Off_Edge_Mult = " << Data_Sets[index].Head_Off_Edge_Mult << ";" << endl;
    cout << "Data_Sets[index].Find_Direction_Mult = " << Data_Sets[index].Find_Direction_Mult << ";" << endl;
    cout << "Data_Sets[index].Rot_Min = " << Data_Sets[index].Rot_Min << ";" << endl;
    cout << "Data_Sets[index].Rot_Max = " << Data_Sets[index].Rot_Max << ";" << endl;
    cout << "Data_Sets[index].Cont_After_Half = " << Data_Sets[index].Cont_After_Half << ";" << endl;
    cout << "Data_Sets[index].Half = " << Data_Sets[index].Half << ";" << endl;
    cout << "Data_Sets[index].On_Left_Rot_Min = " << Data_Sets[index].After_Half_Rot_Min << ";" << endl;
    cout << "Data_Sets[index].After_Half_Rot_Max = " << Data_Sets[index].After_Half_Rot_Max << ";" << endl;
    cout << "Data_Sets[index].On_Left_Rot_Min = " << Data_Sets[index].On_Left_Rot_Min << ";" << endl;
    cout << "Data_Sets[index].On_Left_Rot_Max = " << Data_Sets[index].On_Left_Rot_Max << ";" << endl;
    cout << "Data_Sets[index].On_Right_Rot_Min = " << Data_Sets[index].On_Right_Rot_Min << ";" << endl;
    cout << "Data_Sets[index].On_Right_Rot_Max = " << Data_Sets[index].On_Right_Rot_Max << ";" << endl;
    cout << "Data_Sets[index].Switch_Before_Done = " << Data_Sets[index].Switch_Before_Done << ";" << endl;
    cout << "Data_Sets[index].After_Half_Top_Skip = " << Data_Sets[index].After_Half_Top_Skip << ";" << endl;
    cout << "Robots scored ave: " << Data_Sets[index].Ave_Robots << endl;
    cout << "Time used ave: " << Data_Sets[index].Ave_Time_Taken / COMP_FRAME_RATE / SECONDS_IN_MINUTE << endl;
    cout << "Score ave: " << Data_Sets[index].Ave_Score << endl;
    cout << endl << endl;
}



void Sort_Sets_By_Score()
{
    Data_Set Temp_Set;
    for(int counter1 = 0; counter1 < DATA_SET_COUNT; counter1++)
    {
        for(int counter2 = 0; counter2 < DATA_SET_COUNT - counter1; counter2++)
        {
            if(Data_Sets[counter2].Ave_Score < Data_Sets[counter2 + 1].Ave_Score)
            {
                Temp_Set = Data_Sets[counter2];
                Data_Sets[counter2] = Data_Sets[counter2 + 1];
                Data_Sets[counter2 + 1] = Temp_Set;
            }
        }
    }
/*
    for(int counter1 = 0; counter1 < DATA_SET_COUNT; counter1++)
    {
        Output_Set(counter1);
    }
    */
    //Sleep(500000);

}



void Make_New_Sets(int Matches_Num)
{
    for(int counter = 0; counter < DATA_SET_COUNT; counter++)
    {
        Data_Sets[counter].Results.resize(Matches_Num);
    }
    for(int counter1 = 0; counter1 < 1; counter1++)
    {
    for(int counter = 0; counter < round(DATA_SET_COUNT / 4); counter++)
    {
        int Cur_Set = counter + DATA_SET_COUNT / 2;
        Data_Sets[Cur_Set].Dist_From_Drone_Mult = (Data_Sets[counter * 2].Dist_From_Drone_Mult + Data_Sets[counter * 2 + 1].Dist_From_Drone_Mult) / 2;
        Data_Sets[Cur_Set].Dist_From_Edge_Mult = (Data_Sets[counter * 2].Dist_From_Edge_Mult + Data_Sets[counter * 2 + 1].Dist_From_Edge_Mult) / 2;
        Data_Sets[Cur_Set].Green_Line_Mult = (Data_Sets[counter * 2].Green_Line_Mult + Data_Sets[counter * 2 + 1].Green_Line_Mult) / 2;
        Data_Sets[Cur_Set].Head_Off_Edge_Mult = (Data_Sets[counter * 2].Head_Off_Edge_Mult + Data_Sets[counter * 2 + 1].Head_Off_Edge_Mult) / 2;
        Data_Sets[Cur_Set].Find_Direction_Mult = (Data_Sets[counter * 2].Find_Direction_Mult + Data_Sets[counter * 2 + 1].Find_Direction_Mult) / 2;
        Data_Sets[Cur_Set].Rot_Min = (Data_Sets[counter * 2].Rot_Min + Data_Sets[counter * 2 + 1].Rot_Min) / 2;
        Data_Sets[Cur_Set].Rot_Max = (Data_Sets[counter * 2].Rot_Max + Data_Sets[counter * 2 + 1].Rot_Max) / 2;
        Data_Sets[Cur_Set].Cont_After_Half = round((Data_Sets[counter * 2].Cont_After_Half + Data_Sets[counter * 2 + 1].Cont_After_Half) / 2);
        Data_Sets[Cur_Set].Half = (Data_Sets[counter * 2].Half + Data_Sets[counter * 2 + 1].Half) / 2;

        uniform_int_distribution<int> distribution(0, RAND_MAX);
        uniform_int_distribution<int> change_range(MUT_ON_VARIABLE_MIN, MUT_ON_VARIABLE_MAX);
        uniform_real_distribution<double> rand_change(-2, 2);
        uniform_real_distribution<double> rot_change(-10, 10);
        if(change_range(generator) <= RAND_MUT_CHANCE)
        {
            float Green_Line_Mult_Range = ((GREEN_LINE_MULT_MAX - GREEN_LINE_MULT_MIN) + 1);
            Data_Sets[Cur_Set].Green_Line_Mult = static_cast<float>(distribution(generator)) / RAND_MAX * Green_Line_Mult_Range + GREEN_LINE_MULT_MIN;
        }
        else if(change_range(generator) <= INCREMENT_MUT_CHANCE)
        {
            Data_Sets[Cur_Set].Green_Line_Mult += rand_change(generator);
        }
        if(change_range(generator) <= RAND_MUT_CHANCE)
        {
            float Dist_From_Edge_Mult_Range = ((DIST_FROM_EDGE_MAX - DIST_FROM_EDGE_MIN) + 1);
            Data_Sets[Cur_Set].Dist_From_Edge_Mult = static_cast<float>(distribution(generator)) / RAND_MAX * Dist_From_Edge_Mult_Range + DIST_FROM_EDGE_MIN;
        }
        else if(change_range(generator) <= INCREMENT_MUT_CHANCE)
        {
            Data_Sets[Cur_Set].Dist_From_Edge_Mult += rand_change(generator);
        }
        if(change_range(generator) <= RAND_MUT_CHANCE)
        {
            float Dist_From_Drone_Mult_Range = ((DIST_FROM_DRONE_MAX - DIST_FROM_DRONE_MIN) + 1);
            Data_Sets[Cur_Set].Dist_From_Drone_Mult = static_cast<float>(distribution(generator)) / RAND_MAX * Dist_From_Drone_Mult_Range + DIST_FROM_DRONE_MIN;
        }
        else if(change_range(generator) <= INCREMENT_MUT_CHANCE)
        {
            Data_Sets[Cur_Set].Dist_From_Drone_Mult += rand_change(generator);
        }
        if(change_range(generator) <= RAND_MUT_CHANCE)
        {
            float Head_Off_Edge_Mult_Range = ((HEAD_OFF_EDGE_MAX - HEAD_OFF_EDGE_MIN) + 1);
            Data_Sets[Cur_Set].Head_Off_Edge_Mult = static_cast<float>(distribution(generator)) / RAND_MAX * Head_Off_Edge_Mult_Range + HEAD_OFF_EDGE_MIN;
        }
        else if(change_range(generator) <= INCREMENT_MUT_CHANCE)
        {
            Data_Sets[Cur_Set].Head_Off_Edge_Mult += rand_change(generator);
        }
        if(change_range(generator) <= RAND_MUT_CHANCE)
        {
            float Find_Direction_Mult_Range = ((FIND_DIRECTION_MAX - FIND_DIRECTION_MIN) + 1);
            Data_Sets[Cur_Set].Find_Direction_Mult = static_cast<float>(distribution(generator)) / RAND_MAX * Find_Direction_Mult_Range + FIND_DIRECTION_MIN;
        }
        else if(change_range(generator) <= INCREMENT_MUT_CHANCE)
        {
            Data_Sets[Cur_Set].Find_Direction_Mult += rand_change(generator);
        }
        cout << Cur_Set << " ROT_MIN" << endl;
        if(change_range(generator) <= RAND_MUT_CHANCE)
        {
            do
            {
                float Rot_Min_Range = (ROT_MAX - ROT_MIN);
                Data_Sets[Cur_Set].Rot_Min = static_cast<float>(distribution(generator)) / RAND_MAX * Rot_Min_Range + ROT_MIN;
            }while(Data_Sets[Cur_Set].Rot_Min >= Data_Sets[Cur_Set].Rot_Max || Data_Sets[Cur_Set].Rot_Min > ROT_MAX - 5);
        }
        else if(change_range(generator) <= INCREMENT_MUT_CHANCE)
        {
            int index = 0;
            float Temp;
            do
            {
                Temp = rot_change(generator) + Data_Sets[Cur_Set].Rot_Min;
                index++;
            }while((Temp >= Data_Sets[Cur_Set].Rot_Max || Temp > ROT_MAX - 5) && index < MAX_MUT_TRIES);
            Data_Sets[Cur_Set].Rot_Min = Temp;
        }
        cout << Cur_Set << " ROT_MAX" << endl;
        if(change_range(generator) <= RAND_MUT_CHANCE)
        {
            do
            {
                float Rot_Max_Range = (ROT_MAX - ROT_MIN);
                Data_Sets[Cur_Set].Rot_Max = static_cast<float>(distribution(generator)) / RAND_MAX * Rot_Max_Range + ROT_MIN;
            }while(Data_Sets[Cur_Set].Rot_Max <= Data_Sets[Cur_Set].Rot_Min);
        }
        else if(change_range(generator) <= INCREMENT_MUT_CHANCE)
        {
            cout <<  "in 2" << endl;
            int index = 0;
            float Temp;
            do
            {
                Temp = rot_change(generator) + Data_Sets[Cur_Set].Rot_Max;
                index++;
            }while(Temp <= Data_Sets[Cur_Set].Rot_Min && index < MAX_MUT_TRIES);
            Data_Sets[Cur_Set].Rot_Max = Temp;
        }

        if(change_range(generator) <= 25)
        {
            int Continue_After_Half_Range = (CONTINUE_AFTER_HALF_ON - CONTINUE_AFTER_HALF_OFF);
            Data_Sets[Cur_Set].Cont_After_Half = round(static_cast<float>(distribution(generator)) / RAND_MAX) * Continue_After_Half_Range;
        }
        cout << Cur_Set << " AFTER_MIN" << endl;
        if(change_range(generator) <= RAND_MUT_CHANCE)
        {
            cout << " in 1 " << endl;
            do
            {
                float After_Half_Rot_Min_Range = (AFTER_HALF_ROT_MAX - AFTER_HALF_ROT_MIN);
                Data_Sets[Cur_Set].After_Half_Rot_Min = static_cast<float>(distribution(generator)) / RAND_MAX * After_Half_Rot_Min_Range + AFTER_HALF_ROT_MIN;
                cout << Data_Sets[Cur_Set].After_Half_Rot_Min << " " << Data_Sets[Cur_Set].After_Half_Rot_Max;
            }while(Data_Sets[Cur_Set].After_Half_Rot_Min >= Data_Sets[Cur_Set].After_Half_Rot_Max || Data_Sets[Cur_Set].After_Half_Rot_Max - Data_Sets[Cur_Set].After_Half_Rot_Min >= 360
                   || Data_Sets[Cur_Set].After_Half_Rot_Min > AFTER_HALF_ROT_MAX - 5);
        }
        else if(change_range(generator) <= INCREMENT_MUT_CHANCE)
        {
            cout << " in 2 " << endl;
            int index = 0;
            float Temp;
            do
            {
                Temp = rot_change(generator) + Data_Sets[Cur_Set].After_Half_Rot_Min;
                index++;
            }while((Temp >= Data_Sets[Cur_Set].After_Half_Rot_Max  || Data_Sets[Cur_Set].After_Half_Rot_Max - Temp >= 360
                   || Temp > AFTER_HALF_ROT_MAX - 5) && index < MAX_MUT_TRIES);
            if(index < MAX_MUT_TRIES)
            Data_Sets[Cur_Set].After_Half_Rot_Min = Temp;
        }
        cout << Cur_Set << " AFTER_MAX" << endl;
        if(change_range(generator) <= RAND_MUT_CHANCE)
        {
            cout <<  "in 1" << endl;
            do
            {
                float After_Half_Rot_Max_Range = (AFTER_HALF_ROT_MAX - AFTER_HALF_ROT_MIN);
                Data_Sets[Cur_Set].After_Half_Rot_Max = static_cast<float>(distribution(generator)) / RAND_MAX * After_Half_Rot_Max_Range + AFTER_HALF_ROT_MIN;
            }while(Data_Sets[Cur_Set].After_Half_Rot_Max <= Data_Sets[Cur_Set].After_Half_Rot_Min || Data_Sets[Cur_Set].After_Half_Rot_Max - Data_Sets[Cur_Set].After_Half_Rot_Min >= 360
                || Data_Sets[Cur_Set].After_Half_Rot_Min > AFTER_HALF_ROT_MAX - 5);
        }
        else if(change_range(generator) <= INCREMENT_MUT_CHANCE)
        {
            cout <<  "in 2" << endl;
            float Temp;
            do
            {
                Temp = rot_change(generator) + Data_Sets[Cur_Set].After_Half_Rot_Max;
            }while(Temp <= Data_Sets[Cur_Set].After_Half_Rot_Min || Temp - Data_Sets[Cur_Set].After_Half_Rot_Min >= 360 || Data_Sets[Cur_Set].After_Half_Rot_Min > AFTER_HALF_ROT_MAX - 5);
            Data_Sets[Cur_Set].After_Half_Rot_Max = Temp;
        }

        if(change_range(generator) <= RAND_MUT_CHANCE)
        {
            float Half_Range = ((HALF_MAX - HALF_MIN) + 1);
            Data_Sets[Cur_Set].Half = static_cast<float>(distribution(generator)) / RAND_MAX * Half_Range + HALF_MIN;
        }
        else if(change_range(generator) <= INCREMENT_MUT_CHANCE)
        {
            Data_Sets[Cur_Set].Half += rand_change(generator);
        }
        if(change_range(generator) <= RAND_MUT_CHANCE)
        {
            do
            {
                float Left_Rot_Min_Range = (LEFT_ROT_MAX - LEFT_ROT_MIN);
                Data_Sets[Cur_Set].On_Left_Rot_Min = static_cast<float>(distribution(generator)) / RAND_MAX * Left_Rot_Min_Range + LEFT_ROT_MIN;
                cout << Data_Sets[Cur_Set].On_Left_Rot_Min << " " << Data_Sets[Cur_Set].On_Left_Rot_Max << " <= " << Data_Sets[Cur_Set].On_Left_Rot_Min << " || "
                    << Data_Sets[Cur_Set].On_Left_Rot_Min << " > " << LEFT_ROT_MAX - 5 << endl;
            }while(Data_Sets[Cur_Set].On_Left_Rot_Max <= Data_Sets[Cur_Set].On_Left_Rot_Min || Data_Sets[Cur_Set].On_Left_Rot_Min > LEFT_ROT_MAX - 5);
        }
        else if(change_range(generator) <= INCREMENT_MUT_CHANCE)
        {
            cout <<  "in 2" << endl;
            int index = 0;
            float Temp;
            do
            {
                Temp = rot_change(generator) + Data_Sets[Cur_Set].On_Left_Rot_Min;
                index++;
                cout << index;
            }while((Data_Sets[Cur_Set].On_Left_Rot_Max <= Temp || Temp > LEFT_ROT_MAX - 5) && index < MAX_MUT_TRIES);
            if(index < MAX_MUT_TRIES)
            Data_Sets[Cur_Set].On_Left_Rot_Min = Temp;
        }

        cout << Cur_Set << " LEFT_MAX" << endl;
        if(change_range(generator) <= RAND_MUT_CHANCE)
        {
            do
            {
                float Left_Rot_Max_Range = (LEFT_ROT_MAX - LEFT_ROT_MIN);
                Data_Sets[Cur_Set].On_Left_Rot_Max = static_cast<float>(distribution(generator)) / RAND_MAX * Left_Rot_Max_Range + LEFT_ROT_MIN;
            }while(Data_Sets[Cur_Set].On_Left_Rot_Max <= Data_Sets[Cur_Set].On_Left_Rot_Min);
        }
        else if(change_range(generator) <= INCREMENT_MUT_CHANCE)
        {
            cout <<  "in 2" << endl;
            float Temp;
            do
            {
                Temp = rot_change(generator) + Data_Sets[Cur_Set].On_Left_Rot_Max;
            }while(Data_Sets[Cur_Set].On_Left_Rot_Max <= Temp);
            Data_Sets[Cur_Set].On_Left_Rot_Max = Temp;
        }

        cout << Cur_Set << " RIGHT_MIN" << endl;
        if(change_range(generator) <= RAND_MUT_CHANCE)
        {
            do
            {
                float Right_Rot_Min_Range = (ROT_MAX - ROT_MIN);
                Data_Sets[Cur_Set].On_Right_Rot_Min = static_cast<float>(distribution(generator)) / RAND_MAX * Right_Rot_Min_Range + ROT_MIN;
            }while(Data_Sets[Cur_Set].On_Right_Rot_Max <= Data_Sets[Cur_Set].On_Right_Rot_Min || Data_Sets[Cur_Set].On_Right_Rot_Min > ROT_MAX - 5);
        }
        else if(change_range(generator) <= INCREMENT_MUT_CHANCE)
        {
            cout <<  "in 2" << endl;
            int index = 0;
            float Temp;
            do
            {
                Temp = rot_change(generator) + Data_Sets[Cur_Set].On_Right_Rot_Min;
                index++;
            }while((Data_Sets[Cur_Set].On_Right_Rot_Max <= Temp || Temp > ROT_MAX - 5) && index < MAX_MUT_TRIES);
            if(index < MAX_MUT_TRIES)
            Data_Sets[Cur_Set].On_Right_Rot_Min = Temp;
        }
        cout << Cur_Set << " RIGHT_MAX" << endl;
        if(change_range(generator) <= RAND_MUT_CHANCE)
        {
            do
            {
                float Right_Rot_Max_Range = (ROT_MAX - ROT_MIN);
                Data_Sets[Cur_Set].On_Right_Rot_Max = static_cast<float>(distribution(generator)) / RAND_MAX * Right_Rot_Max_Range + ROT_MIN;
            }while(Data_Sets[Cur_Set].On_Right_Rot_Max <= Data_Sets[Cur_Set].On_Right_Rot_Min);
        }
        else if(change_range(generator) <= INCREMENT_MUT_CHANCE)
        {
            cout <<  "in 2" << endl;
            float Temp;
            do
            {
                Temp = rot_change(generator) + Data_Sets[Cur_Set].On_Right_Rot_Max;
            }while(Temp <= Data_Sets[Cur_Set].On_Right_Rot_Min);
            Data_Sets[Cur_Set].On_Right_Rot_Max = Temp;
        }

        if(change_range(generator) <= INCREMENT_MUT_CHANCE)
        {
            int Switch_Before_Done_Range = (true - false);
            Data_Sets[Cur_Set].Switch_Before_Done = static_cast<int>(distribution(generator)) / RAND_MAX * Switch_Before_Done_Range;
        }

        if(change_range(generator) <= RAND_MUT_CHANCE)
        {
            float After_Half_Top_Skip_Range = ((AFTER_HALF_TOP_SKIP_MAX - AFTER_HALF_TOP_SKIP_MIN) + 1);
            Data_Sets[Cur_Set].After_Half_Top_Skip = static_cast<float>(distribution(generator)) / RAND_MAX * After_Half_Top_Skip_Range + AFTER_HALF_TOP_SKIP_MIN;
        }
        else if(change_range(generator) <= INCREMENT_MUT_CHANCE)
        {
            Data_Sets[Cur_Set].After_Half_Top_Skip += rand_change(generator);
        }

    }
    }








    for(int counter = round(static_cast<float>(DATA_SET_COUNT) / 4 * 3); counter < DATA_SET_COUNT; counter++)
    {
        Gen_Random_Set(counter);
    }
    return;
}



void Gen_Random_Set(int index)
{
    uniform_int_distribution<int> distribution(0, RAND_MAX);

    float Green_Line_Mult_Range = ((GREEN_LINE_MULT_MAX - GREEN_LINE_MULT_MIN) + 1);
    Data_Sets[index].Green_Line_Mult = static_cast<float>(distribution(generator)) / RAND_MAX * Green_Line_Mult_Range + GREEN_LINE_MULT_MIN;

    float Dist_To_Closest_Edge_Mult_Range = ((DIST_FROM_EDGE_MAX - DIST_FROM_EDGE_MIN) + 1);
    Data_Sets[index].Dist_From_Edge_Mult = static_cast<float>(distribution(generator)) / RAND_MAX * Dist_To_Closest_Edge_Mult_Range + DIST_FROM_EDGE_MIN;

    float Dist_From_Drone_Mult_Range = ((DIST_FROM_DRONE_MAX - DIST_FROM_DRONE_MIN) + 1);
    Data_Sets[index].Dist_From_Drone_Mult = static_cast<float>(distribution(generator)) / RAND_MAX * Dist_From_Drone_Mult_Range + DIST_FROM_DRONE_MIN;

    float Head_Off_Edge_Mult_Range = ((HEAD_OFF_EDGE_MAX - HEAD_OFF_EDGE_MIN) + 1);
    Data_Sets[index].Head_Off_Edge_Mult = static_cast<float>(distribution(generator)) / RAND_MAX * Head_Off_Edge_Mult_Range + HEAD_OFF_EDGE_MIN;

    float Find_Direction_Mult_Range = ((FIND_DIRECTION_MAX - FIND_DIRECTION_MIN) + 1);
    Data_Sets[index].Find_Direction_Mult = static_cast<float>(distribution(generator)) / RAND_MAX * Find_Direction_Mult_Range + FIND_DIRECTION_MIN;
    do
    {
        float Rot_Min_Range = (ROT_MAX - ROT_MIN);
        Data_Sets[index].Rot_Min = static_cast<float>(distribution(generator)) / RAND_MAX * Rot_Min_Range + ROT_MIN;
    }while(Data_Sets[index].Rot_Min > 355);

    do
    {
        float Rot_Max_Range = (ROT_MAX - ROT_MIN);
        Data_Sets[index].Rot_Max = static_cast<float>(distribution(generator)) / RAND_MAX * Rot_Max_Range + ROT_MIN;
    }while(Data_Sets[index].Rot_Max < Data_Sets[index].Rot_Min);

    int Continue_After_Half_Range = (CONTINUE_AFTER_HALF_ON - CONTINUE_AFTER_HALF_OFF);
    Data_Sets[index].Cont_After_Half = round(static_cast<float>(distribution(generator)) / RAND_MAX) * Continue_After_Half_Range;

    do
    {
        float After_Half_Rot_Min_Range = (AFTER_HALF_ROT_MAX - AFTER_HALF_ROT_MIN);
        Data_Sets[index].After_Half_Rot_Min = static_cast<float>(distribution(generator)) / RAND_MAX * After_Half_Rot_Min_Range + AFTER_HALF_ROT_MIN;
    }while(Data_Sets[index].After_Half_Rot_Min > AFTER_HALF_ROT_MAX - 5);

    do
    {
        float After_Half_Rot_Max_Range = (AFTER_HALF_ROT_MAX - AFTER_HALF_ROT_MIN);
        Data_Sets[index].After_Half_Rot_Max = static_cast<float>(distribution(generator)) / RAND_MAX * After_Half_Rot_Max_Range + AFTER_HALF_ROT_MIN;
    }while(Data_Sets[index].After_Half_Rot_Max < Data_Sets[index].After_Half_Rot_Min || Data_Sets[index].After_Half_Rot_Max - Data_Sets[index].After_Half_Rot_Min > 360);

    float Half_Range = ((HALF_MAX - HALF_MIN) + 1);
    Data_Sets[index].Half = static_cast<float>(distribution(generator)) / RAND_MAX * Half_Range + HALF_MIN;

    do
    {
        float Left_Rot_Min_Range = ((LEFT_ROT_MAX - LEFT_ROT_MIN) + 1);
        Data_Sets[index].On_Left_Rot_Min = static_cast<float>(distribution(generator)) / RAND_MAX * Left_Rot_Min_Range + LEFT_ROT_MIN;
    }while(Data_Sets[index].On_Left_Rot_Min > LEFT_ROT_MAX - 5);

    do
    {
        float Left_Rot_Max_Range = ((LEFT_ROT_MAX - LEFT_ROT_MIN) + 1);
        Data_Sets[index].On_Left_Rot_Max = static_cast<float>(distribution(generator)) / RAND_MAX * Left_Rot_Max_Range + LEFT_ROT_MIN;
    }while(Data_Sets[index].On_Left_Rot_Max <= Data_Sets[index].On_Left_Rot_Min);

    do
    {
        float Right_Rot_Min_Range = ((ROT_MAX - ROT_MIN) + 1);
        Data_Sets[index].On_Right_Rot_Min = static_cast<float>(distribution(generator)) / RAND_MAX * Right_Rot_Min_Range + ROT_MIN;
    }while(Data_Sets[index].On_Right_Rot_Min > ROT_MAX - 5);

    do
    {
        float Right_Rot_Max_Range = ((ROT_MAX - ROT_MIN) + 1);
        Data_Sets[index].On_Right_Rot_Max = static_cast<float>(distribution(generator)) / RAND_MAX * Right_Rot_Max_Range + ROT_MIN;
    }while(Data_Sets[index].On_Right_Rot_Max <= Data_Sets[index].On_Right_Rot_Min);

    int Switch_Before_Done_Range = (true - false);
    Data_Sets[index].Switch_Before_Done = round(static_cast<float>(distribution(generator)) / RAND_MAX * Switch_Before_Done_Range);

    float After_Half_Top_Skip_Range = ((AFTER_HALF_TOP_SKIP_MAX - AFTER_HALF_TOP_SKIP_MIN) + 1);
    Data_Sets[index].After_Half_Top_Skip = static_cast<float>(distribution(generator)) / RAND_MAX * After_Half_Top_Skip_Range + AFTER_HALF_TOP_SKIP_MIN;


    if(Evo_Iteration == 0)
    {
        Data_Sets[index].Green_Line_Mult = 0;
        Data_Sets[index].Dist_From_Edge_Mult = 5;
        Data_Sets[index].Dist_From_Drone_Mult = 5.000;
        Data_Sets[index].Head_Off_Edge_Mult = 50000;
        Data_Sets[index].Find_Direction_Mult = -500;
        Data_Sets[index].Rot_Min = ROT_MIN_START;
        Data_Sets[index].Rot_Max = ROT_MAX_START;
        Data_Sets[index].After_Half_Rot_Min = AFTER_HALF_ROT_MIN_START;
        Data_Sets[index].After_Half_Rot_Max = AFTER_HALF_ROT_MAX_START;
        Data_Sets[index].Cont_After_Half = true;
        Data_Sets[index].Half = 12;
        Data_Sets[index].On_Left_Rot_Min = ON_LEFT_ROT_MIN_START;
        Data_Sets[index].On_Left_Rot_Max = ON_LEFT_ROT_MAX_START;
        Data_Sets[index].On_Right_Rot_Min = ON_RIGHT_ROT_MIN_START;
        Data_Sets[index].On_Right_Rot_Max = ON_RIGHT_ROT_MAX_START;
        Data_Sets[index].Switch_Before_Done = SWITCH_BEFORE_DONE_START;
        Data_Sets[index].After_Half_Top_Skip = 1;
    }




/*
    if(Evo_Iteration == 0)
    {
        if(index % 2 == 0)
            Data_Sets[index].Green_Line_Mult = -4.01532;
        else
            Data_Sets[index].Green_Line_Mult = 4.01532;
        Data_Sets[index].Dist_From_Edge_Mult = 4.15277;
        Data_Sets[index].Dist_From_Drone_Mult = -4.26837;
        Data_Sets[index].Head_Off_Edge_Mult = 0.753624;
        Data_Sets[index].Find_Direction_Mult = -474.289;
        Data_Sets[index].Rot_Min = 228.375;
        Data_Sets[index].Rot_Max =  312.271;
        Data_Sets[index].After_Half_Rot_Min = 383.022;
        Data_Sets[index].After_Half_Rot_Max = 593.807;
        Data_Sets[index].Cont_After_Half = 1;
        Data_Sets[index].Half = 12.4779;
        Data_Sets[index].On_Left_Rot_Min = 242.498;
        Data_Sets[index].On_Left_Rot_Max = 593.807;
        Data_Sets[index].On_Right_Rot_Min = 35.388;
        Data_Sets[index].On_Right_Rot_Max = 348.116;
        Data_Sets[index].Switch_Before_Done = 0;
    }
*/


    return;
}



void Run_Match(int In_Match)
{
    while(Matches[In_Match].Current_Frame < FRAME_LIMIT)
    {
        if(Draw_On_Screen == true)
        {
            MSG msg;
            while(PeekMessage(&msg, global_hwnd, 0, 0, PM_REMOVE))
            {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            if(msg.message == WM_QUIT)
                break;
        }

        Matches[In_Match].Current_Frame++;
        Engine(In_Match);
        if(Draw_On_Screen == true && Matches[In_Match].Current_Frame % (FRAME_SKIP + 1) == 0)
        {
            DrawScreen(hWnd);
        }
    }
    Threads_Done++;
    return;
}



bool Is_Ground_Robot_In_Rotation_Range(int G_R_Index, int In_Match)
{
    Ground_Robot_Data * Target_G_Robot = &Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[G_R_Index];
    float Rotation = static_cast<int>(Target_G_Robot->Rotation) % 360;
    if(Rotation < 0)
        Rotation += 360;
    bool ret = true;
    // See if the ground robot's rotation is in the range we want
    if((Matches[In_Match].Current_Frame - Target_G_Robot->Last_Frame_Reversed) / COMP_FRAME_RATE < Data_Sets[Current_Data_Set].Half)
    {
        if((Rotation < Data_Sets[Current_Data_Set].Rot_Min || Rotation > Data_Sets[Current_Data_Set].Rot_Max))
        {
            ret = false;
        }
    }
    else
    {
        int Number_From_Top = 0;
        for(int counter = 0; counter < GROUND_ROBOT_NUMBER; counter++)
        {
            if(counter != G_R_Index && Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[counter].Y < Target_G_Robot->Y
                && Matches[In_Match].Array_Of_Frames[Matches[In_Match].Current_Frame].Ground_Robots[counter].Still_In == true)
            {
                Number_From_Top = Number_From_Top + 1;
            }
        }
        if(Number_From_Top < Data_Sets[Current_Data_Set].After_Half_Top_Skip)
        {
            ret = true;
        }
        else if((Rotation < Data_Sets[Current_Data_Set].After_Half_Rot_Min || Rotation > Data_Sets[Current_Data_Set].After_Half_Rot_Max) ||
            (Rotation < Data_Sets[Current_Data_Set].After_Half_Rot_Min - 360 || Rotation > Data_Sets[Current_Data_Set].After_Half_Rot_Max - 360))
        {
            ret = false;
        }
    }
    return ret;
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
