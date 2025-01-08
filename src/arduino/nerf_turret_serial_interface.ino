#include <Servo.h>
#include <CommandParser.h>

typedef CommandParser<> MyCommandParser;

#define __JOY 1

MyCommandParser parser;
void cmd_angle(MyCommandParser::Argument *args, char *response);
void cmd_fire(MyCommandParser::Argument *args, char *response);
void cmd_rotate(MyCommandParser::Argument *args, char *response);
void cmd_feed(MyCommandParser::Argument *args, char *response);

void action_fire(void);
void action_tilt(int amount);

Servo s_fire;
Servo s_rotate;
Servo s_feed;
Servo s_tilt;

int pushbutton = 0;
int val_A0 = 0;
int val_A1 = 0;
const int deadzone = 10;
const int movement_delta_Y = 2;
const int movement_delta_X = 1;
const int center_joystick = 512;
const int deadzone_pos = center_joystick + deadzone;
const int deadzone_neg = center_joystick - deadzone;

int current_tilt_angle = 1900; // 0'
int current_rotation = 1500;   // center

void setup()
{
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  s_fire.attach(3);
  s_rotate.attach(9);
  // s_feed.attach(10);
  s_tilt.attach(11);
  pinMode(2, INPUT_PULLUP); // push button

  parser.registerCommand("angle", "i", &cmd_angle);
  parser.registerCommand("fire", "i", &cmd_fire);
  parser.registerCommand("rotate", "i", &cmd_rotate);
  parser.registerCommand("feed", "i", &cmd_feed);

  s_fire.writeMicroseconds(1500);
  s_rotate.writeMicroseconds(current_rotation);
  // s_feed.writeMicroseconds(1300);
  s_tilt.writeMicroseconds(current_tilt_angle);
}

void loop()
{
  if (Serial.available())
  {
    char line[128];
    size_t lineLength = Serial.readBytesUntil('\n', line, 127);
    line[lineLength] = '\0';

    char response[MyCommandParser::MAX_RESPONSE_SIZE];
    parser.processCommand(line, response);
    Serial.println(response);
  }
  digitalWrite(LED_BUILTIN, HIGH);

  if (__JOY)
  {
    pushbutton = digitalRead(2);
    if (pushbutton == 0)
    {
      action_fire();
      pushbutton = 0;
    }

    val_A0 = analogRead(A0); // X
    val_A1 = analogRead(A1); // Y

    if (val_A0 > deadzone_pos || val_A0 < deadzone_neg)
    {
      val_A0 > center_joystick ? action_rotate(+movement_delta_X) : action_rotate(-(movement_delta_X));
    }

    if (val_A1 > deadzone_pos || val_A1 < deadzone_neg)
    {
      val_A1 > center_joystick ? action_tilt(+movement_delta_Y) : action_tilt(-(movement_delta_Y));
    }
  }

  delay(5);
}

void action_fire(void)
{
  Serial.println("firing dart");
  s_fire.writeMicroseconds(1750);
  delay(250);
  s_feed.attach(10);
  s_feed.writeMicroseconds(1575);
  delay(915);
  s_feed.detach();
  s_fire.writeMicroseconds(1500);
}

void action_tilt(int amount)
{
  int new_angle = current_tilt_angle - amount;
  if (new_angle < 1000 || new_angle > 1900)
  { // out of bounds
    return;
  }
  else
  {
    s_tilt.writeMicroseconds(new_angle);
    current_tilt_angle = new_angle;
    return;
  }
}

void action_rotate(int amount)
{
  int new_rotation = current_rotation + amount;
  if (new_rotation < 700 || new_rotation > 2300) // out of bounds
    return;
  else
  {
    s_rotate.writeMicroseconds(new_rotation);
    current_rotation = new_rotation;
  }
}

void cmd_fire(MyCommandParser::Argument *args, char *response)
{ // 1550 is ok
  action_fire();
  strlcpy(response, "success", MyCommandParser::MAX_RESPONSE_SIZE);
}

void cmd_angle(MyCommandParser::Argument *args, char *response)
{ // 1900 0'  -- 1000 90'
  Serial.print("Setting Angle:");
  Serial.println((int32_t)args[0].asInt64);
  s_tilt.writeMicroseconds((int32_t)args[0].asInt64);
  strlcpy(response, "success", MyCommandParser::MAX_RESPONSE_SIZE);
}

void cmd_rotate(MyCommandParser::Argument *args, char *response)
{ // 700 90' CW  -- 2300 90' CCW
  Serial.print("Setting rotation:");
  Serial.println((int32_t)args[0].asInt64);
  s_rotate.writeMicroseconds((int32_t)args[0].asInt64);
  strlcpy(response, "success", MyCommandParser::MAX_RESPONSE_SIZE);
}

void cmd_feed(MyCommandParser::Argument *args, char *response)
{
  Serial.print("Setting feed:");
  Serial.println((int32_t)args[0].asInt64);
  if ((int32_t)args[0].asInt64 == 0 && s_feed.attached())
    s_feed.detach();
  else
  {
    s_feed.attach(10);
    s_feed.writeMicroseconds((int32_t)args[0].asInt64);
  }

  strlcpy(response, "success", MyCommandParser::MAX_RESPONSE_SIZE);
}
