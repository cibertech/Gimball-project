#include <Servo.h> //library for servo motor

char SerialinData[20];
char *SerialinParse[20];
int Serialindex = 0;
boolean SerialstringComplete = false;

Servo Servo1; //Servo Up/Down
Servo Servo2; //Servo Left/Right
int angServo1 = 90;  
int angServo2 = 90; 

void setup() {
  Serial.begin(9600);
  Servo1.attach(2); 
  Servo2.attach(3); 
  Servo1.write(angServo1); 
  Servo2.write(angServo2);
  delay(1000);
  Serial.print("1,");
  Serial.println(angServo1);
  Serial.print("2,");
  Serial.println(angServo2);
}

void loop() {

  if (SerialstringComplete) {
    ParseSerialData();
    memset(SerialinData, 0, sizeof(SerialinData));//clear the SerialinData string
    SerialstringComplete = false; 
  }

  if (Serial.available() && SerialstringComplete == false){
    char inChar = Serial.read();    
    if (inChar == '\n') 
    {
      Serialindex = 0;
      SerialstringComplete = true;
    }
    else
    {
      SerialinData[Serialindex] = inChar; 
      Serialindex++;
    }
   }

   
}

void ParseSerialData(){
    char *p = SerialinData;
    char *str;   
    int count = 0;
    
    while ((str = strtok_r(p, ",", &p)) )//!= "\n")//!= NULL)
    {   
      SerialinParse[count] = str;
      count++;      
    }
    if(count >= 2)//
    {
      char *func = SerialinParse[0];
      char *prop1 = SerialinParse[1];    
      switch(*func)
      {
        case '1': Servo1_Set(prop1); break;//Serial.println("OK"); break;//
        case '2': Servo2_Set(prop1); break;//Serial.println("OK"); break;//
      }
      //Serial.println("OK");
    }
  }

void Servo1_Set(char *prop1){  
  switch(*prop1)
    {
      case '+':
        if(angServo1 <= 125){
          //for (int i = 0; i <= 3; i++) {
            angServo1 += 1;
            Servo1.write(angServo1);
            //delay(10);
          //}
        }  
      break;   
      case '-':
        if(angServo1 >= 45){
          //for (int i = 0; i <= 3; i++) {
            angServo1 -= 1;
            Servo1.write(angServo1);
            //delay(10);
          //}
        }
      }
}

void Servo2_Set(char *prop1){
    switch(*prop1)
    {
      case '+':
        if(angServo2 <= 170){
          //for (int i = 0; i <= 3; i++) {
            angServo2 += 1;
            Servo2.write(angServo2);
            //delay(10);
          //}
        }
      break;   
      case '-':
        if(angServo2 >= 10){
          //for (int i = 0; i <= 3; i++) {
            angServo2 -= 1;
            Servo2.write(angServo2);
            //delay(10);
          //}
        }
      }
}
