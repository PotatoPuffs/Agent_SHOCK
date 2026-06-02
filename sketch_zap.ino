
bool right = false;
char incomingByte = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(6, OUTPUT);
  pinMode(5, OUTPUT);
  pinMode(4, OUTPUT);

}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available()) {
    // read the incoming byte:
    incomingByte = Serial.read();
    //if (incomingByte == '\n') {
   
    }
      
    if (incomingByte == 'R')
      {
        digitalWrite(6, LOW);
        digitalWrite(5, LOW);
        digitalWrite(4, LOW);

        digitalWrite(6, HIGH);
        Serial.println("zap");
        //message = "";
      }

    if (incomingByte == 'C')
      {
        digitalWrite(6, LOW);
        digitalWrite(5, LOW);
        digitalWrite(4, LOW);

        digitalWrite(5, HIGH);
        Serial.println("zap");
        //message = "";
      }

    if (incomingByte == 'L')
      {
        digitalWrite(6, LOW);
        digitalWrite(5, LOW);
        digitalWrite(4, LOW);

        digitalWrite(4, HIGH);
        Serial.println("zap");
        //message = "";
      }

    if (incomingByte == 'N') {
      digitalWrite(6, LOW);
      digitalWrite(5, LOW);
      digitalWrite(4, LOW);
    }
  }
  
  
  
  /*right = !right;

  if (right == true)
    {
      digitalWrite(6, LOW);
      digitalWrite(4, HIGH);
      //digitalWrite(5, HIGH);
      //right = false;
      Serial.println("right");
      //message = "";
    }
  else if (right == false) {
    digitalWrite(6, HIGH);
    digitalWrite(4, LOW);
    //digitalWrite(5, LOW);
    //right = true;
    Serial.println("left");
  }
  delay(800);*/

