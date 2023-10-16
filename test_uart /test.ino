#include <Wire.h> 
#include <LiquidCrystal_I2C.h>
LiquidCrystal_I2C lcd(0x27,16,2); 

void setup()
{
  lcd.init();                    
  lcd.backlight();
  Serial.begin(9600);
}

void loop()
{
  if (Serial.available() > 0) {
    String str = Serial.readString();
    lcd.setCursor(2,0);
    lcd.print(str);
    Serial.println(str);
  }
}