// Compile : gcc -Wall uart-send.c -o uart-send -lwiringPi
 
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <wiringPi.h>
#include <wiringSerial.h>
 
int main() {
 
	int fd;
 
	printf("Raspberry's sending : \n");
 
	while(1) {
		if((fd = serialOpen ("/dev/ttyACM0", 9600)) < 0 ){
			fprintf (stderr, "Unable to open serial device: %s\n", strerror (errno));
		}
        if (wiringPiSetup () == -1)
        {
            fprintf (stdout, "Unable to start wiringPi: %s\n", strerror (errno)) ;
            return 1 ;
        }

		serialPuts(fd, "3");
		serialFlush(fd);
		fflush(stdout);

        do{
            char c = serialGetchar(fd);
            printf("%c\n",c);
            fflush (stdout);
        }while(serialDataAvail(fd));
	}
	return 0;
}