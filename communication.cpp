// Client side C/C++ program to demonstrate Socket
// programming
#include <arpa/inet.h>
#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>

#include <sys/types.h>
#include <netinet/in.h>
#include <signal.h>
#define MAX_LEN 512

int client_socket;

void send_message(int client_socket);
void catch_ctrl_c(int signal);

bool connectServer()
{
	if((client_socket=socket(AF_INET,SOCK_STREAM,0))==-1)
	{
		return false;
	}
	struct sockaddr_in client;
	client.sin_family=AF_INET;
	client.sin_port=htons(10000); // Port no. of server
	client.sin_addr.s_addr=INADDR_ANY;
	bzero(&client.sin_zero,0);

	if((connect(client_socket,(struct sockaddr *)&client,sizeof(struct sockaddr_in)))==-1)
	{
        return false;
	}
	signal(SIGINT, catch_ctrl_c);

			
	return true;
}

// Handler for "Ctrl + C"
void catch_ctrl_c(int signal) 
{
	char str[MAX_LEN]="#exit";
	send(client_socket,str,sizeof(str),0);
	close(client_socket);
	exit(signal);
}

// Send message to everyone
void sendServer(char* message, size_t size)
{
    int bytesSent = send( client_socket, message, size, 0 );
}

void closeConnection(){
    char str[MAX_LEN]="#exit";
	send(client_socket,str,sizeof(str),0);
    close(client_socket);
}
