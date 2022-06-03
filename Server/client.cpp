#include <bits/stdc++.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <errno.h>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <signal.h>
#include <mutex>
#define MAX_LEN 512
using namespace std;

bool exit_flag=false;
int client_socket;

void recv_message(int client_socket);
void catch_ctrl_c(int signal);

int main()
{
	if((client_socket=socket(AF_INET,SOCK_STREAM,0))==-1)
	{
		perror("socket: ");
		exit(-1);
	}
	signal(SIGINT, catch_ctrl_c);
	struct sockaddr_in client;
	client.sin_family=AF_INET;
	client.sin_port=htons(10000); // Port no. of server
	client.sin_addr.s_addr=INADDR_ANY;
	bzero(&client.sin_zero,0);

	if((connect(client_socket,(struct sockaddr *)&client,sizeof(struct sockaddr_in)))==-1)
	{
		perror("connect: ");
		exit(-1);
	}
	cout <<"\n\t*********CHAT ROOM***********\n";

	recv_message(client_socket);
	return 0;
}


// Receive message
void recv_message(int client_socket)
{
	while(1)
	{
		char str[MAX_LEN];
		recv(client_socket,str,sizeof(str),0);
		cout<<str<<endl;
		fflush(stdout);
	}	
}

void catch_ctrl_c(int signal) 
{
	char str[MAX_LEN]="#exit";
	send(client_socket,str,sizeof(str),0);
	exit_flag=true;
	close(client_socket);
	exit(signal);
}