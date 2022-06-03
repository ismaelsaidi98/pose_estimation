#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include "communication.cpp"
#include <stdio.h>
#include <vector>
#include <map> 
#include <string>

#define EPS 1e-6

template <class T>
using Vec1D = std::vector<T>;
template <class T>
using Vec2D = std::vector<Vec1D<T>>;
template <class T>
using Vec3D = std::vector<Vec2D<T>>;

std::map<std::string, int> bodyParts = {{"nose", 0}, {"left_eye",2}, {"right_eye",1}, {"left_ear",4}, {"right_ear",3}, {"left_shoulder",6}, {"right_shoulder",5}, {"left_elbow",8}, {"right_elbow",7}, {"left_wrist",10}, {"right_wrist",9}, {"left_hip",12}, {"right_hip",11}, {"left_knee",14}, {"right_knee",13}, {"left_ankle",16}, {"right_ankle",15}, {"neck",17}};
std::string displayedParts[4] = {"left_shoulder", "right_shoulder", "left_wrist", "right_wrist"};

bool server = false;
void display_data(const Vec2D<int> &objects, const Vec3D<float> &normalized_peaks, const gint &frame_number){
    if(!(frame_number%60)){
        for(int i = 0; i < objects.size(); i++){
            int absence = 0;
            for(int parts : objects[i]){
                absence += parts;
            }
            if(absence >= -8){
                std::string info = "Person " + std::to_string(i+1) + "\n";
                for(int j = 0; j < sizeof(displayedParts)/sizeof(std::string); j++){
                    const int indexPart = bodyParts[displayedParts[j]];
                    if(objects[i][indexPart] == 0){
                        info += displayedParts[j] + ":: X:" + std::to_string(normalized_peaks[indexPart][i][1]) +" Y:" + std::to_string(1-normalized_peaks[indexPart][i][0]) + "\n";

                    } else{
                        info += displayedParts[j] + ":: N/A\n";
                    }
                }
                info += "-----------------------------------\n";
                char infoBuffer[info.length() + 1];
                strcpy(infoBuffer, info.c_str());
                g_print("%s", infoBuffer);
                if(server) sendServer(infoBuffer, sizeof(infoBuffer));
            }
        }
    }
}

bool startServer(bool useServer){
    if(useServer){
        server = connectServer();
        return server;
    }
    return false;
}

void closeServer(){
    closeConnection();
}