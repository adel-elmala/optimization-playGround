#include "logger.h"


int main(){

    logStartLine("log 1!");
    int a = 599;
    logInfo("this is info!:%d\n",a);
    logInfo("this is info!:\n");
    logWarning("this is warning!\n");
    logError("this is error!\n");
    logEndLine();

    logStartLine("log 2!");
    logInfo("this is info!\n");
    logWarning("this is warning!\n");
    logError("this is error!\n");
    logEndLine();


    logStartLine("log 3!");
    logInfo("this is info!\n");
    logWarning("this is warning!,%s\n","hello");
    logError("this is error!\n");
    logEndLine();


}