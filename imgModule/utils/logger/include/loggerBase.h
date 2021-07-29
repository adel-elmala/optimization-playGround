// // // // // // // // // // // // // // // // 
//    Simple logger                          //
//    Author:  Adel refat elmala             //
//    Email :  adel.elmala2020@gmail.com     //
//    github:  github.com/adel-elmala        //
// // // // // // // // // // // // // // // //

#ifndef _LOGGER_H_
#define _LOGGER_H_


void logInfo(const char * str,const char * fileName,int lineNumber,...);

void logWarning(const char * str,const char * fileName,int lineNumber,...);

void logError(const char * str,const char * fileName,int lineNumber,...);

void logStartLine(const char * str);
void logEndLine(void);




#endif