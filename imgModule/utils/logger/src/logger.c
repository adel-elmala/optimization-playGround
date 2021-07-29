// // // // // // // // // // // // // // // // 
//    Simple logger                          //
//    Author:  Adel refat elmala             //
//    Email :  adel.elmala2020@gmail.com     //
//    github:  github.com/adel-elmala        //
// // // // // // // // // // // // // // // //

#include <stdio.h>
#include "loggerBase.h"
#include <stdarg.h>


#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_RESET   "\x1b[0m"


void logInfo(const char * str,const char * fileName,int lineNumber,...)
{
    va_list listptr; /* declare list pointer */
    va_start(listptr, lineNumber); 
    
    // fprintf(stdout,ANSI_COLOR_GREEN   "INFO:[%s][l:%d]:"   ANSI_COLOR_RESET "%s" "\n",fileName,lineNumber,str,listptr);
    fprintf(stdout,ANSI_COLOR_GREEN   "INFO:[%s][l:%d]:"   ANSI_COLOR_RESET ,fileName,lineNumber);
    vfprintf(stdout,str,listptr);

    va_end(listptr);
}

void logWarning(const char * str,const char * fileName,int lineNumber,...)
{
    va_list listptr; /* declare list pointer */
    va_start(listptr, lineNumber); 
   
    // fprintf(stderr,ANSI_COLOR_YELLOW  "WARNING:[%s][l:%d]:"  ANSI_COLOR_RESET "%s" "\n",fileName,lineNumber,str);
    fprintf(stderr,ANSI_COLOR_YELLOW  "WARNING:[%s][l:%d]:"  ANSI_COLOR_RESET,fileName,lineNumber);
    vfprintf(stdout,str,listptr);

    va_end(listptr);

}
void logError(const char * str,const char * fileName,int lineNumber,...)
{
    va_list listptr; /* declare list pointer */
    va_start(listptr, lineNumber); 
   
    // fprintf(stderr,ANSI_COLOR_RED     "ERROR:[%s][l:%d]:"     ANSI_COLOR_RESET "%s" "\n",fileName,lineNumber,str);
    fprintf(stderr,ANSI_COLOR_RED     "ERROR:[%s][l:%d]:"     ANSI_COLOR_RESET ,fileName,lineNumber);
    vfprintf(stdout,str,listptr);

    va_end(listptr);

}



void logStartLine(const char * str)
{
    fprintf(stdout,"==================================\n");
    fprintf(stdout,"\t%s\n",str);
    fprintf(stdout,"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
}



void logEndLine(void)
{
    fprintf(stdout,"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
}
