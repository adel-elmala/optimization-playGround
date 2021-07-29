// // // // // // // // // // // // // // // // 
//    Simple logger                          //
//    Author:  Adel refat elmala             //
//    Email :  adel.elmala2020@gmail.com     //
//    github:  github.com/adel-elmala        //
// // // // // // // // // // // // // // // //

#ifndef _LOGGERBASE_H_
#define _LOGGERBASE_H_

#include "loggerBase.h"

// helper api macros 
#define logInfo(str,...) logInfo(str,__FILE__,__LINE__,## __VA_ARGS__)
#define logWarning(str,...) logWarning(str,__FILE__,__LINE__,## __VA_ARGS__)
#define logError(str,...) logError(str,__FILE__,__LINE__,## __VA_ARGS__)
 


#endif