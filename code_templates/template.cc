/*
>>> FILE OVERVIEW <<<
- 

>>> MISC COMMENTS <<<
- 

>>> FILE CONTENTS <<<
- File Overview, Imports, Global Variables
*/
// Helper Functions
bool isPrime(int num);
int countPrimesInRange(int startNum, int endNum);

// Main Function
int main();

// ----- Imports ----------------------------------------------------------------------------------
#include <iostream>     // For cout
#include <cstdio>       // I want meh color coding! >:D
#include <thread>       // For threads
#include <functional>   // For std::function
#include <queue>        // For std::queue
#include <semaphore>    // For semaphores
#include <atomic>       // For atomic bool
#include <climits>      // For INT_MAX (Uncapping qWork)

// ----- Global Variables -------------------------------------------------------------------------
using namespace std;    // Removes the need for std:: everywhere, cause no...
                        // Except at move calls because compiler whines about it...
                        // Also because no one wants STDs everywhere :D

/* 
====================================================================================================
END File Overview, Imports, Global Variables
START Helper Functions
====================================================================================================
*/

bool isPrime(int num) {
    /*
    Determines whether or not num is a prime number
    */
    // ----- Determine False Cases ----------------------------------------------------------------
    if(num < 2) {
        return false;
    }

    for(int i=2; i<num; i++) {
        if(num % i == 0) {
            return false;
        }
    }

    // ----- If Not False, Then True --------------------------------------------------------------
    return true;
}


int countPrimesInRange(int startNum, int endNum) {
    /*
    Count the total prime numbers within a defined range
    */
    // ----- Catch Faulty Range -------------------------------------------------------------------
    if (endNum < startNum) {
        return 0;
    }

    // ----- Instantiate Necessary Variables ------------------------------------------------------
    int totalPrimes = 0;

    // ----- Count All Primes ---------------------------------------------------------------------
    for(int i=startNum; i <= endNum; i++) {
        if (isPrime(i)) {
            totalPrimes++;
        }
    }

    // ----- Return Results -----------------------------------------------------------------------
    return totalPrimes;
}

/* 
====================================================================================================
END Helper Functions
START Main Function
====================================================================================================
*/

int main() {
    /*
    Prints how many primes are within a range
    Depends on PrimeUtils.java
    */
    // ----- Instantiate Necessary Variables ------------------------------------------------------
    int startNum = 10;
    int endNum = 50;

    // ----- Count Total Primes -------------------------------------------------------------------
    int totalPrimes = countPrimesInRange(startNum, endNum);

    // ----- Print Results ------------------------------------------------------------------------
    std::string output = "Serial: Primes in the range [" + std::to_string(startNum) + ", " + std::to_string(endNum) + "] is " + std::to_string(totalPrimes) + "\n";
    std::cout << output;
}

/* 
====================================================================================================
END Main Function
====================================================================================================
*/