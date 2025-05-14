#pragma once
#include <iostream>
#include <string>
#include "Color.cpp"
using namespace std;


void raise_error(string mainMsg, string subMsg) {
    cerr << FG_RED << mainMsg << ": " << RESET_COLOR << subMsg << endl;
    throw runtime_error(mainMsg);
}
