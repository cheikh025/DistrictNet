/* C++ program to check if a given PointSC lies inside a given polygon
   From public domain: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/  */

#ifndef POLYGON_H
#define POLYGON_H

#define MY_EPSILON 0.00001
#include "Point.h"
#include <vector>
using namespace std;

// Given three colinear points p, q, r, the function checks if PointSC q lies on line segment 'pr' 
bool onSegment(PointSC p, PointSC q, PointSC r);

// To find orientation of ordered triplet (p, q, r). 
// The function returns following values 
// 0 --> p, q and r are colinear 
// 1 --> Clockwise 
// 2 --> Counterclockwise 
int orientation(PointSC p, PointSC q, PointSC r);

// The function that returns true if line segment 'p1q1' and 'p2q2' intersect. 
bool doIntersect(PointSC p1, PointSC q1, PointSC p2, PointSC q2);

// Returns true if the PointSC p lies inside the polygon[] with n vertices 
bool isInside(vector <PointSC> & polygon, PointSC p);

#endif
