#ifndef BLOCK_H
#define BLOCK_H

#include "Point.h"
#include <vector>
#include <set>
#include <string>
using namespace std;

class BlockSC
{
public:
	
	int id;												// ID of the BlockSC									// ID of the zone (from JSON input)
	//string zone_name;									// Name of the zone
	int nbInhabitants ;									// Number of inhabitants
	
	vector <PointSC> verticesPoints;						// Copy of the points that delimit the BlockSC
	vector <std::array<double, 2>> verticesLongLat;		// Lat/Long of the points that delimit the BlockSC
	
	double minX = 1.e30;								// Minimum X coordinate of the vertices of the BlockSC
	double maxX = -1.e30;								// Maximum X coordinate of the vertices of the BlockSC
	double minY = 1.e30;								// Minimum Y coordinate of the vertices of the BlockSC
	double maxY = -1.e30;								// Maximum Y coordinate of the vertices of the BlockSC
	
	set <int> adjacentBlocks;							// Adjacent blocks
	vector <double> distanceEucl;						// Distance to the other blocks: Euclidian
	
	double area;										// Area of the BlockSC
	double recArea;										// Area of the enclosing rectangle
	double perimeter = 0.0;								// Perimeter of the BlockSC
	double density;										// Density of the BlockSC
	double distDepot;		  			    			// Distance from each depot
	double distReferencePoint;							// Distance from the reference PointSC
	
	map<int,vector<vector<PointSC>>> trainScenarios;				// Scenarios for each target size of distrcit - used to train 
	map<int,vector<vector<PointSC>>> testScenarios;				// Scenarios for each target size of distrcit - used to evaluate algorithm

	// overloaded comparison operator (to order by increasing distance from the reference PointSC)
	bool operator <(const BlockSC& b) {return (distReferencePoint < b.distReferencePoint);}

	// Distance of a PointSC to the closest PointSC in the BlockSC
	double distance(const PointSC & p1);
};

// JSON output
void to_json(json& j, const BlockSC& b);

#endif
