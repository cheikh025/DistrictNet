#include "Block.h"

// Distance of a PointSC to the closest PointSC in the BlockSC
double BlockSC::distance(const PointSC & p1)
{
	double minDistance = 1.e30;
	for (int i2 = 0; i2 < verticesPoints.size(); i2++)
	{
		PointSC & p2 = verticesPoints[i2];
		PointSC & p3 = verticesPoints[(i2 + 1) % verticesPoints.size()];
		double dist = PointSC::distance(p1, p2, p3);
		if (dist < minDistance) minDistance = dist;
	}
	return minDistance;
};

void to_json(json& j, const BlockSC& b) 
{
	json jprop = json{
	{"ID", b.id},
	//{"NAME", b.zone_name},
	{"POPULATION", b.nbInhabitants},
	{"AREA", b.area},
	{"PERIMETER", b.perimeter},
	{"DENSITY", b.density},
	{"LIST_ADJACENT", b.adjacentBlocks},
	{"DIST_EUCL", b.distanceEucl},
	{"POINTS", b.verticesPoints}
	};
	
	json jgeom = json{
	{"type", "Polygon"},
	{"coordinates", {b.verticesLongLat}}
	};

	j = json{
	{"type", "Feature"},
	{"properties", jprop},
	{"geometry", jgeom}
	};
}
