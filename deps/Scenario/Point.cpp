#include "Point.h"

void to_json(json& j, const PointSC& p) 
{
	j = json{p.x,p.y};
}
