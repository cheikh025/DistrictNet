#include "Params.h"
#include <cstdlib>
#include <iostream>
#include <filesystem>

using namespace std;

vector<string> Split(const string &s, const char &c)
{
	string buff{""};
	vector<string> v;

	for (auto n : s)
	{
		if (n != c)
			buff += n;
		else if (n == c && buff != "")
		{
			v.push_back(buff);
			buff = "";
		}
	}
	if (buff != "")
		v.push_back(buff);

	return v;
}

ParamsSC::ParamsSC(string Instance_evalution)
{

	vector<string> DataConfig = Split(Instance_evalution, '_');
	dataName = DataConfig[0];
	//dataName = string(argv[1]).c_str();
	cout << "CITY NAME: " << dataName << endl;

	targetSizeOfDistrict = stoi(DataConfig[3]);

	depotPosition = DataConfig[1];

	sizeTrainingSet =  stoi(DataConfig[4]);
	cout << "NB Scenarios PER BlockSC: " << sizeTrainingSet << endl;
    
	
	instanceSize  = stoi(DataConfig[2]);
	int randomNumber = rand();
	seed = randomNumber % 10;
	generator.seed(seed);
	cout << "SEED Scenarios: " << seed << endl;

	// Define data paths and read datafile in JSON format
	instanceName = "data/geojson/" + dataName;
	readBlocksJSON();

	outputName = dataName;

	probaCustomerDemandPerTargetSizeDistrict = map<int, double> {{3, 0.004}, {6, 0.002}, {12, 0.001}, {20, 0.0006}, {30, 0.0004}};
	// check if the target size of district is in the map
	if (probaCustomerDemandPerTargetSizeDistrict.find(targetSizeOfDistrict) == probaCustomerDemandPerTargetSizeDistrict.end())
		// create a new entry in the map with value = 96/(8000*targetSizeOfDistrict)
		probaCustomerDemandPerTargetSizeDistrict[targetSizeOfDistrict] = 96. / (8000. * targetSizeOfDistrict);
	probaCustomerDemand = probaCustomerDemandPerTargetSizeDistrict[targetSizeOfDistrict];

	cout << "PROBA CUSTOMER DEMAND: " << probaCustomerDemand << endl;

	cout << "FINISHED READING DATA" << endl;

}

void ParamsSC::readBlocksJSON()
{
// Parsing all blocks in the JSON file
	ifstream inputFile(instanceName + ".geojson");
	if (inputFile.is_open())
	{
		json j;
		inputFile >> j;
		inputFile.close();
		int blockID = 0;
		for (auto &blockIN : j.at("features"))
		{
				
				BlockSC block = BlockSC();
				block.id = blockIN.at("properties").at("ID");
				//block.zone_name = blockIN.at("properties").at("NAME");
				block.nbInhabitants = blockIN.at("properties").at("POPULATION");
				auto &polyIN = blockIN.at("properties").at("POINTS");
				//auto &polyIN = blockIN.at("geometry").at("coordinates")[0];
				// check if it is a polygon or a multipolygon
				if (polyIN.size() == 1)
					polyIN = polyIN[0];
				for (array<double, 2> longLatInfo : polyIN)
				{
					PointSC myPoint = {longLatInfo[0], longLatInfo[1]};
					block.verticesPoints.push_back(myPoint);
					block.minX = min<double>(myPoint.x, block.minX);
					block.maxX = max<double>(myPoint.x, block.maxX);
					block.minY = min<double>(myPoint.y, block.minY);
					block.maxY = max<double>(myPoint.y, block.maxY);
	
				}
				block.distReferencePoint = block.distance({0, 0});
				block.perimeter = blockIN.at("properties").at("PERIMETER");
				block.area = blockIN.at("properties").at("AREA");
				blocks.push_back(block);
		}

		referenceLongLat = j.at("metadata").at("REFERENCE_LONGLAT");
	}
	else
		throw std::invalid_argument("Impossible to open instance JSON file: " + instanceName + ".geojson");
}
void ParamsSC::exportBlockScenarios()
{
		vector<BlockSC> currentBlocks = vector<BlockSC> (blocks);
		currentBlocks.resize(instanceSize);

		maxX = -INFINITY;
		maxY = -INFINITY;

		minX = INFINITY;
		minY = INFINITY;

		for (BlockSC block : currentBlocks)
		{
			maxX = max(maxX,block.maxX);
			maxY = max(maxY, block.maxY);

			minX = min(minX, block.minX);
			minY = min(minY, block.minY);
		}

		//depotPosition = "C";
          
		PointSC depotPoint;


		if (depotPosition == "C")  depotPoint = {(minX + maxX) / 2.0, (minY + maxY) / 2.0 };
		if (depotPosition == "NW") depotPoint = { minX, maxY };
		if (depotPosition == "NE") depotPoint = { maxX, maxY };
		if (depotPosition == "SW") depotPoint = { minX, minY };
		if (depotPosition == "SE") depotPoint = { maxX, minY };
			for (int nbBlock=0 ; nbBlock < currentBlocks.size(); nbBlock++)
			{
				BlockSC* block = &currentBlocks[nbBlock];
				block->distDepot = block->distance(depotPoint);
			}

			auto temp = wgs84::fromCartesian({referenceLongLat[1], referenceLongLat[0]}, {1000. * depotPoint.x, 1000. * depotPoint.y});
			array<double,2> depotLongLat = array<double,2> ({temp[1], temp[0]});

				//int targetSizeOfDistrict = 3;
				double probaCustomer = probaCustomerDemandPerTargetSizeDistrict[targetSizeOfDistrict];
				
				int maxSizeOfDistrict = ceil(1.2 * targetSizeOfDistrict);
				int minSizeOfDistrict = floor(0.8 * targetSizeOfDistrict);

				int numberOfDistricts = floor(instanceSize / targetSizeOfDistrict);

				string instanceName = dataName + "_" + depotPosition + "_" + to_string(instanceSize) + "_" + to_string(targetSizeOfDistrict);

				string outputName = "deps/Scenario/output/" + instanceName + ".json";
				ofstream myfile;
					myfile.open(outputName);
					if (myfile.is_open())
					{
						json jblocks;
						for (BlockSC block : currentBlocks)
						{
							json jBlock = {
								{"ID", block.id},
								{"Scenarios", block.trainScenarios[targetSizeOfDistrict] },
								{"DEPOT_DIST", block.distDepot},
							};
							jblocks += jBlock;
						}	

						myfile << json{{"blocks", jblocks},
									   {"metadata", {
										   	{"TARGET_SIZE_DISTRICT", targetSizeOfDistrict}, 
									   		{"MAX_SIZE_DISTRICT", maxSizeOfDistrict},
											{"MIN_SIZE_DISTRICT", minSizeOfDistrict},
											{"NUMBER_OF_DISTRICTS", numberOfDistricts},
											{"DEPOT_XY", depotPoint},
											{"DEPOT_LONGLAT", depotLongLat},
											{"PROBA_CUSTOMER_DEMAND", probaCustomer},
									   }}};
						myfile.close();
						cout << "Instance " << instanceName << " created" << endl;
					}
					else
						throw std::invalid_argument("Impossible to open output file: " + outputName);

}