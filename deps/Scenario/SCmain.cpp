#include "Params.h"
#include "SCmain.h"
#include <jlcxx/jlcxx.hpp>

void SCmain(string Instance_evalution)
{
	/* READ INSTANCE FILE AND INITIALIZE DATA STRUCTURES */
	ParamsSC ParamsSC(Instance_evalution);

	/* FOR EACH DISTRICT SELECTED IN THE TRAINING SET */
	ParamsSC.startTime = clock();

		//float probaCustomerDemand = 0.004;
		//int targetSizeOfDistrict = 3;
		for (int b = 0; b < ParamsSC.blocks.size(); b++)
		{	
			vector<vector<PointSC>> trainScenarios = vector<vector<PointSC>>();
		
			for (int s = 0; s < ParamsSC.sizeTrainingSet; s++)
			{
				// Pick the number of points in the BlockSC following a binomial distribution based on the number of inhabitants
				vector<PointSC> evaluationPoints;
				binomial_distribution<> distB(ParamsSC.blocks[b].nbInhabitants, ParamsSC.probaCustomerDemand);
				int nbCustomersSampledBlock = distB(ParamsSC.generator);
                if (nbCustomersSampledBlock <3) nbCustomersSampledBlock = 3;
				for (int i = 0; i < nbCustomersSampledBlock; i++)
				{
					uniform_real_distribution<> distX(ParamsSC.blocks[b].minX, ParamsSC.blocks[b].maxX);
					uniform_real_distribution<> distY(ParamsSC.blocks[b].minY, ParamsSC.blocks[b].maxY);

					PointSC randomPoint = {distX(ParamsSC.generator), distY(ParamsSC.generator)};

					while (!isInside(ParamsSC.blocks[b].verticesPoints, randomPoint))
						randomPoint = {distX(ParamsSC.generator), distY(ParamsSC.generator)};
					
					evaluationPoints.push_back(randomPoint);
				}

				trainScenarios.push_back(evaluationPoints);
			}

			ParamsSC.blocks[b].trainScenarios[ParamsSC.targetSizeOfDistrict] = trainScenarios;
		}

	ParamsSC.endTime = clock();

	/* EXPORT DISTRICTS TO FILE */
	ParamsSC.exportBlockScenarios();
}


// Wrap the function for Julia
JLCXX_MODULE define_julia_module(jlcxx::Module& module) {
	module.method("SCmain", &SCmain);
}