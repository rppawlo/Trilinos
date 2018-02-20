// @HEADER
// @HEADER

#ifndef PANZER_L2_PROJECTION_IMPL_HPP
#define PANZER_L2_PROJECTION_IMPL_HPP

#include "Panzer_BasisDescriptor.hpp"
#include "Panzer_TpetraLinearObjContainer.hpp"
#include "Panzer_TpetraLinearObjFactory.hpp"
#include "Panzer_BlockedTpetraLinearObjFactory.hpp"
#include "Panzer_ElementBlockIdToPhysicsIdMap.hpp"
#include "Panzer_DOFManagerFactory.hpp"
#include "Panzer_BlockedDOFManagerFactory.hpp"
#include "Panzer_IntrepidFieldPattern.hpp"
#include "Panzer_WorksetDescriptor.hpp"
#include "Panzer_WorksetContainer.hpp"
#include "Panzer_Workset.hpp"

namespace panzer {

  template<typename LO, typename GO>
  Teuchos::RCP<panzer::LinearObjContainer>
  buildMassMatrix(const panzer::BasisDescriptor& basisDescriptor,
                  const panzer::IntegrationDescriptor& integrationDescriptor,
                  const Teuchos::RCP<Teuchos::MpiComm<int>>& comm,
                  const Teuchos::RCP<const panzer::ConnManager<LO,GO>> connManager,
                  const std::vector<std::string> elementBlockNames,
                  const Teuchos::RCP<panzer::WorksetContainer>& worksetContainer)
  {
    // Build DOF Manager
    Teuchos::RCP<panzer::DOFManager<LO,GO>> targetGlobalIndexer =
      Teuchos::rcp(new panzer::DOFManager<LO,GO>(Teuchos::rcp_const_cast<panzer::ConnManager<LO,GO>>(connManager),*(comm->getRawMpiComm())));
    for (const auto& eBlock : elementBlockNames) {
      std::vector<shards::CellTopology> topologies;
      connManager->getElementBlockTopologies(topologies);
      std::vector<std::string> ebNames;
      connManager->getElementBlockIds(ebNames);
      const auto search = std::find(ebNames.cbegin(),ebNames.cend(),eBlock);
      TEUCHOS_ASSERT(search != ebNames.cend());
      const int index = std::distance(ebNames.cbegin(),search);
      const auto& cellTopology = topologies[index];

      auto intrepidBasis = panzer::createIntrepid2Basis<PHX::Device,double,double>(basisDescriptor.getType(),basisDescriptor.getOrder(),cellTopology);
      Teuchos::RCP<const panzer::FieldPattern> fieldPattern(new panzer::Intrepid2FieldPattern(intrepidBasis));
      targetGlobalIndexer->addField(basisDescriptor.getType(),fieldPattern);
    }
    targetGlobalIndexer->buildGlobalUnknowns();

    // Check workset needs are correct

    // Allocate the owned matrix
    std::vector<Teuchos::RCP<const panzer::UniqueGlobalIndexer<LO,GO>>> indexers;
    indexers.push_back(targetGlobalIndexer);

    panzer::BlockedTpetraLinearObjFactory<panzer::Traits,double,LO,GO,panzer::TpetraNodeType> factory(comm,indexers);

    auto ownedMatrix = factory.getTpetraMatrix(0,0);

    ownedMatrix->resumeFill();
    ownedMatrix->setAllToScalar(0.0);


    // Loop over element blocks and fills mass matrix
    for (const auto& block : elementBlockNames) {

      // Based on descriptor, there should only be one workset
      panzer::WorksetDescriptor wd(block,panzer::WorksetSizeType::ALL_ELEMENTS,true,false);
      const auto& worksets = worksetContainer->getWorksets(wd);
      TEUCHOS_ASSERT(worksets->size() == 1);
      const auto& workset = (*worksets)[0];
      const auto& basisValues = workset.getBasisValues(basis,integrationDescriptor);

      const auto& weighted_basis = basisValues.weighted_basis_scalar;
      const int numBasisPoints = static_cast<int>(weighted_basis.extent(1));

      Kokkos::parallel_for(workset.numOwnedCells(),KOKKOS_LAMBDA (const int& cell) {
        for (int qp=0; qp < numQP; ++qp) {
          double tmp(0.0);
          for (int i=0; i < numBasisPoints; ++i) {
            for (int j=0; j < numBasisPoints; ++j) {

              //ownedMatrix = ;
            }
          }
        }
      });

    }


    // return ownedMatrix;
    return Teuchos::null;
  }


    /*
  void L2Projection::setIntegrationDescriptor(const panzer::IntegrationDescriptor& id)
  { integrationDescriptor_ = id; }

  void L2Projection::setSourceValues(const Teuchos::RCP<panzer::LinearObjContainer>& sourceVector,
                                     const Teuchos::RCP<panzer::UniqueGlobalIndexerBase >& sourceDOFManager)
  {

  }

  void L2Projection::setTargetValues(const Teuchos::RCP<panzer::LinearObjContainer>& targetVector,
                                     const Teuchos::RCP<panzer::UniqueGlobalIndexerBase >& targetDOFManager)
  {

  }

  void L2Projection::requestProjectionVectorToScalarField(const std::string sourceVectorFieldName,
                                                          const Teuchos::RCP<panzer::BasisDescriptor>& basis,
                                                          const std::vector<std::string>& targetScalarFieldNames)
  {


    // Need ADReordering

  }

  void L2Projection::buildObjects()
  {
    TEUCHOS_ASSERT(nonnull());

  }

  // Teuchos::RCP<Tpetra::CrsMatrix>
  // L2Projection::buildMassMatrixForSingleFieldNodalScalarProjection(const BasisDescriptor& basis,
  //                                                                  const IntegrationDescriptor& int_rule,
  //                                                                  const Teuchos::RCP<Mesh>& mesh,
  //                                                                  const TeuchosRCP<std::vector<panzer::Workset>>& worksets)
  // {
  // }

  // Teuchos::RCP<Tpetra::MultiVector<>>
  // buildRHSMultiVector()
  // {

  // }
  */

}

#endif
