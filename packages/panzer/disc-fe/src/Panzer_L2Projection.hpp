// @HEADER
// @HEADER

#ifndef PANZER_L2_PROJECTION_HPP
#define PANZER_L2_PROJECTION_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_DefaultMpiComm.hpp"
#include <vector>
#include <string>

namespace Teuchos {
  template<typename T> class MpiComm;
}

namespace panzer {

  class LinearObjContainer;
  class BasisDescriptor;
  class IntegrationDescriptor;
  template<typename LO,typename GO> class ConnManager;
  class WorksetContainer;

  /** \brief Creates a mass matrix for an L2 projection of a scalar field(s) onto the basis.

      \param[in] basis (required) Basis that field values will be projected to. To project to multiple fields at the same time, send in more than one basis (or the same basis repeated).
      \param[in] integrationDescriptor (required) Integration descriptor for the projection.
      \param[in] mesh (required)
      \param[in] worksetContainer (optional) If the user has already allocated worksets for the corresponding mesh/element blocks, we can used them instead of reallocating the same ones.
      \returns Filled Matrix in a LinearObjectContainer
  */
  template<typename LO, typename GO>
  Teuchos::RCP<panzer::LinearObjContainer>
  buildMassMatrix(const panzer::BasisDescriptor& basis,
                  const panzer::IntegrationDescriptor& integrationDescriptor,
                  const Teuchos::RCP<Teuchos::MpiComm<int>>& comm,
                  const Teuchos::RCP<const panzer::ConnManager<LO,GO>> connManager,
                  const std::vector<std::string> elementBlocks,
                  const Teuchos::RCP<panzer::WorksetContainer>& worksetContainer = nullptr);




  /** \brief This class provides general utilities to perform a L2
      projections. It can be used to build the Mass matrix and RHS
      vectors.

      Users can perform projections in multiple ways. They could
      formulate a projection that does multiple field values all at
      once. If projection multiple fields to the same basis, another
      possibility is to create a mass matrix for a single field
      projection and reuse the matrix for each of the fields. In this
      case, performance can be improved further via using multiple
      right-hand-sides (one per field to project) with the Petra
      MultiVector concept. Users can also choose between consistent
      and lumped mass matrix formulations. This class provides the
      tools to try all of these avenues.

   */
  // class L2Projection {

  //   Teuchos::RCP<const panzer::UniqueGlobalIndexerBase> globalIndexer_;
  //   Teuchos::RCP<const panzer::LinearObjFactory<panzer::Traits> > linObjFactory_;

  //   Teuchos::RCP<const panzer::UniqueGlobalIndexerBase> massGlobalIndexer_;
  //   Teuchos::RCP<const panzer::LinearObjFactory<panzer::Traits> > massLOF_;

  //   std::vector<std::vector<std::string>> scatterScalarNames_;
  //   std::vector<std::string> gatherVectorNames_;

  //   panzer::IntegrationDescriptor integrationDescriptor_;

  // public:

  //   using LO = int;
  //   using GO = panzer::Ordinal64;

  //   //! How to handle inverting the mass matrix
  //   enum class ProjectionType {
  //     Consistent, //! Consistent mass matrix - requires linear solve to invert.
  //     Lumped //! Mass matrix inverse is approximated with inverse row sums. Only requires element-wise multiplication to apply the inverse.
  //   };

  //   L2Projection() = default;
  //   ~L2Projection() = default;

  //   /** \brief Sets the integration order for the problem

  //       \param[in] integrationOrder The integration order.
  //    */
  //   void setIntegrationDescriptor(const panzer::IntegrationDescriptor& id);

  //   /** \brief Sets the source Tpetra vector and corresponding dof manager.

  //       \param[in] sourceVector Tpetra vector with source values to project
  //       \param[in] soruceDOFManager DOFManager to access fields from the Teptra vector
  //    */
  //   void setSourceValues(const Teuchos::RCP<panzer::LinearObjContainer>& sourceVector,
  //                        const Teuchos::RCP<panzer::UniqueGlobalIndexerBase >& sourceDOFManager);

  //   /** \brief Sets the source Tpetra vector and corresponding dof manager.

  //       \param[in] sourceVector Tpetra vector with source values to project
  //       \param[in] sourceDOFManager DOFManager to access fields from the Teptra vector
  //    */
  //   void setTargetValues(const Teuchos::RCP<panzer::LinearObjContainer>& targetVector,
  //                        const Teuchos::RCP<panzer::UniqueGlobalIndexerBase >& targetDOFManager);

  //   /** \brief Requests that a project from a vector basis to a set of scalar bases be performed.
  //    */
  //   void requestProjectionVectorToScalarField(const std::string sourceVectorFieldName,
  //                                             const Teuchos::RCP<panzer::BasisDescriptor>& basis,
  //                                             const std::vector<std::string>& targetScalarFieldNames);

  //   void buildObjects();
  // };

}

#endif
