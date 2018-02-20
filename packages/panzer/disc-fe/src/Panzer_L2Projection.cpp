// @HEADER
// @HEADER

#include "PanzerDiscFE_config.hpp"
#include "Panzer_L2Projection.hpp"
#include "Panzer_L2Projection_impl.hpp"

template
Teuchos::RCP<panzer::LinearObjContainer>
panzer::buildMassMatrix<int,panzer::Ordinal64>(const panzer::BasisDescriptor& basis,
                                               const panzer::IntegrationDescriptor& integrationDescriptor,
                                               const Teuchos::RCP<Teuchos::MpiComm<int>>& comm,
                                               const Teuchos::RCP<const panzer::ConnManager<int,panzer::Ordinal64>> connManager,
                                               const std::vector<std::string> elementBlockNames,
                                               const Teuchos::RCP<panzer::WorksetContainer>& worksetContainer);
