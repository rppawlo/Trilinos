// @HEADER
// *****************************************************************************
//           Panzer: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//
// Copyright 2011 NTESS and the Panzer contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include <Teuchos_ConfigDefs.hpp>
#include <Teuchos_UnitTestHarness.hpp>

#include "Panzer_BasisDescriptor.hpp"
#include "Panzer_IntegrationDescriptor.hpp"
#include "Panzer_LocalMeshInfo.hpp"
#include "Panzer_Workset.hpp"

#include "Shards_BasicTopologies.hpp"
#include "Shards_CellTopology.hpp"

namespace panzer {

namespace {

LocalMeshPartition buildQuadPartition()
{
  LocalMeshPartition partition;

  partition.num_owned_cells = 2;
  partition.num_ghstd_cells = 0;
  partition.num_virtual_cells = 0;
  partition.subcell_index = -1;
  partition.subcell_dimension = 2;
  partition.element_block_name = "block_0";
  partition.sideset_name = "";
  partition.cell_topology = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >()));

  partition.local_cells = PHX::View<panzer::LocalOrdinal*>("local_cells",2);
  auto local_cells_host = Kokkos::create_mirror_view(partition.local_cells);
  local_cells_host(0) = 0;
  local_cells_host(1) = 1;
  Kokkos::deep_copy(partition.local_cells,local_cells_host);

  partition.cell_nodes = PHX::View<double***>("cell_nodes",2,4,2);
  auto cell_nodes_host = Kokkos::create_mirror_view(partition.cell_nodes);

  cell_nodes_host(0,0,0) = 0.0; cell_nodes_host(0,0,1) = 0.0;
  cell_nodes_host(0,1,0) = 1.0; cell_nodes_host(0,1,1) = 0.0;
  cell_nodes_host(0,2,0) = 1.0; cell_nodes_host(0,2,1) = 1.0;
  cell_nodes_host(0,3,0) = 0.0; cell_nodes_host(0,3,1) = 1.0;

  cell_nodes_host(1,0,0) = 1.0; cell_nodes_host(1,0,1) = 0.0;
  cell_nodes_host(1,1,0) = 2.0; cell_nodes_host(1,1,1) = 0.0;
  cell_nodes_host(1,2,0) = 2.0; cell_nodes_host(1,2,1) = 1.0;
  cell_nodes_host(1,3,0) = 1.0; cell_nodes_host(1,3,1) = 1.0;

  Kokkos::deep_copy(partition.cell_nodes,cell_nodes_host);

  partition.has_connectivity = false;

  return partition;
}

} // end anonymous namespace

TEUCHOS_UNIT_TEST(workset_cache, integration_values_by_descriptor)
{
  WorksetDetails workset;
  WorksetOptions options;
  workset.setup(buildQuadPartition(), options);

  const IntegrationDescriptor descriptor(2, IntegrationDescriptor::VOLUME);
  const IntegrationDescriptor other_descriptor(3, IntegrationDescriptor::VOLUME);

  const IntegrationValues2<double> * values = &workset.getIntegrationValues(descriptor);
  TEST_EQUALITY(workset.int_rules.size(), 1);
  TEST_EQUALITY(workset.int_rules[0].get(), values);

  const IntegrationValues2<double> * values_again = &workset.getIntegrationValues(descriptor);
  TEST_EQUALITY(values, values_again);
  TEST_EQUALITY(workset.int_rules.size(), 1);
  TEST_EQUALITY(workset.int_rules[0].get(), values_again);

  const IntegrationValues2<double> * other_values = &workset.getIntegrationValues(other_descriptor);
  TEST_INEQUALITY(values, other_values);
  TEST_EQUALITY(workset.int_rules.size(), 2);
  TEST_EQUALITY(workset.int_rules[1].get(), other_values);

  const IntegrationValues2<double> * lazy_values = &workset.getIntegrationValues(descriptor, true);
  TEST_INEQUALITY(values, lazy_values);
  TEST_EQUALITY(workset.int_rules.size(), 3);
  TEST_EQUALITY(workset.int_rules[2].get(), lazy_values);

  const IntegrationValues2<double> * lazy_values_again = &workset.getIntegrationValues(descriptor, true);
  TEST_EQUALITY(lazy_values, lazy_values_again);
  TEST_EQUALITY(workset.int_rules.size(), 3);
  TEST_EQUALITY(workset.int_rules[2].get(), lazy_values_again);
}

TEUCHOS_UNIT_TEST(workset_cache, basis_values_by_descriptor)
{
  WorksetDetails workset;
  WorksetOptions options;
  workset.setup(buildQuadPartition(), options);

  const IntegrationDescriptor integration_descriptor(2, IntegrationDescriptor::VOLUME);
  const BasisDescriptor basis_descriptor(1, "HGrad");
  const BasisDescriptor other_basis_descriptor(2, "HGrad");

  BasisValues2<double> * values = &workset.getBasisValues(basis_descriptor, integration_descriptor);
  TEST_EQUALITY(workset.bases.size(), 1);
  TEST_EQUALITY(workset.bases[0].get(), values);

  BasisValues2<double> * values_again = &workset.getBasisValues(basis_descriptor, integration_descriptor);
  TEST_EQUALITY(values, values_again);
  TEST_EQUALITY(workset.bases.size(), 1);
  TEST_EQUALITY(workset.bases[0].get(), values_again);

  BasisValues2<double> * other_values = &workset.getBasisValues(other_basis_descriptor, integration_descriptor);
  TEST_INEQUALITY(values, other_values);
  TEST_EQUALITY(workset.bases.size(), 2);
  TEST_EQUALITY(workset.bases[1].get(), other_values);

  BasisValues2<double> * lazy_values = &workset.getBasisValues(basis_descriptor, integration_descriptor, true);
  TEST_INEQUALITY(values, lazy_values);
  TEST_EQUALITY(workset.bases.size(), 3);
  TEST_EQUALITY(workset.bases[2].get(), lazy_values);

  BasisValues2<double> * lazy_values_again = &workset.getBasisValues(basis_descriptor, integration_descriptor, true);
  TEST_EQUALITY(lazy_values, lazy_values_again);
  TEST_EQUALITY(workset.bases.size(), 3);
  TEST_EQUALITY(workset.bases[2].get(), lazy_values_again);

  const IntegrationDescriptor cv_side_descriptor(0, IntegrationDescriptor::CV_SIDE);
  BasisValues2<double> * cv_side_values = nullptr;
  TEST_NOTHROW(cv_side_values = &workset.getBasisValues(basis_descriptor, cv_side_descriptor, true));
  TEST_ASSERT(cv_side_values != nullptr);
}

TEUCHOS_UNIT_TEST(workset_cache, point_and_basis_point_values_by_descriptor)
{
  WorksetDetails workset;
  WorksetOptions options;
  workset.setup(buildQuadPartition(), options);

  const BasisDescriptor basis_descriptor(1, "HGrad");
  const auto point_descriptor = basis_descriptor.getPointDescriptor();

  PointValues2<double> * point_values = &workset.getPointValues(point_descriptor);
  PointValues2<double> * point_values_again = &workset.getPointValues(point_descriptor);
  TEST_EQUALITY(point_values, point_values_again);

  // The point-value maps are private, so check the returned object identity.
  const BasisValues2<double> * basis_values = &workset.getBasisValues(basis_descriptor, point_descriptor);
  const BasisValues2<double> * basis_values_again = &workset.getBasisValues(basis_descriptor, point_descriptor);
  TEST_EQUALITY(basis_values, basis_values_again);

  const BasisValues2<double> * lazy_basis_values = &workset.getBasisValues(basis_descriptor, point_descriptor, true);
  const BasisValues2<double> * lazy_basis_values_again = &workset.getBasisValues(basis_descriptor, point_descriptor, true);
  TEST_EQUALITY(lazy_basis_values, lazy_basis_values_again);
  TEST_INEQUALITY(basis_values, lazy_basis_values);
}

} // end namespace panzer
