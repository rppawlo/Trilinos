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
#include <Teuchos_RCP.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include "Panzer_CellData.hpp"
#include "Panzer_IntegrationRule.hpp"
#include "Panzer_IntegrationValues2.hpp"
#include "Panzer_ArrayTraits.hpp"
#include "Panzer_CommonArrayFactories.hpp"
#include "Panzer_SubcellConnectivity.hpp"
#include "Panzer_LocalMeshInfo.hpp"

using Teuchos::RCP;
using Teuchos::rcp;
using panzer::IntegrationRule;

namespace panzer {

  static PHX::MDField<double,Cell,NODE,Dim>
  buildUnitQuadNodeCoordinates(const int num_cells,
                               const int num_nodes)
  {
    const int base_cell_dimension = 2;
    panzer::MDFieldArrayFactory af("prefix_",true);
    auto node_coordinates = af.buildStaticArray<double,Cell,NODE,Dim>("nc",num_cells,num_nodes,base_cell_dimension);

    typedef panzer::ArrayTraits<double,PHX::MDField<double> >::size_type size_type;
    const size_type x = 0;
    const size_type y = 1;
    auto node_coordinates_k = node_coordinates.get_view();
    Kokkos::parallel_for("initialize unit quad node coords",node_coordinates.extent(0),
                         KOKKOS_LAMBDA (const int cell) {
      node_coordinates_k(cell,0,x) = 0.0;
      node_coordinates_k(cell,0,y) = 0.0;
      node_coordinates_k(cell,1,x) = 1.0;
      node_coordinates_k(cell,1,y) = 0.0;
      node_coordinates_k(cell,2,x) = 1.0;
      node_coordinates_k(cell,2,y) = 1.0;
      node_coordinates_k(cell,3,x) = 0.0;
      node_coordinates_k(cell,3,y) = 1.0;
    });

    return node_coordinates;
  }

  template <typename ViewA, typename ViewB>
  static void
  testFloatingEqualityRank2(const ViewA& expected,
                            const ViewB& actual,
                            const double tolerance,
                            Teuchos::FancyOStream& out,
                            bool& success)
  {
    TEST_EQUALITY(expected.extent_int(0), actual.extent_int(0));
    TEST_EQUALITY(expected.extent_int(1), actual.extent_int(1));
    for(int i=0; i<expected.extent_int(0); ++i)
      for(int j=0; j<expected.extent_int(1); ++j)
        TEST_FLOATING_EQUALITY(expected(i,j), actual(i,j), tolerance);
  }

  template <typename ViewA, typename ViewB>
  static void
  testFloatingEqualityRank3(const ViewA& expected,
                            const ViewB& actual,
                            const double tolerance,
                            Teuchos::FancyOStream& out,
                            bool& success)
  {
    TEST_EQUALITY(expected.extent_int(0), actual.extent_int(0));
    TEST_EQUALITY(expected.extent_int(1), actual.extent_int(1));
    TEST_EQUALITY(expected.extent_int(2), actual.extent_int(2));
    for(int i=0; i<expected.extent_int(0); ++i)
      for(int j=0; j<expected.extent_int(1); ++j)
        for(int k=0; k<expected.extent_int(2); ++k)
          TEST_FLOATING_EQUALITY(expected(i,j,k), actual(i,j,k), tolerance);
  }

  template <typename ViewA, typename ViewB>
  static void
  testFloatingEqualityRank4(const ViewA& expected,
                            const ViewB& actual,
                            const double tolerance,
                            Teuchos::FancyOStream& out,
                            bool& success)
  {
    TEST_EQUALITY(expected.extent_int(0), actual.extent_int(0));
    TEST_EQUALITY(expected.extent_int(1), actual.extent_int(1));
    TEST_EQUALITY(expected.extent_int(2), actual.extent_int(2));
    TEST_EQUALITY(expected.extent_int(3), actual.extent_int(3));
    for(int i=0; i<expected.extent_int(0); ++i)
      for(int j=0; j<expected.extent_int(1); ++j)
        for(int k=0; k<expected.extent_int(2); ++k)
          for(int l=0; l<expected.extent_int(3); ++l)
            TEST_FLOATING_EQUALITY(expected(i,j,k,l), actual(i,j,k,l), tolerance);
  }

  TEUCHOS_UNIT_TEST(integration_values, volume)
  {
    Teuchos::RCP<shards::CellTopology> topo =
       Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData< shards::Quadrilateral<4> >()));

    const int num_cells = 20;
    const int base_cell_dimension = 2;
    const panzer::CellData cell_data(num_cells,topo);

    const int cubature_degree = 2;
    RCP<IntegrationRule> int_rule =
      rcp(new IntegrationRule(cubature_degree, cell_data));

    panzer::MDFieldArrayFactory af("prefix_",true);

    const int num_nodes = int_rule->topology->getNodeCount();
    PHX::MDField<double,Cell,NODE,Dim> node_coordinates
        = af.buildStaticArray<double,Cell,NODE,Dim>("nc",num_cells, num_nodes, base_cell_dimension);

    // Set up node coordinates.  Here we assume the following
    // ordering.  This needs to be consistent with shards topology,
    // otherwise we will get negative determinates

    // 3(0,1)---2(1,1)
    //   |    0  |
    //   |       |
    // 0(0,0)---1(1,0)

    typedef panzer::ArrayTraits<double,PHX::MDField<double> >::size_type size_type;
    const size_type x = 0;
    const size_type y = 1;
    auto node_coordinates_k = node_coordinates.get_view();
    Kokkos::parallel_for("initialize node coords",node_coordinates.extent(0),
                         KOKKOS_LAMBDA (const int cell) {
      node_coordinates_k(cell,0,x) = 0.0;
      node_coordinates_k(cell,0,y) = 0.0;
      node_coordinates_k(cell,1,x) = 1.0;
      node_coordinates_k(cell,1,y) = 0.0;
      node_coordinates_k(cell,2,x) = 1.0;
      node_coordinates_k(cell,2,y) = 1.0;
      node_coordinates_k(cell,3,x) = 0.0;
      node_coordinates_k(cell,3,y) = 1.0;
    });

    panzer::IntegrationValues2<double> int_values("prefix_");
    int_values.setup(int_rule, node_coordinates);

    TEST_EQUALITY(int_values.getCubaturePoints().extent(1), 4);
    double realspace_x_coord = (1.0/std::sqrt(3.0) + 1.0) / 2.0;
    double realspace_y_coord = (1.0/std::sqrt(3.0) + 1.0) / 2.0;
    auto tmp_coords = int_values.getCubaturePoints();
    auto tmp_coords_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),tmp_coords.get_view());
    TEST_FLOATING_EQUALITY(tmp_coords_host(0,0,0),
                           realspace_x_coord, 1.0e-8);
    TEST_FLOATING_EQUALITY(tmp_coords_host(0,0,1),
                           realspace_y_coord, 1.0e-8);
  }

  TEUCHOS_UNIT_TEST(integration_values, lazy_getter_cache_control)
  {
    Teuchos::RCP<shards::CellTopology> topo =
       Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData< shards::Quadrilateral<4> >()));

    const int num_cells = 2;
    const int base_cell_dimension = 2;
    const panzer::CellData cell_data(num_cells,topo);

    const int cubature_degree = 2;
    RCP<IntegrationRule> int_rule =
      rcp(new IntegrationRule(cubature_degree, cell_data));

    const int num_nodes = int_rule->topology->getNodeCount();
    PHX::MDField<double,Cell,NODE,Dim> node_coordinates("nc","nc",num_cells,num_nodes,base_cell_dimension);

    typedef panzer::ArrayTraits<double,PHX::MDField<double> >::size_type size_type;
    const size_type x = 0;
    const size_type y = 1;
    auto node_coordinates_k = node_coordinates.get_view();
    Kokkos::parallel_for("initialize node coords for lazy cache test",node_coordinates.extent(0),
                         KOKKOS_LAMBDA (const int cell) {
      node_coordinates_k(cell,0,x) = 0.0;
      node_coordinates_k(cell,0,y) = 0.0;
      node_coordinates_k(cell,1,x) = 1.0;
      node_coordinates_k(cell,1,y) = 0.0;
      node_coordinates_k(cell,2,x) = 1.0;
      node_coordinates_k(cell,2,y) = 1.0;
      node_coordinates_k(cell,3,x) = 0.0;
      node_coordinates_k(cell,3,y) = 1.0;
    });

    panzer::IntegrationValues2<double> int_values("prefix_");
    int_values.setup(int_rule, node_coordinates);

    TEST_EQUALITY(int_values.jac.extent(0), 0);

    auto uncached_jac = int_values.getJacobian(false);
    TEST_EQUALITY(uncached_jac.extent(0), num_cells);
    TEST_EQUALITY(int_values.jac.extent(0), 0);

    auto cached_jac = int_values.getJacobian(true);
    TEST_EQUALITY(cached_jac.extent(0), num_cells);
    TEST_EQUALITY(int_values.jac.extent(0), num_cells);
    TEST_EQUALITY(cached_jac.get_view().data(), int_values.jac.get_view().data());

    auto cached_jac_again = int_values.getJacobian(true);
    TEST_EQUALITY(cached_jac_again.get_view().data(), int_values.jac.get_view().data());

    auto cached_ref_points = int_values.getUniformCubaturePointsRef(true, false, false);
    auto cached_weights = int_values.getUniformCubatureWeightsRef(true, false, false);
    TEST_EQUALITY(cached_ref_points.extent_int(0), int_rule->num_points);
    TEST_EQUALITY(cached_weights.extent_int(0), int_rule->num_points);
    TEST_EQUALITY(int_values.cub_points.extent_int(0), int_rule->num_points);
    TEST_EQUALITY(int_values.cub_weights.extent_int(0), int_rule->num_points);

    int_values.releaseCachedData();
    TEST_EQUALITY(int_values.cub_points.size(), 0);
    TEST_EQUALITY(int_values.cub_weights.size(), 0);
    TEST_EQUALITY(int_values.jac.size(), 0);
    TEST_EQUALITY(int_values.jac_det.size(), 0);
    TEST_EQUALITY(int_values.weighted_measure.size(), 0);

    auto rebuilt_cached_jac = int_values.getJacobian(true);
    TEST_EQUALITY(rebuilt_cached_jac.extent(0), num_cells);
    TEST_EQUALITY(int_values.jac.extent(0), num_cells);
    TEST_EQUALITY(rebuilt_cached_jac.get_view().data(), int_values.jac.get_view().data());

    panzer::IntegrationValues2<double> det_values("prefix_");
    det_values.setup(int_rule, node_coordinates);

    auto uncached_det_with_cached_dependencies = det_values.getJacobianDeterminant(false, false, true);
    TEST_EQUALITY(uncached_det_with_cached_dependencies.extent(0), num_cells);
    TEST_EQUALITY(det_values.jac_det.extent(0), 0);
    TEST_EQUALITY(det_values.jac.extent(0), num_cells);

    panzer::IntegrationValues2<double> det_values_default_cache("prefix_");
    det_values_default_cache.setup(int_rule, node_coordinates);

    auto cached_det_default = det_values_default_cache.getJacobianDeterminant(true, false);
    TEST_EQUALITY(cached_det_default.extent(0), num_cells);
    TEST_EQUALITY(det_values_default_cache.jac_det.extent(0), num_cells);
    TEST_EQUALITY(det_values_default_cache.jac.extent(0), num_cells);

    panzer::IntegrationValues2<double> det_values_without_dependencies("prefix_");
    det_values_without_dependencies.setup(int_rule, node_coordinates);

    auto cached_det_without_dependencies = det_values_without_dependencies.getJacobianDeterminant(true, false, false);
    TEST_EQUALITY(cached_det_without_dependencies.extent(0), num_cells);
    TEST_EQUALITY(det_values_without_dependencies.jac_det.extent(0), num_cells);
    TEST_EQUALITY(det_values_without_dependencies.jac.extent(0), 0);

    panzer::IntegrationValues2<double> det_values_with_dependencies("prefix_");
    det_values_with_dependencies.setup(int_rule, node_coordinates);

    auto cached_det_with_dependencies = det_values_with_dependencies.getJacobianDeterminant(true, false, true);
    TEST_EQUALITY(cached_det_with_dependencies.extent(0), num_cells);
    TEST_EQUALITY(det_values_with_dependencies.jac_det.extent(0), num_cells);
    TEST_EQUALITY(det_values_with_dependencies.jac.extent(0), num_cells);

    panzer::IntegrationValues2<double> weighted_values("prefix_");
    weighted_values.setup(int_rule, node_coordinates);

    auto uncached_weighted_with_cached_dependencies = weighted_values.getWeightedMeasure(false, false, true);
    TEST_EQUALITY(uncached_weighted_with_cached_dependencies.extent(0), num_cells);
    TEST_EQUALITY(weighted_values.weighted_measure.extent(0), 0);
    TEST_EQUALITY(weighted_values.jac_det.extent(0), num_cells);

    panzer::IntegrationValues2<double> weighted_values_default_cache("prefix_");
    weighted_values_default_cache.setup(int_rule, node_coordinates);

    auto cached_weighted_default = weighted_values_default_cache.getWeightedMeasure(true, false);
    TEST_EQUALITY(cached_weighted_default.extent(0), num_cells);
    TEST_EQUALITY(weighted_values_default_cache.weighted_measure.extent(0), num_cells);
    TEST_EQUALITY(weighted_values_default_cache.jac_det.extent(0), num_cells);
    TEST_EQUALITY(weighted_values_default_cache.jac.extent(0), num_cells);

    panzer::IntegrationValues2<double> weighted_values_without_dependencies("prefix_");
    weighted_values_without_dependencies.setup(int_rule, node_coordinates);

    auto cached_weighted_without_dependencies = weighted_values_without_dependencies.getWeightedMeasure(true, false, false);
    TEST_EQUALITY(cached_weighted_without_dependencies.extent(0), num_cells);
    TEST_EQUALITY(weighted_values_without_dependencies.weighted_measure.extent(0), num_cells);
    TEST_EQUALITY(weighted_values_without_dependencies.jac_det.extent(0), 0);

    panzer::IntegrationValues2<double> weighted_values_with_dependencies("prefix_");
    weighted_values_with_dependencies.setup(int_rule, node_coordinates);

    auto cached_weighted_with_dependencies = weighted_values_with_dependencies.getWeightedMeasure(true, false, true);
    TEST_EQUALITY(cached_weighted_with_dependencies.extent(0), num_cells);
    TEST_EQUALITY(weighted_values_with_dependencies.weighted_measure.extent(0), num_cells);
    TEST_EQUALITY(weighted_values_with_dependencies.jac_det.extent(0), num_cells);

    weighted_values_with_dependencies.releaseCachedData();
    TEST_EQUALITY(weighted_values_with_dependencies.weighted_measure.size(), 0);
    TEST_EQUALITY(weighted_values_with_dependencies.jac_det.size(), 0);
    TEST_EQUALITY(weighted_values_with_dependencies.jac.size(), 0);

    auto rebuilt_cached_weighted = weighted_values_with_dependencies.getWeightedMeasure(true, false, true);
    TEST_EQUALITY(rebuilt_cached_weighted.extent(0), num_cells);
    TEST_EQUALITY(weighted_values_with_dependencies.weighted_measure.extent(0), num_cells);
    TEST_EQUALITY(weighted_values_with_dependencies.jac_det.extent(0), num_cells);
    TEST_EQUALITY(weighted_values_with_dependencies.jac.extent(0), num_cells);
  }

  TEUCHOS_UNIT_TEST(integration_values, geometry_metric_release_rebuild)
  {
    Teuchos::RCP<shards::CellTopology> topo =
       Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData< shards::Quadrilateral<4> >()));

    const int num_cells = 2;
    const panzer::CellData cell_data(num_cells,topo);

    const int cubature_degree = 2;
    RCP<IntegrationRule> int_rule =
      rcp(new IntegrationRule(cubature_degree, cell_data));

    const int num_nodes = int_rule->topology->getNodeCount();
    auto node_coordinates = buildUnitQuadNodeCoordinates(num_cells,num_nodes);

    panzer::IntegrationValues2<double> int_values("prefix_");
    int_values.setup(int_rule, node_coordinates);

    auto cached_jacobian = int_values.getJacobian(true);
    auto cached_jacobian_inverse = int_values.getJacobianInverse(true, false, true);
    auto cached_jacobian_determinant = int_values.getJacobianDeterminant(true, false, true);
    auto cached_weighted_measure = int_values.getWeightedMeasure(true, false, true);
    auto cached_covarient = int_values.getCovarientMatrix(true, false, true);
    auto cached_contravarient = int_values.getContravarientMatrix(true, false, true);
    auto cached_norm_contravarient = int_values.getNormContravarientMatrix(true, false, true);

    auto cached_jacobian_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),cached_jacobian.get_view());
    auto cached_jacobian_inverse_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),cached_jacobian_inverse.get_view());
    auto cached_jacobian_determinant_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),cached_jacobian_determinant.get_view());
    auto cached_weighted_measure_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),cached_weighted_measure.get_view());
    auto cached_covarient_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),cached_covarient.get_view());
    auto cached_contravarient_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),cached_contravarient.get_view());
    auto cached_norm_contravarient_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),cached_norm_contravarient.get_view());

    int_values.releaseCachedData();
    TEST_EQUALITY(int_values.jac.size(), 0);
    TEST_EQUALITY(int_values.jac_inv.size(), 0);
    TEST_EQUALITY(int_values.jac_det.size(), 0);
    TEST_EQUALITY(int_values.weighted_measure.size(), 0);
    TEST_EQUALITY(int_values.covarient.size(), 0);
    TEST_EQUALITY(int_values.contravarient.size(), 0);
    TEST_EQUALITY(int_values.norm_contravarient.size(), 0);
    TEST_EQUALITY(int_values.node_coordinates.extent(0), num_cells);

    TEST_NOTHROW(int_values.getJacobian(true));
    TEST_NOTHROW(int_values.getJacobianInverse(true, false, true));
    TEST_NOTHROW(int_values.getJacobianDeterminant(true, false, true));
    TEST_NOTHROW(int_values.getWeightedMeasure(true, false, true));
    TEST_NOTHROW(int_values.getCovarientMatrix(true, false, true));
    TEST_NOTHROW(int_values.getContravarientMatrix(true, false, true));
    TEST_NOTHROW(int_values.getNormContravarientMatrix(true, false, true));

    auto rebuilt_jacobian = int_values.getJacobian(true);
    auto rebuilt_jacobian_inverse = int_values.getJacobianInverse(true, false, true);
    auto rebuilt_jacobian_determinant = int_values.getJacobianDeterminant(true, false, true);
    auto rebuilt_weighted_measure = int_values.getWeightedMeasure(true, false, true);
    auto rebuilt_covarient = int_values.getCovarientMatrix(true, false, true);
    auto rebuilt_contravarient = int_values.getContravarientMatrix(true, false, true);
    auto rebuilt_norm_contravarient = int_values.getNormContravarientMatrix(true, false, true);

    TEST_EQUALITY(rebuilt_jacobian.extent(0), num_cells);
    TEST_EQUALITY(rebuilt_jacobian_inverse.extent(0), num_cells);
    TEST_EQUALITY(rebuilt_jacobian_determinant.extent(0), num_cells);
    TEST_EQUALITY(rebuilt_weighted_measure.extent(0), num_cells);
    TEST_EQUALITY(rebuilt_covarient.extent(0), num_cells);
    TEST_EQUALITY(rebuilt_contravarient.extent(0), num_cells);
    TEST_EQUALITY(rebuilt_norm_contravarient.extent(0), num_cells);

    auto rebuilt_jacobian_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rebuilt_jacobian.get_view());
    auto rebuilt_jacobian_inverse_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rebuilt_jacobian_inverse.get_view());
    auto rebuilt_jacobian_determinant_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rebuilt_jacobian_determinant.get_view());
    auto rebuilt_weighted_measure_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rebuilt_weighted_measure.get_view());
    auto rebuilt_covarient_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rebuilt_covarient.get_view());
    auto rebuilt_contravarient_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rebuilt_contravarient.get_view());
    auto rebuilt_norm_contravarient_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rebuilt_norm_contravarient.get_view());

    testFloatingEqualityRank4(cached_jacobian_host, rebuilt_jacobian_host, 1.0e-14, out, success);
    testFloatingEqualityRank4(cached_jacobian_inverse_host, rebuilt_jacobian_inverse_host, 1.0e-14, out, success);
    testFloatingEqualityRank2(cached_jacobian_determinant_host, rebuilt_jacobian_determinant_host, 1.0e-14, out, success);
    testFloatingEqualityRank2(cached_weighted_measure_host, rebuilt_weighted_measure_host, 1.0e-14, out, success);
    testFloatingEqualityRank4(cached_covarient_host, rebuilt_covarient_host, 1.0e-14, out, success);
    testFloatingEqualityRank4(cached_contravarient_host, rebuilt_contravarient_host, 1.0e-14, out, success);
    testFloatingEqualityRank2(cached_norm_contravarient_host, rebuilt_norm_contravarient_host, 1.0e-14, out, success);
  }

  TEUCHOS_UNIT_TEST(integration_values, control_volume)
  {
    Teuchos::RCP<shards::CellTopology> topo =
       Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData< shards::Quadrilateral<4> >()));

    const int num_cells = 20;
    const int base_cell_dimension = 2;
    const panzer::CellData cell_data(num_cells,topo);

    std::string cv_type = "volume";
    RCP<IntegrationRule> int_rule_vol =
      rcp(new IntegrationRule(cell_data, cv_type));

    panzer::MDFieldArrayFactory af("prefix_",true);

    cv_type = "side";
    RCP<IntegrationRule> int_rule_side =
      rcp(new IntegrationRule(cell_data, cv_type));

    const int num_nodes = int_rule_vol->topology->getNodeCount();
    PHX::MDField<double,Cell,NODE,Dim> node_coordinates
        = af.buildStaticArray<double,Cell,NODE,Dim>("nc",num_cells, num_nodes, base_cell_dimension);

    // Set up node coordinates.  Here we assume the following
    // ordering.  This needs to be consistent with shards topology,
    // otherwise we will get negative determinates

    // 3(0,1)---2(1,1)
    //   |    0  |
    //   |       |
    // 0(0,0)---1(1,0)

    typedef panzer::ArrayTraits<double,PHX::MDField<double> >::size_type size_type;
    const size_type x = 0;
    const size_type y = 1;
    auto node_coordinates_k = node_coordinates.get_view();
    Kokkos::parallel_for("initialize node coords",node_coordinates.extent(0),
                         KOKKOS_LAMBDA (const int cell) {
      node_coordinates_k(cell,0,x) = 0.0;
      node_coordinates_k(cell,0,y) = 0.0;
      node_coordinates_k(cell,1,x) = 1.0;
      node_coordinates_k(cell,1,y) = 0.0;
      node_coordinates_k(cell,2,x) = 1.0;
      node_coordinates_k(cell,2,y) = 1.0;
      node_coordinates_k(cell,3,x) = 0.0;
      node_coordinates_k(cell,3,y) = 1.0;
    });


    panzer::IntegrationValues2<double> int_values_vol("prefix_");
    int_values_vol.setup(int_rule_vol, node_coordinates);

    panzer::IntegrationValues2<double> int_values_side("prefix_");
    int_values_side.setup(int_rule_side, node_coordinates);

    TEST_EQUALITY(int_values_vol.getCubaturePoints().extent(1), 4);
    TEST_EQUALITY(int_values_side.getCubaturePoints().extent(1), 4);
    TEST_EQUALITY(int_values_side.getWeightedNormals().extent(1), 4);
    double realspace_x_coord_1 = 0.25;
    double realspace_y_coord_1 = 0.25;
    auto tmp_coords = int_values_vol.getCubaturePoints();
    auto tmp_coords_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),tmp_coords.get_view());
    TEST_FLOATING_EQUALITY(tmp_coords_host(0,0,0),
                           realspace_x_coord_1, 1.0e-8);
    TEST_FLOATING_EQUALITY(tmp_coords_host(0,0,1),
                           realspace_y_coord_1, 1.0e-8);
    double realspace_x_coord_2 = 0.5;
    double realspace_y_coord_2 = 0.25;
    auto tmp_side_coords = int_values_side.getCubaturePoints();
    auto tmp_side_coords_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),tmp_side_coords.get_view());
    TEST_FLOATING_EQUALITY(tmp_side_coords_host(0,0,0),
                           realspace_x_coord_2, 1.0e-8);
    TEST_FLOATING_EQUALITY(tmp_side_coords_host(0,0,1),
                           realspace_y_coord_2, 1.0e-8);

  }

  TEUCHOS_UNIT_TEST(integration_values, control_volume_boundary)
  {
    Teuchos::RCP<shards::CellTopology> topo =
       Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData< shards::Quadrilateral<4> >()));

    const int num_cells = 2;
    const int base_cell_dimension = 2;
    const int cell_side = 1;
    const panzer::CellData cell_data(num_cells,cell_side,topo);

    std::string cv_type = "boundary";
    RCP<IntegrationRule> int_rule_bc =
      rcp(new IntegrationRule(cell_data, cv_type));

    panzer::MDFieldArrayFactory af("prefix_",true);

    const int num_nodes = int_rule_bc->topology->getNodeCount();
    PHX::MDField<double,Cell,NODE,Dim> node_coordinates
        = af.buildStaticArray<double,Cell,NODE,Dim>("nc",num_cells, num_nodes, base_cell_dimension);

    // Set up node coordinates.  Here we assume the following
    // ordering.  This needs to be consistent with shards topology,
    // otherwise we will get negative determinates

    // 3(0,1)---2(1,1)
    //   |    0  |
    //   |       |
    // 0(0,0)---1(1,0)

    typedef panzer::ArrayTraits<double,PHX::MDField<double> >::size_type size_type;
    const size_type x = 0;
    const size_type y = 1;
    auto node_coordinates_k = node_coordinates.get_view();
    Kokkos::parallel_for("initialize node coords",node_coordinates.extent(0),
                         KOKKOS_LAMBDA (const int cell) {
      node_coordinates_k(cell,0,x) = 0.0;
      node_coordinates_k(cell,0,y) = 0.0;
      node_coordinates_k(cell,1,x) = 1.0;
      node_coordinates_k(cell,1,y) = 0.0;
      node_coordinates_k(cell,2,x) = 1.0;
      node_coordinates_k(cell,2,y) = 1.0;
      node_coordinates_k(cell,3,x) = 0.0;
      node_coordinates_k(cell,3,y) = 1.0;
    });


    panzer::IntegrationValues2<double> int_values_bc("prefix_");
    int_values_bc.setup(int_rule_bc, node_coordinates);

    auto ip_coords_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),int_values_bc.getCubaturePoints().get_view());

    out << "Points:\n";
    for(int i=0; i<ip_coords_host.extent_int(0); ++i){
      for(int j=0; j<ip_coords_host.extent_int(1); ++j){
        out << "("<<i<<", "<<j<<"): ";
        for(int k=0; k<ip_coords_host.extent_int(2); ++k)
          out << ip_coords_host(i,j,k) << " ";
        out << "\n";
      }
    }

    TEST_EQUALITY(int_values_bc.getCubaturePoints().extent(1), 2);
    double realspace_x_coord_1 = 1.0;
    double realspace_y_coord_1 = 0.25;
    TEST_FLOATING_EQUALITY(ip_coords_host(0,0,0),
                           realspace_x_coord_1, 1.0e-8);
    TEST_FLOATING_EQUALITY(ip_coords_host(0,0,1),
                           realspace_y_coord_1, 1.0e-8);

  }

  TEUCHOS_UNIT_TEST(integration_values, control_volume_release_rebuild)
  {
    Teuchos::RCP<shards::CellTopology> topo =
       Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData< shards::Quadrilateral<4> >()));

    const int num_cells = 2;
    const panzer::CellData cell_data(num_cells,topo);

    std::string cv_type = "volume";
    RCP<IntegrationRule> int_rule_vol =
      rcp(new IntegrationRule(cell_data, cv_type));

    cv_type = "side";
    RCP<IntegrationRule> int_rule_side =
      rcp(new IntegrationRule(cell_data, cv_type));

    const int num_nodes = int_rule_vol->topology->getNodeCount();
    auto node_coordinates = buildUnitQuadNodeCoordinates(num_cells,num_nodes);

    panzer::IntegrationValues2<double> int_values_vol("prefix_");
    int_values_vol.setup(int_rule_vol, node_coordinates);

    auto cached_vol_points = int_values_vol.getCubaturePoints(true);
    auto cached_vol_ref_points = int_values_vol.getCubaturePointsRef(true);
    TEST_EQUALITY(cached_vol_points.extent(0), num_cells);
    TEST_EQUALITY(cached_vol_ref_points.extent(0), num_cells);
    auto cached_vol_points_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),cached_vol_points.get_view());
    auto cached_vol_ref_points_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),cached_vol_ref_points.get_view());

    int_values_vol.releaseCachedData();
    TEST_EQUALITY(int_values_vol.ip_coordinates.size(), 0);
    TEST_EQUALITY(int_values_vol.ref_ip_coordinates.size(), 0);
    TEST_EQUALITY(int_values_vol.node_coordinates.extent(0), num_cells);

    TEST_NOTHROW(int_values_vol.getCubaturePoints(true));
    TEST_NOTHROW(int_values_vol.getCubaturePointsRef(true));
    auto rebuilt_vol_points = int_values_vol.getCubaturePoints(true);
    auto rebuilt_vol_ref_points = int_values_vol.getCubaturePointsRef(true);
    TEST_EQUALITY(rebuilt_vol_points.extent(0), num_cells);
    TEST_EQUALITY(rebuilt_vol_points.extent_int(1), int_rule_vol->num_points);
    TEST_EQUALITY(rebuilt_vol_ref_points.extent(0), num_cells);
    TEST_EQUALITY(rebuilt_vol_ref_points.extent_int(1), int_rule_vol->num_points);
    auto rebuilt_vol_points_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rebuilt_vol_points.get_view());
    auto rebuilt_vol_ref_points_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rebuilt_vol_ref_points.get_view());
    testFloatingEqualityRank3(cached_vol_points_host, rebuilt_vol_points_host, 1.0e-14, out, success);
    testFloatingEqualityRank3(cached_vol_ref_points_host, rebuilt_vol_ref_points_host, 1.0e-14, out, success);

    panzer::IntegrationValues2<double> int_values_side("prefix_");
    int_values_side.setup(int_rule_side, node_coordinates);

    TEST_NOTHROW(int_values_side.getCubaturePoints(true));
    TEST_NOTHROW(int_values_side.getCubaturePointsRef(true));
    TEST_NOTHROW(int_values_side.getWeightedNormals(true));
    auto cached_side_points = int_values_side.getCubaturePoints(true);
    auto cached_side_ref_points = int_values_side.getCubaturePointsRef(true);
    auto cached_side_weighted_normals = int_values_side.getWeightedNormals(true);
    auto cached_side_points_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),cached_side_points.get_view());
    auto cached_side_ref_points_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),cached_side_ref_points.get_view());
    auto cached_side_weighted_normals_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),cached_side_weighted_normals.get_view());

    int_values_side.releaseCachedData();
    TEST_EQUALITY(int_values_side.ip_coordinates.size(), 0);
    TEST_EQUALITY(int_values_side.ref_ip_coordinates.size(), 0);
    TEST_EQUALITY(int_values_side.weighted_normals.size(), 0);
    TEST_EQUALITY(int_values_side.node_coordinates.extent(0), num_cells);

    TEST_NOTHROW(int_values_side.getCubaturePoints(true));
    TEST_NOTHROW(int_values_side.getCubaturePointsRef(true));
    TEST_NOTHROW(int_values_side.getWeightedNormals(true));
    auto rebuilt_side_points = int_values_side.getCubaturePoints(true);
    auto rebuilt_side_ref_points = int_values_side.getCubaturePointsRef(true);
    auto rebuilt_side_weighted_normals = int_values_side.getWeightedNormals(true);
    TEST_EQUALITY(rebuilt_side_points.extent(0), num_cells);
    TEST_EQUALITY(rebuilt_side_points.extent_int(1), int_rule_side->num_points);
    TEST_EQUALITY(rebuilt_side_ref_points.extent(0), num_cells);
    TEST_EQUALITY(rebuilt_side_ref_points.extent_int(1), int_rule_side->num_points);
    auto rebuilt_side_points_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rebuilt_side_points.get_view());
    auto rebuilt_side_ref_points_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rebuilt_side_ref_points.get_view());
    auto rebuilt_side_weighted_normals_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rebuilt_side_weighted_normals.get_view());
    testFloatingEqualityRank3(cached_side_points_host, rebuilt_side_points_host, 1.0e-14, out, success);
    testFloatingEqualityRank3(cached_side_ref_points_host, rebuilt_side_ref_points_host, 1.0e-14, out, success);
    testFloatingEqualityRank3(cached_side_weighted_normals_host, rebuilt_side_weighted_normals_host, 1.0e-14, out, success);
  }

  TEUCHOS_UNIT_TEST(integration_values, surface_quadrature){

    /*

      We are going to apply a periodic boundary condition set at an angle to make sure we can align points across the boundary.
      The mesh takes the form:

      0-----1-----2
       \    |    /
        \   |   /
         3--4--5

      Cell 0: 0,3,4,1
      Cell 1: 1,4,5,2

      Face connectivity will be:
      face ID, Cell 0, Cell 1, (Node 0, Node 1)
      0:  0,  1 (0,3) = (5,2)
      1:  0, -1 (3,4)
      2:  0,  1 (4,1) = (1,4)
      3:  0, -1 (1,0)
      4:  1, -1 (4,5)
      5:  1, -1 (2,1)

     */

    // First we build the mesh
    panzer::LocalMeshPartition mesh;
    {
      mesh.num_owned_cells = 2;
      mesh.num_ghstd_cells = 0;
      mesh.num_virtual_cells = 0;
      mesh.cell_topology = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData< shards::Quadrilateral<4> >()));
      mesh.local_cells = PHX::View<panzer::LocalOrdinal*>("local_cells",2);
      auto local_cells_host = Kokkos::create_mirror_view(mesh.local_cells);
      local_cells_host(0) = 0;
      local_cells_host(1) = 1;
      Kokkos::deep_copy(mesh.local_cells,local_cells_host);
      mesh.cell_nodes = PHX::View<double***>("cell_nodes",2,4,2);

      PHX::View<double**> coordinates("coordinates",6,2);
      auto coordinates_host = Kokkos::create_mirror_view(coordinates);
      coordinates_host(0,0) = 1.; coordinates_host(0,1) = 3.;
      coordinates_host(1,0) = 3.; coordinates_host(1,1) = 3.;
      coordinates_host(2,0) = 5.; coordinates_host(2,1) = 3.;
      coordinates_host(3,0) = 2.; coordinates_host(3,1) = 1.;
      coordinates_host(4,0) = 3.; coordinates_host(4,1) = 1.;
      coordinates_host(5,0) = 4.; coordinates_host(5,1) = 1.;
      Kokkos::deep_copy(coordinates,coordinates_host);

      auto cell_nodes_host = Kokkos::create_mirror_view(mesh.cell_nodes);
#define SET_NC(cell,node,vertex) cell_nodes_host(cell,node,0) = coordinates_host(vertex,0); cell_nodes_host(cell,node,1) = coordinates_host(vertex,1);
      SET_NC(0,0, 0);
      SET_NC(0,1, 3);
      SET_NC(0,2, 4);
      SET_NC(0,3, 1);
      SET_NC(1,0, 1);
      SET_NC(1,1, 4);
      SET_NC(1,2, 5);
      SET_NC(1,3, 2);
#undef SET_NC
      Kokkos::deep_copy(mesh.cell_nodes,cell_nodes_host);

      mesh.face_to_cells = PHX::View<panzer::LocalOrdinal*[2]>("face_to_cells",6);
      auto face_to_cells_host = Kokkos::create_mirror_view(mesh.face_to_cells);
      face_to_cells_host(0,0) = 0; face_to_cells_host(0,1) =  1;
      face_to_cells_host(1,0) = 0; face_to_cells_host(1,1) = -1;
      face_to_cells_host(2,0) = 0; face_to_cells_host(2,1) =  1;
      face_to_cells_host(3,0) = 0; face_to_cells_host(3,1) = -1;
      face_to_cells_host(4,0) = 1; face_to_cells_host(4,1) = -1;
      face_to_cells_host(5,0) = 1; face_to_cells_host(5,1) = -1;
      Kokkos::deep_copy(mesh.face_to_cells,face_to_cells_host);

      mesh.face_to_lidx = PHX::View<panzer::LocalOrdinal*[2]>("face_to_lidx",6);
      auto face_to_lidx_host = Kokkos::create_mirror_view(mesh.face_to_lidx);
      face_to_lidx_host(0,0) = 0; face_to_lidx_host(0,1) =  2;
      face_to_lidx_host(1,0) = 1; face_to_lidx_host(1,1) = -1;
      face_to_lidx_host(2,0) = 2; face_to_lidx_host(2,1) =  0;
      face_to_lidx_host(3,0) = 3; face_to_lidx_host(3,1) = -1;
      face_to_lidx_host(4,0) = 1; face_to_lidx_host(4,1) = -1;
      face_to_lidx_host(5,0) = 3; face_to_lidx_host(5,1) = -1;
      Kokkos::deep_copy(mesh.face_to_lidx,face_to_lidx_host);

      mesh.cell_to_faces = PHX::View<panzer::LocalOrdinal**>("cell_to_faces",2,4);
      auto cell_to_faces_host = Kokkos::create_mirror_view(mesh.cell_to_faces);
      cell_to_faces_host(0,0) = 0; cell_to_faces_host(0,1) = 1; cell_to_faces_host(0,2) = 2; cell_to_faces_host(0,3) = 3;
      cell_to_faces_host(1,0) = 2; cell_to_faces_host(1,1) = 4; cell_to_faces_host(1,2) = 0; cell_to_faces_host(1,3) = 5;
      Kokkos::deep_copy(mesh.cell_to_faces,cell_to_faces_host);
    }

    const auto id = panzer::IntegrationDescriptor(2, panzer::IntegrationDescriptor::SURFACE);
    auto int_rule = Teuchos::rcp(new IntegrationRule(id, mesh.cell_topology, 2, 6));

    // Fill node coordinates
    panzer::MDFieldArrayFactory af("prefix_",true);
    auto node_coordinates = af.buildStaticArray<double,Cell,NODE,Dim>("node_coordinates",2,4,2);
    {
      auto node_coordinates_host = Kokkos::create_mirror_view(node_coordinates.get_static_view());
      auto cell_nodes_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),mesh.cell_nodes);
      for(int i=0; i<mesh.cell_nodes.extent_int(0); ++i)
        for(int j=0; j<mesh.cell_nodes.extent_int(1); ++j)
          for(int k=0; k<mesh.cell_nodes.extent_int(2); ++k)
            node_coordinates_host(i,j,k) = cell_nodes_host(i,j,k);
      Kokkos::deep_copy(node_coordinates.get_static_view(),node_coordinates_host);
    }

    panzer::IntegrationValues2<double> int_values("prefix_");
    int_values.setup(int_rule, node_coordinates);

    // Make sure the integration values fail to construct without the subcell connectivity
    TEST_THROW(int_values.evaluateValues(node_coordinates), std::logic_error);

    // Build the subcell connectivity
    auto connectivity = Teuchos::rcp(new panzer::FaceConnectivity);
    connectivity->setup(mesh);

    TEST_THROW(int_values.setupPermutations(connectivity,-1), std::logic_error);

    // Now we try again
    int_values.evaluateValues(node_coordinates,-1,connectivity,0);

    // This test will focus on normals and points
    const auto normals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),int_values.getSurfaceNormals().get_static_view());
    const auto rotation_matrices = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),int_values.getSurfaceRotationMatrices().get_static_view());
    const auto points = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),int_values.getCubaturePoints().get_static_view());
    const auto ref_points = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),int_values.getCubaturePointsRef().get_static_view());
    const int num_points_per_cell = points.extent_int(1);
    const int num_points_per_face = num_points_per_cell / 4;

    for(int cell=0; cell<2; ++cell){
      out << "CELL " << cell << ":\n";
      for(int face=0; face<4; ++face){
        out << "\tFACE " << face << ":\n";
        for(int face_point=0; face_point<num_points_per_face; ++face_point){
          const int point = face*num_points_per_face + face_point;
          out << "\t\tPOINT  "<<face_point<<": ("<<points(cell,point,0)<<", "<<points(cell,point,1)<<")\n";
          out << "\t\tNORMAL "<<face_point<<": ("<<normals(cell,point,0)<<", "<<normals(cell,point,1)<<")\n";
        }
      }
    }

    // We need to make sure that face 0 and 2 have aligned quadrature points

    const double tolerance = 1.e-14;

    // Face 0
    {
      const int cell_0 = connectivity->cellForSubcellHost(0,0);
      const int cell_1 = connectivity->cellForSubcellHost(0,1);

      const int lidx_0 = connectivity->localSubcellForSubcellHost(0,0);
      const int lidx_1 = connectivity->localSubcellForSubcellHost(0,1);

      TEST_EQUALITY(cell_0,0);
      TEST_EQUALITY(cell_1,1);

      TEST_EQUALITY(lidx_0,0);
      TEST_EQUALITY(lidx_1,2);

      const double sqrt5 = std::sqrt(5.);
      const double normal_0[2] = {-2./sqrt5,-1./sqrt5};
      const double normal_1[2] = { 2./sqrt5,-1./sqrt5};

      // Note that y will be equal, but x will be different
      out << "Comparing cell_0 "<<cell_0<<" to cell_1 "<<cell_1<<"\n";

      for(int face_point=0; face_point<num_points_per_face; ++face_point){
        const int point_0 = lidx_0*num_points_per_face+face_point;
        const int point_1 = lidx_1*num_points_per_face+face_point;

        out << "Comparing point_0 "<<point_0<<" to point_1 "<<point_1<<"\n";

        TEST_ASSERT(std::fabs(points(cell_0,point_0,0) - points(cell_1,point_1,0)) > 0.5);
        TEST_FLOATING_EQUALITY(points(cell_0,point_0,1), points(cell_1,point_1,1), tolerance);

        TEST_FLOATING_EQUALITY(normals(cell_0,point_0,0), normal_0[0], tolerance);
        TEST_FLOATING_EQUALITY(normals(cell_0,point_0,1), normal_0[1], tolerance);

        TEST_FLOATING_EQUALITY(normals(cell_1,point_1,0), normal_1[0], tolerance);
        TEST_FLOATING_EQUALITY(normals(cell_1,point_1,1), normal_1[1], tolerance);

      }
    }

    // Face 2
    {
      const int cell_0 = connectivity->cellForSubcellHost(2,0);
      const int cell_1 = connectivity->cellForSubcellHost(2,1);

      const int lidx_0 = connectivity->localSubcellForSubcellHost(2,0);
      const int lidx_1 = connectivity->localSubcellForSubcellHost(2,1);

      TEST_EQUALITY(cell_0,0);
      TEST_EQUALITY(cell_1,1);

      TEST_EQUALITY(lidx_0,2);
      TEST_EQUALITY(lidx_1,0);

      const double normal_0[2] = { 1.,0.};
      const double normal_1[2] = {-1.,0.};

      out << "Comparing cell_0 "<<cell_0<<" to cell_1 "<<cell_1<<"\n";

      for(int face_point=0; face_point<num_points_per_face; ++face_point){
        const int point_0 = lidx_0*num_points_per_face+face_point;
        const int point_1 = lidx_1*num_points_per_face+face_point;

        out << "Comparing point_0 "<<point_0<<" to point_1 "<<point_1<<"\n";

        TEST_FLOATING_EQUALITY(points(cell_0,point_0,0), points(cell_1,point_1,0), tolerance);
        TEST_FLOATING_EQUALITY(points(cell_0,point_0,1), points(cell_1,point_1,1), tolerance);

        TEST_FLOATING_EQUALITY(normals(cell_0,point_0,0), normal_0[0], tolerance);
        TEST_FLOATING_EQUALITY(normals(cell_0,point_0,1), normal_0[1], tolerance);

        TEST_FLOATING_EQUALITY(normals(cell_1,point_1,0), normal_1[0], tolerance);
        TEST_FLOATING_EQUALITY(normals(cell_1,point_1,1), normal_1[1], tolerance);

      }
    }

    int_values.releaseCachedData();
    TEST_EQUALITY(int_values.ip_coordinates.size(), 0);
    TEST_EQUALITY(int_values.ref_ip_coordinates.size(), 0);
    TEST_EQUALITY(int_values.surface_normals.size(), 0);
    TEST_EQUALITY(int_values.surface_rotation_matrices.size(), 0);
    TEST_EQUALITY(int_values.node_coordinates.extent(0), 2);

    TEST_NOTHROW(int_values.getCubaturePoints(true));
    TEST_NOTHROW(int_values.getCubaturePointsRef(true));
    TEST_NOTHROW(int_values.getSurfaceNormals(true));
    TEST_NOTHROW(int_values.getSurfaceRotationMatrices(true));

    auto rebuilt_points = int_values.getCubaturePoints(true);
    auto rebuilt_ref_points = int_values.getCubaturePointsRef(true);
    auto rebuilt_normals = int_values.getSurfaceNormals(true);
    auto rebuilt_rotation_matrices = int_values.getSurfaceRotationMatrices(true);
    TEST_EQUALITY(rebuilt_points.extent(0), 2);
    TEST_EQUALITY(rebuilt_points.extent_int(1), int_rule->num_points);
    TEST_EQUALITY(rebuilt_ref_points.extent(0), 2);
    TEST_EQUALITY(rebuilt_ref_points.extent_int(1), int_rule->num_points);
    const auto rebuilt_points_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rebuilt_points.get_static_view());
    const auto rebuilt_ref_points_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rebuilt_ref_points.get_static_view());
    const auto rebuilt_normals_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rebuilt_normals.get_static_view());
    const auto rebuilt_rotation_matrices_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),rebuilt_rotation_matrices.get_static_view());
    testFloatingEqualityRank3(points, rebuilt_points_host, 1.0e-14, out, success);
    testFloatingEqualityRank3(ref_points, rebuilt_ref_points_host, 1.0e-14, out, success);
    testFloatingEqualityRank3(normals, rebuilt_normals_host, 1.0e-14, out, success);
    testFloatingEqualityRank4(rotation_matrices, rebuilt_rotation_matrices_host, 1.0e-14, out, success);
  }
}
