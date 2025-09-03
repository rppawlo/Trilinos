// @HEADER
// *****************************************************************************
//                           Sacado Package
//
// Copyright 2006 NTESS and the Sacado contributors.
// SPDX-License-Identifier: LGPL-2.1-or-later
// *****************************************************************************
// @HEADER

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_UnitTestRepository.hpp"
#include "Teuchos_GlobalMPISession.hpp"

#include "Kokkos_Macros.hpp"

// ROGER REVISIT WHAT WE CAN ENABLE FOR DFAD
// TEMPORARY DISABLE FOR DEBUGGING
#define SACADO_TEST_DFAD 0
#include "Fad_KokkosTests.hpp"

// Instantiate tests for SYCL device.  We can only test DFad if UVM is enabled.
using Kokkos::Experimental::SYCL;
VIEW_FAD_TESTS_D( SYCL )

// Tests special size alignment for SFad on SYCL is correct
TEUCHOS_UNIT_TEST(Kokkos_View_Fad, SFadSYCLAligned)
{

  
  std::cout << "\nROGER vec_length=" << Kokkos::TeamPolicy<Kokkos::Experimental::SYCL>::vector_length_max();
  
  
  const int StaticDim = 64;
  const int Stride = 32;
  const int LocalDim = 2;
  typedef Sacado::Fad::SFad<double,StaticDim> FadType;
  typedef Kokkos::LayoutContiguous<Kokkos::LayoutLeft,Stride> Layout;
  typedef Kokkos::Experimental::SYCL Device;
  typedef Kokkos::View<FadType*,Layout,Device> ViewType;

  typedef typename ViewType::traits TraitsType;
  typedef Kokkos::Impl::ViewMapping< TraitsType , typename TraitsType::specialize > MappingType;
  const int view_static_dim = MappingType::FadStaticDimension;
  TEUCHOS_TEST_EQUALITY(view_static_dim, StaticDim, out, success);

  typedef typename Kokkos::ThreadLocalScalarType<ViewType>::type local_fad_type;
  const bool issfd = is_sfad<local_fad_type>::value;
  const int static_dim = Sacado::StaticSize<local_fad_type>::value;
  TEUCHOS_TEST_EQUALITY(issfd, true, out, success);
  TEUCHOS_TEST_EQUALITY(static_dim, LocalDim, out, success);

  const size_t num_rows = 11;
  const size_t fad_size = StaticDim;

  ViewType v("v", num_rows, fad_size+1);
  const size_t span = v.span();
  TEUCHOS_TEST_EQUALITY(span, num_rows*(StaticDim+1), out, success);
}

TEUCHOS_UNIT_TEST(Kokkos_View_Fad, SFadSYCLNotAligned)
{
  const int StaticDim = 50;
  const int Stride = 32;
  const int LocalDim = 0;
  typedef Sacado::Fad::SFad<double,StaticDim> FadType;
  typedef Kokkos::LayoutContiguous<Kokkos::LayoutLeft,Stride> Layout;
  typedef Kokkos::Experimental::SYCL Device;
  typedef Kokkos::View<FadType*,Layout,Device> ViewType;

  typedef typename ViewType::traits TraitsType;
  typedef Kokkos::Impl::ViewMapping< TraitsType , typename TraitsType::specialize > MappingType;
  const int view_static_dim = MappingType::FadStaticDimension;
  TEUCHOS_TEST_EQUALITY(view_static_dim, StaticDim, out, success);

  typedef typename Kokkos::ThreadLocalScalarType<ViewType>::type local_fad_type;
  const bool issfd = is_sfad<local_fad_type>::value;
  const int static_dim = Sacado::StaticSize<local_fad_type>::value;
  TEUCHOS_TEST_EQUALITY(issfd, false, out, success);
  TEUCHOS_TEST_EQUALITY(static_dim, LocalDim, out, success);

  const size_t num_rows = 11;
  const size_t fad_size = StaticDim;

  ViewType v("v", num_rows, fad_size+1);
  const size_t span = v.span();
  TEUCHOS_TEST_EQUALITY(span, num_rows*(StaticDim+1), out, success);
}

int main( int argc, char* argv[] ) {
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);

  Kokkos::InitializationSettings init_args;
  init_args.set_device_id(0);
  Kokkos::initialize( init_args );
  Kokkos::print_configuration(std::cout);

  int res = Teuchos::UnitTestRepository::runUnitTestsFromMain(argc, argv);

  Kokkos::finalize();

  return res;
}
