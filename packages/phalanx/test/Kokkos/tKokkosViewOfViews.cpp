// @HEADER
// ************************************************************************
//
//        Phalanx: A Partial Differential Equation Field Evaluation
//       Kernel for Flexible Management of Complex Dependency Chains
//                    Copyright 2008 Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Roger Pawlowski (rppawlo@sandia.gov), Sandia
// National Laboratories.
//
// ************************************************************************
// @HEADER

#include "Teuchos_Assert.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Phalanx_KokkosDeviceTypes.hpp"
#include <Phalanx_any.hpp>
#include "Phalanx_Print.hpp"

#include "Sacado.hpp"
#include "Kokkos_View_Fad.hpp"
#include "Kokkos_DynRankView.hpp"
#include "Kokkos_DynRankView_Fad.hpp"
#include "KokkosSparse_CrsMatrix.hpp"


/** This is a demonstration/tutorial on how to create a View-of-Views
    on Device without UVM. Typically used in physics-based blocked
    nonlinear systems.
*/
namespace phalanx_test {

  TEUCHOS_UNIT_TEST(kokkos, ViewOfViewsNoUVM)
  {
    const int num_cells = 10;
    const int num_pts = 4;

    // Inner view is non-owning so we don't need to call dtors inner views on device.
    using inner_view_type = Kokkos::View<double**,PHX::Device,Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    // Outer view is created without initializing
    int num_inner_views = 3;
    auto outer = Kokkos::View<inner_view_type*,PHX::Device>(Kokkos::view_alloc("outer view",Kokkos::WithoutInitializing),num_inner_views);

    // Inner views are created somewhere in the application with
    // normal ownership semantics.
    { 
      Kokkos::View<double**,PHX::Device> a("a",num_cells,num_pts);
      // Kokkos::View<double**,PHX::Device> b("a",num_cells,num_pts);
      // Kokkos::View<double**,PHX::Device> c("a",num_cells,num_pts);
      // Kokkos::deep_copy(a,1.0);
      // Kokkos::deep_copy(a,2.0);
      // Kokkos::deep_copy(a,3.0);

      // Assign these to the device using placement new on device.
      Kokkos::parallel_for(Kokkos::RangePolicy<PHX::ExecSpace>(0,1),KOKKOS_LAMBDA (const int )
      {
        new (&outer(0)) inner_view_type (a);
      },"inner view assignment");

    }

    /*

    using Kokkos::Cuda;
    using Kokkos::CudaSpace;
    using Kokkos::CudaUVMSpace;
    using Kokkos::View;
    using Kokkos::view_alloc;
    using Kokkos::WithoutInitializing;
    
    using inner_view_type = View<double*, CudaSpace>;
    using outer_view_type = View<inner_view_type*, CudaUVMSpace>;
    
    const int numOuter = 5;
    const int numInner = 4;
    outer_view_type outer (view_alloc (std::string ("Outer"), WithoutInitializing), numOuter);

    // Create inner Views on host, outside of a parallel region, uninitialized
    for (int k = 0; k < numOuter; ++k) {
      const std::string label = std::string ("Inner ") + std::to_string (k);
      new (&outer[k]) inner_view_type (view_alloc (label, WithoutInitializing), numInner);
    }
    
    // Outer and inner views are now ready for use on device
    
    Kokkos::RangePolicy<Cuda, int> range (0, numOuter);
    Kokkos::parallel_for ("my kernel label", range, 
                          KOKKOS_LAMBDA (const int i) {  
                            for (int j = 0; j < numInner; ++j) {
                              device_outer[i][j] = 10.0 * double (i) + double (j);
                            }
                          }
                          });
  
  // Fence before deallocation on host, to make sure 
  // that the device kernel is done first.
  // Note the new fence syntax that requires an instance.
  // This will work with other CUDA streams, etc.
  Cuda ().fence ();
  
  // Destroy inner Views, again on host, outside of a parallel region.
  for (int k = 0; k < 5; ++k) {
    outer[k].~inner_view_type ();
  }
  
  // You're better off disposing of outer immediately.
  outer = outer_view_type ();

    */
    
  }
}
