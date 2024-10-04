#include "Kokkos_Core.hpp"
#include "Phalanx_Print.hpp"
#include <type_traits>
#include <limits>

template<typename ExecSpace,typename Layout,bool PrintIndexing>
void runMappingQuery()
{
  {
    constexpr int num_cells = 4;
    constexpr int num_basis = 2;
    constexpr int num_qp = 3;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    if (num_cells > std::numeric_limits<int>::max())
      throw std::runtime_error("ERROR: num_cells must fit in an int!");
#endif
    
    Kokkos::View<double***,Layout,ExecSpace> a("a",num_cells,num_basis,num_qp);
    Kokkos::View<double***,Layout,ExecSpace> b("b",num_cells,num_basis,num_qp);
    Kokkos::View<double***,Layout,ExecSpace> c("c",num_cells,num_basis,num_qp);

    std::string is_default_layout = " (not default layout for exec space)";
    if (std::is_same_v<typename ExecSpace::array_layout,Layout>) {
      is_default_layout = " (is default layout for exec space)";
    }

    std::cout << "\nRange Policy: "
	      << PHX::print<ExecSpace>() << ", " << PHX::print<Layout>()
	      << is_default_layout
	      << std::endl;
    Kokkos::RangePolicy<ExecSpace> range_policy(0,num_cells);
    Kokkos::parallel_for("test roger",range_policy,KOKKOS_LAMBDA(const int cell) {
      for (int basis=0; basis < num_basis; ++basis) {
	for (int qp=0; qp < num_qp; ++qp) {
	  c(cell,basis,qp) = a(cell,basis,qp) * 2.0 * b(cell,basis,qp) + 1.0;
	  if constexpr (PrintIndexing) {
#if defined(KOKKOS_ENABLE_SYCL)
	    static_assert(SYCL_EXT_ONEAPI_FREE_FUNCTION_QUERIES);
	    auto item = sycl::ext::oneapi::experimental::this_nd_item<3>();
	    sycl::ext::oneapi::experimental::printf("kokkos_index(%i,%i,%i) = %u, sycl_group_range[gridDim](%u,%u,%u), sycl_group[BlockIdx](%u,%u,%u), sycl_local_range[blockDim](%u,%u,%u), sycl_local_id[threadIdx](%u,%u,%u)\n",
						    cell,basis,qp,a.impl_map().m_impl_offset(cell,basis,qp),
						    item.get_group_range(0),item.get_group_range(1),item.get_group_range(2),
						    item.get_group(0),item.get_group(1),item.get_group(2),
						    item.get_local_range(0),item.get_local_range(1),item.get_local_range(2),
						    item.get_local_id(0),item.get_local_id(1),item.get_local_id(2));
#endif
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
	    Kokkos::printf("kokkos_index(%i,%i,%i) = %i, gridDim(%u,%u,%u), blockIdx(%u,%u,%u), blockDim(%u,%u,%u), threadIdx(%u,%u,%u)\n",
                           cell,basis,qp,int(a.impl_map().m_impl_offset(cell,basis,qp)),
                           gridDim.x,gridDim.y,gridDim.z,
                           blockIdx.x,blockIdx.y,blockIdx.z,
                           blockDim.x,blockDim.y,blockDim.z,
                           threadIdx.x,threadIdx.y,threadIdx.z);
#endif
	  }
	}
      }
    });
    ExecSpace().fence();

    std::cout << "\nTeam Policy: "
	      << PHX::print<ExecSpace>() << ", " << PHX::print<Layout>()
	      << is_default_layout
	      << std::endl;
    Kokkos::TeamPolicy<ExecSpace> team_policy(num_cells,Kokkos::AUTO(),32);
    Kokkos::parallel_for("test roger",team_policy,KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<ExecSpace>::member_type& team) {
      const int cell = team.league_rank();
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,num_basis), [&] (const int basis) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,0,num_qp), [&] (const int qp) {
          c(cell,basis,qp) = a(cell,basis,qp) * 2.0 * b(cell,basis,qp) + 1.0;
	  if constexpr (PrintIndexing) {
#if defined(KOKKOS_ENABLE_SYCL)
	    static_assert(SYCL_EXT_ONEAPI_FREE_FUNCTION_QUERIES);
	    auto item = sycl::ext::oneapi::experimental::this_nd_item<3>();
	    sycl::ext::oneapi::experimental::printf("kokkos_index(%i,%i,%i) = %u, sycl_group_range[gridDim](%u,%u,%u), sycl_group[BlockIdx](%u,%u,%u), sycl_local_range[blockDim](%u,%u,%u), sycl_local_id[threadIdx](%u,%u,%u)\n",
						    cell,basis,qp,a.impl_map().m_impl_offset(cell,basis,qp),
						    item.get_group_range(0),item.get_group_range(1),item.get_group_range(2),
						    item.get_group(0),item.get_group(1),item.get_group(2),
						    item.get_local_range(0),item.get_local_range(1),item.get_local_range(2),
						    item.get_local_id(0),item.get_local_id(1),item.get_local_id(2));
#endif
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
	    Kokkos::printf("kokkos_index(%i,%i,%i) = %i, gridDim(%u,%u,%u), blockIdx(%u,%u,%u), blockDim(%u,%u,%u), threadIdx(%u,%u,%u)\n",
                           cell,basis,qp,int(a.impl_map().m_impl_offset(cell,basis,qp)),
                           gridDim.x,gridDim.y,gridDim.z,
                           blockIdx.x,blockIdx.y,blockIdx.z,
                           blockDim.x,blockDim.y,blockDim.z,
                           threadIdx.x,threadIdx.y,threadIdx.z);
#endif
	  }
        });
      });
    });
    ExecSpace().fence();

    std::cout << "\nMDRange Policy: "
	      << PHX::print<ExecSpace>() << ", " << PHX::print<Layout>()
	      << is_default_layout
	      << std::endl;
    Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<3>> mdr_policy({0,0,0},{num_cells,num_basis,num_qp});
    Kokkos::parallel_for("test roger",mdr_policy,KOKKOS_LAMBDA(const int cell, const int basis, const int qp) {
      c(cell,basis,qp) = a(cell,basis,qp) * 2.0 * b(cell,basis,qp) + 1.0;
      if constexpr (PrintIndexing) {
#if defined(KOKKOS_ENABLE_SYCL)
        static_assert(SYCL_EXT_ONEAPI_FREE_FUNCTION_QUERIES);
	auto item = sycl::ext::oneapi::experimental::this_nd_item<3>();
	sycl::ext::oneapi::experimental::printf("kokkos_index(%i,%i,%i) = %u, sycl_group_range[gridDim](%u,%u,%u), sycl_group[BlockIdx](%u,%u,%u), sycl_local_range[blockDim](%u,%u,%u), sycl_local_id[threadIdx](%u,%u,%u)\n",
						cell,basis,qp,a.impl_map().m_impl_offset(cell,basis,qp),
						item.get_group_range(0),item.get_group_range(1),item.get_group_range(2),
						item.get_group(0),item.get_group(1),item.get_group(2),
						item.get_local_range(0),item.get_local_range(1),item.get_local_range(2),
						item.get_local_id(0),item.get_local_id(1),item.get_local_id(2));
#endif
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
        Kokkos::printf("kokkos_index(%i,%i,%i) = %i, gridDim(%u,%u,%u), blockIdx(%u,%u,%u), blockDim(%u,%u,%u), threadIdx(%u,%u,%u)\n",
                       cell,basis,qp,int(a.impl_map().m_impl_offset(cell,basis,qp)),
                       gridDim.x,gridDim.y,gridDim.z,
                       blockIdx.x,blockIdx.y,blockIdx.z,
                       blockDim.x,blockDim.y,blockDim.z,
                       threadIdx.x,threadIdx.y,threadIdx.z);
#endif
      }
    });
    ExecSpace().fence();

  }

}

int main() {
  Kokkos::initialize();

  using ExecSpace = Kokkos::DefaultExecutionSpace;
  runMappingQuery<ExecSpace,ExecSpace::array_layout,true>();
  // runMappingQuery<ExecSpace,Kokkos::LayoutLeft,true>();
  // runMappingQuery<ExecSpace,Kokkos::LayoutRight,true>();

  Kokkos::finalize();
  return 0;
}
