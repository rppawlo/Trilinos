// @HEADER
// *****************************************************************************
//                           Sacado Package
//
// Copyright 2006 NTESS and the Sacado contributors.
// SPDX-License-Identifier: LGPL-2.1-or-later
// *****************************************************************************
// @HEADER

#ifndef SACADO_FAD_EXP_EXPRASSIGN_HPP
#define SACADO_FAD_EXP_EXPRASSIGN_HPP

#include <Kokkos_Abort.hpp>

namespace Sacado {

  namespace Fad {
  namespace Exp {

#ifndef SACADO_FAD_DERIV_LOOP
#if defined(SACADO_VIEW_CUDA_HIERARCHICAL_DFAD) && !defined(SACADO_DISABLE_CUDA_IN_KOKKOS) && ( defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) )
#define SACADO_FAD_DERIV_LOOP(I,SZ) for (int I=threadIdx.x; I<SZ; I+=blockDim.x)
#elif  defined(SACADO_VIEW_CUDA_HIERARCHICAL_DFAD) && !defined(SACADO_DISABLE_CUDA_IN_KOKKOS) && defined(__SYCL_DEVICE_ONLY__)
#define SACADO_FAD_DERIV_LOOP(I,SZ) for (int I=sycl::ext::oneapi::experimental::this_nd_item<2>().get_local_id(0); I<SZ; I+=sycl::ext::oneapi::experimental::this_nd_item<2>().get_local_range(0))    
#else
#define SACADO_FAD_DERIV_LOOP(I,SZ) for (int I=0; I<SZ; ++I)
#endif
#endif

#ifndef SACADO_FAD_THREAD_SINGLE
#if (defined(SACADO_VIEW_CUDA_HIERARCHICAL) || defined(SACADO_VIEW_CUDA_HIERARCHICAL_DFAD)) && !defined(SACADO_DISABLE_CUDA_IN_KOKKOS) && ( defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) )
#define SACADO_FAD_THREAD_SINGLE if (threadIdx.x == 0)
#elif (defined(SACADO_VIEW_CUDA_HIERARCHICAL) || defined(SACADO_VIEW_CUDA_HIERARCHICAL_DFAD)) && !defined(SACADO_DISABLE_CUDA_IN_KOKKOS) && defined(__SYCL_DEVICE_ONLY__)
#define SACADO_FAD_THREAD_SINGLE if (sycl::ext::oneapi::experimental::this_nd_item<2>().get_local_id(0) == 0)
#else
#define SACADO_FAD_THREAD_SINGLE /* */
#endif
#endif

    //! Class that implements various forms of expression assignments
    /*!
     * GeneralFad uses this class to implement =, +=, -=, *=, and /= for
     * GeneralFad and expression right-hand-sides.  This design allows
     * partial specializations of this class to be easily written without
     * having to specialize GeneralFad itself.  For examples, specializations
     * for various derivative storage schemes may be desired.
     */
    template <typename DstType, typename Enabled = void>
    class ExprAssign {
    public:

      //! Typename of values
      typedef typename DstType::value_type value_type;

      //! Implementation of dst = x
      template <typename SrcType>
      SACADO_INLINE_FUNCTION
      static void assign_equal(DstType& dst, const SrcType& x)
      {
        const int xsz = x.size();

        if (xsz != dst.size())
          dst.resizeAndZero(xsz);

        const int sz = dst.size();

        // For ViewStorage, the resize above may not in fact resize the
        // derivative array, so it is possible that sz != xsz at this point.
        // The only valid use case here is sz > xsz == 0, so we use sz in the
        // assignment below

        if (sz) {
          if (x.hasFastAccess()) {
            SACADO_FAD_DERIV_LOOP(i,sz)
              dst.fastAccessDx(i) = x.fastAccessDx(i);
          }
          else
            SACADO_FAD_DERIV_LOOP(i,sz)
              dst.fastAccessDx(i) = x.dx(i);
        }

        dst.val() = x.val();
      }

      //! Implementation of dst += x
      template <typename SrcType>
      SACADO_INLINE_FUNCTION
      static void assign_plus_equal(DstType& dst, const SrcType& x)
      {
        const int xsz = x.size(), sz = dst.size();

#if defined(SACADO_DEBUG)
        if ((xsz != sz) && (xsz != 0) && (sz != 0))
	  Kokkos::abort("Fad Error:  Attempt to assign with incompatible sizes");
#endif
	
        if (xsz) {
          if (sz) {
            if (x.hasFastAccess())
              SACADO_FAD_DERIV_LOOP(i,sz)
                dst.fastAccessDx(i) += x.fastAccessDx(i);
            else
              for (int i=0; i<sz; ++i)
                dst.fastAccessDx(i) += x.dx(i);
          }
          else {
            dst.resizeAndZero(xsz);
            if (x.hasFastAccess())
              SACADO_FAD_DERIV_LOOP(i,xsz)
                dst.fastAccessDx(i) = x.fastAccessDx(i);
            else
              SACADO_FAD_DERIV_LOOP(i,xsz)
                dst.fastAccessDx(i) = x.dx(i);
          }
        }

        dst.val() += x.val();
      }

      //! Implementation of dst -= x
      template <typename SrcType>
      SACADO_INLINE_FUNCTION
      static void assign_minus_equal(DstType& dst, const SrcType& x)
      {
        const int xsz = x.size(), sz = dst.size();

#if defined(SACADO_DEBUG)
        if ((xsz != sz) && (xsz != 0) && (sz != 0))
	  Kokkos::abort("Fad Error:  Attempt to assign with incompatible sizes");
#endif

        if (xsz) {
          if (sz) {
            if (x.hasFastAccess())
              SACADO_FAD_DERIV_LOOP(i,sz)
                dst.fastAccessDx(i) -= x.fastAccessDx(i);
            else
              SACADO_FAD_DERIV_LOOP(i,sz)
                dst.fastAccessDx(i) -= x.dx(i);
          }
          else {
            dst.resizeAndZero(xsz);
            if (x.hasFastAccess())
              SACADO_FAD_DERIV_LOOP(i,xsz)
                dst.fastAccessDx(i) = -x.fastAccessDx(i);
            else
              SACADO_FAD_DERIV_LOOP(i,xsz)
                dst.fastAccessDx(i) = -x.dx(i);
          }
        }

        dst.val() -= x.val();
      }

      //! Implementation of dst *= x
      template <typename SrcType>
      SACADO_INLINE_FUNCTION
      static void assign_times_equal(DstType& dst, const SrcType& x)
      {
        const int xsz = x.size(), sz = dst.size();
        const value_type xval = x.val();
        const value_type v = dst.val();

#if defined(SACADO_DEBUG)
        if ((xsz != sz) && (xsz != 0) && (sz != 0))
	  Kokkos::abort("Fad Error:  Attempt to assign with incompatible sizes");
#endif

        if (xsz) {
          if (sz) {
            if (x.hasFastAccess())
              SACADO_FAD_DERIV_LOOP(i,sz)
                dst.fastAccessDx(i) = v*x.fastAccessDx(i) + dst.fastAccessDx(i)*xval;
            else
              SACADO_FAD_DERIV_LOOP(i,sz)
                dst.fastAccessDx(i) = v*x.dx(i) + dst.fastAccessDx(i)*xval;
          }
          else {
            dst.resizeAndZero(xsz);
            if (x.hasFastAccess())
              SACADO_FAD_DERIV_LOOP(i,xsz)
                dst.fastAccessDx(i) = v*x.fastAccessDx(i);
            else
              SACADO_FAD_DERIV_LOOP(i,xsz)
                dst.fastAccessDx(i) = v*x.dx(i);
          }
        }
        else {
          if (sz) {
            SACADO_FAD_DERIV_LOOP(i,sz)
              dst.fastAccessDx(i) *= xval;
          }
        }

        dst.val() *= xval;
      }

      //! Implementation of dst /= x
      template <typename SrcType>
      SACADO_INLINE_FUNCTION
      static void assign_divide_equal(DstType& dst, const SrcType& x)
      {
        const int xsz = x.size(), sz = dst.size();
        const value_type xval = x.val();
        const value_type v = dst.val();

#if defined(SACADO_DEBUG)
        if ((xsz != sz) && (xsz != 0) && (sz != 0))
	  Kokkos::abort("Fad Error:  Attempt to assign with incompatible sizes");
#endif

        if (xsz) {
          const value_type xval2 = xval*xval;
          if (sz) {
            if (x.hasFastAccess())
              SACADO_FAD_DERIV_LOOP(i,sz)
                dst.fastAccessDx(i) =
                ( dst.fastAccessDx(i)*xval - v*x.fastAccessDx(i) ) / xval2;
            else
              SACADO_FAD_DERIV_LOOP(i,sz)
                dst.fastAccessDx(i) =
                ( dst.fastAccessDx(i)*xval - v*x.dx(i) ) / xval2;
          }
          else {
            dst.resizeAndZero(xsz);
            if (x.hasFastAccess())
              SACADO_FAD_DERIV_LOOP(i,xsz)
                dst.fastAccessDx(i) = - v*x.fastAccessDx(i) / xval2;
            else
              SACADO_FAD_DERIV_LOOP(i,xsz)
                dst.fastAccessDx(i) = -v*x.dx(i) / xval2;
          }
        }
        else {
          if (sz) {
            SACADO_FAD_DERIV_LOOP(i,sz)
              dst.fastAccessDx(i) /= xval;
          }
        }

        dst.val() /= xval;
      }

    };

    //! Specialization of ExprAssign for statically sized storage types
    /*!
     * This simplifies the logic considerably in the static, fixed case,
     * making the job easier for the compiler to optimize the code.  In
     * this case, dst.size() always equals x.size().
     */
    template <typename DstType>
    class ExprAssign<DstType,
                     typename std::enable_if<Sacado::IsStaticallySized<DstType>::value>::type> {
    public:

      //! Typename of values
      typedef typename DstType::value_type value_type;

      //! Implementation of dst = x
      template <typename SrcType>
      SACADO_INLINE_FUNCTION
      static void assign_equal(DstType& dst, const SrcType& x)
      {
        const int sz = dst.size();
        SACADO_FAD_DERIV_LOOP(i,sz)
          dst.fastAccessDx(i) = x.fastAccessDx(i);
        dst.val() = x.val();
      }

      //! Implementation of dst += x
      template <typename SrcType>
      SACADO_INLINE_FUNCTION
      static void assign_plus_equal(DstType& dst, const SrcType& x)
      {





// 	auto item = sycl::ext::oneapi::experimental::this_nd_item<3>();
// 	sycl::ext::oneapi::experimental::printf("sycl_group_range[gridDim](%u,%u,%u), sycl_group[BlockIdx](%u,%u,%u), sycl_loca\
// l_range[blockDim](%u,%u,%u), sycl_local_id[threadIdx](%u,%u,%u)\n",
// 						item.get_group_range(0),item.get_group_range(1),item.get_group_range(2),
// 						item.get_group(0),item.get_group(1),item.get_group(2),
// 						item.get_local_range(0),item.get_local_range(1),item.get_local_range(2),
// 						item.get_local_id(0),item.get_local_id(1),item.get_local_id(2));
// 	Kokkos::printf("ROGER x=%i, sz=%i\n",int(x.size()),int(dst.size()));


	

	
        const int sz = dst.size();
        SACADO_FAD_DERIV_LOOP(i,sz)
          dst.fastAccessDx(i) += x.fastAccessDx(i);
        dst.val() += x.val();
      }

      //! Implementation of dst -= x
      template <typename SrcType>
      SACADO_INLINE_FUNCTION
      static void assign_minus_equal(DstType& dst, const SrcType& x)
      {
        const int sz = dst.size();
        SACADO_FAD_DERIV_LOOP(i,sz)
          dst.fastAccessDx(i) -= x.fastAccessDx(i);
        dst.val() -= x.val();
      }

      //! Implementation of dst *= x
      template <typename SrcType>
      SACADO_INLINE_FUNCTION
      static void assign_times_equal(DstType& dst, const SrcType& x)
      {
        const int sz = dst.size();
        const value_type xval = x.val();
        const value_type v = dst.val();
        SACADO_FAD_DERIV_LOOP(i,sz)
          dst.fastAccessDx(i) = v*x.fastAccessDx(i) + dst.fastAccessDx(i)*xval;
        dst.val() *= xval;
      }

      //! Implementation of dst /= x
      template <typename SrcType>
      SACADO_INLINE_FUNCTION
      static void assign_divide_equal(DstType& dst, const SrcType& x)
      {
        const int sz = dst.size();
        const value_type xval = x.val();
        const value_type xval2 = xval*xval;
        const value_type v = dst.val();
        SACADO_FAD_DERIV_LOOP(i,sz)
          dst.fastAccessDx(i) =
          ( dst.fastAccessDx(i)*xval - v*x.fastAccessDx(i) )/ xval2;
        dst.val() /= xval;
      }

    };

  } // namespace Exp
  } // namespace Fad

} // namespace Sacado

#endif // SACADO_FAD_EXP_EXPRASSIGN_HPP
