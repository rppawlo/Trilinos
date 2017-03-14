// @HEADER
// ************************************************************************
//
//                           Intrepid2 Package
//                 Copyright (2007) Sandia Corporation
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
// Questions? Contact Kyungjoo Kim  (kyukim@sandia.gov), or
//                    Mauro Perego  (mperego@sandia.gov)
//
// ************************************************************************
// @HEADER

/** \file   Intrepid_HGRAD_TRI_Cn_FEM_Def.hpp
    \brief  Definition file for FEM basis functions of degree n for H(grad) functions on TRI.
    \author Created by R. Kirby and P. Bochev and D. Ridzal.
            Kokkorized by Kyungjoo Kim
*/

#ifndef __INTREPID2_HGRAD_TRI_CN_FEM_DEF_HPP__
#define __INTREPID2_HGRAD_TRI_CN_FEM_DEF_HPP__

namespace Intrepid2 {

  // -------------------------------------------------------------------------------------
  
  namespace Impl {

    template<EOperator opType>
    template<typename outputViewType,
             typename inputViewType,
             typename workViewType,
             typename vinvViewType>
    KOKKOS_INLINE_FUNCTION
    void
    Basis_HGRAD_TRI_Cn_FEM::Serial<opType>::
    getValues(       outputViewType output,
               const inputViewType  input,
                     workViewType   work,
               const vinvViewType   vinv,
               const ordinal_type   operatorDn ) {
      ordinal_type opDn = operatorDn;

      const ordinal_type 
        card = vinv.dimension(0),
        npts = input.dimension(0);

      // compute order
      ordinal_type order = 0;
      for (ordinal_type p=0;p<Parameters::MaxOrder;++p) {
        if (card == (p+1)*(p+2)/2) { 
          order = p; 
          break;
        }
      }
      
      switch (opType) {
      case OPERATOR_VALUE: {
        Kokkos::DynRankView<typename workViewType::value_type,
            typename workViewType::memory_space> phis(work.data(), card, npts);
        
        Impl::Basis_HGRAD_TRI_Cn_FEM_ORTH::
          Serial<opType>::getValues(phis, input, order);
        
        for (ordinal_type i=0;i<card;++i)
          for (ordinal_type j=0;j<npts;++j) {
            output(i,j) = 0.0;
            for (ordinal_type k=0;k<card;++k)
              output(i,j) += vinv(k,i)*phis(k,j);
          }
        break;
      }
      case OPERATOR_GRAD:
      case OPERATOR_D1:
      case OPERATOR_D2:
      case OPERATOR_D3:
      case OPERATOR_D4:
      case OPERATOR_D5:
      case OPERATOR_D6:
      case OPERATOR_D7:
      case OPERATOR_D8:
      case OPERATOR_D9:
      case OPERATOR_D10:
        opDn = getOperatorOrder(opType);
      case OPERATOR_Dn: {
        const ordinal_type dkcard = opDn + 1;
        Kokkos::DynRankView<typename workViewType::value_type,
            typename workViewType::memory_space> phis(work.data(), card, npts, dkcard);

        Impl::Basis_HGRAD_TRI_Cn_FEM_ORTH::
          Serial<opType>::getValues(phis, input, order, opDn);
        
        for (ordinal_type i=0;i<card;++i)
          for (ordinal_type j=0;j<npts;++j)
            for (ordinal_type k=0;k<dkcard;++k) {
              output(i,j,k) = 0.0;
              for (ordinal_type l=0;l<card;++l)
                output(i,j,k) += vinv(l,i)*phis(l,j,k);
            }
        break;
      }
      case OPERATOR_CURL: { // only works in 2d. first component is -d/dy, second is d/dx
        const ordinal_type d2card = 2;
        Kokkos::DynRankView<typename workViewType::value_type,
            typename workViewType::memory_space> phis(work.data(), card, npts, d2card);

        Impl::Basis_HGRAD_TRI_Cn_FEM_ORTH::
          Serial<opType>::getValues(phis, input, order, 1);

        for (ordinal_type i=0;i<card;++i)
          for (ordinal_type j=0;j<npts;++j) {
            output(i,j,0) = 0.0;
            for (ordinal_type l=0;l<card;++l)
              output(i,j,0) += vinv(l,i)*phis(l,j,1);
            output(i,j,1) = 0.0;
            for (ordinal_type l=0;l<card;++l)
              output(i,j,1) -= vinv(l,i)*phis(l,j,0);
          }
        break;
      }
      default: {
        INTREPID2_TEST_FOR_ABORT( true,
                                  ">>> ERROR (Basis_HGRAD_TRI_Cn_FEM): Operator type not implemented");
      }
      }
    }

    template<typename SpT, ordinal_type numPtsPerEval,
             typename outputValueValueType, class ...outputValueProperties,
             typename inputPointValueType,  class ...inputPointProperties,
             typename vinvValueType,        class ...vinvProperties>
    void
    Basis_HGRAD_TRI_Cn_FEM::
    getValues(       Kokkos::DynRankView<outputValueValueType,outputValueProperties...> outputValues,
               const Kokkos::DynRankView<inputPointValueType, inputPointProperties...>  inputPoints,
               const Kokkos::DynRankView<vinvValueType,       vinvProperties...>        vinv,
               const EOperator operatorType ) {
      typedef          Kokkos::DynRankView<outputValueValueType,outputValueProperties...>         outputValueViewType;
      typedef          Kokkos::DynRankView<inputPointValueType, inputPointProperties...>          inputPointViewType;
      typedef          Kokkos::DynRankView<vinvValueType,       vinvProperties...>                vinvViewType;
      typedef typename ExecSpace<typename inputPointViewType::execution_space,SpT>::ExecSpaceType ExecSpaceType;
      
      // loopSize corresponds to cardinality
      const auto loopSizeTmp1 = (inputPoints.dimension(0)/numPtsPerEval);
      const auto loopSizeTmp2 = (inputPoints.dimension(0)%numPtsPerEval != 0);
      const auto loopSize = loopSizeTmp1 + loopSizeTmp2;
      Kokkos::RangePolicy<ExecSpaceType,Kokkos::Schedule<Kokkos::Static> > policy(0, loopSize);
      
      switch (operatorType) {
      case OPERATOR_VALUE: {
        typedef Functor<outputValueViewType,inputPointViewType,vinvViewType,
            OPERATOR_VALUE,numPtsPerEval> FunctorType;
        Kokkos::parallel_for( policy, FunctorType(outputValues, inputPoints, vinv) );
      break;
      }
      case OPERATOR_CURL: {
        typedef Functor<outputValueViewType,inputPointViewType,vinvViewType,
            OPERATOR_CURL,numPtsPerEval> FunctorType;
        Kokkos::parallel_for( policy, FunctorType(outputValues, inputPoints, vinv) );
        break;
      }
      case OPERATOR_GRAD:
      case OPERATOR_D1:
      case OPERATOR_D2:
      case OPERATOR_D3:
      case OPERATOR_D4:
      case OPERATOR_D5:
      case OPERATOR_D6:
      case OPERATOR_D7:
      case OPERATOR_D8:
      case OPERATOR_D9:
      case OPERATOR_D10: {
        typedef Functor<outputValueViewType,inputPointViewType,vinvViewType,
            OPERATOR_Dn,numPtsPerEval> FunctorType;
        Kokkos::parallel_for( policy, FunctorType(outputValues, inputPoints, vinv,
                                                  getOperatorOrder(operatorType)) );
        break;
      }
      default: {
        INTREPID2_TEST_FOR_EXCEPTION( true , std::invalid_argument,
                                      ">>> ERROR (Basis_HGRAD_TRI_Cn_FEM): Operator type not implemented" );
        break;
      }
      }
    }
  }
  
  // -------------------------------------------------------------------------------------
  template<typename SpT, typename OT, typename PT>
  Basis_HGRAD_TRI_Cn_FEM<SpT,OT,PT>::
  Basis_HGRAD_TRI_Cn_FEM( const ordinal_type order,
                          const EPointType   pointType ) {

    this->basisCardinality_  = (order+1)*(order+2)/2; // bigN
    this->basisDegree_       = order; // small n
    this->basisCellTopology_ = shards::CellTopology(shards::getCellTopologyData<shards::Triangle<3> >() );
    this->basisType_         = BASIS_FEM_FIAT;
    this->basisCoordinates_  = COORDINATES_CARTESIAN;

    const ordinal_type card = this->basisCardinality_;

    // points are computed in the host and will be copied
    Kokkos::DynRankView<typename scalarViewType::value_type,typename SpT::array_layout,Kokkos::HostSpace>
      dofCoords("Hgrad::Tri::Cn::dofCoords", card, 2);
    
    // construct lattice
    const ordinal_type offset = 0;
    PointTools::getLattice( odfCoords, 
                            this->basisCellTopology_,
                            order, offset,
                            pointType );
    
    this->dofCoords_ = Kokkos::create_mirror_view(typename SpT::memory_space(), dofCoords);
    Kokkos::deep_copy(this->dofCoords_, dofCoords);

    // form Vandermonde matrix.  Actually, this is the transpose of the VDM,
    // so we transpose on copy below.
    const ordinal_type lwork = card*card;
    Kokkos::DynRankView<PT,Kokkos::LayoutLeft,Kokkos::HostSpace>
      vmat("Hgrad::Tri::Cn::vmat", card, card), 
      work("Hgrad::Tri::Cn::work", lwork), 
      ipiv("Hgrad::Tri::Cn::ipiv", card);
  
    Impl::Basis_HGRAD_TRI_Cn_ORTH::
      getValues<Kokkos::HostSpace::execution_space,Parameters::MaxNumPtsPerBasisEval>
      (vmat, dofCoords, order, OPERATOR_VALUE);

    ordinal_type info = 0;
    Teuchos::LAPACK<ordinal_type,PT> lapack;
    
    lapack.GETRF(card, card,
                 vmat.data(), vmat.stride_1(),
                 (ordinal_type*)ipiv.data(),
                 &info);

    INTREPID2_TEST_FOR_EXCEPTION( info != 0,
                                  std::runtime_error ,
                                  ">>> ERROR: (Intrepid2::Basis_HGRAD_LINE_Cn_FEM) lapack.GETRF returns nonzero info." );

    lapack.GETRI(card,
                 vmat.data(), vmat.stride_1(),
                 (ordinal_type*)ipiv.data(),
                 work.data(), lwork,
                 &info);

    INTREPID2_TEST_FOR_EXCEPTION( info != 0,
                                  std::runtime_error ,
                                  ">>> ERROR: (Intrepid2::Basis_HGRAD_LINE_Cn_FEM) lapack.GETRI returns nonzero info." );

    // create host mirror
    Kokkos::DynRankView<typename scalarViewType::value_type,typename SpT::array_layout,Kokkos::HostSpace>
      vinv("Hgrad::Line::Cn::vinv", card, card);

    for (ordinal_type i=0;i<card;++i)
      for (ordinal_type j=0;j<card;++j)
        vinv(i,j) = vmat(j,i);

    this->vinv_ = Kokkos::create_mirror_view(typename SpT::memory_space(), vinv);
    Kokkos::deep_copy(this->vinv_ , vinv);

    // initialize tags
    {
      // Basis-dependent initializations
      const ordinal_type tagSize  = 4;        // size of DoF tag, i.e., number of fields in the tag
      const ordinal_type posScDim = 0;        // position in the tag, counting from 0, of the subcell dim 
      const ordinal_type posScOrd = 1;        // position in the tag, counting from 0, of the subcell ordinal
      const ordinal_type posDfOrd = 2;        // position in the tag, counting from 0, of DoF ordinal relative to the subcell
      
      constexpr ordinal_type maxCard = (Parameters::MaxOrder+1)*(Parameters::MaxOrder+2)/2; 
      ordinal_type tags[maxCard][4];

      const ordinal_type 
        edgeDof = (order - 1),
        intrDof = (order > 2 ? (order-2)*(order-1)/2 : 0);
      
      // compute bicentric coordinates (whatever it is called)
      auto bicentric = [](const double x, 
                          const double y,
                          double &xi0, double &xi1, double &xi2) {
        xi0 = 1.0 - x - y;
        xi1 = x;
        xi2 = y;
      } 

      double xi0, xi1, xi2;
      const double eps = threshold();

      ordinal_type eid[3] = {}, iid = 0;
      for (ordinal_type i=0;i<card;++i) {
        const double x = dofCoords(i,0), y = dofCoords(i,1);
        bicentric(x, y,  xi0, xi1, xi2);
        
        // vertex
             if ((std::abs(xi0) - 1.0) < eps) { // vert 0
          tags[i][0] = 0; // vertex dof
          tags[i][1] = 0; // vertex id 
          tags[i][2] = 0; // local dof id
          tags[i][3] = 1; // total vert dof
        } 
        else if ((std::abs(xi1) - 1.0) < eps) { // vert 1
          tags[i][0] = 0; // vertex dof
          tags[i][1] = 1; // vertex id 
          tags[i][2] = 0; // local dof id
          tags[i][3] = 1; // total vert dof
        } 
        else if ((std::abs(xi2) - 1.0) < eps) { // vert 2
          tags[i][0] = 0; // vertex dof
          tags[i][1] = 2; // vertex id 
          tags[i][2] = 0; // local dof id
          tags[i][3] = 1; // total vert dof
        } 
        else if (std::abs(xi2) < eps) { // edge 0
          tags[i][0] = 1; // edge dof
          tags[i][1] = 0; // edge id 
          tags[i][2] = eid[0]++; // local dof id
          tags[i][3] = edgeDof; // total vert dof
        } 
        else if (std::abs(xi0) < eps) { // edge 1
          tags[i][0] = 1; // edge dof
          tags[i][1] = 1; // edge id 
          tags[i][2] = eid[1]++; // local dof id
          tags[i][3] = edgeDof; // total vert dof
        } 
        else if (std::abs(xi1) < eps) { // edge 2
          tags[i][0] = 1; // edge dof
          tags[i][1] = 2; // edge id 
          tags[i][2] = eid[2]++; // local dof id
          tags[i][3] = edgeDof; // total vert dof
        }
        else if (std::abs(xi1) < eps) { // intr
          tags[i][0] = 2; // intr dof
          tags[i][1] = 0; // intr id 
          tags[i][2] = iid++; // local dof id
          tags[i][3] = edgeDof; // total vert dof
        }
      }

      ordinal_type_array_1d_host tagView(&tags[0][0], card*4);
      
      // Basis-independent function sets tag and enum data in tagToOrdinal_ and ordinalToTag_ arrays:
      // tags are constructed on host
      this->setOrdinalTagData(this->tagToOrdinal_,
                              this->ordinalToTag_,
                              tagView,
                              this->basisCardinality_,
                              tagSize,
                              posScDim,
                              posScOrd,
                              posDfOrd);
    }
  }

}// namespace Intrepid2
#endif
