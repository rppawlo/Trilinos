// @HEADER
// ***********************************************************************
//
//           Panzer: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//                 Copyright (2011) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
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
// Questions? Contact Roger P. Pawlowski (rppawlo@sandia.gov) and
// Eric C. Cyr (eccyr@sandia.gov)
// ***********************************************************************
// @HEADER

#ifndef PANZER_RESPONSE_SCATTER_EVALUATOR_EXTREMEVALUE_IMPL_HPP
#define PANZER_RESPONSE_SCATTER_EVALUATOR_EXTREMEVALUE_IMPL_HPP

#include <iostream>
#include <string>
#include <limits>

#include "PanzerDiscFE_config.hpp"

#include "Phalanx_Evaluator_Macros.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_DataLayout_MDALayout.hpp"

#include "Panzer_CellData.hpp"
#include "Panzer_PointRule.hpp"
#include "Panzer_CommonArrayFactories.hpp"
#include "Panzer_ResponseBase.hpp"
#include "Panzer_Dimension.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

#include "Thyra_SpmdVectorBase.hpp"
#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_DefaultMpiComm.hpp"
#include "Kokkos_ViewFactory.hpp"

namespace panzer {

template<typename EvalT, typename Traits, typename LO, typename GO>
ResponseScatterEvaluator_PointBase<EvalT,Traits,LO,GO>::
ResponseScatterEvaluator_PointBase(
  const std::string & responseName,
  MPI_Comm comm,
  const std::string & fieldName,
  const int fieldComponent,
  const Teuchos::Array<double>& point,
  const IntegrationRule & ir,
  const Teuchos::RCP<const PureBasis>& basis,
  const Teuchos::RCP<const panzer::UniqueGlobalIndexer<LO,GO> > & indexer,
  const Teuchos::RCP<PointScatterBase> & pointScatter)
  : responseName_(responseName)
  , comm_(comm)
  , fieldName_(fieldName)
  , fieldComponent_(fieldComponent)
  , point_(point)
  , basis_(basis)
  , topology_(ir.topology)
  , globalIndexer_(indexer)
  , scatterObj_(pointScatter)
  , cellIndex_(0)
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  // the field manager will allocate all of these fields
  field_ = PHX::MDField<const ScalarT,Cell,BASIS>(fieldName,basis_->functional);
  this->addDependentField(field_);

  num_basis = basis->cardinality();
  num_dim = basis->dimension();
  TEUCHOS_ASSERT(num_dim == static_cast<size_t>(point_.size()));

  basis_values_ = Kokkos::DynRankView<double,PHX::Device>(
    "basis_values", 1, num_basis, 1); // Cell, Basis, Point

  // build dummy target tag
  std::string dummyName =
    ResponseBase::buildLookupName(responseName) + " dummy target";
  RCP<PHX::DataLayout> dl_dummy = rcp(new PHX::MDALayout<panzer::Dummy>(0));
  scatterHolder_ = rcp(new PHX::Tag<ScalarT>(dummyName,dl_dummy));
  this->addEvaluatedField(*scatterHolder_);

  std::string n = "Point Response Scatter: " + responseName;
  this->setName(n);
}

template<typename EvalT, typename Traits, typename LO, typename GO>
void ResponseScatterEvaluator_PointBase<EvalT,Traits,LO,GO>::
preEvaluate(typename Traits::PreEvalData d)
{
  // extract linear object container
  responseObj_ =
    Teuchos::rcp_dynamic_cast<Response_Point<EvalT> >(
      d.gedc.getDataObject(ResponseBase::buildLookupName(responseName_)),
      true);
}

template<typename ScalarType, typename DeviceType>
class BuildProcessBoundingBox
{
public:
  typedef DeviceType  execution_space;
  typedef ScalarType  value_type[];
  const unsigned value_count;

  // Cell coordinates
  const Kokkos::View<const double***,PHX::Device> coords_;
  const Kokkos::View<const double*,PHX::Device> initial_bb_;
  const int rank_;

  typedef typename execution_space::size_type size_type;
  
  BuildProcessBoundingBox(const unsigned int in_value_count,
                          const Kokkos::View<const double***,PHX::Device> coords,
                          const Kokkos::View<const double*,PHX::Device> bb,
                          
                          const int rank) :
    value_count(in_value_count),
    coords_(coords),
    initial_bb_(bb),
    rank_(rank)
  {}
  
  KOKKOS_INLINE_FUNCTION
  void init( ScalarType bb[] ) const
  {
    for (int i=0; i < value_count; ++i)
      bb[i] = initial_bb_[i];
  }
  
  KOKKOS_INLINE_FUNCTION
  void join( volatile ScalarType dst[],
             const volatile ScalarType src[] ) const
  {
    const int num_dims = coords_.extent(2);
    for (int dim = 0; dim < num_dims; ++dim ) {
      const int offset = dim * 2;
      if (dst[offset] > src[offset]) dst[offset] = src[offset];  // min
      if (dst[offset+1] < src[offset+1]) dst[offset+1] = src[offset+1]; // max
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( const size_type& cell, ScalarType dst[] ) const
  {
    const int num_vertices = coords_.extent(1);
    const int num_dims = coords_.extent(2);

    for ( int v = 0; v < num_vertices; ++v ) {
      for ( int dim = 0; dim < num_dims; ++dim ) {
        const int offset = dim * 2;
        if ( dst[offset] > coords_(cell,v,dim) ) dst[offset] = coords_(cell,v,dim); // min
        if ( dst[offset+1] < coords_(cell,v,dim) ) dst[offset+1] = coords_(cell,v,dim); // max
        //printf("rank=%i,cell=%i,v=%i,offset=%i,dst[offset]=%g\n",rank_,static_cast<int>(cell),v,offset,dst[offset]);
      }
    }
  }
};


template<typename EvalT, typename Traits, typename LO, typename GO>
void ResponseScatterEvaluator_PointBase<EvalT,Traits,LO,GO>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field_,fm);



  Teuchos::MpiComm<int> comm(comm_);
  const int rank = comm.getRank();
  //const int nprocs = comm.getSize();  
  //printf("size=%i, rank=%i\n",nprocs,rank);

  const int dim = (*(d.worksets_))[0].cell_vertex_coordinates.extent(2);


  // Temporarily create test points
  Kokkos::View<double**,PHX::Device> pts("all points",4,dim);
  {
    auto h_pts = Kokkos::create_mirror_view(pts);
    h_pts(0,0) = -0.9;
    h_pts(0,1) = -0.9;
    h_pts(1,0) = -0.8;
    h_pts(1,1) = -0.8;
    h_pts(2,0) = 0.2;
    h_pts(2,1) = 0.2;
    h_pts(3,0) = 0.8;
    h_pts(3,1) = 0.8;
    Kokkos::deep_copy(pts,h_pts);
  }

  // 1. Coarse Global (MPI) search: create bounding boxes and export
  // potential matched to all processes.
  Kokkos::View<double*,PHX::Device> initial_bb("bounding box",dim*2);
  Kokkos::View<double*,PHX::Device> bb("bounding box",dim*2);
  for (int d=0; d < dim; ++d) {
    const int index = d * 2;
    initial_bb(index) = std::numeric_limits<double>::max();
    initial_bb(index+1) = std::numeric_limits<double>::lowest();
  }

  Kokkos::View<double***,PHX::Device> cell_bbs;

  for (const auto& workset : *(d.worksets_)) {
    const auto& coordinates = workset.cell_vertex_coordinates.get_static_view();
    Kokkos::RangePolicy<PHX::Device> policy(0,coordinates.extent(0));
    Kokkos::parallel_reduce(policy,BuildProcessBoundingBox<double,PHX::Device>((2*dim),coordinates,initial_bb,rank),bb);
    Kokkos::fence();
    Kokkos::deep_copy(initial_bb,bb);
    Kokkos::fence();
  }

  // for (const auto& workset : *(d.worksets_)) {
  //   const auto& coordinates = workset.cell_vertex_coordinates.get_static_view();
  //   for (int cell=0; cell < coordinates.extent(0); ++cell) {
  //     for (int vertex=0; vertex < coordinates.extent(1); ++vertex) {
  //       for (int dim=0; dim < coordinates.extent(2); ++dim) {
  //         const int min_index = dim*2;
  //         const int max_index = dim*2+1;
  //         if (bb(min_index) > coordinates(cell,vertex,dim)) bb(min_index) = coordinates(cell,vertex,dim);
  //         if (bb(max_index) < coordinates(cell,vertex,dim)) bb(max_index) = coordinates(cell,vertex,dim);
  //       }
  //     }
  //   }
  // }  

  if (std::is_same<EvalT,panzer::Traits::Residual>::value) {
    if (dim == 3)
      printf("global bounding box rank=%i, x_min=%g,x_max=%g,y_min=%g,y_max=%g,z_min=%g,z_max=%g\n",rank,bb(0),bb(1),bb(2),bb(3),bb(4),bb(5));
    else
      printf("global bounding box rank=%i, x_min=%g,x_max=%g,y_min=%g,y_max=%g\n",rank,bb(0),bb(1),bb(2),bb(3));
  }

  std::vector<int> candidate_pts;
  for (int pt=0; pt < static_cast<int>(pts.extent(0)); ++pt) {
    bool is_candidate = true;
    for (int d=0; d < dim; ++d) {
      const int min_index = d*2;
      const int max_index = d*2+1;
      if ( (pts(pt,d) < bb(min_index)) || (pts(pt,d) > bb(max_index)) ) {
        is_candidate = false;
        break;
      }
    }
    if (is_candidate) candidate_pts.push_back(pt);
  }
  
  if (std::is_same<EvalT,panzer::Traits::Residual>::value) {
    for (const auto& p : candidate_pts) {
      if (dim ==2)
        printf("global search candidate pt: rank=%i, %i(%g,%g)\n",rank,p,pts(p,0),pts(p,1));
      else
        printf("global search candidate pt: rank=%i, %i(%g,%g,%g)\n",rank,p,pts(p,0),pts(p,1),pts(p,2));
    }
  }

  // 2. Coarse Local search: this is on host for now (assuming small
  // number of points). This is very inefficient. It should use a
  // kd-tree. Eventually use DTK to improve efficiency. Map: key is
  // point the index, value is a vector of matching cell candidates: each cell candidate is a pair of ints, the first is the workset
  // index and the second the local cell index

  std::map<int,std::vector<std::pair<int,int>>> candidate_cells;
  for (const auto& workset : *(d.worksets_)) {
    const auto& coordinates = workset.cell_vertex_coordinates.get_static_view();
    for (int cell=0; cell < coordinates.extent(0); ++cell) {
      
      // Form a local cell bounding box and check for possible inclusion
      Kokkos::View<double*[2],PHX::Device> cell_bb("cell_bb",dim);
      for (int d=0; d < dim; ++d) {
        cell_bb(d,0) = std::numeric_limits<double>::max();
        cell_bb(d,1) = std::numeric_limits<double>::lowest();
      }
      for (int vertex=0; vertex < coordinates.extent(1); ++vertex) {
        for (int d=0; d < dim; ++d) {
          const int min_index = d*2;
          const int max_index = d*2+1;
          if (cell_bb(d,0) > coordinates(cell,vertex,d)) cell_bb(d,0) = coordinates(cell,vertex,d);
          if (cell_bb(d,1) < coordinates(cell,vertex,d)) cell_bb(d,1) = coordinates(cell,vertex,d);
        }
      }

      // Very inefficient
      for (int pt=0; pt < candidate_pts.size(); ++pt) {
        bool in_cell_bb = true;
        for (int d=0; d < dim; ++d) {
          if ( (pts(candidate_pts[pt],d) < cell_bb(d,0)) || (pts(candidate_pts[pt],d) > cell_bb(d,1)) ) {
            in_cell_bb = false;
            break;
          }
        }
        if (in_cell_bb) {
          
        }
      }

    }
  }

  // 3. Fine Local search

  // 4. Reconcile and remove duplicate identifications (node on
  // element boundary could be found by 2 elements).


}

template<typename EvalT, typename Traits, typename LO, typename GO>
bool ResponseScatterEvaluator_PointBase<EvalT,Traits,LO,GO>::
computeBasisValues(typename Traits::EvalData d)
{
  typedef Intrepid2::CellTools<PHX::exec_space> CTD;
  typedef Intrepid2::FunctionSpaceTools<PHX::exec_space> FST;

  const int num_points = 1; // Always a single point in this evaluator!
  Kokkos::DynRankView<int,PHX::Device> inCell("inCell", this->wda(d).cell_vertex_coordinates.extent_int(0), num_points);
  Kokkos::DynRankView<double,PHX::Device> physical_points_cell("physical_points_cell", this->wda(d).cell_vertex_coordinates.extent_int(0), num_points, num_dim);
  for (panzer::index_t cell(0); cell < d.num_cells; ++cell)
    for (size_t dim=0; dim<num_dim; ++dim)
      physical_points_cell(cell,0,dim) = point_[dim];

  const double tol = 1.0e-12;
  CTD::checkPointwiseInclusion(inCell,
                               physical_points_cell,
                               this->wda(d).cell_vertex_coordinates.get_view(),
                               *topology_,
                               tol);

  // Find which cell contains our point
  cellIndex_ = -1;
  bool havePoint = false;
  for (index_t cell=0; cell<static_cast<int>(d.num_cells); ++cell) {
    // CTD::checkPointwiseInclusion(inCell,
    //                              physical_points_cell,
    //                              this->wda(d).cell_vertex_coordinates,
    //                              *topology_,
    //                              cell);

    if (inCell(cell,0) == 1) {
      cellIndex_ = cell;
      havePoint = true;
      break;
    }
  }

  // If no cell does, we're done
  if (!havePoint) {
    return false;
  }

  // Map point to reference frame
  const size_t num_vertex = this->wda(d).cell_vertex_coordinates.dimension_1();
  Kokkos::DynRankView<double,PHX::Device> cell_coords(
    "cell_coords", 1, num_vertex, num_dim); // Cell, Basis, Dim
  for (size_t i=0; i<num_vertex; ++i) {
    for (size_t j=0; j<num_dim; ++j) {
      cell_coords(0,i,j) = this->wda(d).cell_vertex_coordinates(cellIndex_,i,j);
    }
  }
  Kokkos::DynRankView<double,PHX::Device> physical_points(
    "physical_points", 1, 1, num_dim); // Cell, Point, Dim
  for (size_t i=0; i<num_dim; ++i)
    physical_points(0,0,i) = physical_points_cell(0,0,i);
  Kokkos::DynRankView<double,PHX::Device> reference_points(
     "reference_points", 1, 1, num_dim); // Cell, Point, Dim
  CTD::mapToReferenceFrame(reference_points, physical_points, cell_coords,
                           *topology_);
  Kokkos::DynRankView<double,PHX::Device> reference_points_cell(
    "reference_points_cell", 1, num_dim); // Point, Dim
  for (size_t i=0; i<num_dim; ++i)
    reference_points_cell(0,i) = reference_points(0,0,i);

  // Compute basis functions at point
  if (basis_->getElementSpace() == PureBasis::CONST ||
      basis_->getElementSpace() == PureBasis::HGRAD) {

    // Evaluate basis at reference values
    Kokkos::DynRankView<double,PHX::Device>
      ref_basis_values("ref_basis_values", num_basis, 1); // Basis, Point
    basis_->getIntrepid2Basis()->getValues(ref_basis_values,
                                           reference_points_cell,
                                           Intrepid2::OPERATOR_VALUE);

    // Apply transformation to physical frame
    FST::HGRADtransformVALUE<double>(basis_values_, ref_basis_values);

  }
  else if (basis_->getElementSpace() == PureBasis::HCURL ||
           basis_->getElementSpace() == PureBasis::HDIV) {

    // Evaluate basis at reference values
    Kokkos::DynRankView<double,PHX::Device> ref_basis_values(
      "ref_basis_values", num_basis, 1, num_dim); // Basis, Point, Dim
    basis_->getIntrepid2Basis()->getValues(ref_basis_values,
                                           reference_points_cell,
                                           Intrepid2::OPERATOR_VALUE);

    // Apply transformation to physical frame
    Kokkos::DynRankView<double,PHX::Device> jac
      ("jac", 1, 1, num_dim, num_dim); // Cell, Point, Dim, Dim
    CTD::setJacobian(jac, reference_points, cell_coords, *topology_);
    Kokkos::DynRankView<double,PHX::Device> basis_values_vec(
      "basis_values_vec", 1, num_basis, 1, num_dim); // Cell, Basis, Point, Dim
    if (basis_->getElementSpace() == PureBasis::HCURL) {
      Kokkos::DynRankView<double,PHX::Device> jac_inv(
        "jac_inv", 1, 1, num_dim, num_dim); // Cell, Point, Dim, Dim
      CTD::setJacobianInv(jac_inv, jac);
      FST::HCURLtransformVALUE<double>(basis_values_vec, jac_inv,
                                       ref_basis_values);
    }
    else {
      Kokkos::DynRankView<double,PHX::Device> jac_det(
        "jac_det", 1, 1); // Cell Point
      CTD::setJacobianDet(jac_det, jac);
      FST::HDIVtransformVALUE<double>(basis_values_vec, jac, jac_det,
                                      ref_basis_values);
    }

    // Compute element orientations
    std::vector<double> orientation;
    globalIndexer_->getElementOrientation(cellIndex_, orientation);
    std::string blockId = this->wda(d).block_id;
    int fieldNum = globalIndexer_->getFieldNum(fieldName_);
    const std::vector<int> & elmtOffset =
      globalIndexer_->getGIDFieldOffsets(blockId,fieldNum);

    // Extract component of basis
    for (size_t i=0; i<num_basis; ++i) {
      int offset = elmtOffset[i];
      basis_values_(0,i,0) =
        orientation[offset] * basis_values_vec(0,i,0,fieldComponent_);
    }

  }

  return true;
}

template<typename EvalT, typename Traits, typename LO, typename GO>
void ResponseScatterEvaluator_PointBase<EvalT,Traits,LO,GO>::
evaluateFields(typename Traits::EvalData d)
{
  // Compute basis values at point
  const bool havePoint = computeBasisValues(d);

  if (!havePoint)
    return;

  // Get field coefficients for cell
  Kokkos::DynRankView<ScalarT,PHX::Device> field_coeffs =
    Kokkos::createDynRankView(field_.get_static_view(), "field_val",
                              1, num_basis); // Cell, Basis
  for (size_t i=0; i<num_basis; ++i)
    field_coeffs(0,i) = field_(cellIndex_,i);

  // Evaluate FE interpolant at point
  Kokkos::DynRankView<ScalarT,PHX::Device> field_val =
    Kokkos::createDynRankView(field_coeffs, "field_val", 1, 1); // Cell, Point
  Intrepid2::FunctionSpaceTools<PHX::exec_space>::evaluate(
    field_val, field_coeffs, basis_values_);
  responseObj_->value = field_val(0,0);
  responseObj_->have_point = true;
}

template <typename LO, typename GO>
void ResponseScatterEvaluator_Point<panzer::Traits::Jacobian,panzer::Traits,LO,GO>::
evaluateFields(panzer::Traits::EvalData d)
{
  using Teuchos::RCP;
  using Teuchos::rcp_dynamic_cast;
  using Thyra::SpmdVectorBase;

  TEUCHOS_ASSERT(this->scatterObj_!=Teuchos::null);

  Base::evaluateFields(d);

  // grab local data for inputing
  Teuchos::ArrayRCP<double> local_dgdx;
  RCP<SpmdVectorBase<double> > dgdx =
    rcp_dynamic_cast<SpmdVectorBase<double> >(this->responseObj_->getGhostedVector());
  dgdx->getNonconstLocalData(ptrFromRef(local_dgdx));
  TEUCHOS_ASSERT(!local_dgdx.is_null());

  this->scatterObj_->scatterDerivative(this->responseObj_->value,
                                       this->cellIndex_,
                                       this->responseObj_->have_point,
                                       d,this->wda,
                                       local_dgdx);
}

}

#endif
